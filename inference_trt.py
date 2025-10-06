# %%
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

notebook_dir = os.getcwd()
print("Current notebook dir:", notebook_dir)

from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file

# %% [markdown]
# ## Model Initialization

# %%
model_path = "models/BAGEL-7B-MoT/"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# %% [markdown]
# ## Model Loading and Multi GPU Infernece Preparing

# %%
max_mem_per_gpu = "31GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')

# import inspect

# device = "cuda" if torch.cuda.is_available() else "cpu"

# vit_module = model.vit_model.to(device).eval()
# llm_module = model.language_model.to(device).eval()
# print("Bagel.forward:", inspect.signature(model.forward))
# print("ViT.forward:", inspect.signature(vit_module.forward))
# print("LLM.forward:", inspect.signature(llm_module.forward))
# print("LLM.forward_inference:", inspect.signature(llm_module.forward_inference))

# %% [markdown]
# ## Export to ONNX

# %%
import os
import torch
import torch.onnx as onnx

os.makedirs("onnx", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

B = 1
T = 64

# 1) Dummy for LLM (Qwen2ForCausalLM)
vocab_size = tokenizer.vocab_size

dummy_packed_query_sequence = torch.randint(0, vocab_size, (B, T), dtype=torch.long, device=device)
dummy_query_lens = torch.full((B,), T, dtype=torch.long, device=device)  # 或 dummy_attention_mask.sum(dim=1)
dummy_packed_query_position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
dummy_packed_query_indexes = torch.zeros((B, T), dtype=torch.long, device=device)
dummy_past_key_values = None
dummy_key_values_lens = None
dummy_packed_key_value_indexes = torch.zeros((B, T), dtype=torch.long, device=device)
dummy_update_past_key_values = True
dummy_is_causal = True
dummy_mode = "und"
dummy_packed_vae_token_indexes = None
dummy_packed_text_indexes = None

# Build a clean, single-device LLM instance and load weights from the checkpoint
# ...existing code...
llm_for_export = Qwen2ForCausalLM(llm_config).eval().to(device, dtype=torch.float16)

ckpt_path = os.path.join(model_path, "ema.safetensors")
print(f"Loading weights from {ckpt_path} ...")
sd = load_file(ckpt_path)  # CPU tensors

# Filter the LLM sub-keys and strip the "language_model." prefix
llm_state = {}
prefix = "language_model."
for k, v in sd.items():
    if k.startswith(prefix):
        new_k = k[len(prefix):]
        llm_state[new_k] = v.to(device=device, dtype=torch.float16)

missing, unexpected = llm_for_export.load_state_dict(llm_state, strict=False)
print(f"Loaded LLM weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

# 导出 LLM 子模块（Qwen2ForCausalLM）
llm_onnx = "onnx/qwen2_llm.onnx"
torch.onnx.export(
    llm_for_export,
    (
        dummy_packed_query_sequence,
        dummy_query_lens, 
        dummy_packed_query_position_ids,
        dummy_packed_query_indexes,
        dummy_past_key_values,
        dummy_key_values_lens,
        dummy_packed_key_value_indexes,
        dummy_update_past_key_values,
        dummy_is_causal,
        dummy_mode,
        dummy_packed_vae_token_indexes,
        dummy_packed_text_indexes,
    ),
    llm_onnx,
    input_names=[
        "packed_query_sequence", 
        "query_lens", 
        "packed_query_position_ids", 
        "packed_query_indexes",
        "past_key_values",
        "key_values_lens",
        "packed_key_value_indexes",
        "update_past_key_values",
        "is_causal",
        "mode",
        "packed_vae_token_indexes",
        "packed_text_indexes",
    ],
    output_names=["logits"],
    dynamic_axes={
        "packed_query_sequence": {0: "batch", 1: "sequence"},
        "query_lens": {0: "batch"},
        "packed_query_position_ids": {0: "batch", 1: "sequence"},
        "packed_query_indexes": {0: "batch", 1: "sequence"},
        "packed_key_value_indexes": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"LLM ONNX saved to {llm_onnx}")

# %%
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 2) Dummy for ViT (SigLIP): 980x980, 14 的倍数
vit_H = vit_W = 980
patch_size = 14
N = (vit_H // patch_size) * (vit_W // patch_size)

dummy_packed_pixel_values = torch.randn(B, 3, vit_H, vit_W, dtype=torch.float32, device=device)
dummy_packed_flattened_position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
dummy_cu_seqlens = torch.tensor([0, N], dtype=torch.int32, device=device)
dummy_max_seqlen = N

vit_for_export = SiglipVisionModel(vit_config).eval().to(device, dtype=torch.float16)

ckpt_path = os.path.join(model_path, "ema.safetensors")
print(f"Loading weights from {ckpt_path} ...")
sd = load_file(ckpt_path)  # CPU tensors

vit_state = {}
vit_prefix = "vit_model."
for k, v in sd.items():
    if k.startswith(vit_prefix):
        new_k = k[len(vit_prefix):]
        vit_state[new_k] = v.to(device=device, dtype=torch.float16)

if not vit_state:
    print("Fallback: try prefix 'vision_model.'")
    for k, v in sd.items():
        if k.startswith("vision_model."):
            vit_state[k] = v.to(device=device, dtype=torch.float16)

# 把子模块移到单设备上（避免 accelerate 的分片/CPU offload 干扰导出）
vit_module = model.vit_model.to(device).eval()

# 导出 ViT 子模块
vit_onnx = "onnx/siglip_vit.onnx"
torch.onnx.export(
    vit_module,
    (dummy_packed_pixel_values, dummy_packed_flattened_position_ids, dummy_cu_seqlens, dummy_max_seqlen),
    vit_onnx,
    input_names=["packed_pixel_values", "packed_flattened_position_ids", "cu_seqlens", "max_seqlen"],
    output_names=["vit_features"],
    dynamic_axes={
        "packed_pixel_values": {0: "batch", 2: "height", 3: "width"},
        "packed_flattened_position_ids": {0: "batch", 1: "tokens"},
        "cu_seqlens": {0: "batch_plus_one"},
        "vit_features": {0: "batch", 1: "tokens"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"ViT ONNX saved to {vit_onnx}")

# %% [markdown]
# ## Convert to TensorRT

# %%


# %%
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

onnx_path = "resnet50_pytorch.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model!")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
# config.set_flag(trt.BuilderFlag.FP16)  # FP16

serialized_engine = builder.build_serialized_network(network, config)
assert serialized_engine is not None, "Failed to build TensorRT engine!"

with open("resnet_engine_pytorch.trt", "wb") as f:
    f.write(serialized_engine)

# %% [markdown]
# ## Inferencer Preparing 

# %%
import tensorrt as trt
import numpy as np
import torch

def trt_infer(engine_path, inputs_dict):
    # 伪代码，实际需用 trt.Runtime, trt.IExecutionContext, cuda bindings 等
    # 返回 numpy 数组或 torch.Tensor
    pass

class TRTLLMWrapper(torch.nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.engine_path = engine_path
        # 加载 TensorRT 引擎等

    def forward(self, input_ids, attention_mask, packed_query_position_ids, packed_query_indexes):
        # 调用 TensorRT 推理
        # 返回 logits (torch.Tensor)
        logits = trt_infer(self.engine_path, {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "packed_query_position_ids": packed_query_position_ids.cpu().numpy(),
            "packed_query_indexes": packed_query_indexes.cpu().numpy(),
        })
        return torch.from_numpy(logits).to(input_ids.device)


class TRTViTWrapper(torch.nn.Module):
    def __init__(self, engine_path):
        super().__init__()
        self.engine_path = engine_path

    def forward(self, pixel_values):
        vit_features = trt_infer(self.engine_path, {
            "pixel_values": pixel_values.cpu().numpy(),
        })
        return torch.from_numpy(vit_features).to(pixel_values.device)

# %%
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Image Generation

# %%
inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)

# %%
prompt = "A teenage boy in a deep-blue ninja outfit is leaping across Edo-style rooftops at sunset, facing forward with a clear front view of his face. Warm golden sunlight glows on the red tiled roofs, and a flock of crows flies in the distance. Red paper lanterns hang below in the narrow streets. The scene is cel-shaded anime style, with strong motion, layered silhouettes, and a cinematic sense of speed and atmosphere typical of Japanese TV animation."

print(prompt)
print('-' * 10)
output_dict = inferencer(text=prompt, **inference_hyper)
display(output_dict['image'])

# %% [markdown]
# ## Editing

# %%
inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

# %%
image = Image.open('test_images/__castorice_honkai_and_1_more_drawn_by_houkisei__c767800bb2e5210319e753aaebc0855c.jpg')
prompt = 'Take off all her clothes, exposing her private parts.'

display(image)
print(prompt)
print('-'*10)
output_dict = inferencer(image=image, text=prompt, **inference_hyper)
display(output_dict['image'])

# %% [markdown]
# ## Understanding

# %%
inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)

# %%
image = Image.open('test_images/meme.jpg')
prompt = "Can someone explain what’s funny about this meme??"

display(image)
print(prompt)
print('-'*10)
output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
print(output_dict['text'])


