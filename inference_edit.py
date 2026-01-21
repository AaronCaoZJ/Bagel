# %%
import torch
print(torch.cuda.is_available())

# %%
import os
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
# model_path = "/root/Miko_share/Kokoro/models/BAGEL-7B-MoT/"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
model_path = "/home/zhijun/Code/Bagel/models/BAGEL-7B-MoT/"

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

vae_model = vae_model.to("cuda:0", dtype=torch.bfloat16).eval()

_original_vae_encode = vae_model.encode
def _safe_vae_encode(x):
    device = next(vae_model.parameters()).device
    if x.device != device:
        x = x.to(device)
    return _original_vae_encode(x)
vae_model.encode = _safe_vae_encode

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

# %%
# print(model)

# %% [markdown]
# ## Inferencer Preparing 

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
# ## TaylorSeer/SpeCa Support for Inference Acceleration

# %%
# inference_hyper=dict(
#     cfg_text_scale=4.0,
#     cfg_img_scale=2.0,
#     cfg_interval=[0.0, 1.0],
#     timestep_shift=3.0,
#     num_timesteps=50,
#     cfg_renorm_min=0.0,
#     cfg_renorm_type="text_channel",
#     enable_taylorseer=True,  # Enable TaylorSeer
#     enable_speca=True,  # Enable SpeCa
#     taylor_first_enhance=6,  # TaylorSeer parameter
#     taylor_max_order=6,
#     speca_base_threshold=0.5,   # SpeCa parameter
#     speca_decay_rate=0.05,
#     speca_min_taylor_steps=2,
#     speca_max_taylor_steps=6,
#     speca_error_metric="relative_l2"
# )

inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=41,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    enable_taylorseer=True,  # Enable TaylorSeer
    taylor_first_enhance=10,  # TaylorSeer parameter
    taylor_max_order=6,
    taylor_fresh_threshold=3
)

# inference_hyper=dict(
#     cfg_text_scale=4.0,
#     cfg_img_scale=2.0,
#     cfg_interval=[0.0, 1.0],
#     timestep_shift=3.0,
#     num_timesteps=50,
#     cfg_renorm_min=0.0,
#     cfg_renorm_type="text_channel",
# )
# %%
# ...existing code...
# image = Image.open('/root/Miko_share/Bagel/test_images/__castorice_honkai_and_1_more_drawn_by_houkisei__c767800bb2e5210319e753aaebc0855c.jpg')
image = Image.open('/test_images/20251222-113154.jpg')
prompt = '人物在沙滩上玩水，浪花打湿了裙摆.'

print(prompt)
# Save generated image to outputs/ with inference time in filename
import time
from datetime import datetime
from pathlib import Path

start_time = time.time()  # 记录推理开始时间

output_dict = inferencer(image=image, text=prompt, **inference_hyper)

end_time = time.time()    # 记录推理结束时间
inference_duration = end_time - start_time
print(f"Inference time: {inference_duration:.2f} seconds")

img = output_dict.get('image')
if img is None:
    raise RuntimeError("No image found in output_dict")

# If a list/batch returned, take the first item
if isinstance(img, (list, tuple)):
    img = img[0]

# Convert torch.Tensor -> PIL.Image
if isinstance(img, torch.Tensor):
    t = img.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.shape[0] in (1, 3):
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = t.numpy()
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        maxv = arr.max() if arr.size else 1.0
        if maxv <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
        else:
            arr = np.clip(arr, 0, 255).astype("uint8")
    elif arr.dtype != "uint8":
        arr = arr.astype("uint8")
    pil_img = Image.fromarray(arr)
elif isinstance(img, Image.Image):
    pil_img = img
else:
    try:
        arr = np.array(img)
        if arr.dtype != "uint8":
            maxv = arr.max() if arr.size else 1.0
            if maxv <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
            else:
                arr = np.clip(arr, 0, 255).astype("uint8")
        pil_img = Image.fromarray(arr)
    except Exception as e:
        raise RuntimeError(f"Cannot convert image to PIL: {e}")

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"edit_{inference_duration:.2f}.png"
pil_img.save(out_path)
print(f"Saved image to: {out_path}")
