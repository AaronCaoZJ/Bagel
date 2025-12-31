# new add
import time
import psutil
import platform  
import atexit   

pynvml_available = False
if platform.system() == "Linux" or platform.system() == "Windows":
    try:
        from pynvml import *
        nvmlInit()
        pynvml_available = True
        print("pynvml (NVIDIA GPU monitoring library) initialized successfully.")
        
        def shutdown_pynvml():
            print("Shutting down pynvml...")
            nvmlShutdown()
        atexit.register(shutdown_pynvml) # register close pynvml when it quit
        
    except Exception as e:
        print(f"Warning: pynvml could not be initialized. Detailed GPU stats via pynvml will not be available. Error: {e}")
        if "NVML Shared Library Not Found" in str(e):
            print("pynvml error hint: NVML shared library not found. If you have an NVIDIA GPU and drivers, ensure the library is accessible.")
        elif "Driver Not Loaded" in str(e):
            print("pynvml error hint: NVIDIA driver is not loaded. Please check your GPU driver installation.")

import gradio as gr
import numpy as np
import os
import torch

# 需要安装对应版本的torchao https://github.com/pytorch/ao/issues/2919
from torchao.quantization import quantize_
from torchao.quantization import (
    float8_dynamic_activation_float8_weight, float8_weight_only,
    int8_weight_only, int4_weight_only, int8_dynamic_activation_int8_weight
)

import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

from scripts.export_precision_report import export_precision_report
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
args = parser.parse_args()

# Model Initialization
model_path = args.model_path #Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT to models/BAGEL-7B-MoT

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64, # 96
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


print("Starting model loading and device map configuration...")

# --- ram & vram helps functions ---
def get_gpu_memory_stats_pynvml(device_id=0):
    if not pynvml_available:
        return f"GPU-{device_id} (pynvml): Not available."
    try:
        handle = nvmlDeviceGetHandleByIndex(device_id)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / (1024**3)
        used_gb = info.used / (1024**3)
        # free_gb = info.free / (1024**3) # It can be calculated by the sum already used
        return f"GPU-{device_id} (pynvml): Total: {total_gb:.2f} GB, Used (Overall): {used_gb:.2f} GB"
    except NVMLError as e:
        return f"GPU-{device_id} (pynvml) Error: {e}"

def get_gpu_memory_stats_pytorch(device_id=0):
    if not torch.cuda.is_available():
        return "PyTorch: CUDA not available."
    if device_id < 0 or device_id >= torch.cuda.device_count():
        return f"PyTorch GPU-{device_id}: Invalid device ID."
    
    allocated_gb = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device_id) / (1024**3) # PyTorch Reserved Total vram
    
    # try gets pynvml info 
    total_capacity_str_pt = ""
    if pynvml_available:
        try:
            handle = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(handle)
            total_gb_pt = info.total / (1024**3)
            total_capacity_str_pt = f"Total Capacity: {total_gb_pt:.2f} GB, "
        except NVMLError:
            pass # If the acquisition fails, the total capacity will not be displayed

    return (f"PyTorch GPU-{device_id}: {total_capacity_str_pt}"
            f"Allocated: {allocated_gb:.2f} GB, Reserved: {reserved_gb:.2f} GB")

def get_system_ram_stats():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    used_gb = mem.used / (1024**3)
    percent_used = mem.percent
    return (f"System RAM: Total: {total_gb:.2f} GB, Available: {available_gb:.2f} GB, "
            f"Used (Overall): {used_gb:.2f} GB ({percent_used}%)")

def get_process_ram_stats():
    process = psutil.Process(os.getpid()) # get the current Python process
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)  # Resident Set Size (Actual physical memory usage)
    return f"App Process RAM (RSS): {rss_gb:.2f} GB"

def get_all_memory_stats_for_gradio_display():
    """Prepare the string of memory/video memory statistics for Gradio display"""
    stats_lines = []
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        stats_lines.append("**GPU VRAM Usage:**")
        for i in range(torch.cuda.device_count()):
            stats_lines.append(get_gpu_memory_stats_pynvml(i))
            stats_lines.append(get_gpu_memory_stats_pytorch(i))
            if i < torch.cuda.device_count() - 1: # If there are multiple GPUs, add a separator
                 stats_lines.append("---")
    else:
        stats_lines.append("**GPU VRAM Usage:** CUDA not available or no GPUs found.")
    
    stats_lines.append("\n**CPU RAM Usage:**")
    stats_lines.append(get_system_ram_stats())
    stats_lines.append(get_process_ram_stats())
    
    return "\n".join(s for s in stats_lines if s)
# --- ram & vram helps functions end ---


# ram & vram setting
cpu_mem_for_offload = "0GiB"
gpu_mem_per_device = "31GiB" # TOCHANGE

max_memory_config = {i: gpu_mem_per_device for i in range(torch.cuda.device_count())}
if torch.cuda.device_count() == 0: # If there is no GPU, a basic configuration is also required
    max_memory_config["cpu"] = cpu_mem_for_offload
else:
    max_memory_config["cpu"] = cpu_mem_for_offload # Add a budget for the CPU

print(f"Using max_memory_config: {max_memory_config}")

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory_config, # Use the configuration that includes the CPU memory budget
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print("Device map after infer_auto_device_map (with CPU budget):")
for k, v_map in device_map.items(): # Check info
    print(f"  {k}: {v_map}")

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

# already have same_device_modules 
if torch.cuda.device_count() > 0:
    first_device_key = same_device_modules[0]
    default_target_device = "cuda:0" # The default target is the first GPU
    first_module_target_device = device_map.get(first_device_key, default_target_device)
    
    print(f"Target device for same_device_modules (based on {first_device_key}): {first_module_target_device}")

    for k_module in same_device_modules:
        if k_module in device_map:
            if device_map[k_module] != first_module_target_device:
                print(f"  Moving {k_module} from {device_map[k_module]} to {first_module_target_device} (same_device_modules)")
                device_map[k_module] = first_module_target_device
        else: # If the module is not in the automatically generated map but you want it to be on a specific device
            print(f"  Assigning {k_module} (from same_device_modules) to {first_module_target_device} as it was not in initial map.")
            device_map[k_module] = first_module_target_device 
elif torch.cuda.device_count() == 0 and "cpu" in max_memory_config: # without GPU
    print("No CUDA devices found. Assigning same_device_modules to CPU.")
    for k_module in same_device_modules:
        device_map[k_module] = "cpu"


print("Device map after same_device_modules logic:")
for k, v_map in device_map.items():
    print(f"  {k}: {v_map}")

# make sure no 'disk'  (backup)
keys_to_change_to_cpu = []
for module_name, device_assignment in device_map.items():
    if device_assignment == "disk":
        keys_to_change_to_cpu.append(module_name)

if keys_to_change_to_cpu:
    print(f"Manually changing the following layers from 'disk' to 'cpu': {keys_to_change_to_cpu}")
    for module_name in keys_to_change_to_cpu:
        device_map[module_name] = "cpu"
    print("Final device_map before loading checkpoint (after disk override):")
    for k, v_map in device_map.items():
        print(f"  {k}: {v_map}")
else:
    print("No layers assigned to 'disk' by infer_auto_device_map, or they were already handled. Final device_map is as above.")


# adjust layers more clearly&detail to GPU
# make sure，The device_map only contains GPU indexes (such as 0) or 'cpu'.
print("\nStarting custom device_map modifications to maximize GPU utilization...")
print("Device map state BEFORE custom modifications:")
for k_map_item, v_map_item in device_map.items():
    print(f"  {k_map_item}: {v_map_item}")


# -- Key tuning parameters Start --
# 1. Try to move more LLM Transformer layers (layers 11 to 27) to GPU 0
# These layers are currently on the CPU. There are a total of 17 such layers (ranging from 11 to 27).
# You can set the number of LLM layers that you wish to move from the CPU to GPU 0.
# Please start the experiment with a smaller value, such as 5 or 8, and then gradually increase it.
# If set to 17, all layers 11-27 will be attempted to be moved.

NUM_ADDITIONAL_LLM_LAYERS_TO_GPU = 5  # <--- 5 fit for 24GB Vram for TEST layers(like: 5, 8, 10, 12, 15, 17)

# 2. Whether to attempt to move the 'norm' and 'lm_head' layers of LLM to GPU 0 (if they are on the CPU)
# It is usually recommended to place them on the same device as the last layer of the LLM.

TRY_MOVE_LLM_NORM_HEAD_TO_GPU = True # <--- Default True, Turn to False,If you don't want to remove

# 3. (Optional) Whether to attempt to move 'vit_model' to GPU 0 (if it is on the CPU)
# This is usually considered only after the LLM layer has been successfully moved to the GPU 
# And there is still a considerable amount of video memory left.
    
TRY_MOVE_VIT_MODEL_TO_GPU = False   # <--- Default False , can be test

# --- Adjust end ---


# run LLM layers move
moved_llm_layers_count = 0
if NUM_ADDITIONAL_LLM_LAYERS_TO_GPU > 0:
    print(f"\nAttempting to move up to {NUM_ADDITIONAL_LLM_LAYERS_TO_GPU} LLM layers (11 to {10 + NUM_ADDITIONAL_LLM_LAYERS_TO_GPU}) to GPU 0...")
    for i in range(NUM_ADDITIONAL_LLM_LAYERS_TO_GPU):
        layer_idx = 11 + i  # From layer 11 to start
        if layer_idx > 27:  # language_model.model.layers Max to 27
            print(f"  Reached max layer index (27). Stopped LLM layer promotion.")
            break
        layer_name = f"language_model.model.layers.{layer_idx}"
        
        if device_map.get(layer_name) == 'cpu':
            print(f"  Promoting LLM layer '{layer_name}' from 'cpu' to GPU 0.")
            device_map[layer_name] = 0  # move to GPU 0
            moved_llm_layers_count += 1
        elif layer_name in device_map:
            print(f"  LLM Layer '{layer_name}' is already on device '{device_map[layer_name]}'. Skipping promotion.")
        else:
            print(f"  Warning: LLM Layer '{layer_name}' not found in device_map. Cannot promote.")
    print(f"Successfully promoted {moved_llm_layers_count} LLM layers to GPU 0.")
else:
    print("\nSkipping promotion of additional LLM layers based on NUM_ADDITIONAL_LLM_LAYERS_TO_GPU setting.")

# run LLM norm  & lm_head move
if TRY_MOVE_LLM_NORM_HEAD_TO_GPU:
    print("\nAttempting to move LLM 'norm' and 'lm_head' to GPU 0 (if on CPU)...")
    llm_aux_modules = ["language_model.model.norm", "language_model.model.lm_head"]
    # you can choose other modules with LLM，eg: rotary_emb, norm_moe_gen，if there are on the CPU
    # llm_aux_modules.append("language_model.model.rotary_emb")
    # llm_aux_modules.append("language_model.model.norm_moe_gen")

    for module_name in llm_aux_modules:
        if device_map.get(module_name) == 'cpu':
            print(f"  Promoting '{module_name}' from 'cpu' to GPU 0.")
            device_map[module_name] = 0
        elif module_name in device_map:
            print(f"  Module '{module_name}' is already on device '{device_map[module_name]}'. Skipping promotion.")
        else:
            print(f"  Warning: Module '{module_name}' not found in device_map. Cannot promote.")
else:
    print("\nSkipping promotion of LLM 'norm' and 'lm_head' based on TRY_MOVE_LLM_NORM_HEAD_TO_GPU setting.")

# （option）run vit_model move
if TRY_MOVE_VIT_MODEL_TO_GPU:
    print("\nAttempting to move 'vit_model' to GPU 0 (if on CPU)...")
    vit_module_name = "vit_model"
    if device_map.get(vit_module_name) == 'cpu':
        print(f"  Promoting '{vit_module_name}' from 'cpu' to GPU 0.")
        device_map[vit_module_name] = 0
    elif vit_module_name in device_map:
        print(f"  Module '{vit_module_name}' is already on device '{device_map[vit_module_name]}'. Skipping promotion.")
    else:
        print(f"  Warning: Module '{vit_module_name}' not found in device_map. Cannot promote.")
else:
    print("\nSkipping promotion of 'vit_model' based on TRY_MOVE_VIT_MODEL_TO_GPU setting.")


print("\nFinal device_map after all custom modifications:")
for k_map_item, v_map_item in device_map.items():
    print(f"  {k_map_item}: {v_map_item}")
print("--- End of custom device_map modifications ---")

# adjust gpu vram end

# Quantization process
print("[QUANT-LOAD-BF16] Loading BF16 model for conversion...")
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"), 
    device_map=device_map,
    offload_buffers=False, # 禁用缓冲区卸载
    offload_folder="offload",
    dtype=torch.bfloat16,  # 必须先加载为BF16
    force_hooks=False,  # 禁用钩子强制转换，允许使用compile
).eval()

print("[QUANT-TO-LOW-PRECISION] Converting model to native FP8 compute (Linear layers)...")
quantize_(model, float8_dynamic_activation_float8_weight())  # TOCHANGE: 可选其他量化方案如int8_weight_only

# --- 内存布局优化 ---
# 将language_model转换为channels_last内存格式
print("[QUANT-OPTIMIZE] Converting to channels_last memory format...")
if hasattr(model, 'language_model'):
    model.language_model = model.language_model.to(memory_format=torch.channels_last)

# --- 配置torch编译器 ---
print("[QUANT-COMPILE] Configuring torch._dynamo for Bagel model compatibility...")
import torch._dynamo
# 允许捕获模型中使用.item()的标量输出（Bagel模型需要此功能）
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = False
print("[QUANT-COMPILE] ✅ capture_scalar_outputs enabled for .item() support")

# --- 启用图缓存 ---
# 启用fx_graph_cache避免相同输入形状时重复编译，一般编译最多8种不同尺寸的输入后会复用已编译的kernel
print("[QUANT-COMPILE] Checking language_model device distribution...")
torch._inductor.config.fx_graph_cache = True

# --- 检查设备分布并选择编译策略 ---
# 检查language_model的参数分布在哪些GPU上
if hasattr(model, 'language_model'):
    devices_used = set()
    for name, param in model.language_model.named_parameters():
        if param.device.type == 'cuda':
            devices_used.add(param.device.index)
    print(f"[QUANT-COMPILE] language_model spans devices: {sorted(devices_used)}")
    
    if len(devices_used) > 1:
        print("[QUANT-COMPILE] ⚠️  PROBLEM: language_model spans multiple GPUs")
        print("[QUANT-COMPILE] torch.compile CANNOT optimize cross-device modules")
        
        # 遍历每一层，根据其所在设备单独编译
        for i, layer in enumerate(model.language_model.model.layers):
            layer_device = next(layer.parameters()).device
            print(f"[QUANT-COMPILE] Compiling layer {i} on {layer_device}...", end=" ")
            model.language_model.model.layers[i] = torch.compile(
                layer,
                mode="max-autotune",
                fullgraph=True
            )
            print("✅")
        print("[QUANT-COMPILE] ✅ All layers compiled individually")
    else:
        print(f"[QUANT-COMPILE] ✅ language_model is on single device, can compile as whole")
        model.language_model = torch.compile(
            model.language_model,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False   # dynamic=True可以支持动态shape，稍牺牲极致速度，可以使用简化的warmup
        )
        print("[QUANT-COMPILE] ✅ language_model compiled successfully")

# import gc
# print("[QUANT-MEMORY-CHECK] Quantization done. Cleaning up memory...")
# gc.collect()
# torch.cuda.empty_cache()
# print(f"[QUANT-MEMORY-CHECK] Current Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
# print(f"[QUANT-MEMORY-CHECK] Current Memory Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# export_precision_report(model, "model_precision_report.txt")

# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

# # --- 简化的warmup，适用于dynamic=True编译 ---
# print("[WARMUP] Triggering torch.compile (dynamic mode)...")
# try:
#     # 只需一次warmup，任意尺寸即可触发编译
#     with torch.no_grad():
#         _ = inferencer(
#             text="Single warmup",
#             num_timesteps=10,
#             max_think_token_n=64,
#             think=False,
#             cfg_text_scale=4.0,
#             cfg_interval=[0.4, 1.0],
#             timestep_shift=3.0,
#             cfg_renorm_min=1.0,
#             cfg_renorm_type="global",
#             image_shapes=(1024, 1024),
#         )
#     print("[WARMUP] ✅ Dynamic kernel compiled, all sizes ready")
# except Exception as e:
#     print(f"[WARMUP] Warning: {e}")

# --- Warmup （3阶段：快速预热+完整文本生图+图像编辑）---
print("==========================================================")
print("[WARMUP] Starting warmup to trigger torch.compile AUTOTUNE...")
print("[WARMUP] Precompiling for multiple modes and sizes...")
print("==========================================================")

# 启用详细的编译日志
import os
os.environ['TORCH_LOGS'] = '+dynamo,recompiles'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_CUDAGRAPHS_VERBOSE'] = '0'
import warnings
warnings.filterwarnings('ignore', message='.*cudagraphs.*')
print("[WARMUP] Enabled verbose compilation logging...")

try:
    # 定义需要预热的常见尺寸
    common_sizes = [(1024, 1024), (1024, 768), (768, 1024)]
    
    import time
    total_warmup_start = time.time()
    
    # ===== 阶段1：快速预热所有尺寸（10步） =====
    # 使用较少的步数（10步）快速遍历所有尺寸
    print("\n[WARMUP] Phase 1: Text-to-Image quick precompilation (10 timesteps)...")
    print(f"[WARMUP] Will compile kernels for {len(common_sizes)} different sizes...")
    
    warmup_quick_config = dict(
        text="Quick warmup",      # 简单提示词
        num_timesteps=10,
        max_think_token_n=64,
        think=False,
        cfg_text_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    )
    
    for size_idx, size in enumerate(common_sizes):
        print(f"  [{size_idx+1}/{len(common_sizes)}] Compiling {size[0]}x{size[1]}...", end=" ")
        warmup_config = {**warmup_quick_config, "image_shapes": size}
        start = time.time()
        with torch.no_grad():
            _ = inferencer(**warmup_config)
        elapsed = time.time() - start
        print(f"T2I💖 {elapsed:.1f}s")
    
    # ===== 阶段2：完整步数预热（50步） =====
    # 让max-autotune模式有足够时间选择最优kernel
    print("\n[WARMUP] Phase 2: Text-to-Image full precompilation (50 timesteps)...")
    print("[WARMUP] Running full generation to let max-autotune select best kernels...")
    
    warmup_full_config = dict(
        text="A warmup test image",
        num_timesteps=50,          # 完整的50步
        max_think_token_n=1024,    # 标准token数
        think=False,
        cfg_text_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024), # 最常用尺寸
    )
    
    # 运行2次：第1次编译，第2次验证缓存生效
    for run in range(2):
        start = time.time()
        with torch.no_grad():
            _ = inferencer(**warmup_full_config)
        elapsed = time.time() - start
        
        if run == 0:
            print(f"  → Run 1: {elapsed:.1f}s (compiling + optimizing)")
        else:
            print(f"  → Run 2: {elapsed:.1f}s (using cached kernels)")
    
    # ===== 阶段3：预热图像编辑模式（50步） =====
    print("\n[WARMUP] Phase 3: Image Editing precompilation (50 timesteps)...")
    print("[WARMUP] Compiling kernels for image-to-image pipeline...")
    
    # 创建虚拟图像用于warmup（无需真实内容，只是触发编译）
    from PIL import Image
    dummy_image_1024x768 = Image.new('RGB', (1024, 768), color='gray')
    dummy_image_768x1024 = Image.new('RGB', (768, 1024), color='gray')
    dummy_image_1024x1024 = Image.new('RGB', (1024, 1024), color='gray')
    
    edit_sizes = [
        (dummy_image_1024x768, "1024x768"),  # 横向编辑
        (dummy_image_768x1024, "768x1024"),  # 纵向编辑
        (dummy_image_1024x1024, "1024x1024"),  # 标准尺寸编辑
    ]
    
    warmup_edit_config = dict(
        text="Edit this image",
        num_timesteps=50,
        max_think_token_n=1024,
        think=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    )
    
    for dummy_img, size_name in edit_sizes:
        print(f"  Compiling image editing @ {size_name}...", end=" ")
        start = time.time()
        with torch.no_grad():
            _ = inferencer(image=dummy_img, **warmup_edit_config)
        elapsed = time.time() - start
        print(f"I2I💖 {elapsed:.1f}s")
    
    # 总结warmup结果
    total_elapsed = time.time() - total_warmup_start
    print(f"\n{'='*60}")
    print(f"[WARMUP] ✅ All modes precompiled in {total_elapsed/60:.1f} minutes")
    print(f"[WARMUP] ✅ Text-to-Image: 5 sizes @ 10 steps + 1024x1024 @ 50 steps")
    print(f"[WARMUP] ✅ Image Editing: 2 sizes @ 50 steps")
    print(f"[WARMUP] ✅ Subsequent user requests will be fast (using cached kernels)")
    print(f"{'='*60}\n")
    
except Exception as e:
    # 如果warmup失败，不应该阻止程序继续运行
    # 用户的第一次请求会触发编译（稍慢但不影响功能）
    print(f"\n[WARMUP] ⚠️  Warning: Warmup encountered an error: {e}")
    import traceback
    traceback.print_exc()
    print("[WARMUP] ⚠️  Model will compile during first user request (may be slower).")
    print("="*60 + "\n")
else:
    print("[WARMUP] ⏭️  Skipping warmup (SKIP_WARMUP=True)")
    print("="*60 + "\n")

# Text to Image function with thinking option and hyperparameters
def text_to_image(prompt, show_thinking=False, cfg_text_scale=4.0, cfg_interval=0.4, 
                 timestep_shift=3.0, num_timesteps=50, 
                 cfg_renorm_min=1.0, cfg_renorm_type="global", 
                 max_think_token_n=1024, do_sample=False, text_temperature=0.3,
                 seed=0, image_ratio="1:1"):
    # Set seed for reproducibility
    set_seed(seed)

    if image_ratio == "1:1":
        image_shapes = (1024, 1024)
    elif image_ratio == "4:3":
        image_shapes = (768, 1024)
    elif image_ratio == "3:4":
        image_shapes = (1024, 768) 
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=image_shapes,
    )
    # --- add：record start_time ---
    start_time = time.time()
    
    # Call inferencer with or without think parameter based on user choice
    result = inferencer(text=prompt, think=show_thinking, **inference_hyper)
    
    # --- add：record end_time ---
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f"{duration:.2f} seconds"
    print(f"The image takes time: {duration_str}") # conslog
    
    return result["image"], result.get("text", None), duration_str


# Image Understanding function with thinking option and hyperparameters
def image_understanding(image: Image.Image, prompt: str, show_thinking=False, 
                        do_sample=False, text_temperature=0.3, max_new_tokens=512):
    # --- add：record start_time ---
    start_time = time.time()
    
    if image is None:
        return "Please upload an image."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        do_sample=do_sample,
        text_temperature=text_temperature,
        max_think_token_n=max_new_tokens, # Set max_length
    )
    
    # Use show_thinking parameter to control thinking process
    result = inferencer(image=image, text=prompt, think=show_thinking, 
                        understanding_output=True, **inference_hyper)

    # --- add：record end_time ---
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f"{duration:.2f} seconds"
    print(f"The image takes time: {duration_str}") # conslog
    
    return result["text"], duration_str


# Image Editing function with thinking option and hyperparameters
def edit_image(image: Image.Image, prompt: str, show_thinking=False, cfg_text_scale=4.0, 
              cfg_img_scale=2.0, cfg_interval=0.0, 
              timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=1.0, 
              cfg_renorm_type="text_channel", max_think_token_n=1024, 
              do_sample=False, text_temperature=0.3, seed=0):
    # Set seed for reproducibility
    set_seed(seed)

    # --- add：record start_time ---
    start_time = time.time()
    
    if image is None:
        return "Please upload an image.", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = pil_img2rgb(image)
    
    # Set hyperparameters
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    # Include thinking parameter based on user choice
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)

    # --- add：record end_time ---
    end_time = time.time()
    duration = end_time - start_time
    duration_str = f"{duration:.2f} seconds"
    print(f"The image takes time: {duration_str}") # conslog
    
    return result["image"], result.get("text", ""), duration_str


# Helper function to load example images
def load_example_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None


# Gradio UI 
with gr.Blocks() as demo:
    gr.Markdown("""
<div>
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="380"/>
</div>
""")

    with gr.Tab("📝 Text to Image"):
        txt_input = gr.Textbox(
            label="Prompt", 
            value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
        )
        
        with gr.Row():
            show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, 
                                   label="Seed", info="0 for random seed, positive for reproducible results")
                    image_ratio = gr.Dropdown(choices=["1:1", "4:3", "3:4"], 
                                                value="1:1", label="Image Ratio", 
                                                info="The longer size is fixed to 1024")
                    
                with gr.Row():
                    cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                             label="CFG Text Scale", info="Controls how strongly the model follows the text prompt (4.0-8.0)")
                    cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, 
                                           label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                
                with gr.Row():
                    cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                value="global", label="CFG Renorm Type", 
                                                info="If the genrated image is blurry, use 'global'")
                    cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                             label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                            label="Timesteps", info="Total denoising steps")
                    timestep_shift = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, interactive=True,
                                             label="Timestep Shift", info="Higher values for layout, lower for details")
                
                # Thinking parameters in a single row
                thinking_params = gr.Group(visible=False)
                with thinking_params:
                    with gr.Row():
                        do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                    label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                  label="Temperature", info="Controls randomness in text generation")
        
        thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        img_output = gr.Image(label="Generated Image")
        gen_btn = gr.Button("Generate", variant="primary")
        
        # --- add：A text box used to display the generation time ---
        generation_time_output = gr.Textbox(label="Processing Time", interactive=False)
        # --- add end ---
        
        # Dynamically show/hide thinking process box and parameters
        def update_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        show_thinking.change(
            fn=update_thinking_visibility,
            inputs=[show_thinking],
            outputs=[thinking_output, thinking_params]
        )
        
        # Process function based on thinking option and hyperparameters
        def process_text_to_image(prompt, show_thinking, cfg_text_scale, 
                                 cfg_interval, timestep_shift, 
                                 num_timesteps, cfg_renorm_min, cfg_renorm_type, 
                                 max_think_token_n, do_sample, text_temperature, seed, image_ratio):
            image, thinking, duration_str = text_to_image(
                prompt, show_thinking, cfg_text_scale, cfg_interval,
                timestep_shift, num_timesteps, 
                cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed, image_ratio
            )
            return image, thinking if thinking else "", duration_str 
        
        gr.on(
            triggers=[gen_btn.click, txt_input.submit],
            fn=process_text_to_image,
            inputs=[
                txt_input, show_thinking, cfg_text_scale, 
                cfg_interval, timestep_shift, 
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed, image_ratio
            ],
            # --- 这里是关键修改：确保 outputs 列表包含三个组件 ---
            outputs=[img_output, thinking_output, generation_time_output]
            # --- 修改结束 ---
        )

    with gr.Tab("🖌️ Image Edit"):
        with gr.Row():
            with gr.Column(scale=1):
                edit_image_input = gr.Image(label="Input Image", value=load_example_image('test_images/20251222-113154.jpg'))
                edit_prompt = gr.Textbox(
                    label="Prompt",
                    value="人物在沙滩上奔跑，穿着夏日服装，背景是阳光明媚的海滩和蓝天白云，整体氛围轻松愉快，色彩鲜艳。"
                )
            
            with gr.Column(scale=1):
                edit_image_output = gr.Image(label="Result")
                edit_thinking_output = gr.Textbox(label="Thinking Process", visible=False)
        
        with gr.Row():
            edit_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Group():
                with gr.Row():
                    edit_seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, interactive=True,
                                        label="Seed", info="0 for random seed, positive for reproducible results")
                    edit_cfg_text_scale = gr.Slider(minimum=1.0, maximum=8.0, value=4.0, step=0.1, interactive=True,
                                                  label="CFG Text Scale", info="Controls how strongly the model follows the text prompt")
                
                with gr.Row():
                    edit_cfg_img_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, interactive=True,
                                                 label="CFG Image Scale", info="Controls how much the model preserves input image details")
                    edit_cfg_interval = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                label="CFG Interval", info="Start of CFG application interval (end is fixed at 1.0)")
                    
                with gr.Row():
                    edit_cfg_renorm_type = gr.Dropdown(choices=["global", "local", "text_channel"], 
                                                     value="text_channel", label="CFG Renorm Type", 
                                                     info="If the genrated image is blurry, use 'global")
                    edit_cfg_renorm_min = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True,
                                                  label="CFG Renorm Min", info="1.0 disables CFG-Renorm")
                
                with gr.Row():
                    edit_num_timesteps = gr.Slider(minimum=10, maximum=100, value=50, step=5, interactive=True,
                                                 label="Timesteps", info="Total denoising steps")
                    edit_timestep_shift = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.5, interactive=True,
                                                  label="Timestep Shift", info="Higher values for layout, lower for details")
                
                
                # Thinking parameters in a single row
                edit_thinking_params = gr.Group(visible=False)
                with edit_thinking_params:
                    with gr.Row():
                        edit_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                        edit_max_think_token_n = gr.Slider(minimum=64, maximum=4006, value=1024, step=64, interactive=True,
                                                         label="Max Think Tokens", info="Maximum number of tokens for thinking")
                        edit_text_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, interactive=True,
                                                        label="Temperature", info="Controls randomness in text generation")
        
        edit_btn = gr.Button("Submit", variant="primary")
        # --- add：A text box used to display the generation time ---
        edit_generation_time_output = gr.Textbox(label="Processing Time", interactive=False)
        # --- add end ---
        
        # Dynamically show/hide thinking process box for editing
        def update_edit_thinking_visibility(show):
            return gr.update(visible=show), gr.update(visible=show)
        
        edit_show_thinking.change(
            fn=update_edit_thinking_visibility,
            inputs=[edit_show_thinking],
            outputs=[edit_thinking_output, edit_thinking_params]
        )
        
        # Process editing with thinking option and hyperparameters
        def process_edit_image(image, prompt, show_thinking, cfg_text_scale, 
                              cfg_img_scale, cfg_interval, 
                              timestep_shift, num_timesteps, cfg_renorm_min, 
                              cfg_renorm_type, max_think_token_n, do_sample, 
                              text_temperature, seed):
            edited_image, thinking, duration_str = edit_image(
                image, prompt, show_thinking, cfg_text_scale, cfg_img_scale, 
                cfg_interval, timestep_shift, 
                num_timesteps, cfg_renorm_min, cfg_renorm_type,
                max_think_token_n, do_sample, text_temperature, seed
            )
            
            return edited_image, thinking if thinking else "", duration_str
        
        gr.on(
            triggers=[edit_btn.click, edit_prompt.submit],
            fn=process_edit_image,
            inputs=[
                edit_image_input, edit_prompt, edit_show_thinking, 
                edit_cfg_text_scale, edit_cfg_img_scale, edit_cfg_interval,
                edit_timestep_shift, edit_num_timesteps, 
                edit_cfg_renorm_min, edit_cfg_renorm_type,
                edit_max_think_token_n, edit_do_sample, edit_text_temperature, edit_seed
            ],
            outputs=[edit_image_output, edit_thinking_output, edit_generation_time_output]
        )

    with gr.Tab("🖼️ Image Understanding"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Input Image", value=load_example_image('test_images/meme.jpg'))
                understand_prompt = gr.Textbox(
                    label="Prompt", 
                    value="Can someone explain what's funny about this meme??"
                )
            
            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Result", lines=20)
                
        
        with gr.Row():
            understand_show_thinking = gr.Checkbox(label="Thinking", value=False)
        
        # Add hyperparameter controls in an accordion
        with gr.Accordion("Inference Hyperparameters", open=False):
            with gr.Row():
                understand_do_sample = gr.Checkbox(label="Sampling", value=False, info="Enable sampling for text generation")
                understand_text_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, interactive=True,
                                                     label="Temperature", info="Controls randomness in text generation (0=deterministic, 1=creative)")
                understand_max_new_tokens = gr.Slider(minimum=64, maximum=4096, value=512, step=64, interactive=True,
                                                   label="Max New Tokens", info="Maximum length of generated text, including potential thinking")
        
        img_understand_btn = gr.Button("Submit", variant="primary")
        # --- add：A text box used to display the generation time ---
        understand_generation_time_output = gr.Textbox(label="Processing Time", interactive=False)
        # --- add end ---
        
        # Process understanding with thinking option and hyperparameters
        def process_understanding(image, prompt, show_thinking, do_sample, 
                                 text_temperature, max_new_tokens):
            result, duration_str = image_understanding(
                image, prompt, show_thinking, do_sample, 
                text_temperature, max_new_tokens
            )
            return result, duration_str
        
        gr.on(
            triggers=[img_understand_btn.click, understand_prompt.submit],
            fn=process_understanding,
            inputs=[
                img_input, understand_prompt, understand_show_thinking,
                understand_do_sample, understand_text_temperature, understand_max_new_tokens
            ],
            outputs=[txt_output, understand_generation_time_output]
        )

    gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"
      alt="BAGEL Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"
      alt="BAGEL Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow" 
        alt="BAGEL on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"
      alt="BAGEL Demo"
    />
  </a>
  <a href="https://discord.gg/Z836xxzy">
    <img
      src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"
      alt="BAGEL Discord"
    />
  </a>
  <a href="mailto:bagel@bytedance.com">
    <img
      src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"
      alt="BAGEL Email"
    />
  </a>
</div>
""")
    # --- add ram/vram Stats tab ---
    with gr.Tab("📊 System Monitor"):
        with gr.Column():
            memory_stats_display = gr.Markdown("Check RAM/VRAM Stats")
            refresh_button = gr.Button("🔄 Check RAM/VRAM Stats")
            
           # When the button is clicked, the function get_all_memory_stats_for_gradio_display is called.
           # And update its return value to the memory_stats_display component
            refresh_button.click(
                fn=get_all_memory_stats_for_gradio_display,
                inputs=None, 
                outputs=[memory_stats_display]
            )
    # --- ram/vram Stats end ---

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        inbrowser=True,
    )