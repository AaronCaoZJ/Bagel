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
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
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
    max_latent_size=64,
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

# # Model Loading and Multi GPU Infernece Preparing
# device_map = infer_auto_device_map(
#     model,
#     max_memory={i: "24GiB" for i in range(torch.cuda.device_count())},
#     no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
# )

# same_device_modules = [
#     'language_model.model.embed_tokens',
#     'time_embedder',
#     'latent_pos_embed',
#     'vae2llm',
#     'llm2vae',
#     'connector',
#     'vit_pos_embed'
# ]

# if torch.cuda.device_count() == 1:
#     first_device = device_map.get(same_device_modules[0], "cuda:0")
#     for k in same_device_modules:
#         if k in device_map:
#             device_map[k] = first_device
#         else:
#             device_map[k] = "cuda:0"
# else:
#     first_device = device_map.get(same_device_modules[0])
#     for k in same_device_modules:
#         if k in device_map:
#             device_map[k] = first_device

# --- new changes ---
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
# ram & vram setting
# ram & vram setting  edit by your spec
# If you have 60GB CPU vram，there are 55GiB (Leave some vram)

cpu_mem_for_offload = "16GiB"
gpu_mem_per_device = "31GiB" # Your GPU Vram

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

# key point 2：make sure no 'disk'  (backup)
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
# --- fix model loadding end ---

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