#%% Imports
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

import os
import gc
import warnings
import argparse
from typing import Dict, Tuple, Optional, List

import gradio as gr
import numpy as np

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/zhijun/Code/Bagel/triton"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
import torch
import torch._dynamo
import torch._inductor
import random
from PIL import Image

# TorchAO quantization imports
# 需要安装对应版本的torchao https://github.com/pytorch/ao/issues/2919
from torchao.quantization import quantize_
from torchao.quantization import (
    float8_dynamic_activation_float8_weight, float8_weight_only,
    int8_weight_only, int4_weight_only, int8_dynamic_activation_int8_weight
)
# Accelerate and model imports
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Local imports
# from scripts.export_precision_report import export_precision_report
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

#%% Constants and Configuration
# Image ratio configurations
IMAGE_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (768, 1024),
    "3:4": (1024, 768)
}

# Default inference hyperparameters
DEFAULT_TEXT2IMG_PARAMS = {
    "cfg_text_scale": 4.0,
    "cfg_interval": 0.4,
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
    "max_think_token_n": 1024,
    "do_sample": False,
    "text_temperature": 0.3,
}

DEFAULT_IMG2IMG_PARAMS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": 0.0,
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
    "max_think_token_n": 1024,
    "do_sample": False,
    "text_temperature": 0.3,
}

DEFAULT_UNDERSTANDING_PARAMS = {
    "do_sample": False,
    "text_temperature": 0.3,
    "max_new_tokens": 512,
}

# Warmup configurations (simplified for dynamic mode)
# Dynamic compilation adapts to any size automatically

# GPU memory configuration - ADJUST THESE FOR YOUR HARDWARE
GPU_MEM_PER_DEVICE = "31GiB"
CPU_MEM_FOR_OFFLOAD = "0GiB"
NUM_LLM_LAYERS_TO_GPU = 5  # Number of additional LLM layers to move to GPU
MOVE_LLM_NORM_HEAD_TO_GPU = True
MOVE_VIT_MODEL_TO_GPU = False

# compile模式选择：
# - "max-autotune": 最激进优化，但可能与 accelerate hooks 冲突
# - "reduce-overhead": 平衡模式，减少 Python 开销
# - "default": 最安全模式
COMPILE_MODE = "default"

# Modules that should be on the same device
SAME_DEVICE_MODULES = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

#%% Memory Statistics Functions
def get_gpu_memory_stats_pynvml(device_id: int = 0) -> str:
    """Get GPU memory stats using pynvml."""
    if not pynvml_available:
        return f"GPU-{device_id} (pynvml): Not available"
    try:
        # from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, NVMLError
        handle = nvmlDeviceGetHandleByIndex(device_id)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / (1024**3)
        used_gb = info.used / (1024**3)
        # free_gb = info.free / (1024**3) # It can be calculated by the sum already used
        return f"GPU-{device_id} (pynvml): Total: {total_gb:.2f} GB, Used: {used_gb:.2f} GB"
    except NVMLError as e:
        return f"GPU-{device_id} (pynvml) Error: {e}"


def get_gpu_memory_stats_pytorch(device_id: int = 0) -> str:
    """Get GPU memory stats using PyTorch."""
    if not torch.cuda.is_available():
        return "PyTorch: CUDA not available"
    if device_id < 0 or device_id >= torch.cuda.device_count():
        return f"PyTorch GPU-{device_id}: Invalid device ID"
    
    allocated_gb = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device_id) / (1024**3)
    
    total_capacity_str = ""
    if pynvml_available:
        try:
            # from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            handle = nvmlDeviceGetHandleByIndex(device_id)
            info = nvmlDeviceGetMemoryInfo(handle)
            total_gb = info.total / (1024**3)
            total_capacity_str = f"Total: {total_gb:.2f} GB, "
        except:
            pass
    
    return (f"PyTorch GPU-{device_id}: {total_capacity_str}"
            f"Allocated: {allocated_gb:.2f} GB, Reserved: {reserved_gb:.2f} GB")


def get_system_ram_stats() -> str:
    """Get system RAM statistics."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    used_gb = mem.used / (1024**3)
    return (f"System RAM: Total: {total_gb:.2f} GB, Available: {available_gb:.2f} GB, "
            f"Used: {used_gb:.2f} GB ({mem.percent}%)")


def get_process_ram_stats() -> str:
    """Get current process RAM usage."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024**3)
    return f"App Process RAM (RSS): {rss_gb:.2f} GB"


def get_all_memory_stats() -> str:
    """Compile all memory statistics for display."""
    stats_lines = []
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        stats_lines.append("**GPU VRAM Usage:**")
        for i in range(torch.cuda.device_count()):
            stats_lines.append(get_gpu_memory_stats_pynvml(i))
            stats_lines.append(get_gpu_memory_stats_pytorch(i))
            if i < torch.cuda.device_count() - 1:
                stats_lines.append("---")
    else:
        stats_lines.append("**GPU VRAM Usage:** CUDA not available")
    
    stats_lines.append("\n**CPU RAM Usage:**")
    stats_lines.append(get_system_ram_stats())
    stats_lines.append(get_process_ram_stats())
    
    return "\n".join(stats_lines)

#%% Model Device Map Configuration
def create_device_map(model):
    """Create and configure device map for model distribution."""
    max_memory_config = {i: GPU_MEM_PER_DEVICE for i in range(torch.cuda.device_count())}
    max_memory_config["cpu"] = CPU_MEM_FOR_OFFLOAD
    
    print(f"Memory configuration: {max_memory_config}")
    
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory_config,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    # Ensure same_device_modules are on the same device
    if torch.cuda.device_count() > 0:
        first_module = SAME_DEVICE_MODULES[0]
        target_device = device_map.get(first_module, "cuda:0")
        print(f"Assigning same_device_modules to: {target_device}")
        for module in SAME_DEVICE_MODULES:
            device_map[module] = target_device
    else:
        for module in SAME_DEVICE_MODULES:
            device_map[module] = "cpu"
    
    # Prevent disk offloading
    for module_name, device in list(device_map.items()):
        if device == "disk":
            print(f"⚠️ Moving {module_name} from 'disk' to 'cpu'")
            device_map[module_name] = "cpu"
    
    return device_map


def optimize_device_map(device_map):
    """Optimize device map to maximize GPU utilization."""
    if torch.cuda.device_count() == 0:
        return device_map
    
    print("\n=== Optimizing device map ===")
    
    # Move additional LLM layers to GPU
    moved_count = 0
    if NUM_LLM_LAYERS_TO_GPU > 0:
        print(f"Moving up to {NUM_LLM_LAYERS_TO_GPU} LLM layers to GPU 0...")
        for i in range(NUM_LLM_LAYERS_TO_GPU):
            layer_idx = 11 + i
            if layer_idx > 27:
                break
            layer_name = f"language_model.model.layers.{layer_idx}"
            if device_map.get(layer_name) == 'cpu':
                device_map[layer_name] = 0
                moved_count += 1
        print(f"✅ Moved {moved_count} LLM layers to GPU 0")
    
    # Move LLM auxiliary modules
    if MOVE_LLM_NORM_HEAD_TO_GPU:
        print("Moving LLM norm and lm_head to GPU 0...")
        for module in ["language_model.model.norm", "language_model.model.lm_head"]:
            if device_map.get(module) == 'cpu':
                device_map[module] = 0
    
    # Optionally move VIT model
    if MOVE_VIT_MODEL_TO_GPU and device_map.get("vit_model") == 'cpu':
        print("Moving vit_model to GPU 0...")
        device_map["vit_model"] = 0
    
    print("=== Device map optimization complete ===\n")
    return device_map

#%% Model Loading and Quantization
def load_model(model_path: str):
    """Load model configuration and initialize empty model."""
    # Load configurations
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
    
    # Initialize empty model
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # print("Converting VAE to bfloat16 on CUDA...")
    # vae_model = vae_model.to(dtype=torch.bfloat16, device="cuda:0")
    # print(f"VAE model device:{next(vae_model.parameters()).device},dtype:{next(vae_model.parameters()).dtype}")

    return model, vae_model, tokenizer, new_token_ids


def apply_quantization_and_compile(model):
    """Apply quantization and torch.compile optimizations."""
    print("[🧩 QUANT] Converting to FP8 quantization...")
    quantize_(model, float8_dynamic_activation_float8_weight())
    
    print("[🧩 QUANT] Converting to channels_last memory format...")
    if hasattr(model, 'language_model'):
        model.language_model = model.language_model.to(memory_format=torch.channels_last)
    
    print("[🧩 QUANT] Configuring torch._dynamo...")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = False
    
    # Enable persistent FX graph cache with custom directory
    torch._inductor.config.fx_graph_cache = True
    
    # Configure compilation logging
    os.environ['TORCH_LOGS'] = '+dynamo,recompiles'
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'
    os.environ['TORCH_CUDAGRAPHS_VERBOSE'] = '0'
    warnings.filterwarnings('ignore', message='.*cudagraphs.*')
    
    if hasattr(model, 'language_model'):
        devices_used = {param.device.index for name, param in model.language_model.named_parameters() 
                       if param.device.type == 'cuda'}
        
        print(f"[🛠️ COMPILE] language_model on {len(devices_used)} GPU(s), compiling layers individually...")
        print(f"[🛠️ COMPILE] Using mode='{COMPILE_MODE}'")
        
        compiled_count = 0
        failed_count = 0
        
        for i, layer in enumerate(model.language_model.model.layers):
            try:
                model.language_model.model.layers[i] = torch.compile(
                    layer, mode=COMPILE_MODE, fullgraph=False, dynamic=True
                )
                compiled_count += 1
            except Exception as e:
                print(f"[🛠️ COMPILE] ⚠️ Layer {i} compilation failed: {e}")
                failed_count += 1
        
        if failed_count > 0:
            print(f"[🛠️ COMPILE] ⚠️ {failed_count} layers failed, {compiled_count} layers compiled")
        else:
            print(f"[🛠️ COMPILE] ✅ All {compiled_count} layers compiled (dynamic mode)")
    
    # 温和清理：只清理量化过程的临时数据，不影响编译后的计算图
    # torch.compile的计算图会持久化到磁盘缓存，empty_cache()不会删除它们
    print("[🛁 CLEANUP] Clearing temporary tensors from quantization...")
    gc.collect()  # 回收Python对象
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 只清理未使用的显存碎片，不影响模型本身
    print(f"[🛁 CLEANUP] Memory after cleanup: {get_all_memory_stats()}")
    
    return model

#%% Model Warmup
def warmup_model(inferencer):
    """
    简化预热：启用dynamic=True后，只需预热一次即可支持所有尺寸
    Dynamic compilation will automatically adapt to any input size
    """
    print("\n" + "="*60)
    print("[WARMUP] Starting simplified warmup (dynamic mode)...")
    print("="*60)
    
    total_start = time.time()
    
    try:
        # 单次文生图预热
        print(f"\n[🔥 WARMUP] Text-to-image warmup (10 steps, 1024x1024)...")
        start = time.time()
        with torch.no_grad():
            _ = inferencer(
                text="Warmup test",
                image_shapes=(1024, 1024),
                num_timesteps=10,
                max_think_token_n=512,
                think=False,
                cfg_text_scale=4.0,
                cfg_interval=[0.4, 1.0],
                timestep_shift=3.0,
            )
        print(f"  ✅ Done in {time.time() - start:.1f}s (supports all sizes now)")
        
        # 清理warmup产生的激活值和中间tensor（不会影响编译好的kernel）
        # 编译的计算图已经持久化，这里只清理推理产生的临时数据
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [🛁 CLEANUP] Memory after T2I warmup: {get_gpu_memory_stats_pytorch(0)}")
        
        # 单次图生图预热
        print(f"\n[🔥 WARMUP] Image-to-image warmup (10 steps, 1024x768)...")
        dummy_img = Image.new('RGB', (1024, 768), color='gray')
        start = time.time()
        with torch.no_grad():
            _ = inferencer(
                image=dummy_img,
                text="Edit warmup",
                num_timesteps=10,
                max_think_token_n=1024,
                think=False,
                cfg_text_scale=4.0,
                cfg_img_scale=2.0,
                cfg_interval=[0.0, 1.0],
                timestep_shift=3.0,
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
            )
        print(f"  ✅ Done in {time.time() - start:.1f}s (supports all sizes now)")
        
        # 清理warmup产生的激活值（编译好的kernel仍在缓存中）
        del dummy_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [🛁 CLEANUP] Memory after I2I warmup: {get_gpu_memory_stats_pytorch(0)}")
        
        # 最终清理：只清理warmup的激活值，保留编译的计算图和kernel缓存
        # torch._inductor.config.fx_graph_cache=True 会把编译结果存在磁盘
        # empty_cache() 只释放未使用的显存，不会删除编译好的代码
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()  # 安全：只清理空闲显存碎片
                    torch.cuda.synchronize()  # 同步以确保操作完成
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"[🔥 WARMUP] ✅ Complete in {total_time:.1f}s")
        print(f"[🔥 WARMUP] ✅ Dynamic mode ready for any size")
        print(f"{'='*60}")
        print(f"\n[FINAL MEMORY] {get_all_memory_stats()}\n")
        
    except Exception as e:
        print(f"\n[🔥 WARMUP] ⚠️ Warning: {e}")
        import traceback
        traceback.print_exc()
        print("[WARMUP] Model will compile during first user request\n")

#%% Utility Functions
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_example_image(image_path: str) -> Optional[Image.Image]:
    """Load an example image, return None if failed."""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading example image: {e}")
        return None

#%% Inference Functions
def text_to_image(
    inferencer,
    prompt: str,
    show_thinking: bool = False,
    image_ratio: str = "1:1",
    seed: int = 0,
    **kwargs
) -> Tuple[Image.Image, str, str]:
    """Generate image from text prompt."""
    set_seed(seed)
    
    inference_params = {
        "text": prompt,
        "think": show_thinking,
        "image_shapes": IMAGE_RATIOS[image_ratio],
        "max_think_token_n": kwargs.get("max_think_token_n", 1024) if show_thinking else 1024,
        "do_sample": kwargs.get("do_sample", False) if show_thinking else False,
        "text_temperature": kwargs.get("text_temperature", 0.3) if show_thinking else 0.3,
        "cfg_text_scale": kwargs.get("cfg_text_scale", 4.0),
        "cfg_interval": [kwargs.get("cfg_interval", 0.4), 1.0],
        "timestep_shift": kwargs.get("timestep_shift", 3.0),
        "num_timesteps": kwargs.get("num_timesteps", 50),
        "cfg_renorm_min": kwargs.get("cfg_renorm_min", 0.0),
        "cfg_renorm_type": kwargs.get("cfg_renorm_type", "global"),
    }
    
    start_time = time.time()
    result = inferencer(**inference_params)
    duration = time.time() - start_time
    
    return result["image"], result.get("text", ""), f"{duration:.2f} seconds"


def image_understanding(
    inferencer,
    image: Image.Image,
    prompt: str,
    show_thinking: bool = False,
    **kwargs
) -> Tuple[str, str]:
    """Understand and describe image content."""
    if image is None:
        return "Please upload an image.", "0.00 seconds"
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = pil_img2rgb(image)
    
    inference_params = {
        "image": image,
        "text": prompt,
        "think": show_thinking,
        "understanding_output": True,
        "do_sample": kwargs.get("do_sample", False),
        "text_temperature": kwargs.get("text_temperature", 0.3),
        "max_think_token_n": kwargs.get("max_new_tokens", 512),
    }
    
    start_time = time.time()
    result = inferencer(**inference_params)
    duration = time.time() - start_time
    
    return result["text"], f"{duration:.2f} seconds"


def edit_image(
    inferencer,
    image: Image.Image,
    prompt: str,
    show_thinking: bool = False,
    seed: int = 0,
    **kwargs
) -> Tuple[Image.Image, str, str]:
    """Edit image based on text prompt."""
    set_seed(seed)
    
    if image is None:
        return None, "Please upload an image.", "0.00 seconds"
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = pil_img2rgb(image)
    
    inference_params = {
        "image": image,
        "text": prompt,
        "think": show_thinking,
        "max_think_token_n": kwargs.get("max_think_token_n", 1024) if show_thinking else 1024,
        "do_sample": kwargs.get("do_sample", False) if show_thinking else False,
        "text_temperature": kwargs.get("text_temperature", 0.3) if show_thinking else 0.3,
        "cfg_text_scale": kwargs.get("cfg_text_scale", 4.0),
        "cfg_img_scale": kwargs.get("cfg_img_scale", 2.0),
        "cfg_interval": [kwargs.get("cfg_interval", 0.0), 1.0],
        "timestep_shift": kwargs.get("timestep_shift", 3.0),
        "num_timesteps": kwargs.get("num_timesteps", 50),
        "cfg_renorm_min": kwargs.get("cfg_renorm_min", 0.0),
        "cfg_renorm_type": kwargs.get("cfg_renorm_type", "text_channel"),
    }
    
    start_time = time.time()
    result = inferencer(**inference_params)
    duration = time.time() - start_time
    
    return result["image"], result.get("text", ""), f"{duration:.2f} seconds"

#%% Gradio Interface
def create_gradio_interface(inferencer):
    """Create and configure Gradio interface."""
    
    with gr.Blocks() as demo:
        gr.Markdown("""
<div>
  <img src="https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/banner.png" alt="BAGEL" width="380"/>
</div>
""")
        
        # Text to Image Tab
        with gr.Tab("📝 Text to Image"):
            txt_input = gr.Textbox(
                label="Prompt",
                value="A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
            )
            
            with gr.Row():
                show_thinking = gr.Checkbox(label="Thinking", value=False)
            
            with gr.Accordion("Inference Hyperparameters", open=False):
                with gr.Row():
                    seed = gr.Slider(0, 1000000, 0, step=1, label="Seed")
                    image_ratio = gr.Dropdown(["1:1", "4:3", "3:4"], value="1:1", label="Image Ratio")
                
                with gr.Row():
                    cfg_text_scale = gr.Slider(1.0, 8.0, 4.0, step=0.1, label="CFG Text Scale")
                    cfg_interval = gr.Slider(0.0, 1.0, 0.4, step=0.1, label="CFG Interval")
                
                with gr.Row():
                    cfg_renorm_type = gr.Dropdown(["global", "local", "text_channel"], value="global", label="CFG Renorm Type")
                    cfg_renorm_min = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Renorm Min")
                
                with gr.Row():
                    num_timesteps = gr.Slider(10, 100, 50, step=5, label="Timesteps")
                    timestep_shift = gr.Slider(1.0, 5.0, 3.0, step=0.5, label="Timestep Shift")
                
                thinking_params = gr.Group(visible=False)
                with thinking_params:
                    with gr.Row():
                        do_sample = gr.Checkbox(label="Sampling", value=False)
                        max_think_token_n = gr.Slider(64, 4006, 1024, step=64, label="Max Think Tokens")
                        text_temperature = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Temperature")
            
            thinking_output = gr.Textbox(label="Thinking Process", visible=False)
            img_output = gr.Image(label="Generated Image")
            gen_btn = gr.Button("Generate", variant="primary")
            generation_time = gr.Textbox(label="Processing Time", interactive=False)
            
            show_thinking.change(
                lambda x: (gr.update(visible=x), gr.update(visible=x)),
                inputs=[show_thinking],
                outputs=[thinking_output, thinking_params]
            )
            
            def process_t2i(*args):
                params = dict(zip([
                    "prompt", "show_thinking", "cfg_text_scale", "cfg_interval",
                    "timestep_shift", "num_timesteps", "cfg_renorm_min", "cfg_renorm_type",
                    "max_think_token_n", "do_sample", "text_temperature", "seed", "image_ratio"
                ], args))
                return text_to_image(inferencer, **params)
            
            gr.on(
                [gen_btn.click, txt_input.submit],
                process_t2i,
                inputs=[txt_input, show_thinking, cfg_text_scale, cfg_interval, timestep_shift,
                       num_timesteps, cfg_renorm_min, cfg_renorm_type, max_think_token_n,
                       do_sample, text_temperature, seed, image_ratio],
                outputs=[img_output, thinking_output, generation_time]
            )
        
        # Image Edit Tab
        with gr.Tab("🖌️ Image Edit"):
            with gr.Row():
                with gr.Column(scale=1):
                    edit_img_input = gr.Image(label="Input Image", value=load_example_image('test_images/20251222-113154.jpg'))
                    edit_prompt = gr.Textbox(label="Prompt", value="人物在沙滩上奔跑，穿着夏日服装，背景是阳光明媚的海滩和蓝天白云")
                
                with gr.Column(scale=1):
                    edit_img_output = gr.Image(label="Result")
                    edit_thinking_output = gr.Textbox(label="Thinking Process", visible=False)
            
            with gr.Row():
                edit_show_thinking = gr.Checkbox(label="Thinking", value=False)
            
            with gr.Accordion("Inference Hyperparameters", open=False):
                with gr.Row():
                    edit_seed = gr.Slider(0, 1000000, 0, step=1, label="Seed")
                    edit_cfg_text_scale = gr.Slider(1.0, 8.0, 4.0, step=0.1, label="CFG Text Scale")
                
                with gr.Row():
                    edit_cfg_img_scale = gr.Slider(1.0, 4.0, 2.0, step=0.1, label="CFG Image Scale")
                    edit_cfg_interval = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Interval")
                
                with gr.Row():
                    edit_cfg_renorm_type = gr.Dropdown(["global", "local", "text_channel"], value="text_channel", label="CFG Renorm Type")
                    edit_cfg_renorm_min = gr.Slider(0.0, 1.0, 0.0, step=0.1, label="CFG Renorm Min")
                
                with gr.Row():
                    edit_num_timesteps = gr.Slider(10, 100, 50, step=5, label="Timesteps")
                    edit_timestep_shift = gr.Slider(1.0, 10.0, 3.0, step=0.5, label="Timestep Shift")
                
                edit_thinking_params = gr.Group(visible=False)
                with edit_thinking_params:
                    with gr.Row():
                        edit_do_sample = gr.Checkbox(label="Sampling", value=False)
                        edit_max_think_token_n = gr.Slider(64, 4006, 1024, step=64, label="Max Think Tokens")
                        edit_text_temperature = gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Temperature")
            
            edit_btn = gr.Button("Submit", variant="primary")
            edit_time = gr.Textbox(label="Processing Time", interactive=False)
            
            edit_show_thinking.change(
                lambda x: (gr.update(visible=x), gr.update(visible=x)),
                inputs=[edit_show_thinking],
                outputs=[edit_thinking_output, edit_thinking_params]
            )
            
            def process_edit(*args):
                params = dict(zip([
                    "image", "prompt", "show_thinking", "cfg_text_scale", "cfg_img_scale",
                    "cfg_interval", "timestep_shift", "num_timesteps", "cfg_renorm_min",
                    "cfg_renorm_type", "max_think_token_n", "do_sample", "text_temperature", "seed"
                ], args))
                return edit_image(inferencer, **params)
            
            gr.on(
                [edit_btn.click, edit_prompt.submit],
                process_edit,
                inputs=[edit_img_input, edit_prompt, edit_show_thinking, edit_cfg_text_scale,
                       edit_cfg_img_scale, edit_cfg_interval, edit_timestep_shift,
                       edit_num_timesteps, edit_cfg_renorm_min, edit_cfg_renorm_type,
                       edit_max_think_token_n, edit_do_sample, edit_text_temperature, edit_seed],
                outputs=[edit_img_output, edit_thinking_output, edit_time]
            )
        
        # Image Understanding Tab
        with gr.Tab("🖼️ Image Understanding"):
            with gr.Row():
                with gr.Column(scale=1):
                    understand_img_input = gr.Image(label="Input Image", value=load_example_image('test_images/meme.jpg'))
                    understand_prompt = gr.Textbox(label="Prompt", value="Can someone explain what's funny about this meme??")
                
                with gr.Column(scale=1):
                    understand_output = gr.Textbox(label="Result", lines=20)
            
            with gr.Row():
                understand_show_thinking = gr.Checkbox(label="Thinking", value=False)
            
            with gr.Accordion("Inference Hyperparameters", open=False):
                with gr.Row():
                    understand_do_sample = gr.Checkbox(label="Sampling", value=False)
                    understand_temperature = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Temperature")
                    understand_max_tokens = gr.Slider(64, 4096, 512, step=64, label="Max New Tokens")
            
            understand_btn = gr.Button("Submit", variant="primary")
            understand_time = gr.Textbox(label="Processing Time", interactive=False)
            
            def process_understand(*args):
                params = dict(zip([
                    "image", "prompt", "show_thinking", "do_sample",
                    "text_temperature", "max_new_tokens"
                ], args))
                return image_understanding(inferencer, **params)
            
            gr.on(
                [understand_btn.click, understand_prompt.submit],
                process_understand,
                inputs=[understand_img_input, understand_prompt, understand_show_thinking,
                       understand_do_sample, understand_temperature, understand_max_tokens],
                outputs=[understand_output, understand_time]
            )
        
        # System Monitor Tab
        with gr.Tab("📊 System Monitor"):
            memory_stats = gr.Markdown("Click button to check RAM/VRAM stats")
            refresh_btn = gr.Button("🔄 Check RAM/VRAM Stats")
            refresh_btn.click(get_all_memory_stats, outputs=[memory_stats])
        
        gr.Markdown("""
<div style="display: flex; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">
  <a href="https://bagel-ai.org/"><img src="https://img.shields.io/badge/BAGEL-Website-0A66C2?logo=safari&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2505.14683"><img src="https://img.shields.io/badge/BAGEL-Paper-red?logo=arxiv&logoColor=red"/></a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT"><img src="https://img.shields.io/badge/BAGEL-Hugging%20Face-orange?logo=huggingface&logoColor=yellow"/></a>
  <a href="https://demo.bagel-ai.org/"><img src="https://img.shields.io/badge/BAGEL-Demo-blue?logo=googleplay&logoColor=blue"/></a>
  <a href="https://discord.gg/Z836xxzy"><img src="https://img.shields.io/badge/BAGEL-Discord-5865F2?logo=discord&logoColor=purple"/></a>
  <a href="mailto:bagel@bytedance.com"><img src="https://img.shields.io/badge/BAGEL-Email-D14836?logo=gmail&logoColor=red"/></a>
</div>
""")
    
    return demo

#%% Main Execution
parser = argparse.ArgumentParser(description="BAGEL Model Gradio Interface")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
args = parser.parse_args()

print("="*60)
print("BAGEL Model Initialization")
print("="*60)

# Load model
print("\n[1/5] Loading model configuration...")
model, vae_model, tokenizer, new_token_ids = load_model(args.model_path)

# Create transforms
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# Configure device map
print("\n[2/5] Configuring device map...")
device_map = create_device_map(model)
device_map = optimize_device_map(device_map)
print("Device map:")
for k, v_map in device_map.items():
    print(f"  {k}: {v_map}")

# Load checkpoint
print("\n[3/5] Loading model checkpoint...")
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(args.model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=False,
    offload_folder="offload",
    dtype=torch.bfloat16,
    force_hooks=False,
).eval()

# Apply quantization and compilation
print("\n[4/5] Applying quantization and compilation...")
model = apply_quantization_and_compile(model)

# Create inferencer
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

# Warmup
print("\n[5/5] Warming up model...")
warmup_model(inferencer)

# Create Gradio interface
print("\n" + "="*60)
print("Starting Gradio Interface")
print("="*60)
demo = create_gradio_interface(inferencer)

if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )
