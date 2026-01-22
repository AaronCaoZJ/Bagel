#%% OK for w or w/o fp8 + compile + taylorseer
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
import sys
from typing import Dict, Tuple, Optional, List

import gradio as gr
import numpy as np

# ============================================================
# 重要：这些环境变量必须在 import torch 之前设置！
# ============================================================
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/zhijun/Code/Bagel/triton"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"  # 启用缓存

# 编译日志配置（在 import torch 前设置才有效）
# os.environ['TORCH_LOGS'] = '+dynamo,+recompiles,+graph_breaks'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
os.environ['TORCH_CUDAGRAPHS_VERBOSE'] = '0'

import torch
import torch._dynamo
import torch._inductor
# torch.compiler.reset()  # 清理之前的编译缓存

import random
from PIL import Image

# TorchAO quantization imports
# 需要安装对应版本的torchao https://github.com/pytorch/ao/issues/2919
from torchao.quantization import quantize_
from torchao.quantization import (
    float8_dynamic_activation_float8_weight, float8_weight_only,
    int8_weight_only, int4_weight_only, int8_dynamic_activation_int8_weight
)
# SafeTensors for model loading
from safetensors.torch import load_file

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

# Warmup configurations (simplified for dynamic mode)
# Dynamic compilation adapts to any size automatically

# compile模式选择：
# - "max-autotune": 最激进优化
# - "reduce-overhead": 平衡模式，减少 Python 开销
# - "default": 最安全模式
COMPILE_MODE = "default"
USE_FULLGRAPH = False  # Whether to use fullgraph compilation for the entire language model

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
    
    if torch.cuda.is_available():
        stats_lines.append("**GPU VRAM Usage (cuda:0):**")
        stats_lines.append(get_gpu_memory_stats_pynvml(0))
        stats_lines.append(get_gpu_memory_stats_pytorch(0))
    else:
        stats_lines.append("**GPU VRAM Usage:** CUDA not available")
    
    stats_lines.append("\n**CPU RAM Usage:**")
    stats_lines.append(get_system_ram_stats())
    stats_lines.append(get_process_ram_stats())
    
    return "\n".join(stats_lines)

#%% Model Loading and Quantization
def load_model(model_path: str):
    """
    Optimized loading: Initialize on Meta device -> Cast to BF16 (Meta) -> Materialize on GPU.
    """
    import torch
    from safetensors.torch import load_file
    
    print(f"Loading model from {model_path}...")

    # 1. 加载配置
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
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # =========================================================
    # ⚡️ 优化核心：Meta Device + Manual BF16 Cast + to_empty
    # =========================================================
    print("🚀 Initializing model on Meta device (zero memory)...")
    
    with torch.device("meta"):
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)

    print("🚀 Materializing model directly to GPU VRAM (bfloat16)...")
    
    # 1. 先将 Meta 模型转换为 BF16。
    #    这一步非常快，因为它只修改元数据（shape/stride/dtype），不涉及实际数据搬运。
    model = model.to(dtype=torch.bfloat16)
    
    # 2. 调用 to_empty 分配显存。
    #    此时 PyTorch 看到模型已经是 BF16，就会直接分配 BF16 的显存。
    model.to_empty(device="cuda:0")

    # 3. 执行结构修改 (必须在 to_empty 之后)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)

    # =========================================================
    # ⚡️ 优化加载：Direct Storage -> GPU VRAM
    # =========================================================
    print("🚀 Loading weights directly to GPU VRAM...")
    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    
    # 利用 PCIe 直接传输
    state_dict = load_file(checkpoint_path, device="cuda:0")
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # 处理 VAE (手动转到 GPU)
    print("Moving VAE to GPU...")
    vae_model = vae_model.to(dtype=torch.bfloat16, device="cuda:0")
    
    # VAE Safe Encode Patch (保持原样)
    _original_vae_encode = vae_model.encode
    def _safe_vae_encode(x):
        device = next(vae_model.parameters()).device
        if x.device != device:
            x = x.to(device)
        return _original_vae_encode(x)
    vae_model.encode = _safe_vae_encode
    
    print(f"✅ Model loaded successfully on {model.device}")
    
    # 清理
    torch.cuda.empty_cache()

    return model, vae_model, tokenizer, new_token_ids


def compile_bagel_single_gpu(model, mode="default", enable_taylorseer_compile=False, verbose=False):
    """
    为单GPU场景优化的 Bagel 模型编译策略
    
    Args:
        model: Bagel 模型实例
        mode: 编译模式 ('default', 'reduce-overhead', 'max-autotune')
        enable_taylorseer_compile: 是否启用 TaylorSeer 编译模式
        verbose: 是否打印详细信息
    
    Returns:
        编译后的模型
    """
    
    # ============================================================
    # 第0步：环境配置
    # ============================================================
    if verbose:
        print("=" * 60, flush=True)
        print("[🔧 COMPILE] 配置编译环境...", flush=True)
    
    # 核心配置
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.verbose = False
    torch._inductor.config.fx_graph_cache = True
    
    # 性能优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    
    # 过滤无用警告
    warnings.filterwarnings('ignore', message='.*cudagraphs.*')
    
    # 内存优化
    torch.cuda.empty_cache()
    gc.collect()
    
    if verbose:
        print(f"  ✓ 编译模式: {mode}", flush=True)
        print(f"  ✓ TF32: 已启用", flush=True)
        print(f"  ✓ FX Graph Cache: 已启用", flush=True)
        print(f"  ✓ 编译日志: 已启用（查看 stderr）", flush=True)
    
    try:
        compiled_layers = 0
        failed_layers = []
        
        for i, layer in enumerate(model.language_model.model.layers):
            try:
                model.language_model.model.layers[i] = torch.compile(
                    layer,
                    mode=mode,
                    fullgraph=False,
                    dynamic=True,
                )
                compiled_layers += 1
                if verbose and i % 5 == 0:  # 每5层打印一次
                    print(f"  → 已编译 {compiled_layers}/{len(model.language_model.model.layers)} 层...")
            except Exception as layer_e:
                failed_layers.append(i)
                if verbose:
                    print(f"    ⚠️ Layer {i} 编译失败: {layer_e}")
        
        if verbose:
            print(f"  ✅ 逐层编译完成: {compiled_layers}/{len(model.language_model.model.layers)} 层成功")
            if failed_layers:
                print(f"    失败的层: {failed_layers}")
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Language Model 编译失败: {e}")
            print("  → 保持 eager 模式")
    

    skip_modules = [
        ('latent_pos_embed', '图像位置编码（动态形状）'),
        ('vit_pos_embed', 'Vision 位置编码（动态形状）'),
        ('language_model.model.embed_tokens', 'Token 嵌入（查表操作）'),
        ('language_model.lm_head', '输出投影（已由 LM 编译）'),
    ]
    
    if verbose:
        print("\n[🚫 SKIP] 以下模块不编译（保持灵活性）:")
        for _, desc in skip_modules:
            print(f"  • {desc}")
    

    if verbose:
        print("\n[🧹 CLEANUP] 清理临时资源...")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if verbose:
        print("=" * 60)
        print("✅ 编译配置完成！首次推理将触发实际编译（30-60秒）")
        print("   后续推理将直接使用编译缓存（2-4x 加速）")
        print("=" * 60)
    
    return model


def apply_quantization_and_compile(model, enable_taylorseer_compile=False):
    """Apply quantization and torch.compile optimizations.
    
    Args:
        model: Model to quantize and compile
        enable_taylorseer_compile: If True, configure for TaylorSeer compatibility
    """
    print("\n" + "=" * 60)
    print("[🧩 QUANTIZATION] 开始量化...")
    print("=" * 60)
    
    # print("[🧩 QUANT] Converting to FP8 quantization...")
    # quantize_(model, float8_dynamic_activation_float8_weight())
    
    # print("[🧩 QUANT] Converting to channels_last memory format...")
    # if hasattr(model, 'language_model'):
    #     model.language_model = model.language_model.to(memory_format=torch.channels_last)
    
    # print("[🧩 QUANT] 量化完成！")
    
    if enable_taylorseer_compile:
        print("[🛠️ COMPILE] ⚠️  TaylorSeer 模式：使用编译（仅层级别）")
        print("[🛠️ COMPILE] ℹ️  原因：TaylorSeer 的动态控制流（full/Taylor 切换）需要 Python 层处理")
    
    # 清理量化临时数据
    print("[🛁 CLEANUP] 清理量化临时数据...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 使用增强的编译策略
    print("\n[🛠️ COMPILE] 开始编译...")
    model = compile_bagel_single_gpu(model, mode=COMPILE_MODE, enable_taylorseer_compile=enable_taylorseer_compile)
    
    return model

#%% Model Warmup
def warmup_model(inferencer, enable_taylorseer_compile=False):
    """
    简化预热：启用dynamic=True后，只需预热一次即可支持所有尺寸
    Dynamic compilation will automatically adapt to any input size
    
    Args:
        inferencer: The model inferencer
        enable_taylorseer_compile: If True, also warmup TaylorSeer compilation paths
    """
    print("\n" + "="*60)
    print("[WARMUP] Starting simplified warmup (dynamic mode)...")
    print("="*60)
    
    total_start = time.time()
    
    try:
        if not enable_taylorseer_compile:
            # 单次文生图预热
            print(f"\n[🔥 WARMUP] Text-to-image warmup (10 steps, 1024x1024)...", flush=True)
            print(f"[🔥 WARMUP] ⚠️ 首次运行将触发实际编译，请耐心等待...", flush=True)
            print(f"[🔥 WARMUP] 💡 编译日志会输出到 stderr（可能在终端看到大量日志）", flush=True)
            print(f"[🔥 WARMUP] 开始推理...", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
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
        
        else:
            print(f"\n[🔥 WARMUP] TaylorSeer mode: warming up sub-module compilation...")
            print(f"[🔥 WARMUP] Using fixed params: taylor_first_enhance=10, taylor_max_order=6, taylor_fresh_threshold=3")
            
            # Text-to-Image with TaylorSeer
            print(f"[🔥 WARMUP]   - Text-to-Image path (41 steps, 1024x1024)...")
            start = time.time()
            with torch.no_grad():
                _ = inferencer(
                    text="TaylorSeer T2I warmup",
                    image_shapes=(1024, 1024),
                    num_timesteps=41,
                    max_think_token_n=512,
                    think=False,
                    cfg_text_scale=4.0,
                    cfg_interval=[0.4, 1.0],
                    timestep_shift=3.0,
                    enable_taylorseer=True,
                    taylor_first_enhance=10,
                    taylor_max_order=6,
                    taylor_fresh_threshold=3,
                )
            print(f"  ✅ Taylorseer T2I done in {time.time() - start:.1f}s")
            
            # 清理激活值
            del _
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 重新初始化 TaylorSeer 缓存
            print(f"[🛁 CLEANUP]   - Reinitializing TaylorSeer cache...")
            if hasattr(inferencer, 'model'):
                model_to_clear = inferencer.model
            else:
                model_to_clear = inferencer
            
            reset_taylorseer_cache_simple(
                model_to_clear,
                num_steps=41,
                taylor_fresh_threshold=3,
                taylor_first_enhance=10,
                taylor_max_order=6
            )
            
            # 强制同步并清理显存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            print(f"[🛁 CLEANUP]   - TaylorSeer cache cleared between T2I and I2I")
            
            # Image-to-Image with TaylorSeer
            print(f"[🔥 WARMUP]   - Image-to-Image path (41 steps, 1024x768)...")
            dummy_img_ts = Image.new('RGB', (1024, 768), color='gray')
            start = time.time()
            with torch.no_grad():
                _ = inferencer(
                    image=dummy_img_ts,
                    text="TaylorSeer I2I warmup",
                    num_timesteps=41,
                    max_think_token_n=1024,
                    think=False,
                    cfg_text_scale=4.0,
                    cfg_img_scale=2.0,
                    cfg_interval=[0.0, 1.0],
                    timestep_shift=3.0,
                    cfg_renorm_min=0.0,
                    cfg_renorm_type="text_channel",
                    enable_taylorseer=True,
                    taylor_first_enhance=10,
                    taylor_max_order=6,
                    taylor_fresh_threshold=3,
                )
            print(f"  ✅ Taylorseer I2I done in {time.time() - start:.1f}s")

            # 清理warmup产生的激活值
            del dummy_img_ts
            del _
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 最终清理
            print(f"[🛁 CLEANUP]   - Final TaylorSeer cache reset...")
            reset_taylorseer_cache_simple(
                model_to_clear,
                num_steps=41,
                taylor_fresh_threshold=3,
                taylor_first_enhance=10,
                taylor_max_order=6
            )
            print(f"  [🛁 CLEANUP] Memory after TaylorSeer warmup: {get_gpu_memory_stats_pytorch(0)}")

            print(f"  ✅ TaylorSeer warmup complete (both T2I and I2I paths compiled)")
                    
        # 最终清理：只清理warmup的激活值，保留编译的计算图和kernel缓存
        # torch._inductor.config.fx_graph_cache=True 会把编译结果存在磁盘
        # empty_cache() 只释放未使用的显存，不会删除编译好的代码
        gc.collect()
        if torch.cuda.is_available():
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


def reset_taylorseer_cache_simple(model, num_steps=41, taylor_fresh_threshold=3, taylor_first_enhance=10, taylor_max_order=6):
    """
    重置 TaylorSeer 缓存。
    由于 load_model 已优化且 simple_cache_init 只创建字典，此处无需再进行 dtype 强制转换。
    """
    from modeling.cache_utils.taylorseer import simple_cache_init
    
    # 1. 检查路径有效性
    if not (hasattr(model, 'language_model') and hasattr(model.language_model, 'model')):
        return
    
    llm_model = model.language_model.model

    # 2. 构造临时上下文以适配 simple_cache_init 的调用方式
    # simple_cache_init 需要访问 self.language_model.model.layers
    class TempContext:
        def __init__(self, language_model):
            self.language_model = language_model
    
    temp_ctx = TempContext(model.language_model)
    
    # 3. 初始化缓存 (只生成字典配置，不涉及 Tensor 创建)
    # 注意：这里直接使用 simple_cache_init，它比原先的 cache_init 更轻量
    raw_cache_dic, current = simple_cache_init(
        temp_ctx, 
        num_steps=num_steps,
        taylor_fresh_threshold=taylor_fresh_threshold,
        taylor_first_enhance=taylor_first_enhance,
        taylor_max_order=taylor_max_order
    )
    
    # 4. 直接赋值
    llm_model.cache_dic = raw_cache_dic
    llm_model.current = current


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
    
    if kwargs.get("enable_taylorseer", False):
        reset_taylorseer_cache_simple(
            inferencer.model,
            num_steps=kwargs.get("num_timesteps", 41),
            taylor_fresh_threshold=kwargs.get("taylor_fresh_threshold", 3),
            taylor_first_enhance=kwargs.get("taylor_first_enhance", 10),
            taylor_max_order=kwargs.get("taylor_max_order", 6),
        )
    
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
        "enable_taylorseer": kwargs.get("enable_taylorseer", False),
        "taylor_first_enhance": kwargs.get("taylor_first_enhance", 10),
        "taylor_max_order": kwargs.get("taylor_max_order", 6),
        "taylor_fresh_threshold": kwargs.get("taylor_fresh_threshold", 3),
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
    
    if kwargs.get("enable_taylorseer", False):
        reset_taylorseer_cache_simple(
            inferencer.model,
            num_steps=kwargs.get("num_timesteps", 41),
            taylor_fresh_threshold=kwargs.get("taylor_fresh_threshold", 3),
            taylor_first_enhance=kwargs.get("taylor_first_enhance", 10),
            taylor_max_order=kwargs.get("taylor_max_order", 6),
        )
    
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
        "enable_taylorseer": kwargs.get("enable_taylorseer", False),
        "taylor_first_enhance": kwargs.get("taylor_first_enhance", 10),
        "taylor_max_order": kwargs.get("taylor_max_order", 6),
        "taylor_fresh_threshold": kwargs.get("taylor_fresh_threshold", 3),
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
                
                # TaylorSeer Parameters
                with gr.Row():
                    enable_taylorseer = gr.Checkbox(label="Enable TaylorSeer", value=False)
                
                taylorseer_params = gr.Group(visible=False)
                with taylorseer_params:
                    gr.Markdown("**TaylorSeer Configuration** (requires enable_taylorseer=True)")
                    with gr.Row():
                        taylor_first_enhance = gr.Slider(5, 20, 10, step=1, label="First Enhance Steps")
                        taylor_max_order = gr.Slider(3, 10, 6, step=1, label="Max Taylor Order")
                        taylor_fresh_threshold = gr.Slider(1, 10, 3, step=1, label="Fresh Threshold")
                
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
            
            enable_taylorseer.change(
                lambda x: gr.update(visible=x),
                inputs=[enable_taylorseer],
                outputs=[taylorseer_params]
            )
            
            def process_t2i(*args):
                params = dict(zip([
                    "prompt", "show_thinking", "cfg_text_scale", "cfg_interval",
                    "timestep_shift", "num_timesteps", "cfg_renorm_min", "cfg_renorm_type",
                    "max_think_token_n", "do_sample", "text_temperature", "seed", "image_ratio",
                    "enable_taylorseer", "taylor_first_enhance", "taylor_max_order", "taylor_fresh_threshold"
                ], args))
                return text_to_image(inferencer, **params)
            
            gr.on(
                [gen_btn.click, txt_input.submit],
                process_t2i,
                inputs=[txt_input, show_thinking, cfg_text_scale, cfg_interval, timestep_shift,
                       num_timesteps, cfg_renorm_min, cfg_renorm_type, max_think_token_n,
                       do_sample, text_temperature, seed, image_ratio,
                       enable_taylorseer, taylor_first_enhance, taylor_max_order, taylor_fresh_threshold],
                outputs=[img_output, thinking_output, generation_time]
            )
        
        # Image Edit Tab
        with gr.Tab("🖌️ Image Edit"):
            with gr.Row():
                with gr.Column(scale=1):
                    edit_img_input = gr.Image(label="Input Image", value=load_example_image('test_images/display_images/__inugami_korone_hololive_drawn_by_risu_risuuu_q__sample-def4cbd83e9632be79a2badc97747e32.jpg'))
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
                
                # TaylorSeer Parameters
                with gr.Row():
                    edit_enable_taylorseer = gr.Checkbox(label="Enable TaylorSeer", value=False)
                
                edit_taylorseer_params = gr.Group(visible=False)
                with edit_taylorseer_params:
                    gr.Markdown("**TaylorSeer Configuration** (requires enable_taylorseer=True)")
                    with gr.Row():
                        edit_taylor_first_enhance = gr.Slider(5, 20, 10, step=1, label="First Enhance Steps")
                        edit_taylor_max_order = gr.Slider(3, 10, 6, step=1, label="Max Taylor Order")
                        edit_taylor_fresh_threshold = gr.Slider(1, 10, 3, step=1, label="Fresh Threshold")
                
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
            
            edit_enable_taylorseer.change(
                lambda x: gr.update(visible=x),
                inputs=[edit_enable_taylorseer],
                outputs=[edit_taylorseer_params]
            )
            
            def process_edit(*args):
                params = dict(zip([
                    "image", "prompt", "show_thinking", "cfg_text_scale", "cfg_img_scale",
                    "cfg_interval", "timestep_shift", "num_timesteps", "cfg_renorm_min",
                    "cfg_renorm_type", "max_think_token_n", "do_sample", "text_temperature", "seed",
                    "enable_taylorseer", "taylor_first_enhance", "taylor_max_order", "taylor_fresh_threshold"
                ], args))
                return edit_image(inferencer, **params)
            
            gr.on(
                [edit_btn.click, edit_prompt.submit],
                process_edit,
                inputs=[edit_img_input, edit_prompt, edit_show_thinking, edit_cfg_text_scale,
                       edit_cfg_img_scale, edit_cfg_interval, edit_timestep_shift,
                       edit_num_timesteps, edit_cfg_renorm_min, edit_cfg_renorm_type,
                       edit_max_think_token_n, edit_do_sample, edit_text_temperature, edit_seed,
                       edit_enable_taylorseer, edit_taylor_first_enhance, edit_taylor_max_order, edit_taylor_fresh_threshold],
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
parser.add_argument("--model_path", type=str, default="/root/Miko_share/Kokoro/models/BAGEL-7B-MoT")
args = parser.parse_args()

print("="*60)
print("BAGEL Model Initialization")
print("="*60)

# Load model (直接加载，不使用accelerate)
print("\n[1/3] Loading model configuration and weights...")
model, vae_model, tokenizer, new_token_ids = load_model(args.model_path)
model = model.eval()

# Create transforms
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# Apply quantization and compilation
print("\n[2/3] Applying quantization and compilation...")
ENABLE_TAYLORSEER = True  # Set to True to enable TaylorSeer-compatible compilation
model = apply_quantization_and_compile(model, enable_taylorseer_compile=ENABLE_TAYLORSEER)

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
print("\n[3/3] Warming up model...")
warmup_model(inferencer, enable_taylorseer_compile=ENABLE_TAYLORSEER)

# 注册退出清理函数
def cleanup_on_exit():
    print("\nCleaning up resources...")
    global model, vae_model, inferencer
    try:
        if 'model' in globals():
            del model
        if 'vae_model' in globals():
            del vae_model
        if 'inferencer' in globals():
            del inferencer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Resources cleaned up successfully.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

atexit.register(cleanup_on_exit)

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
