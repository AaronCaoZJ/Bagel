import importlib
import os
from datetime import datetime
import time
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer
import random
import numpy as np
import gc

# ==============================================================================
# 1. 参数定义区: 定义您想测试的所有参数组合
# ==============================================================================
parameter_sets = [
    {'taylor_max_order': -1, 'taylor_first_enhance': -1, 'taylor_fresh_threshold': 1, 'enable_taylorseer': False},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 2, 'enable_taylorseer': True},
]

prompt = "In a futuristic city at night, a silver-haired cyborg girl leans against a towering skyscraper beneath a massive neon billboard. The glowing lights reflect cool tones on her metallic limbs and synthetic skin. She wears a tight tactical jacket, her expression emotionless, with red lights glowing faintly in her cybernetic eyes. The city is filled with flying vehicles, giant holographic ads, and pulsating neon signs. A thin mist fills the air. The image is in cyberpunk anime style, with high-contrast lighting and rich sci-fi details."
# prompt = "In a dazzling starry sky, a pink-haired girl in a gorgeous magical girl outfit raises a crystal-tipped wand high into the air. Her skirt flutters in the wind, and her hair is adorned with star-shaped accessories. The wand emits a radiant magical glow. She is surrounded by floating feathered wing effects and stardust particles, with a faint glowing magic circle beneath her. The image is in the magical girl anime style, with dreamy colors, sparkling light effects, and a heroic, elegant atmosphere."
# prompt = "A teenage boy in a deep-blue ninja outfit is leaping across Edo-style rooftops at sunset, facing forward with a clear front view of his face. Warm golden sunlight glows on the red tiled roofs, and a flock of crows flies in the distance. Red paper lanterns hang below in the narrow streets. The scene is cel-shaded anime style, with strong motion, layered silhouettes, and a cinematic sense of speed and atmosphere typical of Japanese TV animation."
# prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
# ==============================================================================
# 2. 定义模型加载函数
# ==============================================================================
def load_model_once():
    """
    一次性加载模型，返回所有需要的组件
    """
    print("\n" + "="*80)
    print("Loading model (this will only happen once)...")
    print("="*80)
    
    model_path = "/home/zhijun/Code/Bagel/models/BAGEL-7B-MoT/"
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
        vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    max_mem_per_gpu = "31GiB"
    device_map = infer_auto_device_map(
        model, max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules: device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map: device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model, checkpoint=os.path.join(model_path, "ema.safetensors"), device_map=device_map,
        offload_buffers=True, dtype=torch.bfloat16, force_hooks=True, offload_folder="/tmp/offload"
    )
    model = model.eval()
    
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids
    )
    
    print("--- Model loaded successfully. ---")
    return model, vae_model, inferencer, tokenizer, new_token_ids, vae_transform, vit_transform

def clear_taylorseer_cache(model):
    """
    清理taylorseer缓存，确保两次实验之间缓存被充分释放
    """
    print("--- Clearing TaylorSeer cache ---")
    
    # 清理语言模型中的taylorseer相关缓存
    if hasattr(model.language_model.model, 'enable_taylorseer'):
        model.language_model.model.enable_taylorseer = False
    
    if hasattr(model.language_model.model, 'cache_dic'):
        delattr(model.language_model.model, 'cache_dic')
    
    if hasattr(model.language_model.model, 'current'):
        delattr(model.language_model.model, 'current')
    
    # 清理各个层的缓存
    for layer in model.language_model.model.layers:
        if hasattr(layer, 'cache_dic'):
            delattr(layer, 'cache_dic')
        if hasattr(layer, 'current'):
            delattr(layer, 'current')
        if hasattr(layer, 'enable_taylorseer'):
            layer.enable_taylorseer = False
    
    # 强制进行垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("--- TaylorSeer cache cleared successfully ---")

# ==============================================================================
# 3. 定义单次实验函数（不再加载模型）
# ==============================================================================
def run_experiment(params, experiment_index, total_experiments, inferencer):
    """
    使用已加载的模型运行单次实验
    """
    print("\n" + "="*80)
    print(f"Starting Experiment {experiment_index}/{total_experiments} with params: {params}")
    print("="*80)

    output_dir = "/home/zhijun/Code/Bagel/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 设置随机种子 ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 构建超参数并推理 ---
    current_hyper = {
        'cfg_text_scale': 4.0, 'cfg_img_scale': 1.0, 'cfg_interval': [0.4, 1.0],
        'timestep_shift': 3.0, 'num_timesteps': 50, 'cfg_renorm_min': 0.0,
        'cfg_renorm_type': "global",
        'enable_taylorseer':  params.get('enable_taylorseer', True),
    }
    current_hyper.update(params)

    start_time = time.time()
    output_dict = inferencer(text=prompt, **current_hyper)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Inference time: {duration:.2f} seconds")

    # --- 保存图片 ---
    if 'image' in output_dict and output_dict['image']:
        prompt_folder_name = "".join(filter(str.isalnum, prompt))[:50]
        prompt_output_dir = os.path.join(output_dir, prompt_folder_name)
        os.makedirs(prompt_output_dir, exist_ok=True)
        order = current_hyper.get('taylor_max_order', 'na')
        enhance = current_hyper.get('taylor_first_enhance', 'na')
        threshold = current_hyper.get('taylor_fresh_threshold', 'na')
        params_str = f"o{order}_f{enhance}_N{threshold}"
        duration_str = f"time{duration:.2f}s"
        file_name = f"{params_str}_{duration_str}.png"
        save_path = os.path.join(prompt_output_dir, file_name)
        output_dict['image'].save(save_path)
        print(f"Image saved to: {save_path}")

    print(f"--- Experiment {experiment_index} finished. ---")

# ==============================================================================
# 4. 主执行区: 一次性加载模型并顺序运行所有实验
# ==============================================================================
if __name__ == '__main__':
    # 移除多进程设置，不再需要
    # mp.set_start_method('spawn', force=True)

    # 一次性加载模型和相关组件
    model, vae_model, inferencer, tokenizer, new_token_ids, vae_transform, vit_transform = load_model_once()

    # 可选：运行baseline inference (enable_taylorseer=False)
    # 这部分代码被注释掉了，如果需要可以取消注释
    # print("\n" + "="*80)
    # print("Running baseline inference (enable_taylorseer=False)")
    # print("="*80)
    # baseline_params = {'taylor_max_order': -1, 'taylor_first_enhance': -1, 'taylor_fresh_threshold': 1, 'enable_taylorseer': False}
    # run_experiment(baseline_params, 0, len(parameter_sets) + 1, inferencer)

    # ====== 各参数实验循环 ======
    print("\n" + "="*80)
    print(f"Starting {len(parameter_sets)} experiments with the loaded model...")
    print("="*80)
    
    for i, params in enumerate(parameter_sets):
        # 在每次实验开始前清理taylorseer缓存
        clear_taylorseer_cache(model)
        
        # 直接在主进程中调用实验函数，不使用多进程
        run_experiment(params, i + 1, len(parameter_sets), inferencer)
        
        # 实验结束后再次清理缓存以释放内存
        print("--- Post-experiment cleanup ---")
        clear_taylorseer_cache(model)
        
        # 额外的垃圾回收和GPU缓存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("All experiments finished.")
    print("="*80)