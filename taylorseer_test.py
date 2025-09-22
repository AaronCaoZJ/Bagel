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
import multiprocessing as mp

# ==============================================================================
# 1. 参数定义区: 定义您想测试的所有参数组合
# ==============================================================================
parameter_sets = [
    {'taylor_max_order': -1, 'taylor_first_enhance': -1, 'taylor_fresh_threshold': 1, 'enable_taylorseer': False},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 2, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 5, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 6, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 7, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 8, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 9, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 10, 'enable_taylorseer': True}, # N
    {'taylor_max_order': 6, 'taylor_first_enhance': 1, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 1, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 2, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 2, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 3, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 3, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 4, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 4, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},    
    {'taylor_max_order': 6, 'taylor_first_enhance': 6, 'taylor_fresh_threshold': 3, 'enable_taylorseer': True},
    {'taylor_max_order': 6, 'taylor_first_enhance': 6, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True}, # First Enchance
    {'taylor_max_order': 6, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 5, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 4, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 3, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 2, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 1, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True},
    {'taylor_max_order': 0, 'taylor_first_enhance': 5, 'taylor_fresh_threshold': 4, 'enable_taylorseer': True}, # 0
    {'taylor_max_order': 0, 'taylor_first_enhance': 1, 'taylor_fresh_threshold': 2, 'enable_taylorseer': True},
]

prompt = "In a futuristic city at night, a silver-haired cyborg girl leans against a towering skyscraper beneath a massive neon billboard. The glowing lights reflect cool tones on her metallic limbs and synthetic skin. She wears a tight tactical jacket, her expression emotionless, with red lights glowing faintly in her cybernetic eyes. The city is filled with flying vehicles, giant holographic ads, and pulsating neon signs. A thin mist fills the air. The image is in cyberpunk anime style, with high-contrast lighting and rich sci-fi details."
# prompt = "In a dazzling starry sky, a pink-haired girl in a gorgeous magical girl outfit raises a crystal-tipped wand high into the air. Her skirt flutters in the wind, and her hair is adorned with star-shaped accessories. The wand emits a radiant magical glow. She is surrounded by floating feathered wing effects and stardust particles, with a faint glowing magic circle beneath her. The image is in the magical girl anime style, with dreamy colors, sparkling light effects, and a heroic, elegant atmosphere."
# prompt = "A teenage boy in a deep-blue ninja outfit is leaping across Edo-style rooftops at sunset, facing forward with a clear front view of his face. Warm golden sunlight glows on the red tiled roofs, and a flock of crows flies in the distance. Red paper lanterns hang below in the narrow streets. The scene is cel-shaded anime style, with strong motion, layered silhouettes, and a cinematic sense of speed and atmosphere typical of Japanese TV animation."
# prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
# ==============================================================================
# 2. 定义将在子进程中运行的实验函数
# ==============================================================================
def run_experiment(params, experiment_index, total_experiments):
    """
    这个函数包含了单次实验的所有逻辑，它将在一个独立的进程中被执行。
    """
    import modeling.bagel.bagel
    import modeling.cache_utils.taylorseer
    import inferencer
    importlib.reload(modeling.bagel.bagel)
    importlib.reload(modeling.cache_utils.taylorseer)
    importlib.reload(inferencer)
    from modeling.bagel.bagel import Bagel # 重新导入类
    from inferencer import InterleaveInferencer # 重新导入类
    print("\n" + "="*80)
    print(f"Starting Experiment {experiment_index}/{total_experiments} in a new process with params: {params}")
    print("="*80)

    # --- 静态资源准备 ---
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
    output_dir = "/home/zhijun/Code/Bagel/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- 模型加载 ---
    print("--- Loading model for this experiment... ---")
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
    print("--- Model loaded successfully. ---")

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids
    )

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

    print(f"--- Experiment {experiment_index} finished. Process will now exit. ---")

# ==============================================================================
# 3. 主执行区: 启动并管理子进程
# ==============================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # # ====== Baseline inference (enable_taylorseer=False) ======
    # # 只运行一次，不随参数变化
    # import modeling.bagel.bagel
    # import modeling.cache_utils.taylorseer
    # import inferencer
    # importlib.reload(modeling.bagel.bagel)
    # importlib.reload(modeling.cache_utils.taylorseer)
    # importlib.reload(inferencer)
    # from modeling.bagel.bagel import Bagel
    # from inferencer import InterleaveInferencer

    # model_path = "/home/zhijun/Code/Bagel/models/BAGEL-7B-MoT/"
    # llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    # llm_config.qk_norm = True
    # llm_config.tie_word_embeddings = False
    # llm_config.layer_module = "Qwen2MoTDecoderLayer"
    # vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    # vit_config.rope = False
    # vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    # vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    # config = BagelConfig(
    #     visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
    #     vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
    #     latent_patch_size=2, max_latent_size=64,
    # )
    # tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    # tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    # vae_transform = ImageTransform(1024, 512, 16)
    # vit_transform = ImageTransform(980, 224, 14)
    # output_dir = "/home/zhijun/Code/Bagel/outputs"
    # os.makedirs(output_dir, exist_ok=True)

    # print("--- Running baseline inference (enable_taylorseer=False) ---")
    # with init_empty_weights():
    #     language_model = Qwen2ForCausalLM(llm_config)
    #     vit_model = SiglipVisionModel(vit_config)
    #     model = Bagel(language_model, vit_model, config)
    #     model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # max_mem_per_gpu = "31GiB"
    # device_map = infer_auto_device_map(
    #     model, max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    #     no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    # )
    # same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
    # if torch.cuda.device_count() == 1:
    #     first_device = device_map.get(same_device_modules[0], "cuda:0")
    #     for k in same_device_modules: device_map[k] = first_device
    # else:
    #     first_device = device_map.get(same_device_modules[0])
    #     for k in same_device_modules:
    #         if k in device_map: device_map[k] = first_device

    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=os.path.join(model_path, "ema.safetensors"), device_map=device_map,
    #     offload_buffers=True, dtype=torch.bfloat16, force_hooks=True, offload_folder="/tmp/offload"
    # )
    # model = model.eval()

    # inferencer = InterleaveInferencer(
    #     model=model, vae_model=vae_model, tokenizer=tokenizer,
    #     vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids
    # )

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # baseline_hyper = {
    #     'cfg_text_scale': 4.0, 'cfg_img_scale': 1.0, 'cfg_interval': [0.4, 1.0],
    #     'timestep_shift': 3.0, 'num_timesteps': 50, 'cfg_renorm_min': 0.0,
    #     'cfg_renorm_type': "global", 'enable_taylorseer': False,
    # }
    # print("\n--- Running baseline inference (enable_taylorseer=False) ---")
    # start_time = time.time()
    # baseline_output = inferencer(text=prompt, **baseline_hyper)
    # end_time = time.time()
    # duration = end_time - start_time

    # if 'image' in baseline_output and baseline_output['image']:
    #     prompt_folder_name = "".join(filter(str.isalnum, prompt))[:50]
    #     prompt_output_dir = os.path.join(output_dir, prompt_folder_name)
    #     os.makedirs(prompt_output_dir, exist_ok=True)
    #     duration_str = f"time{duration:.2f}s"
    #     file_name = f"baseline_{duration_str}.png"
    #     save_path = os.path.join(prompt_output_dir, file_name)
    #     baseline_output['image'].save(save_path)
    #     print(f"Baseline image saved to: {save_path}")

    # ====== 各参数实验循环 ======
    for i, params in enumerate(parameter_sets):
        p = mp.Process(target=run_experiment, args=(params, i + 1, len(parameter_sets)))
        p.start()
        p.join()

    print("\n" + "="*80)
    print("All experiments finished.")
    print("="*80)