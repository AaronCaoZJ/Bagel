# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache
import math
import re

# 上线--->故事模式上线前  :  全部 global
# 故事模式上线--->第一次大更前  :  全部 global
# 

# 全局变量，控制DEBUG模式是否开启
DEBUG_MODE = True

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = model.device

    def parse_weighted_prompt(self, prompt: str) -> Tuple[str, List[Tuple[int, str, float]]]:
        """
        Parse weighted tags from prompt and rebuild clean prompt.

        This function tracks the position of each weighted tag in the cleaned prompt
        to handle cases where the same tag appears multiple times (weighted and unweighted).

        Supports patterns:
        - (tag:1.5) - simple weighted tag
        - (gan yu:1.2) genshin impact - weighted tag with suffix text
        - (hu tao (genshin impact):2.0) - nested parentheses in tag name

        Args:
            prompt: Input text like "masterpiece, (masterpiece:1.2), solo, (best quality:1.5)"

        Returns:
            clean_prompt: Rebuilt prompt like "masterpiece, masterpiece, solo, best quality"
            weighted_tags: List of (position_in_clean, tag_text, weight) tuples
                          where position_in_clean is the index in the clean_items list
        """
        items = [item.strip() for item in prompt.split(',')]
        clean_items = []
        weighted_tags = []  # Store (position, tag, weight)

        # Pattern explanation:
        # ^\( - starts with opening paren
        # (.+) - capture tag (greedy, will capture everything until last : before number)
        # :([\d.]+) - colon followed by weight (digits and dots)
        # \) - closing paren
        # (.*)$ - optional suffix text after the closing paren
        weight_pattern = re.compile(r'^\((.+):([\d.]+)\)(.*)$')

        for item in items:
            match = weight_pattern.match(item)
            if match:
                tag = match.group(1).strip()
                weight = float(match.group(2))
                suffix = match.group(3).strip()

                # Combine tag with suffix if present
                # e.g., "(gan yu:1) genshin impact" -> "gan yu genshin impact"
                if suffix:
                    full_text = f"{tag} {suffix}"
                else:
                    full_text = tag

                position = len(clean_items)  # Current position in clean_items
                clean_items.append(full_text)
                weighted_tags.append((position, full_text, weight))
            else:
                clean_items.append(item)

        clean_prompt = ', '.join(clean_items)
        return clean_prompt, weighted_tags

    def map_tags_to_token_positions(self, clean_prompt: str, weighted_tags: List[Tuple[int, str, float]],
                                     tokenizer) -> Dict[int, float]:
        """
        Map weighted tags to their token positions after tokenization.

        Uses position information to correctly handle duplicate tags.

        Args:
            clean_prompt: The cleaned prompt string (comma-separated)
            weighted_tags: List of (position_in_clean, tag_text, weight) tuples
            tokenizer: The tokenizer to use

        Returns:
            token_weights: Dict mapping token positions to their weights
        """
        if not weighted_tags:
            return {}

        # Split clean prompt back into items to track positions
        clean_items = [item.strip() for item in clean_prompt.split(',')]

        # Tokenize the full clean prompt
        full_tokens = tokenizer.encode(clean_prompt)

        token_weights = {}

        # Build a cumulative token position map for each item
        # This tells us where each comma-separated item starts in the token sequence
        item_token_starts = []
        current_pos = 0

        for i, item in enumerate(clean_items):
            item_tokens = tokenizer.encode(item if i == 0 else ', ' + item)
            if i == 0:
                item_token_starts.append(0)
            else:
                # Account for comma and space
                current_pos += len(tokenizer.encode(', '))
                item_token_starts.append(current_pos)
                current_pos += len(tokenizer.encode(item))

        # For each weighted tag, use its position to find the exact token range
        for position, tag, weight in weighted_tags:
            if position >= len(item_token_starts):
                continue

            # Get token start position for this item
            token_start = item_token_starts[position]

            # Tokenize just this tag to get its length
            tag_tokens = tokenizer.encode(tag)
            tag_len = len(tag_tokens)

            # Apply weight to all tokens of this tag
            for j in range(tag_len):
                token_pos = token_start + j
                if token_pos < len(full_tokens):
                    token_weights[token_pos] = weight

        return token_weights

    def init_gen_context(self, num_images=1):
        gen_context = {
            'kv_lens': [0] * num_images,
            'ropes': [0] * num_images,
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    # 计算文本的 KV 并更新 gen_context
    @torch.no_grad()
    def update_context_text(self, text, gen_context, enable_tag_weighting=False):
        """
        计算文本的 KV 并更新 gen_context
        """
        # used for interleave data, currently only support 1 data inference,

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # Determine batch size from context
        batch_size = len(kv_lens)

        # Parse weighted tags if enabled
        token_weights = None
        if enable_tag_weighting:
            clean_text, weighted_tags = self.parse_weighted_prompt(text)
            if weighted_tags:
                token_weights = self.map_tags_to_token_positions(clean_text, weighted_tags, self.tokenizer)
                text = clean_text  # Use the cleaned text for tokenization

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text] * batch_size,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
            token_weights=token_weights,
        )

        # Move generation_input tensors to the same device as the model
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    # 计算图像的 KV 并更新 gen_context
    # 特别注意，图像会在VAE和VIT中二选一写入
    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        """
        计算图像的 KV 并更新 gen_context

        特别注意，除首图外。图像会在VAE和VIT中二选一写入
        """
        # used for interleave data, currently only support 1 data inference,

        # assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # Determine batch size from context
        batch_size = len(kv_lens)

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image] * batch_size,
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            # Move generation_input tensors to the same device as the model
            for k, v in generation_input.items():
                if isinstance(v, torch.Tensor):
                    generation_input[k] = v.to(self.device)
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image] * batch_size,
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            # Move generation_input tensors to the same device as the model
            for k, v in generation_input.items():
                if isinstance(v, torch.Tensor):
                    generation_input[k] = v.to(self.device)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",

        num_timesteps=50,
        timestep_shift=3.0,
        is_shortcut=False,
        dt_base=7.0,
        enable_taylorseer=False,
        fresh_threshold=4,
        max_order=6,
        first_enhance=5,
        num_images=1,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape] * num_images,
            new_token_ids=self.new_token_ids,
        ) 
        for k, v in generation_input.items():
            if isinstance(v, torch.Tensor):
                generation_input[k] = v.to(self.device)
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape] * num_images,
        )
        for k, v in generation_input_cfg_text.items():
            if isinstance(v, torch.Tensor):
                generation_input_cfg_text[k] = v.to(self.device)
        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape] * num_images,
        )
        for k, v in generation_input_cfg_img.items():
            if isinstance(v, torch.Tensor):
                generation_input_cfg_img[k] = v.to(self.device)
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            is_shortcut=is_shortcut,
            dt_base=dt_base,
            enable_taylorseer=enable_taylorseer,
            fresh_threshold=fresh_threshold,
            max_order=max_order,
            first_enhance=first_enhance,
        )

        image_list = []
        for i in range(num_images):
            image = self.decode_image(unpacked_latent[i], image_shape)
            image_list.append(image)
        return image_list if num_images > 1 else image_list[0]

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0, 
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        # image_shapes=(1536, 1536),
        story_mode=False,
        vae_pattern=None,
        vit_pattern=None,
        negative_prompt="",
        positive_prompt="",
        device="cuda:0",
        is_shortcut=False,
        dt_base=7.0,
        enable_taylorseer=False,
        fresh_threshold=3,
        max_order=6,
        first_enhance=10,
        num_images=1,
        tag_enhance=False
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context(num_images)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        cfg_negative_context = None
        vit_index = 0
        dt_base = 7.0
        if is_shortcut:
            num_timesteps = 2 ** int(math.log2(num_timesteps))  # Ensure num_timesteps is a power of 2
            print(f"Shortcut mode enabled: num_timesteps set to {num_timesteps}")
            dt_base = float(math.log2(num_timesteps))
        if story_mode:
            vae_index = 0
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            # 处理输入序列
            #
            # 特别注意，cfg_text_context 是一个截止到文本输入前的 context 副本
            # 在每段文本写入前做一次快照；在每张图像写入后也做一次快照。最后它通常停在“最后一段文本之前”（但包含最后一张图像）
            #
            # 而 cfg_img_context ，此项中反而没有任何图像，全部是文本
            # 例：input_lists = [img1, "换衣服", img2, "喝水", img3, "睡觉"]
            # 
            # 结尾时(这里的imgX都是经过vae/vit处理的token)：
            # 
            # main      = [img1, "换衣服", img2, "喝水", img3, "睡觉"]
            # cfg_text  = [img1, "换衣服", img2, "喝水", img3]
            # cfg_img   = ["换衣服", "喝水", "睡觉"]
            for index, input_term in enumerate(input_lists):
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context, enable_tag_weighting=tag_enhance)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context, enable_tag_weighting=tag_enhance)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    if story_mode is not True:
                        gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    else:
                        print("vae pattern:", vae_pattern[vae_index])
                        gen_context = self.update_context_image(input_term, gen_context, vae=(vae_pattern[vae_index] and not understanding_output), vit=vit_pattern[vit_index])
                        vae_index += 1
                        vit_index += 1
                    # image_shapes = input_term.size[::-1]
                    # 2025.10.14 泡泡喵修改，替换使用外部传入尺寸
                    cfg_text_context = deepcopy(gen_context)  # 使用参数中的image_shapes
                        
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            # 如果有负向提示词，创建负向context
            if negative_prompt and not understanding_output:
                cfg_negative_context = deepcopy(cfg_text_context)
                cfg_negative_context = self.update_context_text(negative_prompt, cfg_negative_context)

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                # 如果有负向提示词，使用负向context作为CFG text context
                effective_cfg_text_context = cfg_negative_context if cfg_negative_context is not None else cfg_text_context
                
                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=effective_cfg_text_context,
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    is_shortcut=is_shortcut,
                    dt_base=dt_base,
                    enable_taylorseer=enable_taylorseer,
                    fresh_threshold=fresh_threshold,
                    max_order=max_order,
                    first_enhance=first_enhance,
                    num_images=num_images,
                )

                if isinstance(img, list):
                    output_list.extend(img)
                else:
                    output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict

    