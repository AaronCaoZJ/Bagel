import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm

def consistency_distill(
    self,
    student,                       # student: Bagel (or compatible) model (unfrozen)
    dataloader,                    # yields batch of PIL images (or preprocessed tensors)
    optimizer,
    device: torch.device = torch.device("cuda"),
    epochs: int = 1,
    steps_per_epoch: Optional[int] = None,
    timestep_shift: float = 1.0,
    sample_t_per_step: bool = True,
    cfg_text_scale: float = 1.0,
    cfg_img_scale: float = 1.0,
    new_token_ids: Optional[Dict[str, int]] = None,
    log_interval: int = 50,
    save_path: Optional[str] = None,
):
    """
    Simple consistency distillation loop:
    - teacher = self (frozen, eval)
    - student: trainable model (should have same _forward_flow API)
    - dataloader yields a batch of PIL images or tensors (HWC or CHW).
    - new_token_ids must contain 'start_of_image' and 'end_of_image' token ids used by prepare_vae_latent.
    """

    if new_token_ids is None:
        # WARNING: replace with actual tokenizer tokens in real runs
        new_token_ids = {"start_of_image": 1, "end_of_image": 2}

    self.eval()
    student.train()
    student.to(device)

    # freeze teacher params
    for p in self.parameters():
        p.requires_grad = False

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            # Accept either list of PIL images or tensor batch (C,H,W) or (H,W,C)
            images = batch
            if isinstance(images, (list, tuple)):
                image_sizes = []
                for im in images:
                    if hasattr(im, "size"):  # PIL.Image
                        W, H = im.size
                        image_sizes.append((H, W))
                    else:
                        # assume tensor CHW
                        image_sizes.append((im.shape[1], im.shape[2]))
            else:
                # single tensor batch -> turn into list
                # expect shape (B, C, H, W)
                imgs = images
                images = [imgs[i] for i in range(imgs.shape[0])]
                image_sizes = [(img.shape[1], img.shape[2]) for img in images]

            batch_size = len(images)
            curr_kvlens = [0] * batch_size
            curr_rope = [0] * batch_size

            # prepare generation inputs (uses Bagel.prepare_vae_latent)
            generation_input = self.prepare_vae_latent(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                image_sizes=image_sizes,
                new_token_ids=new_token_ids,
            )

            # move tensors to device
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)

            packed_init_noises = generation_input["packed_init_noises"].to(device)
            packed_vae_position_ids = generation_input["packed_vae_position_ids"].to(device)
            packed_vae_token_indexes = generation_input["packed_vae_token_indexes"].to(device)
            packed_seqlens = generation_input["packed_seqlens"].to(device)
            packed_position_ids = generation_input["packed_position_ids"].to(device)
            packed_indexes = generation_input["packed_indexes"].to(device)
            packed_key_value_indexes = generation_input["packed_key_value_indexes"].to(device)
            key_values_lens = generation_input["key_values_lens"].to(device)
            packed_text_ids = generation_input["packed_text_ids"].to(device)
            packed_text_indexes = generation_input["packed_text_indexes"].to(device)

            # sample timestep t in (0,1) or use schedule (teacher produced using same mapping as generate_image)
            if sample_t_per_step:
                t_raw = torch.rand(1, device=device)
                t = timestep_shift * t_raw / (1 + (timestep_shift - 1) * t_raw)
            else:
                t_raw = torch.tensor([0.5], device=device)
                t = timestep_shift * t_raw / (1 + (timestep_shift - 1) * t_raw)
            timesteps = t.repeat(packed_init_noises.shape[0])  # one timestep per packed_init_noises row
            # make into shaped input expected by _forward_flow (the method expects timestep shaped [batch])
            timestep_tensor = timesteps

            # teacher forward -> v_teacher (no grad)
            with torch.no_grad():
                v_teacher = self._forward_flow(
                    x_t=packed_init_noises,
                    timestep=timestep_tensor,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_indexes=packed_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=None,
                    packed_key_value_indexes=packed_key_value_indexes,
                    cfg_text_scale=cfg_text_scale,
                    cfg_text_packed_position_ids=None,
                    cfg_text_packed_query_indexes=None,
                    cfg_text_key_values_lens=None,
                    cfg_text_past_key_values=None,
                    cfg_text_packed_key_value_indexes=None,
                    cfg_img_scale=cfg_img_scale,
                    cfg_img_packed_position_ids=None,
                    cfg_img_packed_query_indexes=None,
                    cfg_img_key_values_lens=None,
                    cfg_img_past_key_values=None,
                    cfg_img_packed_key_value_indexes=None,
                    cfg_type="parallel",
                    model_pred_cache_dic=None,
                    model_pred_current=None,
                    model_pred_text_cache_dic=None,
                    model_pred_text_current=None,
                    model_pred_img_cache_dic=None,
                    model_pred_img_current=None,
                )
                # ensure detached target
                v_teacher = v_teacher.detach()

            # student forward -> v_student (grad)
            student.to(device)
            student.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                v_student = student._forward_flow(
                    x_t=packed_init_noises,
                    timestep=timestep_tensor,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_indexes=packed_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_seqlens=packed_seqlens,
                    key_values_lens=key_values_lens,
                    past_key_values=None,
                    packed_key_value_indexes=packed_key_value_indexes,
                    cfg_text_scale=cfg_text_scale,
                    cfg_text_packed_position_ids=None,
                    cfg_text_packed_query_indexes=None,
                    cfg_text_key_values_lens=None,
                    cfg_text_past_key_values=None,
                    cfg_text_packed_key_value_indexes=None,
                    cfg_img_scale=cfg_img_scale,
                    cfg_img_packed_position_ids=None,
                    cfg_img_packed_query_indexes=None,
                    cfg_img_key_values_lens=None,
                    cfg_img_past_key_values=None,
                    cfg_img_packed_key_value_indexes=None,
                    cfg_type="parallel",
                    model_pred_cache_dic=None,
                    model_pred_current=None,
                    model_pred_text_cache_dic=None,
                    model_pred_text_current=None,
                    model_pred_img_cache_dic=None,
                    model_pred_img_current=None,
                )

                # loss: MSE between student v and teacher v
                loss = F.mse_loss(v_student, v_teacher)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % log_interval == 0:
                print(f"[distill] epoch={epoch} step={batch_idx} loss={loss.item():.6f}")

            global_step += 1

        # end epoch
        if save_path is not None:
            torch.save(student.state_dict(), f"{save_path}.ep{epoch}.pth")

    # final save
    if save_path is not None:
        torch.save(student.state_dict(), save_path)

    return student