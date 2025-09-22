# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
export PYTHONPATH=/home/zhijun/Code/Bagel:$PYTHONPATH

num_nodes=1
node_rank=0
master_addr=localhost
master_port=12345

vae_path=./models/BAGEL-7B-MoT/ae.safetensors
vit_path=./models/BAGEL-7B-MoT/vit_config.json
llm_path=./models/BAGEL-7B-MoT/llm_config.json
resume_from=./models/BAGEL-7B-MoT/ema.safetensors
output_path=./results
ckpt_path=./checkpoints

cd /home/zhijun/Code/Bagel
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=2 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --model_path ./models/BAGEL-7B-MoT \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --max_latent_size 64  \
  --num_workers 1 \
  --batch_size 64 \