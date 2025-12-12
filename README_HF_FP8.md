---
license: apache-2.0
base_model:
  - ByteDance-Seed/BAGEL-7B-MoT
base_model_relation: quantized
pipeline_tag: any-to-any
tags:
  - fp8
  - bagel
---
**[BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) quantized with [TorchAO](https://github.com/pytorch/ao) float8 weight only quantization, using default Round-to-Nearest algorithm and Symmetric Max-Abs Scaling.**

## ⏳ Notice 
### This model is a pickle `.pt` file because traditionally we serialize and distribute TorchAO quantized model with PyTorch native APIs, specifically:
```python
torch.save(model.state_dict())
state_dict = torch.load("model_fp8.pt", weights_only=True)
model.load_state_dict(state_dict, assign=True)
```
### TorchAO has supported safetensor
.safetensors file TOTO

---

**The following is just an example of using BF16 OG weights and utilizing TorchAO for online dynamic quantization to achieve FP8 inference.**

## 📊 Inference Experiment
### On 2*RTX5090 with 24GiB VRAM
Can save about 39% VRAM, and accelerate model inference for about 10%.
![image](assets/5090.png)
### On 1*H100 with 80GiB VRAM
TODO


## 🍩 Quick Start
### Set up Environment for Bagel
```bash
git clone https://github.com/AaronCaoZJ/BAGEL.git  # Forked from OG ByteDance-Seed/Bagel
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install packaging ninja
pip install flash-attn==2.8.3 --no-build-isolation  # FlashAttention only supports Ampere GPUs or newer
```

### Download Pretrained Checkpoint
```python
from huggingface_hub import snapshot_download
save_dir = "models/BAGEL-7B-MoT"
repo_id = "aaroncaozj/BAGEL-7B-MoT_FP8"
cache_dir = save_dir + "/cache"
snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", ".pt", "*.bin", "*.py", "*.md", "*.txt"],)
```

### Use Gradio WebUI to Play with BAGEL
```bash
# For 32GB+ VRAM GPU or multi GPUs.
python app-torchao-fp8.py
```