Build Faster Bagel
# 🍩 Quick Start
## Set up Environment
```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
# FlashAttention only supports Ampere GPUs or newer 
pip install flash_attn==2.5.8 --no-build-isolation
```
## Download Pretrained Checkpoint
```python
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],)
```
## Use Gradio WebUI to Play with BAGEL
```bash
# For 32GB+ VRAM GPU or multi GPUs.
python app.py
```
```bash
# For 12~32GB VRAM GPU, recommend using NF4 quantization. And use Chinese interface.
python app.py --mode 2 --zh
```
```bash
# For 22~32GB VRAM GPU, not recommended to use INT8 quantization.
python app.py  --mode 3
```

<br>

# 🦄 Use TaylorSeer
## Args
* `taylor_max_order`: Taylor factor (the maximum order of high-order differences between outputs of each layer).
* `taylor_first_enhance`: The step from which to start calculating the Taylor factor.
* `taylor_fresh_threshold`: AKA "N", How many steps to interval for caching and factor refreshing.

## Experiment on img_edit task
* 1024\*1024px picture, inference on 1 H100 machine.
* Using a private fine-tuned Bagel based model.
* Using parameters that balance time and quality, compare with the original 50-step sampling.

| num_steps | max_order | first_enhance | N | time/s |
| :-------: | :------: | :-----------: | :-: | -----: |
| 50 w/o TS | - | - | -- | 25.97 |
| 41 w/ TS | 6 | 10 | 3 | 🥇15.29 |
| 39 w/ TS | 6 | 8 | 2 | 🥈17.06 |


<br>

# 🔥 Use TensorRT
## Install Dependence
### Install tensorrt py package
```bash
python3 -m pip install --upgrade pip
python3 -m pip install wheel
python3 -m pip install --upgrade tensorrt
```
### Install onnx & cuda-python
```bash
# pip install onnx
pip install cuda-python==12.4.0
```
Should see **cudart.py** in current_env/lib/python/site-packages/cuda.

## Convert Model to TensorRT
*ResNet50 as example*
### Export model to ONNX
```python
torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)
```
### Build engine from ONNX
`trtexec` does not install with the tensorrt py package, use tensorrt.Bulider to avoid conflict
```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

onnx_path = "resnet50_pytorch.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model!")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
# config.set_flag(trt.BuilderFlag.FP16)  # FP16

serialized_engine = builder.build_serialized_network(network, config)
assert serialized_engine is not None, "Failed to build TensorRT engine!"

with open("resnet_engine_pytorch.trt", "wb") as f:
    f.write(serialized_engine)
```
If do need to build with trtexec
```python
# step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
if USE_FP16:
    !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt   --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
else:
    !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt 
```

<br>

# 💪 Train & Eval
## Train
```bash
bash scripts/train.sh
```
You can replace the variables in the script with your own before running. 
See [TRAIN](TRAIN.md) for more details.
## Eval
We provide the scripts for evaluating VLM, T2I and Editing benchmarks. 
Please See [EVAL](EVAL.md) for more details.

<br>

# 📊 Benchmarks
## Visual Understanding
| Model | MME | MMBench |   MMMU | MM-Vet | MathVista |
| ------------------- | ----------: | ----------: | -------: | -------: | ----------: |
| Janus-Pro-7B        | -  |     79.2 |     41.0 |     50.0 |           – |
| Qwen2.5-VL-7B      | 2347    |   83.5 | **58.6** |     67.1 |           68.2 |
| **BAGEL**    | **2388**  |  **85.0** |     55.3 | **67.2** |    **73.1** |
## Text-to-Image Generation
| Model        | GenEval | WISE |
| ------------ | --------- | --------- |
| Janus-Pro-7B | 0.80      | 0.35 | 
| SD3-Medium   | 0.74      | - |
| FLUX-1-dev   | 0.82      | 0.50 |
| **BAGEL**    | 0.82  | 0.52  |
| **BAGEL + Rewritter/CoT**    | **0.88**  | **0.70** |
## Image Editing
| Model         | GEdit-Bench-EN (SC) | GEdit-Bench-EN (PQ) | GEdit-Bench-EN (O) | IntelligentBench | KISE-Bench | RISEBench |
| ------------- | ---------------------: | ---------------------: | -------------------: | ------------------: | ------------: | ------------: | 
| Step1X-Edit   | 🥉7.09                | 🥉6.76                | 🥈6.70            | 14.9               |  43.29   |  1.9  |
| Gemini 2.0    | 6.73                  | 6.61                  | 6.32                | 🥈57.6             | 🥈62.41   |  🥈13.3  |
| GPT-4o        | 🥇7.85              | 🥇7.62              | 🥇7.53            | 🥇78.9           | 🥇80.09   |  🥇28.9  |
| **BAGEL**     | 🥈7.36                | 🥈6.83                | 🥉6.52                | 44.0               |  56.21   |  6.1 |
| **BAGEL+CoT** | –                     | –                     | –                   | 🥉55.3             |  🥉60.18   |  🥉11.9 |
