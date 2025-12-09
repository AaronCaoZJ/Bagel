# 🔥 Try TensorRT
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

## Convert ResNet50 to TensorRT
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

# ~~Run Bagel with TensorRT~~
## Install Dependence
1. **Install tensorrt py package**
    ```
    python3 -m pip install --upgrade pip
    python3 -m pip install wheel
    python3 -m pip install --upgrade tensorrt
    
    ```
2. **Install onnx, cuda-python**
    ```
    # pip install onnx
    pip install cuda-python==12.4.0
    ```
    should see **cudart.py** in current_env/lib/python/site-packages/cuda

## Convert Model to TensorRT
### ResNet50 as example
1. **Export model to ONNX**
    ```
    torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)
    ```
2. **Build engine from ONNX**
    * trtexec does not install with the tensorrt py package, use tensorrt.Bulider to avoid conflict
    ```
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
    * If do need to build with trtexec
    ```
    # step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
    if USE_FP16:
        !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt   --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    else:
        !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt 
    ```
    * !!! DataType.FLOAT/HALF big difference !!!



