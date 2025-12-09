from safetensors import safe_open

with safe_open("/home/zhijun/Code/Bagel/models/BAGEL-7B-MoT/ema-FP8.safetensors", framework="pt") as f:
    dtypes = {k: f.get_tensor(k).dtype for k in f.keys()}
    print(set(dtypes.values()))