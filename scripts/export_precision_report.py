import torch

def export_precision_report(model, filename="model_precision_report.txt"):
    print(f"[QUANT-REPORT] Generating precision report to {filename}...")
    
    def get_real_dtype_str(param):
        # 1. 尝试访问 torchao 特有的 _data 属性 (最直接)
        if hasattr(param, "_data") and isinstance(param._data, torch.Tensor):
            dt = param._data.dtype
            if dt == torch.float8_e4m3fn: return "F8_E4M3"
            if dt == torch.float8_e5m2: return "F8_E5M2"
            return f"AO_Data({str(dt)})"

        # 2. 针对 Float8Tensor 的特殊解包逻辑
        # 有些版本的 torchao 把数据放在 _elem 或者通过 __tensor_flatten__ 暴露
        try:
            # 尝试通过这种方式获取底层 tensor
            if hasattr(param, "__tensor_flatten__"):
                field_names, _ = param.__tensor_flatten__()
                if field_names:
                    # 通常第一个字段就是数据，比如 '_data' 或 '_elem'
                    data_attr = getattr(param, field_names[0], None)
                    if isinstance(data_attr, torch.Tensor):
                        dt = data_attr.dtype
                        if dt == torch.float8_e4m3fn: return "F8_E4M3"
                        if dt == torch.float8_e5m2: return "F8_E5M2"
        except:
            pass

        # 3. 如果都拿不到，回退到类名推断
        type_name = type(param).__name__
        if "Float8" in type_name:
            return "F8_E4M3 (Standard for Weights)" # 权重几乎总是 E4M3
            
        # ... 标准 dtype 检查 ...
        dtype = param.dtype
        if dtype == getattr(torch, "float8_e4m3fn", None): return "F8_E4M3"
        if dtype == getattr(torch, "float8_e5m2", None): return "F8_E5M2"
        if dtype == torch.bfloat16: return "BF16"
        if dtype == torch.float16: return "FP16"
        if dtype == torch.float32: return "FP32"
        
        return str(dtype).replace("torch.", "")
    

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'Layer Name':<60}\t{'Shape':<20}\t{'Device':<10}\t{'Real Precision'}\n")
        f.write("-" * 110 + "\n")
        
        for name, param in model.named_parameters():
            shape_str = str(list(param.shape))
            # 使用新的检测函数
            dtype_str = get_real_dtype_str(param)
            device_str = str(param.device)
            f.write(f"{name:<60}\t{shape_str:<20}\t{device_str:<10}\t{dtype_str}\n")

    print(f"[QUANT-REPORT] Report saved successfully.")