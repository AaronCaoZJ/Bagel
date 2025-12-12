import json
from typing import Tuple, Dict, Any

import torch
from safetensors.torch import save_file
from safetensors import safe_open

# 导入 torchao 的 helper（仓库路径：torchao/prototype/safetensors）
from torchao.prototype.safetensors.safetensors_support import (
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)


def _to_cpu_tensors(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """
    确保所有 torch.Tensor 都在 CPU，safetensors 更稳定地处理 CPU tensors。
    非 tensor 保持不变（理论上 flatten 已把所有可序列化项转为 torch.Tensor 或 json-able）
    """
    out = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone().cpu()
        else:
            out[k] = v
    return out


def save_model_to_safetensors(model: torch.nn.Module, path: str) -> None:
    """
    把 model.state_dict() 使用 torchao 的 flatten 工具转换并保存为 .safetensors 文件。
    path: 目标文件路径，比如 "my_model.safetensors"
    """
    # 获取 state dict（包含可能的 tensor 子类）
    state = model.state_dict()

    # flatten -> 返回 (tensors_data_dict, metadata)
    tensors_data_dict, metadata = flatten_tensor_state_dict(state)

    # 确保 tensors 在 cpu（safetensors 在写入时期望 CPU tensors）
    tensors_data_dict = _to_cpu_tensors(tensors_data_dict)

    # metadata 应为 dict[str, str]（flatten 已生成 JSON 字符串）
    # safetensors.save_file 支持传 metadata 参数
    save_file(tensors_data_dict, path, metadata=metadata)
    print(f"Saved {len(tensors_data_dict)} tensors with metadata keys {list(metadata.keys())} to {path}")


def load_model_from_safetensors(
    model: torch.nn.Module, path: str, map_location: str = "cpu"
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    从 safetensors 文件加载并把重构的 state_dict load 到 model。
    返回 (model, leftover)：leftover 是 unflatten 返回的 leftover_state_dict（若有未被重构/匹配的数据）
    """
    # 使用 safe_open 读取文件以同时拿到 metadata
    # framework="pt" 表示使用 pytorch 的张量格式
    tensors_loaded = {}
    metadata = {}
    with safe_open(path, framework="pt", device=map_location) as f:
        metadata = f.metadata()  # dict[str, str]
        # 读取所有 tensors（get_tensor 会返回 torch.Tensor）
        for k in f.keys():
            tensors_loaded[k] = f.get_tensor(k)

    # unflatten -> 恢复为原来的 tensor subclasses state_dict 形式
    reconstructed_state_dict, leftover = unflatten_tensor_state_dict(tensors_loaded, metadata)

    # 将重建的 state_dict load 到 model（strict=False 更稳健）
    model.load_state_dict(reconstructed_state_dict, strict=False)
    print(f"Loaded into model: {len(reconstructed_state_dict)} entries, leftover tensors: {len(leftover)}")
    return model, leftover