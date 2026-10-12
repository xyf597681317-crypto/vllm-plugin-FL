# -*- coding: utf-8 -*-
"""环境探针：AscendC 算子可用性 + 设备占用情况。"""
import os
import torch
import torch_npu  # noqa: F401

print("torch", torch.__version__, "| cwd", os.getcwd(), flush=True)
try:
    print("device_count", torch.npu.device_count(), "| current", torch.npu.current_device(), flush=True)
except Exception as e:
    print("device_count ERR", e, flush=True)

for d in range(3, 5):   # dev id 也扫一下 0..3
    try:
        free, total = torch.npu.mem_get_info(d)
        print(f"  cuda:{d} free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB", flush=True)
    except Exception as e:
        print(f"  cuda:{d} ERR {type(e).__name__}: {e}", flush=True)

print("has _C_ascend:", hasattr(torch.ops, "_C_ascend"), flush=True)
try:
    print("has npu_sparse_attn_sharedkv:", hasattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv"), flush=True)
    print("schema:", torch.ops._C_ascend.npu_sparse_attn_sharedkv.default._schema, flush=True)
except Exception as e:
    print("schema ERR", type(e).__name__, e, flush=True)

try:
    x = torch.ones(64, device=os.environ.get("DSH_NPU", "npu:0")) * 3
    print("small op ok:", float((x + x).sum().item()), flush=True)
except Exception as e:
    print("small op ERR", type(e).__name__, e, flush=True)
print("ENVDONE", flush=True)
