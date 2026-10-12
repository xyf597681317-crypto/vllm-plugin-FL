# -*- coding: utf-8 -*-
"""_probe_kvce_smoke.py —— 设备健康 + 算子最小冒烟（npu 可选）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]

a = torch.ones(1024, device=DEV)
b = (a * 2).sum().item()
print("SMOKE arith ok:", b, flush=True)

from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

d, group, rows = 576, 64, 9
quant_col = d - 64
scale_col = (quant_col + group - 1) // group
concat_col = quant_col + 128 + scale_col
kv_col = ((concat_col + 127) // 128) * 128
num_slots = 20
row_stride = kv_col
print(f"geom: quant_col={quant_col} scale_col={scale_col} concat_col={concat_col}"
      f" kv_col={kv_col}", flush=True)

for tag, nrows in (("1 行", 1), ("2 行", 2), ("9 行", rows)):
    torch.manual_seed(9)
    x = (torch.randn(nrows, d) * 2.0).to(torch.bfloat16).to(DEV)
    slots = list(range(nrows))
    if nrows >= 5:
        slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    cache_fp8 = buf.view(torch.float8_e4m3fn)
    print(f"  [{tag}] 调用前 slots={slots}", flush=True)
    kv_compress_epilog(x, sm, cache_fp8, quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=0,
                       block_stride=row_stride)
    torch.npu.synchronize()
    nz = int((buf.detach().cpu() != 0).sum())
    print(f"  [{tag}] OK  非零字节={nz}", flush=True)
print("SMOKEKVCE DONE", flush=True)
