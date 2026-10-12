# -*- coding: utf-8 -*-
"""_probe_kvce_10.py —— 最小独立探针，逐行 try/except 定位崩在哪一步。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_10.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

# 设备：先不 set_device，只打印当前设备（测试文件就是这样）
DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]
print("DEV =", DEV, flush=True)
print("device_count =", torch.npu.device_count(), flush=True)
print("current_device(before) =", torch.npu.current_device(), flush=True)

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402


def step(tag, fn):
    try:
        r = fn()
        print(f"  OK  {tag}", flush=True)
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc()
        return None


rows, d, group, rscale = 9, 576, 64, 0
kc = 768
torch.manual_seed(9)

x = step("alloc x", lambda: (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV))
print("  current_device(after x) =", torch.npu.current_device(), flush=True)
sm = step("alloc sm", lambda: torch.tensor([-1, 0, 18, 13, 17, -1, 19, 5, 2],
                                           dtype=torch.int32, device=DEV))
buf = step("alloc buf", lambda: torch.zeros(20 * kc, dtype=torch.uint8, device=DEV))
print("  x.device =", x.device, " sm.device =", sm.device, " buf.device =", buf.device,
      flush=True)
step("view fp8", lambda: buf.view(torch.float8_e4m3fn))
step("call op", lambda: kv_compress_epilog(
    x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
    quant_group_size=group, round_scale=rscale, block_stride=kc))
step("synchronize", lambda: torch.npu.synchronize())
nz = int((buf.detach().cpu() != 0).sum())
print("非零字节 =", nz, flush=True)
print("KVCE10DONE", flush=True)
