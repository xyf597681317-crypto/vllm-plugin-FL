# -*- coding: utf-8 -*-
"""_probe_kvce_11.py —— 用**整数索引** set_device，并验证 current_device 真的变了。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_11.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
print("target index =", IDX, " device_count =", torch.npu.device_count(), flush=True)
print("current(before) =", torch.npu.current_device(), flush=True)
torch.npu.set_device(IDX)
print("current(after set_device(int)) =", torch.npu.current_device(), flush=True)
try:
    torch.npu.set_device("npu:%d" % IDX)
    print("current(after set_device(str)) =", torch.npu.current_device(), flush=True)
except Exception as e:  # noqa: BLE001
    print("set_device(str) 报错:", type(e).__name__, e, flush=True)

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:%d" % IDX
rows, d, group, rscale = 9, 576, 64, 0
kc = 768
torch.manual_seed(9)


def step(tag, fn):
    try:
        r = fn()
        print(f"  OK  {tag}", flush=True)
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {tag}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        traceback.print_exc()
        return None


x = step("alloc x", lambda: (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV))
sm = step("alloc sm", lambda: torch.tensor([-1, 0, 18, 13, 17, -1, 19, 5, 2],
                                           dtype=torch.int32, device=DEV))
buf = step("alloc buf", lambda: torch.zeros(20 * kc, dtype=torch.uint8, device=DEV))
print("  x.device =", x.device, " current =", torch.npu.current_device(), flush=True)
step("call op", lambda: kv_compress_epilog(
    x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
    quant_group_size=group, round_scale=rscale, block_stride=kc))
step("synchronize", lambda: torch.npu.synchronize())
print("非零字节 =", int((buf.detach().cpu() != 0).sum()), flush=True)
print("KVCE11DONE", flush=True)
