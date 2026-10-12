# -*- coding: utf-8 -*-
"""发现 AscendC `_C_ascend` 的 .so 装载方式（为 oracle 对照铺路）。

用法（容器内）：DSH_NPU=npu:14 python3 -W ignore _probe_extra_so.py
"""
import glob
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"

print(f"[so] dev={DEV}", flush=True)
cands = [p for p in glob.glob(SP + "/**/*.so", recursive=True)
         if ("ascend" in p.lower() or "_C" in os.path.basename(p))]
print(f"[so] 候选 .so ({len(cands)}):", flush=True)
for p in cands[:30]:
    print(f"   {p}  ({os.path.getsize(p)} B)", flush=True)

print("[so] 试 import vllm_ascend ...", flush=True)
try:
    import vllm_ascend  # noqa: F401
    print("   import vllm_ascend OK", flush=True)
except Exception as ex:  # noqa: BLE001
    print(f"   import vllm_ascend 失败: {type(ex).__name__}: {str(ex)[:150]}", flush=True)


def chk():
    out = []
    for n in ("npu_hc_post", "npu_hc_pre_inv_rms", "npu_causal_conv1d_custom",
              "npu_recurrent_gated_delta_rule"):
        try:
            out.append(f"{n}={hasattr(torch.ops._C_ascend, n)}")
        except Exception as ex:  # noqa: BLE001
            out.append(f"{n}=EXC({type(ex).__name__})")
    return " ".join(out)


print("[so] import 之后:", chk(), flush=True)

print("[so] 试逐个 load_library ...", flush=True)
for p in cands:
    try:
        torch.ops.load_library(p)
        print(f"   load OK: {p}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"   load FAIL: {p}: {type(ex).__name__}: {str(ex)[:90]}", flush=True)
print("[so] load 之后:", chk(), flush=True)
print("[so] done", flush=True)
