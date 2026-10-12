# -*- coding: utf-8 -*-
"""_probe_ops_5.py —— 定位 torch.ops 注册后为何走了 CPU fallback。

跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_5.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
DEV = f"npu:{IDX}"
print("torch:", torch.__version__, " device =", DEV, flush=True)
sys.path.insert(0, "/workspace/FlagGems/src")

INTG = "/workspace/flag_gems_integration"
sys.path.insert(0, INTG)
from _C_ascend_ops_a import npu_hc_pre_v2 as adapter  # noqa: E402
from _C_ascend_sigs import _C_ASCEND_SIGS  # noqa: E402

schema = _C_ASCEND_SIGS["npu_hc_pre_v2"]
if not isinstance(schema, str):
    schema = schema[0]
print("\nschema =", schema, flush=True)

LIB = "_C_ascend_probe"
torch.library.define(f"{LIB}::npu_hc_pre_v2", schema)
print("defined schema ok", flush=True)

called = {"n": 0}


def _wrapped(*a, **k):
    called["n"] += 1
    print("   >>> adapter 被调用，args=", len(a), "first dtype/device:",
          getattr(a[0], "dtype", None), getattr(a[0], "device", None), flush=True)
    return adapter(*a, **k)


for key in ("PrivateUse1", "npu", "CPU"):
    try:
        lib = torch.library.Library(LIB, "IMPL")
        lib.impl("npu_hc_pre_v2", _wrapped, key)
        print(f"registered impl for key={key}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"register key={key} 失败: {type(e).__name__}: {str(e)[:120]}", flush=True)

try:
    torch._C._dispatch_has_kernel_for_dispatch_key(f"{LIB}::npu_hc_pre_v2", "PrivateUse1")
except Exception:  # noqa: BLE001
    pass

torch.manual_seed(0)
hm, d, bs = 4, 16, 8
hmix = 2 * hm + hm * hm
x = torch.randn(bs, hm, d, device=DEV, dtype=torch.float32).to(torch.bfloat16)
fn = torch.randn(hmix, hm * d, device=DEV, dtype=torch.float32)
sc = torch.randn(3, device=DEV, dtype=torch.float32)
bb = torch.randn(hmix, device=DEV, dtype=torch.float32)
print("\nx.device =", x.device, " x.dtype =", x.dtype, flush=True)
print("calling torch.ops.", LIB, ".npu_hc_pre_v2 ...", flush=True)
try:
    out = torch.ops._C_ascend_probe.npu_hc_pre_v2(x, fn, sc, bb, hm, 3, 1e-5, 1e-2)
    print("  调用成功 called=", called["n"], flush=True)
    print("  返回:", type(out), [tuple(t.shape) for t in out] if isinstance(out, tuple) else out.shape,
          flush=True)
except Exception as e:  # noqa: BLE001
    print("  调用失败:", type(e).__name__, str(e)[:200], flush=True)
print("PROBEOPS5DONE", flush=True)
