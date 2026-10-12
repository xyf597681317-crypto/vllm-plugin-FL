# -*- coding: utf-8 -*-
"""_probe_ops_2.py —— 在**Triton 接管开启**（默认）下，看 `_C_ascend` 里到底有什么。

前面 _probe_ops_1.py 显示 `torch.ops._C_ascend` 只有 1 个名字，且打印
`[flag_gems_integration] USE_FLAGGEMS 未开启，跳过 _C_ascend 接管`。
Lead 已说明：**Triton 接管不再依赖 USE_FLAGGEMS，默认开启，关闭用 VLLM_FL_TRITON_OPS=0**，
且新增了 `impl/triton_ops/_C_ascend_extra.py`。
本探针把几种开关组合都打一遍，确认 `npu_hc_pre_v2` 究竟在什么条件下可用。

跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
print("device =", IDX, flush=True)
print("VLLM_FL_TRITON_OPS      =", os.environ.get("VLLM_FL_TRITON_OPS", "(unset)"), flush=True)
print("VLLM_FL_TRITON_OPS_SKIP =", os.environ.get("VLLM_FL_TRITON_OPS_SKIP", "(unset)"), flush=True)
print("USE_FLAGGEMS            =", os.environ.get("USE_FLAGGEMS", "(unset)"), flush=True)

# 1) 找 triton 接管的 patch 模块在哪
import importlib.util  # noqa: E402
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"
cands = [
    os.path.join(SP, "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_triton_ops.py"),
    os.path.join(SP, "vllm_fl/dispatch/backends/vendor/ascend/impl/triton_ops/_C_ascend_extra.py"),
    os.path.join(SP, "vllm_fl/dispatch/backends/vendor/ascend/impl/triton_ops/__init__.py"),
]
for c in cands:
    print(f"  exists={os.path.exists(c)}  {c}", flush=True)

# 2) 看接管模块怎么注册
p = cands[0]
if os.path.exists(p):
    txt = open(p, encoding="utf-8", errors="replace").read()
    print(f"\n== {os.path.basename(p)} 关键行 ==", flush=True)
    for i, ln in enumerate(txt.splitlines(), 1):
        s = ln.strip()
        if any(k in s for k in ("library", "_C_ascend", "registered", "VLLM_FL_TRITON_OPS",
                                "def apply", "def _", "EXTRA_OPS", "hc_pre")):
            print(f"  {i:4d}: {s[:120]}", flush=True)

# 3) 真正注册完再看一次
print("\n== 触发注册后 torch.ops._C_ascend ==", flush=True)
try:
    from vllm_fl.dispatch.backends.vendor.ascend.patches import patch_triton_ops as P  # noqa: E402
    for fn in ("apply", "apply_patch", "install", "register", "patch"):
        f = getattr(P, fn, None)
        if callable(f):
            try:
                f()
                print(f"  调用 {fn}() 成功", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  调用 {fn}() 失败: {type(e).__name__}: {str(e)[:140]}", flush=True)
except Exception as e:  # noqa: BLE001
    print("  import patch_triton_ops 失败:", type(e).__name__, str(e)[:160], flush=True)

mod = torch.ops._C_ascend
names = sorted(n for n in dir(mod) if not n.startswith("_"))
print(f"\n== _C_ascend 名字数 = {len(names)} ==", flush=True)
print("  ", names[:60], flush=True)
for probe in ("npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_inv_rms", "npu_hc_pre_sinkhorn",
              "npu_add_rms_norm_bias", "npu_gemma_rms_norm", "npu_swiglu_group_quant"):
    print(f"   {probe:30s} hasattr={hasattr(mod, probe)}", flush=True)
print("PROBEOPS2DONE", flush=True)
