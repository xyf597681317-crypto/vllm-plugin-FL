# -*- coding: utf-8 -*-
"""_probe_ops_3.py —— 调用 apply_triton_ops_patch() 后，确认 `_C_ascend` 里有什么。

同时验证 npu_hc_pre_v2 入口可用（为写它的数值对拍测试做准备）。
跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_3.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
print("device =", IDX, flush=True)

from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_triton_ops import (  # noqa: E402
    apply_triton_ops_patch,
)

ok = apply_triton_ops_patch()
print("apply_triton_ops_patch() ->", ok, flush=True)

mod = torch.ops._C_ascend
names = sorted(n for n in dir(mod) if not n.startswith("_"))
print(f"\n== _C_ascend 名字数 = {len(names)} ==", flush=True)
for n in names:
    print("   ", n, flush=True)

print("\n== 关心的入口 ==", flush=True)
for p in ("npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_inv_rms", "npu_hc_pre_sinkhorn",
          "npu_add_rms_norm_bias", "npu_gemma_rms_norm", "npu_swiglu_group_quant",
          "npu_sparse_attn_sharedkv", "npu_causal_conv1d_custom"):
    print(f"   {p:34s} hasattr={hasattr(mod, p)}", flush=True)

# 冒烟调用 npu_hc_pre_v2，看 schema 是否可用
print("\n== npu_hc_pre_v2 schema ==", flush=True)
try:
    print("   ", torch.ops._C_ascend.npu_hc_pre_v2._schemas, flush=True)
except Exception as e:  # noqa: BLE001
    print("   取 schema 失败:", type(e).__name__, str(e)[:200], flush=True)

if hasattr(mod, "npu_hc_pre_v2"):
    try:
        torch.manual_seed(0)
        B, S, HM, D = 2, 3, 2, 8
        x = torch.randn(B * S, D, device=f"npu:{IDX}", dtype=torch.float32)
        fn = torch.randn(HM * D, device=f"npu:{IDX}", dtype=torch.float32)
        sc = torch.randn(HM, device=f"npu:{IDX}", dtype=torch.float32)
        bs = torch.randn(HM, device=f"npu:{IDX}", dtype=torch.float32)
        out = torch.ops._C_ascend.npu_hc_pre_v2(x, fn, sc, bs, HM, 3, 1e-6, 1e-6)
        print("   调用成功，返回", type(out), flush=True)
        if isinstance(out, (tuple, list)):
            print("   返回元素:", [tuple(t.shape) for t in out], flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        print("   调用失败:", type(e).__name__, str(e)[:300], flush=True)
        traceback.print_exc()
print("PROBEOPS3DONE", flush=True)
