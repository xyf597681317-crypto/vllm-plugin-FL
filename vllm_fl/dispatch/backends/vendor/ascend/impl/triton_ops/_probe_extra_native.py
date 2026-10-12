# -*- coding: utf-8 -*-
"""baseline 走谁？—— 验证 FL 原生 provider（`vllm_fl.ascend_custom_ops.enable_custom_op()`）。

背景：`patch.py:63 enable_ascend_native_ops()` → `enable_custom_op()`，
docstring 写着 "Require FL's native Qwen GDN provider on an installed Ascend wheel"。
我先前只 load 了**厂商** `vllm_fl/_C_ascend...so`（aclnn 薄封装，aclnn 算子缺失 → 报 not in libopapi.so）。
模型进程走的多半是这条**原生 payload**，不是那个 .so —— 本探针确认。

用法：DSH_NPU=npu:14 python3 -W ignore _probe_extra_native.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
print(f"[native] dev={DEV} SOC_VERSION={os.environ.get('SOC_VERSION')} "
      f"VLLM_VENDOR={os.environ.get('VLLM_VENDOR')}", flush=True)

try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    print("[native] import vllm_fl.ascend_custom_ops OK", flush=True)
except Exception as ex:  # noqa: BLE001
    print(f"[native] import 失败: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    raise SystemExit(1)

try:
    res = enable_custom_op()
    print(f"[native] enable_custom_op() -> {res}", flush=True)
except Exception as ex:  # noqa: BLE001
    print(f"[native] enable_custom_op() 抛异常: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    res = None

NAMES = ["npu_causal_conv1d_custom", "npu_recurrent_gated_delta_rule",
         "npu_fused_gdn_gating", "npu_hc_post", "npu_hc_pre_inv_rms",
         "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o", "npu_dequant_swiglu_quant",
         "npu_hc_pre", "npu_hc_pre_sinkhorn", "npu_rms_norm_dynamic_quant"]
print("[native] _C_ascend 里的存在性:", flush=True)
for n in NAMES:
    try:
        print(f"    {n:36s} {hasattr(torch.ops._C_ascend, n)}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"    {n:36s} EXC {type(ex).__name__}", flush=True)

# 真跑一个最简单的（hc_post），看原生实现是否可用
if res and hasattr(torch.ops._C_ascend, "npu_hc_post"):
    try:
        torch.manual_seed(7)
        n, hc, d = 4, 4, 64
        x = ((torch.rand(1, n, d, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        r = ((torch.rand(1, n, hc, d, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        p = ((torch.rand(1, n, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        c = ((torch.rand(1, n, hc, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        y = torch.ops._C_ascend.npu_hc_post(x, r, p, c)
        torch.npu.synchronize()
        print(f"[native] ★ 原生 npu_hc_post 跑通: shape={tuple(y.shape)} "
              f"|y|max={float(y.abs().max()):.4f}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[native] ★ 原生 npu_hc_post 失败: {type(ex).__name__}: {str(ex)[:220]}", flush=True)
print("[native] done", flush=True)
