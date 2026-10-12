# -*- coding: utf-8 -*-
"""_probe_ops_1.py —— 只读列出容器里 torch.ops._C_ascend 的真实算子集合。

回答三件事：
  1. `_C_ascend` 到底是哪个 .so（vllm-plugin-FL 自带 vs vllm-ascend）；
  2. 三个"真孤儿"（add_rms_norm_bias / gemma_rms_norm / swiglu_group_quant）
     在真实注册表里**存在/不存在**；
  3. `npu_hc_pre_v2` 等我们关心的入口是否存在（决定 hc_pre_v2 测试怎么写）。

跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_1.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
print("device =", IDX, flush=True)

import torch.ops  # noqa: E402

# 1) _C_ascend 的来源
try:
    import vllm_fl  # noqa: F401
    print("vllm_fl:", vllm_fl.__file__, flush=True)
except Exception as e:  # noqa: BLE001
    print("vllm_fl import 失败:", type(e).__name__, str(e)[:120], flush=True)
try:
    import vllm_ascend  # noqa: F401
    print("vllm_ascend:", vllm_ascend.__file__, flush=True)
except Exception as e:  # noqa: BLE001
    print("vllm_ascend import 失败:", type(e).__name__, str(e)[:120], flush=True)

try:
    mod = torch.ops._C_ascend
except Exception as e:  # noqa: BLE001
    print("torch.ops._C_ascend 不存在:", e, flush=True)
    sys.exit(0)

names = sorted(n for n in dir(mod) if not n.startswith("_"))
print(f"\n== _C_ascend 算子总数 = {len(names)} ==", flush=True)
for n in names:
    print("   ", n, flush=True)

print("\n== 三个孤儿 / 关心的入口是否存在 ==", flush=True)
probe = [
    "npu_add_rms_norm_bias", "add_rms_norm_bias",
    "npu_gemma_rms_norm", "gemma_rms_norm",
    "npu_swiglu_group_quant", "swiglu_group_quant",
    "npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_inv_rms", "npu_hc_pre_sinkhorn",
]
for p in probe:
    has = hasattr(mod, p)
    print(f"   {p:34s} hasattr={has}", flush=True)

# 2) 逐个确认能不能真的调用（有 schema 无 kernel 的情况）
print("\n== 有无 PrivateUse1 kernel ==", flush=True)
for p in probe:
    if not hasattr(mod, p):
        continue
    try:
        ok = torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C_ascend::{p}", "PrivateUse1")
    except Exception as e:  # noqa: BLE001
        ok = f"err:{type(e).__name__}"
    print(f"   {p:34s} PrivateUse1_kernel={ok}", flush=True)
print("PROBEOPSDONE", flush=True)
