# -*- coding: utf-8 -*-
"""`Tensor(a!)` 注解是否**承重**：把同一个 adapter 注册到一个不带别名注解的 schema 上，
看 `state` 还会不会被原地写。

背景（已实测）：`_C_ascend::npu_recurrent_gated_delta_rule` 的 schema 标了 `Tensor(a!) state`，
而实现返回的是 `(out, final_state)` 二元组、自己**不写 state**；但走 op 路径时 state 确实
被原地更新、且内容与实现的 final_state 逐元素一致 —— 说明是 torch.library 按 schema 的
可变输入注解在替我们写回。本探针验证这一依赖：**去掉 (a!) 会怎样**。

用法：EXTRA_MODE=triton DSH_NPU=npu:14 python3 -W ignore _probe_extra_alias.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.ops.load_library(SP + "/vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so")

import triton_ops._C_ascend_extra as extra  # noqa: E402

schemas, adapters = extra.build()
lib = torch.library.Library("_C_ascend", "IMPL")
for name, fn in adapters.items():
    lib.impl(name, fn, "PrivateUse1")

# 再定义一个**不带别名注解**的对照 schema，挂同一个 adapter
NOALIAS = ("(Tensor query, Tensor key, Tensor value, Tensor state, "
           "*, Tensor? beta=None, float? scale=None, Tensor? actual_seq_lengths=None, "
           "Tensor? ssm_state_indices=None) -> Tensor")
try:
    torch.library.define("_C_probe_extra::rgdr_noalias", NOALIAS)
except Exception as ex:  # noqa: BLE001
    print(f"[alias] define 失败（可能已存在）: {type(ex).__name__}: {ex}", flush=True)
lib2 = torch.library.Library("_C_probe_extra", "IMPL")
lib2.impl("rgdr_noalias", extra._adapter_npu_recurrent_gated_delta_rule, "PrivateUse1")

T, NK, NV, DK, DV = 32, 2, 2, 32, 32
torch.manual_seed(3)


def U(*s, lo=0.2):
    return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo


query = U(T, NK, DK).to(torch.bfloat16)
key = U(T, NK, DK).to(torch.bfloat16)
value = U(T, NV, DV).to(torch.bfloat16)
beta = U(T, NV).to(torch.bfloat16)
g = U(T, NV, lo=0.1) - 0.05
cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=DEV)
ssm = torch.arange(T, dtype=torch.int32, device=DEV) * NV
sc = float(DK) ** -0.5
S = int(ssm.max().item()) + NV
st0 = U(S, NV, DV, DK).to(torch.bfloat16)

st_alias = st0.clone()
res_alias = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query, key=key, value=value, g=g, beta=beta, state=st_alias,
    scale=sc, actual_seq_lengths=cu, ssm_state_indices=ssm)
torch.npu.synchronize()
n_alias = int((st_alias != st0).sum())

st_no = st0.clone()
try:
    res_no = torch.ops._C_probe_extra.rgdr_noalias(
        query, key, value, st_no, beta=beta, scale=sc,
        actual_seq_lengths=cu, ssm_state_indices=ssm)
    torch.npu.synchronize()
    n_no = int((st_no != st0).sum())
    print(f"[alias] 带 (a!) 的 _C_ascend op   : state 变化元素={n_alias}/{st_alias.numel()}  "
          f"返回 type={type(res_alias).__name__}", flush=True)
    print(f"[alias] 不带 (a!) 的对照 op     : state 变化元素={n_no}/{st_no.numel()}  "
          f"返回 type={type(res_no).__name__}", flush=True)
    print(f"[alias] ⇒ (a!) 注解承重? {n_alias > 0 and n_no == 0}", flush=True)
except Exception as ex:  # noqa: BLE001
    print(f"[alias] 对照 op 抛异常: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    print(f"[alias] （对照信息）带 (a!) 的 op: state 变化元素={n_alias}", flush=True)
print("[alias] done", flush=True)
