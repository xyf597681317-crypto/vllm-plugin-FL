# -*- coding: utf-8 -*-
"""定位 `npu_recurrent_gated_delta_rule` 的 op 路径行为（形状健壮版）。

对照三方：
  A) op 路径             torch.ops._C_ascend.npu_recurrent_gated_delta_rule(..., state=st_op, ...)
  B) impl 直调（同输入新 state）impl.recurrent_gated_delta_rule(query,key,value,beta,st_ref,cu,ssm,g,scale_value=sc)
问三件事：返回的 Tensor 是不是 B 的 out？st_op 有没有被原地写？写的是不是 B 的 final？

用法：EXTRA_MODE=triton DSH_NPU=npu:14 python3 -W ignore _probe_extra_rgdr2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
MODE = os.environ.get("EXTRA_MODE", "triton")
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.ops.load_library(SP + "/vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so")

import triton_ops._C_ascend_extra as extra  # noqa: E402
import triton_ops.recurrent_gated_delta_rule as impl_mod  # noqa: E402

print(f"[rgdr2] 实现模块文件 = {impl_mod.__file__}", flush=True)
schemas, adapters = extra.build()
if MODE == "triton":
    lib = torch.library.Library("_C_ascend", "IMPL")
    for name, fn in adapters.items():
        lib.impl(name, fn, "PrivateUse1")
    print("[rgdr2] 已注册 Triton adapters", flush=True)

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

# ---- A) op 路径
st_a = st0.clone()
res_a = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query, key=key, value=value, g=g, beta=beta, state=st_a,
    scale=sc, actual_seq_lengths=cu, ssm_state_indices=ssm)
torch.npu.synchronize()

# ---- B) impl 直调（同 st0 起点，新张量）
st_b = st0.clone()
out_b, fin_b = impl_mod.recurrent_gated_delta_rule(
    query, key, value, beta, st_b, cu, ssm, g, scale_value=sc)
torch.npu.synchronize()

print(f"[A] op 返回: type={type(res_a).__name__} "
      f"shape={tuple(res_a.shape) if torch.is_tensor(res_a) else '-'} "
      f"dtype={getattr(res_a, 'dtype', '-')}", flush=True)
print(f"[B] impl 返回: out.shape={tuple(out_b.shape)} fin.shape={tuple(fin_b.shape)}", flush=True)
print(f"[B] impl 直调后 st_b 变化元素={int((st_b != st0).sum())}（0 ⇒ 实现不原地写 state）", flush=True)

if torch.is_tensor(res_a) and tuple(res_a.shape) == tuple(out_b.shape):
    d = (res_a.float().cpu() - out_b.float().cpu()).abs().max().item()
    print(f"[A vs B] 返回 Tensor 与 impl.out 逐元素 max|Δ|={d:.3e}", flush=True)
else:
    print(f"[A vs B] 形状不同，无法直接比: op={tuple(res_a.shape) if torch.is_tensor(res_a) else '-'} "
          f"vs out={tuple(out_b.shape)}", flush=True)

n_ch = int((st_a != st0).sum())
print(f"[A] ★ op 路径后 st_a 变化元素={n_ch}/{st_a.numel()}", flush=True)
if tuple(st_a.shape) == tuple(fin_b.shape):
    d2 = (st_a.float().cpu() - fin_b.float().cpu()).abs().max().item()
    print(f"[A vs B] st_a(调用后) 与 impl.final_state 逐元素 max|Δ|={d2:.3e} "
          f"(0 ⇒ 写回的就是实现的 final_state)", flush=True)
print("[rgdr2] done", flush=True)
