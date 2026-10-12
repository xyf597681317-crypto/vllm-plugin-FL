# -*- coding: utf-8 -*-
"""定位 `npu_recurrent_gated_delta_rule` 走 op（adapter）路径时到底发生了什么：
返回的 Tensor 是谁？`state` 被谁改的？原生 .so 实现是否被我的 impl 覆盖？

用法：EXTRA_MODE=triton DSH_NPU=npu:14 python3 -W ignore _probe_extra_rgdr.py
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

schemas, adapters = extra.build()
if MODE == "triton":
    lib = torch.library.Library("_C_ascend", "IMPL")
    for name, fn in adapters.items():
        lib.impl(name, fn, "PrivateUse1")
    print("[rgdr] 已注册 Triton adapters", flush=True)

# 关键：确认注册进去的到底是哪个 callable
print(f"[rgdr] adapters['npu_recurrent_gated_delta_rule'] is CUSTOM? "
      f"{adapters['npu_recurrent_gated_delta_rule'] is extra._adapter_npu_recurrent_gated_delta_rule}",
      flush=True)
print(f"[rgdr] 该 callable 名字={adapters['npu_recurrent_gated_delta_rule'].__name__}", flush=True)

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

# ---- 1) impl 直调（基准）
st_impl = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.bfloat16)
st_impl_before = st_impl.clone()
out_impl, fin_impl = impl_mod.recurrent_gated_delta_rule(
    query, key, value, beta, st_impl, cu, ssm, g, scale_value=sc)
torch.npu.synchronize()
print(f"[1] impl 直调: out.ptr={out_impl.data_ptr()} fin.ptr={fin_impl.data_ptr()} "
      f"st.ptr={st_impl.data_ptr()} | st_impl 变化={int((st_impl != st_impl_before).sum())}",
      flush=True)
print(f"[1] fin 是否就是 st_impl 本身: {fin_impl.data_ptr() == st_impl.data_ptr()}", flush=True)
print(f"[1] out 是否就是 st_impl 本身: {out_impl.data_ptr() == st_impl.data_ptr()}", flush=True)

# ---- 2) op 路径
st_op = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.bfloat16)
st_op_before = st_op.clone()
res_op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query, key=key, value=value, g=g, beta=beta, state=st_op,
    scale=sc, actual_seq_lengths=cu, ssm_state_indices=ssm)
torch.npu.synchronize()
print(f"[2] op 路径: 返回类型={type(res_op).__name__} ptr={res_op.data_ptr() if torch.is_tensor(res_op) else '-'}",
      flush=True)
print(f"[2] 返回 Tensor 与 impl.out 相同? {torch.is_tensor(res_op) and res_op.data_ptr() == out_impl.data_ptr()}"
      f"  与 impl.fin 相同? {torch.is_tensor(res_op) and res_op.data_ptr() == fin_impl.data_ptr()}"
      f"  与 st_op 相同? {torch.is_tensor(res_op) and res_op.data_ptr() == st_op.data_ptr()}", flush=True)
if torch.is_tensor(res_op):
    d_out = (res_op.float().cpu() - out_impl.float().cpu()).abs().max().item()
    d_fin = (res_op.float().cpu() - fin_impl.float().cpu()).abs().max().item()
    d_st = (res_op.float().cpu() - st_op_before.float().cpu()).abs().max().item()
    print(f"[2] 返回 Tensor 与 impl.out 逐元素 max|Δ|={d_out:.3e}; 与 impl.fin={d_fin:.3e}; "
          f"与调用前的 st={d_st:.3e}", flush=True)
n_changed = int((st_op != st_op_before).sum())
d_ch = (st_op.float().cpu() - st_op_before.float().cpu()).abs().max().item()
print(f"[2] ★ st_op 调用后变化元素={n_changed}/{st_op.numel()} max|Δ|={d_ch:.3e}", flush=True)
d_match_fin = (st_op.float().cpu() - fin_impl.float().cpu()).abs().max().item()
print(f"[2] st_op(调用后) 与 impl.fin 逐元素 max|Δ|={d_match_fin:.3e}  "
      f"(0 ⇒ 写回的就是 impl 的 final_state)", flush=True)
print("[rgdr] done", flush=True)
