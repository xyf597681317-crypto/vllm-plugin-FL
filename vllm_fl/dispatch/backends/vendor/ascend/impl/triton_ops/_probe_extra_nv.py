# -*- coding: utf-8 -*-
"""真·原生 vs Triton 同进程逐元素对照（修正上一轮"裁判不可用"的结论）。

上一轮我 load 的是**厂商** `vllm_fl/_C_ascend...so`（aclnn 薄封装 → 报 not in libopapi.so）。
本探针改走模型进程真正的路：`vllm_fl.ascend_custom_ops.enable_custom_op()`
（= `patch.py:63 enable_ascend_native_ops()` 调的那条），它注册的 `_C_ascend` 原生实现**能跑**。

流程（同进程、同输入）：
  1) enable_custom_op() → 跑原生 → 存结果
  2) 按 patch 的方式注册 _C_ascend_extra 的 11 个 adapter（Triton）→ 跑同样的输入 → 存结果
  3) 逐元素比对 max|Δ| / exact

用法：DSH_NPU=npu:14 python3 -W ignore _probe_extra_nv.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_fl.ascend_custom_ops import enable_custom_op  # noqa: E402
import triton_ops._C_ascend_extra as extra  # noqa: E402
import triton_ops.hc_post as hc_post_mod  # noqa: E402
import triton_ops.hc_pre_inv_rms as inv_mod  # noqa: E402
import triton_ops.causal_conv1d as conv_mod  # noqa: E402
import triton_ops.recurrent_gated_delta_rule as rgdr_mod  # noqa: E402

print(f"[nv] dev={DEV} enable_custom_op() -> {enable_custom_op()}", flush=True)


def inputs():
    """构造 4 个算子共用的确定性输入（原生与 Triton 两次跑同一份）。"""
    torch.manual_seed(1234)
    d = {}
    n, hc, dd = 8, 4, 64
    d["x"] = ((torch.rand(1, n, dd, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["r"] = ((torch.rand(1, n, hc, dd, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["p"] = ((torch.rand(1, n, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["c"] = ((torch.rand(1, n, hc, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["irx"] = (torch.randn(64, 4, 512, device=DEV) * 1.3).to(torch.bfloat16)
    d["cx"] = (torch.randn(2 * 1, 64, device=DEV) * 1.5).to(torch.float16)
    d["cw"] = (torch.randn(4, 64, device=DEV) * 0.7).to(torch.float16)
    d["cb"] = (torch.randn(64, device=DEV) * 0.3).to(torch.float16)
    d["cst"] = (torch.randn(2, 3, 64, device=DEV) * 0.5).to(torch.float16)
    d["cci"] = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    T, NK, NV, DK, DV = 32, 2, 2, 32, 32
    d["q"] = ((torch.rand(T, NK, DK, device=DEV) * 2 - 1) * 0.2).to(torch.bfloat16)
    d["k"] = ((torch.rand(T, NK, DK, device=DEV) * 2 - 1) * 0.2).to(torch.bfloat16)
    d["v"] = ((torch.rand(T, NV, DV, device=DEV) * 2 - 1) * 0.2).to(torch.bfloat16)
    d["beta"] = ((torch.rand(T, NV, device=DEV) * 2 - 1) * 0.2).to(torch.bfloat16)
    d["g"] = (torch.rand(T, NV, device=DEV) * 2 - 1) * 0.1 - 0.05
    d["cu"] = torch.tensor([0, 16, 32], dtype=torch.int32, device=DEV)
    d["ssm"] = torch.arange(T, dtype=torch.int32, device=DEV) * NV
    d["state"] = ((torch.rand(64, NV, DV, DK, device=DEV) * 2 - 1) * 0.2).to(torch.bfloat16)
    d["sc"] = float(DK) ** -0.5
    return d


D = inputs()
SENT = -777.0


def run_all(phase):
    """phase: 'native' | 'triton'。返回 dict[name] = tensor/信息。"""
    out = {}
    tag = phase

    def call(fn, *a, **kw):
        return fn(*a, **kw)

    # hc_post
    try:
        y = call(torch.ops._C_ascend.npu_hc_post, D["x"], D["r"], D["p"], D["c"])
        torch.npu.synchronize()
        out["hc_post"] = y.float().cpu()
    except Exception as ex:  # noqa: BLE001
        out["hc_post"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"

    # hc_pre_inv_rms
    try:
        y = call(torch.ops._C_ascend.npu_hc_pre_inv_rms, D["irx"], 1e-6)
        torch.npu.synchronize()
        out["inv_rms"] = y.float().cpu()
    except Exception as ex:  # noqa: BLE001
        out["inv_rms"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"

    # causal_conv1d_custom（output 原地写）
    try:
        ob = torch.full((2, 64), SENT, dtype=torch.float16, device=DEV)
        r = call(torch.ops._C_ascend.npu_causal_conv1d_custom,
                 ob, D["cx"], D["cw"], D["cst"], D["cb"], None, D["cci"],
                 None, None, 1, -1, 1)
        torch.npu.synchronize()
        out["conv1d"] = ob.float().cpu()
        out["conv1d_same_obj"] = bool(torch.is_tensor(r) and r.data_ptr() == ob.data_ptr())
    except Exception as ex:  # noqa: BLE001
        out["conv1d"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"

    # rgdr（state 原地写）—— 原生路径实测会触发 507035 并污染本进程后续所有调用，
    # 故默认跳过（SKIP_RGDR=0 可打开）
    if os.environ.get("SKIP_RGDR", "1") == "0":
        try:
            st = D["state"].clone()
            st_before = st.clone()
            r = call(torch.ops._C_ascend.npu_recurrent_gated_delta_rule,
                     query=D["q"], key=D["k"], value=D["v"], g=D["g"], beta=D["beta"],
                     state=st, scale=D["sc"], actual_seq_lengths=D["cu"], ssm_state_indices=D["ssm"])
            torch.npu.synchronize()
            out["rgdr_out"] = r.float().cpu() if torch.is_tensor(r) else f"type={type(r).__name__}"
            out["rgdr_state_changed"] = int((st != st_before).sum())
            out["rgdr_state"] = st.float().cpu()
        except Exception as ex:  # noqa: BLE001
            out["rgdr_out"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"
    print(f"  [{tag}] " + "  ".join(f"{k}={'EXC' if isinstance(v, str) else 'ok'}"
                                    for k, v in out.items()), flush=True)
    return out


print("== 阶段 1：原生（enable_custom_op 提供的 _C_ascend）==", flush=True)
NAT = run_all("native")
print("== 阶段 2：Triton（注册 _C_ascend_extra 的 adapters）==", flush=True)
schemas, adapters = extra.build()
lib = torch.library.Library("_C_ascend", "IMPL")
for name, fn in adapters.items():
    lib.impl(name, fn, "PrivateUse1")
print(f"  已注册 {len(adapters)} 个 adapter", flush=True)
TRI = run_all("triton")

print("== 逐元素比对（原生 vs Triton）==", flush=True)
for k in NAT:
    a, b = NAT[k], TRI.get(k)
    if isinstance(a, str) or isinstance(b, str):
        print(f"  {k:20s} 一方异常: native={a if isinstance(a, str) else 'ok'} | "
              f"triton={b if isinstance(b, str) else 'ok'}", flush=True)
        continue
    if isinstance(a, (int, bool)) or isinstance(b, (int, bool)):
        print(f"  {k:20s} native={a} triton={b}", flush=True)
        continue
    if tuple(a.shape) != tuple(b.shape):
        print(f"  {k:20s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
        continue
    dd = (a.double() - b.double()).abs()
    md = float(dd.max()) if dd.numel() else 0.0
    ex = float((dd == 0).double().mean()) * 100 if dd.numel() else 100.0
    print(f"  {k:20s} max|Δ|={md:.3e} exact={ex:.4f}% n={dd.numel()}", flush=True)
print("[nv] done", flush=True)
