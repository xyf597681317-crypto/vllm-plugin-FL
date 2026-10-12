# -*- coding: utf-8 -*-
"""真·原生（enable_custom_op）vs Triton 同进程逐元素对照 —— v2（修掉 v1 的两个 harness 缺陷）。

v1 缺陷：
  ① 两个阶段共用同一份输入，而 `npu_causal_conv1d_custom` 会**原地更新 conv_state**
     ⇒ 阶段 2 从已被推进过的 state 起步，比较无效；
  ② 只做两边互比，没有第三方（fp64）判断"谁对"。
v2：每个阶段**重建输入**（同 seed ⇒ 同数值、不同缓冲），并对 `hc_post` / `hc_pre_inv_rms`
    额外与 fp64 参考比，明确哪一侧符合语义。
    另：原生 `npu_recurrent_gated_delta_rule` 实测触发 507035 并污染后续调用 ⇒ 默认跳过。

用法：DSH_NPU=npu:15 python3 -W ignore _probe_extra_nv2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:15")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_fl.ascend_custom_ops import enable_custom_op  # noqa: E402
import triton_ops._C_ascend_extra as extra  # noqa: E402

print(f"[nv2] dev={DEV} enable_custom_op() -> {enable_custom_op()}", flush=True)


def inputs():
    torch.manual_seed(1234)
    d = {}
    n, hc, dd = 8, 4, 64
    d["x"] = ((torch.rand(1, n, dd, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["r"] = ((torch.rand(1, n, hc, dd, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["p"] = ((torch.rand(1, n, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["c"] = ((torch.rand(1, n, hc, hc, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
    d["irx"] = (torch.randn(64, 4, 512, device=DEV) * 1.3).to(torch.bfloat16)
    d["cx"] = (torch.randn(2, 64, device=DEV) * 1.5).to(torch.float16)
    d["cw"] = (torch.randn(4, 64, device=DEV) * 0.7).to(torch.float16)
    d["cb"] = (torch.randn(64, device=DEV) * 0.3).to(torch.float16)
    d["cst"] = (torch.randn(2, 3, 64, device=DEV) * 0.5).to(torch.float16)
    d["cci"] = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    return d


SENT = -777.0
D0 = inputs()          # 只用于算 fp64 参考（与两个阶段同 seed 同数值）


def ref_hc_post(D):
    xd = D["x"].detach().cpu().double()
    rd = D["r"].detach().cpu().double()
    pd = D["p"].detach().cpu().double()
    cd = D["c"].detach().cpu().double()
    B, Dd = xd.reshape(-1, xd.shape[-1]).shape
    H = rd.shape[-2]
    xr = xd.reshape(B, Dd)
    rr = rd.reshape(B, H, Dd)
    pr = pd.reshape(B, H)
    cr = cd.reshape(B, H, H)
    return (xr[:, None, :] + (pr[:, :, None] * rr).sum(dim=1)[:, None, :]
            + torch.einsum("bhi,bhd->bid", cr, rr)).reshape(xd.shape[:-1] + (H, Dd))


def ref_inv_rms(D, eps=1e-6):
    xd = D["irx"].detach().cpu().double()
    A = 1
    for s in xd.shape[:-2]:
        A *= int(s)
    R = int(xd.shape[-2]) * int(xd.shape[-1])
    x2 = xd.reshape(A, R)
    return (1.0 / torch.sqrt((x2 * x2).sum(dim=1) / float(R) + eps))


def run_all(phase):
    D = inputs()                    # ★ 每阶段重建（同 seed 同数值，缓冲全新）
    out = {}
    try:
        y = torch.ops._C_ascend.npu_hc_post(D["x"], D["r"], D["p"], D["c"])
        torch.npu.synchronize()
        out["hc_post"] = y.float().cpu()
    except Exception as ex:  # noqa: BLE001
        out["hc_post"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"
    try:
        y = torch.ops._C_ascend.npu_hc_pre_inv_rms(D["irx"], 1e-6)
        torch.npu.synchronize()
        out["inv_rms"] = y.float().cpu().reshape(-1)
    except Exception as ex:  # noqa: BLE001
        out["inv_rms"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"
    try:
        ob = torch.full((2, 64), SENT, dtype=torch.float16, device=DEV)
        st = D["cst"].clone()
        r = torch.ops._C_ascend.npu_causal_conv1d_custom(
            ob, D["cx"], D["cw"], st, D["cb"], None, D["cci"], None, None, 1, -1, 1)
        torch.npu.synchronize()
        out["conv1d"] = ob.float().cpu()
        out["conv1d_st"] = st.float().cpu()
        out["conv1d_same_obj"] = bool(torch.is_tensor(r) and r.data_ptr() == ob.data_ptr())
    except Exception as ex:  # noqa: BLE001
        out["conv1d"] = f"EXC {type(ex).__name__}: {str(ex)[:120]}"
    print(f"  [{phase}] " + "  ".join(f"{k}={'EXC' if isinstance(v, str) else 'ok'}"
                                      for k, v in out.items()), flush=True)
    return out


print("== 阶段 1：原生 ==", flush=True)
NAT = run_all("native")
print("== 阶段 2：Triton ==", flush=True)
schemas, adapters = extra.build()
lib = torch.library.Library("_C_ascend", "IMPL")
for name, fn in adapters.items():
    lib.impl(name, fn, "PrivateUse1")
print(f"  已注册 {len(adapters)} 个 adapter", flush=True)
TRI = run_all("triton")


def cmp3(name, a, b, ref):
    if isinstance(a, str) or isinstance(b, str):
        print(f"  {name:14s} 异常: native={a if isinstance(a,str) else 'ok'} | "
              f"triton={b if isinstance(b,str) else 'ok'}", flush=True)
        return
    da = (a.double() - ref).abs().max().item()
    db = (b.double() - ref).abs().max().item()
    d3 = (a.double() - b.double()).abs().max().item()
    print(f"  {name:14s} native vs triton max|Δ|={d3:.3e} | native vs fp64={da:.3e} | "
          f"triton vs fp64={db:.3e}   ← 哪个接近 fp64 就是对的", flush=True)


print("== 三方比对（fp64 参考当裁判）==", flush=True)
cmp3("hc_post", NAT["hc_post"], TRI["hc_post"], ref_hc_post(D0).float())
cmp3("inv_rms", NAT["inv_rms"], TRI["inv_rms"], ref_inv_rms(D0).float())
if not isinstance(NAT["conv1d"], str) and not isinstance(TRI["conv1d"], str):
    d3 = (NAT["conv1d"].double() - TRI["conv1d"].double()).abs().max().item()
    ds = (NAT["conv1d_st"].double() - TRI["conv1d_st"].double()).abs().max().item()
    print(f"  conv1d         native vs triton max|Δ|={d3:.3e}  (conv_state 差={ds:.3e})", flush=True)
    print(f"  conv1d         原地写同一对象: native={NAT['conv1d_same_obj']} triton={TRI['conv1d_same_obj']}",
          flush=True)
print("[nv2] done", flush=True)
