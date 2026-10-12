# -*- coding: utf-8 -*-
"""手算最小例：判定 `npu_hc_pre_inv_rms` / `npu_causal_conv1d_custom` 的
「原生 vs 文档语义」差异是谁的问题。

原则（Lead 要求）：**先证我的原生调用约定是对的，再判谁错**。
- 输入取**整数图案**，手算值是解析可得的；同时避开极小 R（`x[2,2,4]` 会把本仓库的
  Triton kernel 直接打成 vector core exception，那是另一个问题，不在本探针范围）。
- 每个用例一个进程（`inv` / `conv`），避免一次崩溃污染后续（见 BACKEND_NOTES 坑 4.2）。
- 三方逐元素打印：手算（fp64，按 op_host/文档语义）/ 原生（enable_custom_op）/ Triton（本仓库）。

用法：
    DSH_NPU=npu:12 python3 -W ignore _probe_native_min.py inv
    DSH_NPU=npu:12 python3 -W ignore _probe_native_min.py conv
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:12")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_fl.ascend_custom_ops import enable_custom_op  # noqa: E402
import triton_ops._C_ascend_extra as extra  # noqa: E402

print(f"[min] dev={DEV} enable_custom_op() -> {enable_custom_op()}", flush=True)

HC, D, ROWS = 4, 32, 2
R = HC * D


def silu(t):
    return t / (1.0 + torch.exp(-t))


# ================================================================ 用例 1
def case_inv(eps):
    """x[2,4,32]（R=128）：row0 只有两个非零 (3,4) ⇒ Σx²=25；row1 有 8 个 1 ⇒ Σx²=8。"""
    xd = torch.zeros(ROWS, HC, D, dtype=torch.float32)
    xd[0, 0, 0], xd[0, 0, 1] = 3.0, 4.0
    xd[1, 0, :8] = 1.0
    hand = (1.0 / torch.sqrt((xd.double() ** 2).sum(dim=(1, 2)) / R + eps))
    hand_perhc = 1.0 / torch.sqrt((xd.double() ** 2).sum(dim=2) / D + eps)   # 假设只按 d 归约
    x = xd.to(torch.bfloat16).to(DEV)
    try:
        n = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, eps)
        torch.npu.synchronize()
        nv = f"shape={tuple(n.shape)} {[round(v, 6) for v in n.detach().cpu().flatten().tolist()]}"
    except Exception as ex:  # noqa: BLE001
        nv = f"EXC {type(ex).__name__}: {str(ex)[:110]}"
    import triton_ops.hc_pre_inv_rms as ir
    t = ir.hc_pre_inv_rms(x, eps)
    torch.npu.synchronize()
    tv = f"shape={tuple(t.shape)} {[round(v, 6) for v in t.detach().cpu().flatten().tolist()]}"
    print(f"  eps={eps}", flush=True)
    print(f"    手算(H1 全 R 归约) = {[round(v, 6) for v in hand.tolist()]}", flush=True)
    print(f"    手算(H2 只按 d)    = {[round(v, 6) for v in hand_perhc.flatten().tolist()][:6]} ...",
          flush=True)
    print(f"    手算倒数 1/H1      = {[round(1 / v, 6) for v in hand.tolist()]}", flush=True)
    print(f"    原生 AscendC       = {nv}", flush=True)
    print(f"    Triton 实现        = {tv}", flush=True)
    print("", flush=True)


# ================================================================ 用例 2
def case_conv(activation):
    """dim=32, width=4, 单 token, state 全零, w 只有"当前 tap"(k=3) 为 1
    ⇒ 文档语义：out[0,j] = x[0,j]（未激活）/ silu(x[0,j])（激活）。"""
    import triton_ops.causal_conv1d as conv_mod
    DIM, W = 32, 4
    xv = torch.arange(1, DIM + 1, dtype=torch.float32).reshape(1, DIM) * 0.25
    w = torch.zeros(W, DIM, dtype=torch.float32)
    w[W - 1, :] = 1.0
    stz = torch.zeros(1, W - 1, DIM, dtype=torch.float32)
    x = xv.to(torch.float16).to(DEV)
    wt = w.to(torch.float16).to(DEV)
    st = stz.to(torch.float16).to(DEV)
    ci = torch.tensor([0], dtype=torch.int32, device=DEV)
    hand = silu(xv) if activation else xv
    SENT = -777.0
    ob = torch.full((1, DIM), SENT, dtype=torch.float16, device=DEV)
    st_n = st.clone()
    try:
        r = torch.ops._C_ascend.npu_causal_conv1d_custom(
            ob, x, wt, st_n, None, None, ci, None, None, activation, -1, 1)
        torch.npu.synchronize()
        nv = [round(v, 5) for v in ob.detach().cpu().flatten().tolist()[:8]]
        nsame = bool(torch.is_tensor(r) and r.data_ptr() == ob.data_ptr())
        nst = [round(v, 5) for v in st_n.detach().cpu().flatten().tolist()[:8]]
    except Exception as ex:  # noqa: BLE001
        nv, nsame, nst = f"EXC {type(ex).__name__}: {str(ex)[:110]}", None, None
    ob2 = torch.full((1, DIM), SENT, dtype=torch.float16, device=DEV)
    st_t = st.clone()
    try:
        r2 = conv_mod.causal_conv1d_fn(
            x, wt, conv_state=st_t, bias_opt=None, query_start_loc_opt=None,
            cache_indices_opt=ci, initial_state_mode_opt=None,
            num_accepted_tokens_opt=None, activation_mode=activation,
            pad_slot_id=-1, run_mode=1, out=ob2)
        torch.npu.synchronize()
        tv = [round(v, 5) for v in ob2.detach().cpu().flatten().tolist()[:8]]
        tsame = bool(torch.is_tensor(r2) and r2.data_ptr() == ob2.data_ptr())
        tst = [round(v, 5) for v in st_t.detach().cpu().flatten().tolist()[:8]]
    except Exception as ex:  # noqa: BLE001
        tv, tsame, tst = f"EXC {type(ex).__name__}: {str(ex)[:110]}", None, None
    print(f"  activation_mode={activation}  x[:8]={[round(v, 5) for v in xv.flatten().tolist()[:8]]}",
          flush=True)
    print(f"    手算(文档语义) = {[round(v, 5) for v in hand.flatten().tolist()[:8]]}", flush=True)
    print(f"    原生 AscendC   = {nv}   原地写={nsame}", flush=True)
    print(f"    Triton 实现    = {tv}   原地写={tsame}", flush=True)
    print(f"    conv_state 前 8：原生={nst} / Triton={tst}", flush=True)
    print("", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "inv"
    if which == "inv":
        print(f"== 用例 1：npu_hc_pre_inv_rms  x[{ROWS},{HC},{D}]  R={R} ==", flush=True)
        for eps in (1e-6, 1e-20):
            case_inv(eps)
    else:
        print("== 用例 2：npu_causal_conv1d_custom  单 token / width=4 / 零 state / 只有当前 tap ==",
              flush=True)
        case_conv(0)
        case_conv(1)
    print("[min] done", flush=True)


if __name__ == "__main__":
    main()
