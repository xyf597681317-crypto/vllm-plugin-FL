# -*- coding: utf-8 -*-
"""task-3 探针：grouped_matmul_swiglu_quant_v2 的 2-D 掩码 store（L129）。

_gmm_v2_kernel 的 store：
    tl.store(Y + rows[:,None]*HALF_N + n_off[None,:], qr.to(int8),
             mask=rm[:,None] & nm[None,:])          ← [BLK_M, BLK_N] 2-D 掩码
值链：load → tl.dot → 逐元素（silu/exp）→ 逐行 max → 除法 → fp16 往返 → rint → int8
本探针用 sentinel 双填充检测「完全没写」的 lane（决定性判据），
并用 fp64 参考统计「写了但值错」的 lane 数（>1 LSB 才算错，避免 fp32/fp64 舍入噪声）。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_gmmv2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
F1, F2 = -99, -77

from triton_ops.grouped_matmul_swiglu_quant_v2 import _gmm_v2_kernel  # noqa: E402
from triton_ops.common import next_pow2  # noqa: E402


def ref_gmm(x, w, ws, xs, sm, I, limit):
    """[M,K]i8, [K,N]i8, [N]f32, [M]f32, [I]f32 -> (y_i8 [M,I], scale [M]) fp64。"""
    xi = x.detach().cpu().double()
    wd = w.detach().cpu().double()[0]          # [K, N]（探针只用 E=1）
    cg = xi @ wd[:, :I]
    cu = xi @ wd[:, I:]
    xsd = xs.detach().cpu().double()
    wsd = ws.detach().cpu().double()[0]        # [N]
    gate = cg * xsd[:, None] * wsd[None, :I]
    up = cu * xsd[:, None] * wsd[None, I:]
    if limit > 0:
        gate = gate.clamp(max=limit)
    sw = (gate / (1.0 + torch.exp(-gate))) * up
    if sm is not None:
        smd = sm.detach().cpu().double()
        if smd.dim() == 2:                     # smooth_scale (E, N/2)
            sw = sw * smd[0][None, :]
        else:                                  # smooth_scale (E,) 单标量
            sw = sw * float(smd[0])
    mx = sw.abs().amax(dim=1)
    mxs = torch.where(mx > 0, mx, torch.ones_like(mx))
    s = mxs / 127.0
    q = sw * (127.0 / mxs)[:, None]
    q16 = q.to(torch.float16).double()
    qr = torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5)).clamp(-128, 127)
    return qr, s


def run_case(M, K, I, limit=0.0, smooth=None, seed=31):
    E = 1
    N = 2 * I
    torch.manual_seed(seed)
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-128, 128, (E, K, N), dtype=torch.int8, device=DEV)
    ws = (torch.rand(E, N, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    xs = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    sm = None
    has_col = has_row = False
    if smooth == "col":
        sm = (torch.rand(E, I, dtype=torch.float32, device=DEV) * 0.5 + 0.75)
        has_col = True
    elif smooth == "row":
        sm = (torch.rand(E, dtype=torch.float32, device=DEV) * 0.5 + 0.75)
        has_row = True
    else:
        sm = x
    starts = torch.zeros(E, dtype=torch.int32, device=DEV)
    ends = torch.full((E,), M, dtype=torch.int32, device=DEV)
    blk_m, blk_k, blk_n = 16, max(next_pow2(K), 16), max(next_pow2(I), 16)
    refq, refs = ref_gmm(x, w, ws, xs, sm if smooth else None, I, limit)

    outs = []
    for f in (F1, F2):
        y = torch.full((M, I), f, dtype=torch.int8, device=DEV)
        ys = torch.full((M,), -1.0e30, dtype=torch.float32, device=DEV)
        _gmm_v2_kernel[(E, triton.cdiv(M, blk_m))](
            x, w, ws, xs, sm, starts, ends, y, ys, K, N, I,
            BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
            HAS_SMOOTH_COL=has_col, HAS_SMOOTH_ROW=has_row,
            LIMIT_ON=bool(limit > 0.0), SWIGLU_LIMIT=float(limit))
        torch.npu.synchronize()
        outs.append((y.detach().cpu(), ys.detach().cpu()))
    dropped = (outs[0][0] == F1) & (outs[1][0] == F2)
    d = (outs[0][0].long() - refq.long()).abs()
    wrong = (~dropped) & (d > 1)
    se = float((outs[0][1].double() - refs).abs().max())
    tag = f"M={M} K={K} I={I:5d} smooth={smooth or '-'} limit={limit}"
    print(f"  [gmm_v2] {tag:44s} blk_n={blk_n:4d} dropped={int(dropped.sum()):5d} "
          f"wrong>1LSB={int(wrong.sum()):5d}/{M * I} scale_err={se:.2e}  "
          f"{'DROP!' if dropped.any() else ('wrong' if wrong.any() else 'clean')}", flush=True)
    if dropped.any():
        r0 = int(dropped.any(dim=1).nonzero()[0])
        print("      r=%d: %s" % (r0, "".join("X" if dropped[r0, c] else "." for c in range(I))), flush=True)


def main():
    print(f"[gmmv2] dev={DEV} triton={triton.__version__}", flush=True)
    cases = [(32, 128, I, 0.0, None) for I in (17, 24, 33, 48, 100, 127, 257, 512)]
    cases += [(32, 128, I, 1.5, None) for I in (33, 100)]
    cases += [(32, 128, I, 0.0, "col") for I in (33, 100)]
    cases += [(32, 128, I, 0.0, "row") for I in (33, 100)]
    cases += [(17, 256, 17, 0.0, None)]
    for M, K, I, lim, sm in cases:
        try:
            run_case(M, K, I, limit=lim, smooth=sm)
        except Exception as ex:  # noqa: BLE001
            print(f"  [gmm_v2] M={M} K={K} I={I} smooth={sm} EXC {type(ex).__name__}: "
                  f"{str(ex)[:140]}", flush=True)
    print("[gmmv2] done", flush=True)


if __name__ == "__main__":
    main()
