# -*- coding: utf-8 -*-
"""task-3 探针：dequant_swiglu_quant 的 store 形态（L168 / L201，报告点 L201）。

两个 kernel 的 store 都是「**标量行号 + 1-D 列 tile + 1-D 掩码**」：
    _dsq_swiglu_kernel:  row = program_id(0)；h = arange(BLK_H)；mask = hm = h < H
    _dsq_quant_kernel :  同上
⇒ 形态上不是「2-D block + 2-D 掩码」，本探针用 sentinel 双填充实测确认
   （非 2 幂 H ∈ {17,24,33,48,100,127,129,1000,7168}），并做端到端逐元素比对
   ＋「同一输入跑两次比较」检测 uninitialized lane。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_dsq.py
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
SENT_F = -12345.0

from triton_ops.dequant_swiglu_quant import (  # noqa: E402
    _dsq_swiglu_kernel, _dsq_quant_kernel, dequant_swiglu_quant)
from triton_ops.common import next_pow2  # noqa: E402


def probe_swiglu(M, H, act_left=False, mode1=False, blk_h=None):
    twh = 2 * H
    blk_h = blk_h or (128 if H > 128 else next_pow2(H))
    torch.manual_seed(41)
    x = torch.randint(-3000, 3000, (M, twh), dtype=torch.int32, device=DEV)
    ws = (torch.rand(twh, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    gid = torch.zeros(M, dtype=torch.int32, device=DEV)
    yb = torch.full((M, H), SENT_F, dtype=torch.float32, device=DEV)
    _dsq_swiglu_kernel[(M,)](
        x, ws, x, x, x, gid, yb, M, H, 0.0, 1.0, 0.0,
        IS_INT32=True, HAS_ASCALE=False, HAS_BIAS=False, HAS_QSCALE=False, HAS_GROUP=False,
        ACTIVATE_LEFT=act_left, SWIGLU_MODE1=mode1, BLK_H=blk_h)
    torch.npu.synchronize()
    got = yb.detach().cpu()
    dropped = (got == SENT_F)
    print(f"  [dsq_swiglu] M={M} H={H:5d} blk_h={blk_h:4d} act_left={int(act_left)} mode1={int(mode1)} "
          f"dropped={int(dropped.sum()):5d}/{M * H}  {'DROP!' if dropped.any() else 'clean'}", flush=True)


def probe_quant(M, H, blk_h=None):
    blk_h = blk_h or (128 if H > 128 else next_pow2(H))
    torch.manual_seed(42)
    yb = (torch.randn(M, H, dtype=torch.float32, device=DEV) * 3.0)
    sc_fill = []
    y_fill = []
    for f in (F1, F2):
        y = torch.full((M, H), f, dtype=torch.int8, device=DEV)
        sc = torch.full((M,), -1.0e30, dtype=torch.float32, device=DEV)
        _dsq_quant_kernel[(M,)](yb, y, sc, M, H, 1.0 / 127.0, BLK_H=blk_h)
        torch.npu.synchronize()
        y_fill.append(y.detach().cpu())
        sc_fill.append(sc.detach().cpu())
    dropped = (y_fill[0] == F1) & (y_fill[1] == F2)
    mx = yb.detach().cpu().double().abs().amax(dim=1)
    refs = mx / 127.0
    se = float((sc_fill[0].double() - refs).abs().max())
    print(f"  [dsq_quant]  M={M} H={H:5d} blk_h={blk_h:4d} "
          f"dropped={int(dropped.sum()):5d}/{M * H} scale_err={se:.2e}  "
          f"{'DROP!' if dropped.any() else 'clean'}", flush=True)


def e2e(M, H, act_left=False, smooth=True, ascale=True, bias=True, seed=43):
    torch.manual_seed(seed)
    twh = 2 * H
    x = torch.randint(-3000, 3000, (M, twh), dtype=torch.int32, device=DEV)
    ws = (torch.rand(twh, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    asc = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.75) if ascale else None
    bi = (torch.randn(twh, dtype=torch.float32, device=DEV) * 0.5) if bias else None
    qs = (torch.rand(H, dtype=torch.float32, device=DEV) * 0.5 + 0.75) if smooth else None

    xf = x.detach().cpu().double() * ws.detach().cpu().double()[None, :]
    if asc is not None:
        xf = xf * asc.detach().cpu().double()[:, None]
    if bi is not None:
        xf = xf + bi.detach().cpu().double()[None, :]
    if act_left:
        a, g = xf[:, :H], xf[:, H:]
    else:
        g, a = xf[:, :H], xf[:, H:]
    y = g * (a / (1.0 + torch.exp(-a)))
    if qs is not None:
        y = y * qs.detach().cpu().double()[None, :]
    mx = y.abs().amax(dim=1)
    refs = torch.where(mx > 0, mx / 127.0, torch.zeros_like(mx))
    refq = torch.round(torch.where(refs[:, None] > 0, y / refs[:, None], torch.zeros_like(y))).clamp(-128, 127)

    res = []
    for pat in (-99, 123):
        junk = torch.full((M, H), pat, dtype=torch.int8, device=DEV)
        del junk
        yq, sc = dequant_swiglu_quant(x, ws, asc, bi, qs, None, None,
                                      activate_left=act_left, quant_mode="dynamic")
        torch.npu.synchronize()
        res.append((yq.detach().cpu(), sc.detach().cpu()))
    d = (res[0][0].long() - refq.long()).abs()
    nd = int((d > 1).sum())
    nondet = int((res[0][0] != res[1][0]).sum())
    se = float((res[0][1].double() - refs).abs().max())
    print(f"  [dsq_e2e]    M={M} H={H:5d} act_left={int(act_left)} smooth={int(smooth)} "
          f"asc={int(ascale)} bias={int(bias)} wrong>1LSB={nd:5d}/{M * H} "
          f"run2diff={nondet:5d} scale_err={se:.2e}  "
          f"{'BAD' if (nd or nondet) else 'OK'}", flush=True)


def main():
    print(f"[dsq] dev={DEV} triton={triton.__version__}", flush=True)
    for H in (17, 24, 33, 48, 100, 127, 129, 256, 1000, 7168):
        blk = 128 if H > 128 else next_pow2(H)
        probe_swiglu(8, H, blk_h=blk)
        probe_quant(8, H, blk_h=blk)
    probe_swiglu(8, 33, act_left=True)
    probe_swiglu(8, 33, mode1=True)
    print("== 端到端 ==", flush=True)
    for H in (17, 24, 33, 48, 100, 127, 129, 1000, 7168):
        e2e(8, H)
    for H in (33, 100):
        e2e(8, H, act_left=True)
        e2e(8, H, smooth=False, ascale=False, bias=False)
    print("[dsq] done", flush=True)


if __name__ == "__main__":
    main()
