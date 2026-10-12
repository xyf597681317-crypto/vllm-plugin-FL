# -*- coding: utf-8 -*-
"""诊断 E：load+store 组合下的丢写，以及候选修复写法的正确性。

判定用「双填充」法（不会被 kernel 合法输出误判）：
    同一 kernel 跑两遍，输出缓冲分别预填 F1=-99 / F2=-77；
    某元素两遍仍等于各自填充值 => store 完全没写（dropped）。
用法（容器内）：python3 -W ignore _probe_moe_ls.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = os.environ.get("MOE_DEV", "npu:1")
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

F1, F2 = -99, -77


def val(d, c):
    return ((d * 7 + c * 3) % 100) - 50


@triton.jit
def ls_A(OUT, X, D, S, BD: tl.constexpr, BC: tl.constexpr):
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m2 = (d < D)[:, None] & (c < S)[None, :]
    x = tl.load(X + d[:, None] * S + c[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(OUT + d[:, None] * S + c[None, :], x.to(tl.int8), mask=m2)


@triton.jit
def ls_B(OUT, X, D, S, BD: tl.constexpr, BC: tl.constexpr):
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    md = d < D
    for c0 in range(0, tl.cdiv(S, BC)):
        c = c0 * BC + tl.arange(0, BC)
        m2 = md[:, None] & (c < S)[None, :]
        x = tl.load(X + d[:, None] * S + c[None, :], mask=m2, other=0.0).to(tl.float32)
        tl.store(OUT + d[:, None] * S + c[None, :], x.to(tl.int8), mask=m2)


@triton.jit
def ls_C(OUT, X, D, S, BLK: tl.constexpr):
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < D * S
    d = o // S
    c = o - d * S
    x = tl.load(X + d * S + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + o, x.to(tl.int8), mask=m)


@triton.jit
def ls_D(OUT, X, D, S, BC: tl.constexpr):
    d = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < S
    x = tl.load(X + d * S + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + d * S + c, x.to(tl.int8), mask=m)


@triton.jit
def ls_E(OUT, X, D, S, ROWS: tl.constexpr, BC: tl.constexpr):
    d0 = tl.program_id(0) * ROWS
    for i in range(ROWS):
        d = d0 + i
        if d < D:
            for c0 in range(0, tl.cdiv(S, BC)):
                c = c0 * BC + tl.arange(0, BC)
                m = c < S
                x = tl.load(X + d * S + c, mask=m, other=0.0).to(tl.float32)
                tl.store(OUT + d * S + c, x.to(tl.int8), mask=m)


def two_fill(launch, D, S):
    outs = []
    for f in (F1, F2):
        o = torch.full((D, S), f, dtype=torch.int8, device=DEV)
        launch(o)
        torch.npu.synchronize()
        outs.append(o.clone())
    dropped = (outs[0] == F1) & (outs[1] == F2)
    return outs[0], dropped


def report(tag, launch, D, S, ref):
    try:
        got, dropped = two_fill(launch, D, S)
    except Exception as ex:  # noqa: BLE001
        print(f"  [{tag:34s}] EXC {type(ex).__name__}: {str(ex)[:140]}", flush=True)
        return
    written = ~dropped
    wrong = written & (got.long() != ref.long())
    line = (f"  [{tag:34s}] dropped={dropped.sum().item():5d} written_wrong={wrong.sum().item():5d} "
            f"rows_with_drop={dropped.any(dim=1).sum().item():3d}")
    print(line, flush=True)
    if dropped.sum().item():
        for i in range(D):
            if dropped[i].any():
                m = "".join("X" if dropped[i, j] else ("." if got[i, j] == ref[i, j] else "#")
                            for j in range(S))
                print(f"        d={i:3d} {m}", flush=True)
                if i > 8:
                    break


if __name__ == "__main__":
    print(f"[probe-ls] dev={DEV}", flush=True)
    D = 32
    for S in (32, 33, 17, 48, 24):
        c = torch.arange(S).view(1, -1)
        r = torch.arange(D).view(-1, 1)
        ref = (val(r, c)).to(torch.int8).npu()
        x = ref.to(torch.float16)
        print(f"\n=== S={S} (D={D}) ===", flush=True)
        report("A 2Dgrid 2Dmask BD=8 BC=64",
               lambda o, S=S, x=x: ls_A[(triton.cdiv(D, 8), triton.cdiv(S, 64))](
                   o, x, D, S, BD=8, BC=64), D, S, ref)
        report("B 1Dgrid+loop 2Dmask BD=8 BC=64",
               lambda o, S=S, x=x: ls_B[(triton.cdiv(D, 8),)](
                   o, x, D, S, BD=8, BC=64), D, S, ref)
        report("C flat 1D BLK=256",
               lambda o, S=S, x=x: ls_C[(triton.cdiv(D * S, 256),)](
                   o, x, D, S, BLK=256), D, S, ref)
        report("D per-row 1D BC=64",
               lambda o, S=S, x=x: ls_D[(D, triton.cdiv(S, 64))](
                   o, x, D, S, BC=64), D, S, ref)
        report("E rowloop ROWS=4 BC=64",
               lambda o, S=S, x=x: ls_E[(triton.cdiv(D, 4),)](
                   o, x, D, S, ROWS=4, BC=64), D, S, ref)
    print("\n[probe-ls] done", flush=True)
