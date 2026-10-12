# -*- coding: utf-8 -*-
"""诊断 D：找出在「非 2 幂行距 + 掩码」下不会丢写的 store 写法。

参考值 ref[d,c] = (7d + 3c) % 100（int8，非负）；缓冲区初始化为 -1，
仍是 -1 的元素 = store 完全没写（丢写）。
用法（容器内）：python3 -W ignore _probe_moe_storevar.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = os.environ.get("MOE_DEV", "npu:1")


@triton.jit
def st_A(OUT, D, S, BD: tl.constexpr, BC: tl.constexpr):
    """现行写法：2-D grid + [BD,BC] 2-D 掩码 store。"""
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m2 = (d < D)[:, None] & (c < S)[None, :]
    v = ((d[:, None] * 7 + c[None, :] * 3) % 100).to(tl.int8)
    tl.store(OUT + d[:, None] * S + c[None, :], v, mask=m2)


@triton.jit
def st_B(OUT, D, S, BD: tl.constexpr, BC: tl.constexpr):
    """1-D grid + 行块内 runtime 列循环，仍是 2-D 掩码 store。"""
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    md = d < D
    for c0 in range(0, tl.cdiv(S, BC)):
        c = c0 * BC + tl.arange(0, BC)
        m2 = md[:, None] & (c < S)[None, :]
        v = ((d[:, None] * 7 + c[None, :] * 3) % 100).to(tl.int8)
        tl.store(OUT + d[:, None] * S + c[None, :], v, mask=m2)


@triton.jit
def st_C(OUT, D, S, BLK: tl.constexpr):
    """整体展平：1-D tile + o//S, o%S，连续 1-D store。"""
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < D * S
    d = o // S
    c = o - d * S
    v = ((d * 7 + c * 3) % 100).to(tl.int8)
    tl.store(OUT + o, v, mask=m)


@triton.jit
def st_D(OUT, D, S, BC: tl.constexpr):
    """每行一个 program（grid 第 0 维 = 行号），行内 1-D 列分块，1-D 掩码 store。"""
    d = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < S
    v = ((d * 7 + c * 3) % 100).to(tl.int8)
    tl.store(OUT + d * S + c, v, mask=m)


@triton.jit
def st_E(OUT, D, S, ROWS: tl.constexpr, BC: tl.constexpr):
    """每 program 处理 ROWS 行；行内 1-D 列循环，全部 1-D 掩码 store。"""
    d0 = tl.program_id(0) * ROWS
    for i in range(ROWS):
        d = d0 + i
        if d < D:
            for c0 in range(0, tl.cdiv(S, BC)):
                c = c0 * BC + tl.arange(0, BC)
                m = c < S
                v = ((d * 7 + c * 3) % 100).to(tl.int8)
                tl.store(OUT + d * S + c, v, mask=m)


def check(tag, fn, D, S, cb_blocks=None, bc=None):
    out = torch.full((D, S), -1, dtype=torch.int8, device=DEV)
    try:
        fn(out)
        torch.npu.synchronize()
    except Exception as ex:  # noqa: BLE001
        print(f"  [{tag}] EXC {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        return
    d = torch.arange(D).view(-1, 1)
    c = torch.arange(S).view(1, -1)
    ref = ((d * 7 + c * 3) % 100).to(torch.int8)
    got = out.detach().cpu()
    dropped = (got == -1)
    wrong = (got != -1) & (got != ref)
    n_bad_rows = (dropped.any(dim=1)).sum().item()
    print(f"  [{tag}] D={D} S={S} dropped={dropped.sum().item()}/{got.numel()} "
          f"wrong={wrong.sum().item()} rows_with_drop={n_bad_rows}", flush=True)
    if dropped.sum().item():
        for i in range(D):
            if dropped[i].any():
                m = "".join("X" if dropped[i, j] else ("." if got[i, j] == ref[i, j] else "#")
                            for j in range(S))
                print(f"      d={i:3d} {m}", flush=True)
                if i > 6:
                    break


if __name__ == "__main__":
    print(f"[storevar] dev={DEV}", flush=True)
    for S in (33, 17, 48, 100, 7168):
        D = 32
        print(f"\n=== S={S} ===", flush=True)
        check("A 2Dgrid+2Dmask  BD=8 BC=64",
              lambda o, S=S: st_A[(triton.cdiv(32, 8), triton.cdiv(S, 64))](
                  o, 32, S, BD=8, BC=64), 32, S)
        check("B 1Dgrid+loop 2Dmask BD=8 BC=64",
              lambda o, S=S: st_B[(triton.cdiv(32, 8),)](o, 32, S, BD=8, BC=64), 32, S)
        check("C flat 1D BLK=256",
              lambda o, S=S: st_C[(triton.cdiv(32 * S, 256),)](o, 32, S, BLK=256), 32, S)
        check("D per-row 1D BC=64",
              lambda o, S=S: st_D[(32, triton.cdiv(S, 64))](o, 32, S, BC=64), 32, S)
        check("E 1D loop ROWS=4 BC=64",
              lambda o, S=S: st_E[(triton.cdiv(32, 4),)](o, 32, S, ROWS=4, BC=64), 32, S)
    print("\n[storevar] done", flush=True)
