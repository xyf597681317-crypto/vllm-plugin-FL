# -*- coding: utf-8 -*-
"""诊断 F：在双填充（-99 / -77）判定下复现真实 kernel 的丢写，并逐步二分触发点。

V0 = 算子里的 _apply_quant_kernel（原样）
V1 = ls_A 基础写法 + src 间址（row = src // K, md &= src >= 0）
V2 = V1 + escale 载入 + 除法
V3 = V2 + fp16 往返 + floor/ceil/where + 饱和（= V0 等价物）
用法（容器内）：python3 -W ignore _probe_moe_bisect.py
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

from triton_ops.moe_init_routing_custom import (  # noqa: E402
    _apply_quant_kernel as _real_apply, _rowscale_kernel as _real_rowscale,
)

F1, F2 = -99, -77


@triton.jit
def v1_srcidx(X, SRC, OUTX, TOTAL, K, COLS, BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], x.to(tl.int8), mask=m2)


@triton.jit
def v2_div(X, SRC, ESCALE, OUTX, TOTAL, K, COLS, BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    s = tl.load(ESCALE + d, mask=md, other=1.0)
    q = x / s[:, None]
    tl.store(OUTX + d[:, None] * COLS + c[None, :], q.to(tl.int8), mask=m2)


@triton.jit
def v3_full(X, SRC, ESCALE, OUTX, TOTAL, K, COLS, BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    s = tl.load(ESCALE + d, mask=md, other=1.0)
    q = x / s[:, None]
    f = q.to(tl.float16).to(tl.float32)
    r = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], r.to(tl.int8), mask=m2)


def two_fill(launch, total, cols):
    outs = []
    for f in (F1, F2):
        o = torch.full((total, cols), f, dtype=torch.int8, device=DEV)
        launch(o)
        torch.npu.synchronize()
        outs.append(o.clone())
    return outs[0], (outs[0] == F1) & (outs[1] == F2)


def main():
    total, k, cols = 32, 2, 33
    torch.manual_seed(0)
    x = (torch.randn(total // k, cols, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    src = torch.arange(total, dtype=torch.int32, device=DEV)
    de = torch.zeros(total, dtype=torch.int32, device=DEV)
    smat = torch.zeros(1, dtype=torch.float32, device=DEV)
    esc = torch.empty(total, dtype=torch.float32, device=DEV)
    cases = []
    for bd, bc in ((8, 64), (2, 64), (8, 32)):
        grid = (triton.cdiv(total, bd), triton.cdiv(cols, bc))
        cases.append((f"V0 real  bd={bd} bc={bc}",
                      lambda o, bd=bd, bc=bc, grid=grid: _real_apply[grid](
                          x, src, de, smat, esc, o, total, k, cols,
                          BLK_D=bd, BLK_C=bc, HAS_SMOOTH=False)))
        cases.append((f"V1 srcidx bd={bd} bc={bc}",
                      lambda o, bd=bd, bc=bc, grid=grid: v1_srcidx[grid](
                          x, src, o, total, k, cols, BLK_D=bd, BLK_C=bc)))
        cases.append((f"V2 div   bd={bd} bc={bc}",
                      lambda o, bd=bd, bc=bc, grid=grid: v2_div[grid](
                          x, src, esc, o, total, k, cols, BLK_D=bd, BLK_C=bc)))
        cases.append((f"V3 full  bd={bd} bc={bc}",
                      lambda o, bd=bd, bc=bc, grid=grid: v3_full[grid](
                          x, src, esc, o, total, k, cols, BLK_D=bd, BLK_C=bc)))
    # esc 必须先算好（V0/V2/V3 都用它）
    _real_rowscale[(triton.cdiv(total, 8),)](x, src, de, smat, esc, total, k, cols,
                                             BLK_D=8, BLK_C=64, HAS_SMOOTH=False)
    torch.npu.synchronize()
    print("esc[:4] =", esc[:4].detach().cpu().tolist(), flush=True)
    for tag, fn in cases:
        try:
            got, dropped = two_fill(fn, total, cols)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}] EXC {type(ex).__name__}: {str(ex)[:150]}", flush=True)
            continue
        print(f"[{tag}] dropped={dropped.sum().item()}", flush=True)
        if dropped.sum().item():
            for i in range(total):
                if dropped[i].any():
                    m = "".join("X" if dropped[i, j] else "." for j in range(cols))
                    print(f"    d={i:3d} {m}", flush=True)
    print("[bisect] done", flush=True)


if __name__ == "__main__":
    main()
