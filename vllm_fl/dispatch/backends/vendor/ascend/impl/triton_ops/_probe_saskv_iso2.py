# -*- coding: utf-8 -*-
"""Probe14：在**全 Python 常量**下对比三种"窗口→行号"写法，并 dump 中间量。

A: shift_w = tl.where(base<0, -base, 0); rw = tl.minimum(tl.maximum(lane-shift_w,0), cnt-1)
B: 纯 Python 常量分支：base<0 → rw = lane-shift(常量)；否则 rw = lane
C: Lead 建议：j = base+lane; vld=(j>=0)&(j<kvlen); js=tl.where(vld,j,0)（负值根本不进取址）

行值编码 KV[row,0,d]=row*2+(d%2)，读出 value//2 即行号。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_iso2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
BS = 16
MB = 4
NROWS = 2 * MB * BS
D = 512
NLANE = 32
BASE = -68
SHIFT = 68        # = -BASE


@triton.jit
def _A(KV, OUT, BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
       DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    shift_w = tl.where(BASEC < 0, -BASEC, 0)
    row0_o = tl.maximum(BASEC, 0)
    cnt = tl.minimum(NLANE, MAX_BLOCKS * BLOCK_SIZE - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    row = row0_o + rw
    off = (row * 1 + 0) * DC
    tl.store(OUT + lane, tl.load(KV + off))


@triton.jit
def _B(KV, OUT, BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
       DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    if BASEC < 0:
        row = lane - (-BASEC)                      # 编译期常量
        row = tl.maximum(row, 0)
    else:
        row = BASEC + lane
    off = (row * 1 + 0) * DC
    tl.store(OUT + lane, tl.load(KV + off))


@triton.jit
def _C(KV, OUT, BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
       DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    j = BASEC + lane                                # 允许为负
    vld = (j >= 0) & (j < MAX_BLOCKS * BLOCK_SIZE)
    js = tl.where(vld, j, 0)                        # 逐 lane 选择，0 恒合法
    off = (js * 1 + 0) * DC
    tl.store(OUT + lane, tl.load(KV + off))


def main():
    kv = torch.zeros(NROWS, 1, D, dtype=torch.float16, device=DEV)
    rid = torch.arange(NROWS, dtype=torch.float32, device=DEV)[:, None] * 2.0
    dpar = (torch.arange(D, dtype=torch.float32, device=DEV)[None, :] % 2)
    kv[:, 0, :] = (rid + dpar).to(torch.float16)
    out = torch.full((NLANE,), -1.0, dtype=torch.float16, device=DEV)
    exp = [max(BASE + i, 0) for i in range(NLANE)]
    for base, sh in ((BASE, SHIFT), (-67, 67), (0, 0), (5, 0), (-4, 4)):
        for name, kern in (("A where+minmax", _A), ("B 常量分支", _B), ("C where(vld,j,0)", _C)):
            out.fill_(-1.0)
            if name.startswith("A"):
                kern[(1,)](kv, out, BASEC=base, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=D,
                           NLANE=NLANE)
            elif name.startswith("B"):
                kern[(1,)](kv, out, BASEC=base, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=D,
                           NLANE=NLANE)
            else:
                kern[(1,)](kv, out, BASEC=base, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=D,
                           NLANE=NLANE)
            torch.npu.synchronize()
            got_rows = [(int(v) // 2) for v in out.detach().cpu().tolist()]
            e = [max(base + i, 0) for i in range(NLANE)]
            nd = sum(1 for a, b in zip(got_rows, e) if a != b)
            print(f"base={base:4d} {name:18s} 不符={nd:2d}/{NLANE}  "
                  f"行号got={got_rows[:10]}", flush=True)
            if nd:
                print(f"                      行号exp={e[:10]}", flush=True)
    print("ISO2DONE", flush=True)


if __name__ == "__main__":
    main()
