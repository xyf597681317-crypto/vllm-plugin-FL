# -*- coding: utf-8 -*-
"""Probe20：决定性 —— 2D tile load（算子真实形态）下，负 base 的两种写法对比。

行值编码：KV[row,0,d] = row*2 + (d%2)（行≤63，fp16 精确，读出//2 = 行号）
构造 (8, 512) tile：tile[l, d] = KV[row_l, 0, d]
写法 W1 = 当前实现（shift + min(max(lane-shift,0), cnt-1)）
写法 W2 = Lead 建议（j=base+lane; vld=(j>=0)&(j<kvlo); js=where(vld,j,0)）
两者都 dump 每个 tile 的第 0 列，反推行号。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_tile2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
MB = 4
BS = 16
NROWS = 2 * MB * BS
D = 512
BLK_K = 8
NLANE = 128


@triton.jit
def _w1(KV, BT, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, NLANEC: tl.constexpr,
        BLKC: tl.constexpr, MBC: tl.constexpr, BSC: tl.constexpr, DC: tl.constexpr,
        BIDX: tl.constexpr):
    lane = tl.arange(0, BLKC)
    d = tl.arange(0, DC)
    kvlo = tl.load(KVLA)
    base = tl.load(S1A) + (kvlo - tl.load(QLENA)) - WINL
    shift = tl.where(base < 0, -base, 0)
    r0 = tl.maximum(base, 0)
    cnt = tl.minimum(NLANEC, kvlo - r0)
    for c in range(0, tl.cdiv(NLANEC, BLKC)):
        ln = c * BLKC + lane
        rw = tl.minimum(tl.maximum(ln - shift, 0), cnt - 1)
        row = r0 + rw
        bidx = tl.minimum(tl.maximum(row // BSC, 0), MBC - 1)
        page = tl.maximum(tl.load(BT + BIDX * MBC + bidx), 0)
        off = (page * BSC + (row % BSC)) * DC
        tile = tl.load(KV + off[:, None] + d[None, :])
        tl.store(OUT + ln, tl.sum(tile * ((d == 0).to(tl.float16))[None, :], axis=1))  # 取 d=0 列


@triton.jit
def _w2(KV, BT, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, NLANEC: tl.constexpr,
        BLKC: tl.constexpr, MBC: tl.constexpr, BSC: tl.constexpr, DC: tl.constexpr,
        BIDX: tl.constexpr):
    lane = tl.arange(0, BLKC)
    d = tl.arange(0, DC)
    kvlo = tl.load(KVLA)
    base = tl.load(S1A) + (kvlo - tl.load(QLENA)) - WINL
    for c in range(0, tl.cdiv(NLANEC, BLKC)):
        ln = c * BLKC + lane
        j = base + ln
        vld = (j >= 0) & (j < kvlo)
        row = tl.where(vld, j, 0)
        bidx = tl.minimum(tl.maximum(row // BSC, 0), MBC - 1)
        page = tl.maximum(tl.load(BT + BIDX * MBC + bidx), 0)
        off = (page * BSC + (row % BSC)) * DC
        tile = tl.load(KV + off[:, None] + d[None, :])
        tl.store(OUT + ln, tl.sum(tile * ((d == 0).to(tl.float16))[None, :], axis=1))  # 取 d=0 列


def main():
    torch.manual_seed(81)
    kv = torch.zeros(NROWS, 1, D, dtype=torch.float16, device=DEV)
    rid = torch.arange(NROWS, dtype=torch.float32, device=DEV)[:, None] * 2.0
    dpar = (torch.arange(D, dtype=torch.float32, device=DEV)[None, :] % 2)
    kv[:, 0, :] = (rid + dpar).to(torch.float16)
    bt = torch.zeros(2, MB, dtype=torch.int32, device=DEV)
    for b in range(2):
        bt[b] = torch.randperm(MB, device=DEV).to(torch.int32) + b * MB
    BTD = bt.detach().cpu()
    out = torch.full((NLANE,), -1.0, dtype=torch.float16, device=DEV)
    print("bt =", BTD.tolist(), flush=True)
    for b, s1v, kvlov, qlenv in ((0, 0, 64, 5), (0, 4, 64, 5), (1, 0, 48, 4)):
        base = s1v + (kvlov - qlenv) - 127
        s1a = torch.tensor(s1v, dtype=torch.int32, device=DEV)
        kvla = torch.tensor(kvlov, dtype=torch.int32, device=DEV)
        qlena = torch.tensor(qlenv, dtype=torch.int32, device=DEV)
        for name, kern in (("W1 现实现", _w1), ("W2 where(vld)", _w2)):
            out.fill_(-1.0)
            kern[(1,)](kv, bt, s1a, kvla, qlena, out, WINL=127, NLANEC=NLANE, BLKC=BLK_K,
                       MBC=MB, BSC=BS, DC=D, BIDX=b)
            torch.npu.synchronize()
            rows = [int(v) // 2 for v in out.detach().cpu().tolist()]
            exp = []
            for ln in range(NLANE):
                j = base + ln
                exp.append(int(BTD[b, j // BS]) * BS + j % BS if 0 <= j < kvlov else -1)
            nd = sum(1 for a, e in zip(rows, exp) if e >= 0 and a != e)
            nv = sum(1 for e in exp if e >= 0)
            print(f"b={b} s1={s1v} kvlo={kvlov} base={base:5d} {name:14s}"
                  f" 有效={nv:3d} 不符={nd:3d}", flush=True)
            if nd:
                bad = [i for i, (a, e) in enumerate(zip(rows, exp)) if e >= 0 and a != e]
                print(f"    首个不符 lane={bad[0]} got={rows[bad[0]]} exp={exp[bad[0]]}"
                      f"  前 12 个不符={bad[:12]}", flush=True)
    print("TILE2DONE", flush=True)


if __name__ == "__main__":
    main()
