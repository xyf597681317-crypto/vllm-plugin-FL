# -*- coding: utf-8 -*-
"""Probe19：逼近真算子 —— 随机块表 + NLANE=128 窗口 + BLK_K 分块 + cnt 两种写法。

KV[row,0,d] = row*2 + (d%2)，读出 value//2 = 行号（行≤63，fp16 精确）。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_real.py
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
NLANE = 128


@triton.jit
def _k(KV, BT, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, NLANEC: tl.constexpr,
       BLKC: tl.constexpr, MBC: tl.constexpr, BSC: tl.constexpr, DSTRIDE: tl.constexpr,
       CNTMODE: tl.constexpr, BIDX: tl.constexpr):
    lane = tl.arange(0, BLKC)
    kvlo = tl.load(KVLA)
    base = tl.load(S1A) + (kvlo - tl.load(QLENA)) - WINL
    for c in range(0, tl.cdiv(NLANEC, BLKC)):
        ln = c * BLKC + lane
        shift = tl.where(base < 0, -base, 0)
        r0 = tl.maximum(base, 0)
        if CNTMODE == 0:
            cnt = tl.minimum(NLANEC, kvlo - r0)
        else:
            cnt = tl.minimum(NLANEC, MBC * BSC - r0)
        rw = tl.minimum(tl.maximum(ln - shift, 0), cnt - 1)
        row = r0 + rw
        bidx = tl.minimum(tl.maximum(row // BSC, 0), MBC - 1)
        page = tl.maximum(tl.load(BT + BIDX * MBC + bidx), 0)
        prow = page * BSC + (row % BSC)
        off = prow * DSTRIDE
        v = tl.load(KV + off)
        tl.store(OUT + ln, v)


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
    print("bt =", BTD.tolist(), flush=True)

    for b, s1v, kvlov, qlenv in ((0, 0, 64, 5), (1, 0, 48, 4), (0, 4, 64, 5)):
        base = s1v + (kvlov - qlenv) - 127
        s1a = torch.tensor(s1v, dtype=torch.int32, device=DEV)
        kvla = torch.tensor(kvlov, dtype=torch.int32, device=DEV)
        qlena = torch.tensor(qlenv, dtype=torch.int32, device=DEV)
        for cntmode in (0, 1):
            out = torch.full((NLANE,), -1.0, dtype=torch.float16, device=DEV)
            _k[(1,)](kv, bt, s1a, kvla, qlena, out, WINL=127, NLANEC=NLANE, BLKC=8,
                     MBC=MB, BSC=BS, DSTRIDE=D, CNTMODE=cntmode, BIDX=b)
            torch.npu.synchronize()
            rows = [int(v) // 2 for v in out.detach().cpu().tolist()]
            exp = []
            for ln in range(NLANE):
                j = base + ln
                if not (0 <= j < kvlov):
                    exp.append(-1)
                else:
                    exp.append(int(BTD[b, j // BS]) * BS + j % BS)
            nd = sum(1 for a, e in zip(rows, exp) if e >= 0 and a != e)
            nvalid = sum(1 for e in exp if e >= 0)
            print(f"b={b} s1={s1v} kvlo={kvlov} base={base:5d} cntMode={cntmode}"
                  f" 有效={nvalid} 不符={nd}", flush=True)
            if nd:
                bad = [i for i, (a, e) in enumerate(zip(rows, exp)) if e >= 0 and a != e]
                print(f"    首个不符 lane={bad[0]} got={rows[bad[0]]} exp={exp[bad[0]]}",
                      flush=True)
                print(f"    got[64:76]={rows[64:76]}", flush=True)
                print(f"    exp[64:76]={exp[64:76]}", flush=True)
    print("REALDONE", flush=True)


if __name__ == "__main__":
    main()
