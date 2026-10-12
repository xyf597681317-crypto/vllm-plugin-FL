# -*- coding: utf-8 -*-
"""Probe16：决定性对比 —— base 由**运行时标量**算出（与算子一致），三种写法。

V1: shift_w = tl.where(base<0, -base, 0); rw = tl.minimum(tl.maximum(lane-shift_w,0), cnt-1)
V2: Lead 建议：j = base+lane; vld=(j>=0)&(j<kvlen); js = tl.where(vld, j, 0)  → 行号 = js
V3: 同 V2，但行号用 tl.maximum(j, 0)（对照）

行值编码 KV[row,0,d]=row*2+(d%2)；读出 value//2 即行号。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_rt.py
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
DC = 512


@triton.jit
def _v1(KV, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    s1 = tl.load(S1A)
    kvlo = tl.load(KVLA)
    base = s1 + (kvlo - tl.load(QLENA)) - WINL
    shift_w = tl.where(base < 0, -base, 0)
    row0_o = tl.maximum(base, 0)
    cnt = tl.minimum(NLANE, kvlo - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    tl.store(OUT + lane, tl.load(KV + (row0_o + rw) * DC))


@triton.jit
def _v2(KV, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    s1 = tl.load(S1A)
    kvlo = tl.load(KVLA)
    base = s1 + (kvlo - tl.load(QLENA)) - WINL
    j = base + lane                       # 允许为负
    vld = (j >= 0) & (j < kvlo)
    js = tl.where(vld, j, 0)              # 逐 lane 选择，0 恒合法
    tl.store(OUT + lane, tl.load(KV + js * DC))


@triton.jit
def _v3(KV, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    s1 = tl.load(S1A)
    kvlo = tl.load(KVLA)
    base = s1 + (kvlo - tl.load(QLENA)) - WINL
    j = base + lane
    js = tl.maximum(j, 0)
    tl.store(OUT + lane, tl.load(KV + js * DC))


def main():
    kv = torch.zeros(NROWS, 1, D, dtype=torch.float16, device=DEV)
    rid = torch.arange(NROWS, dtype=torch.float32, device=DEV)[:, None] * 2.0
    dpar = (torch.arange(D, dtype=torch.float32, device=DEV)[None, :] % 2)
    kv[:, 0, :] = (rid + dpar).to(torch.float16)
    out = torch.full((NLANE,), -1.0, dtype=torch.float16, device=DEV)

    for tag, s1v, kvlov, qlenv in (("s1=0 kvlo=64 qlen=5", 0, 64, 5),
                                   ("s1=1 kvlo=64 qlen=5", 1, 64, 5),
                                   ("s1=0 kvlo=48 qlen=4", 0, 48, 4),
                                   ("s1=4 kvlo=64 qlen=5", 4, 64, 5),
                                   ("s1=0 kvlo=192 qlen=8", 0, 192, 8)):
        s1a = torch.tensor(s1v, dtype=torch.int32, device=DEV)
        kvla = torch.tensor(kvlov, dtype=torch.int32, device=DEV)
        qlena = torch.tensor(qlenv, dtype=torch.int32, device=DEV)
        base = s1v + (kvlov - qlenv) - 127
        exp = [max(base + i, 0) for i in range(NLANE)]
        exp = [x if (0 <= (base + i) < kvlov) else 0 for i, x in enumerate(exp)]
        for name, kern in (("V1 shift+minmax", _v1), ("V2 where(vld,j,0)", _v2),
                           ("V3 max(j,0)", _v3)):
            out.fill_(-1.0)
            kern[(1,)](kv, s1a, kvla, qlena, out, WINL=127, BLOCK_SIZE=BS, MAX_BLOCKS=MB,
                       DC=DC, NLANE=NLANE)
            torch.npu.synchronize()
            rows = [int(v) // 2 for v in out.detach().cpu().tolist()]
            nd = sum(1 for a, b in zip(rows, exp) if a != b)
            print(f"{tag:22s} base={base:5d} {name:20s} 不符={nd:2d}/{NLANE}"
                  f"  got={rows[:12]}", flush=True)
        print(f"{'':22s} {'':13s} {'期望':20s}      exp={exp[:12]}", flush=True)
    print("RTDONE", flush=True)


if __name__ == "__main__":
    main()
