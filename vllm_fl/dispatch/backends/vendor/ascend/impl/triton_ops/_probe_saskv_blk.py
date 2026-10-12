# -*- coding: utf-8 -*-
"""Probe18：同一窗口逻辑下，扫 BLK（=BLK_K）大小 + 真实行值编码，找塌陷条件。

KV[row,0,d] = row*2 + (d%2)，读出 value//2 = 行号。
base 由运行时标量算出（与算子一致）。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_blk.py
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


@triton.jit
def _w(KV, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, NLANEC: tl.constexpr,
       DC: tl.constexpr, BASEV: tl.constexpr):
    lane = tl.arange(0, NLANEC)                 # 0..NLANE-1（= 窗口内偏移）
    s1 = tl.load(S1A)
    kvlo = tl.load(KVLA)
    base = s1 + (kvlo - tl.load(QLENA)) - WINL
    j = base + lane
    vld = (j >= 0) & (j < kvlo)
    js = tl.where(vld, j, 0)
    tl.store(OUT + lane, tl.load(KV + js * DC), mask=lane < NLANEC)


@triton.jit
def _w2(KV, S1A, KVLA, QLENA, OUT, WINL: tl.constexpr, NLANEC: tl.constexpr,
        DC: tl.constexpr):
    """更接近算子：加一个与 D 无关的逐 lane 头部（q 也不读，只验证寻址）"""
    lane = tl.arange(0, NLANEC)
    kvlo = tl.load(KVLA)
    base = tl.load(S1A) + (kvlo - tl.load(QLENA)) - WINL
    shift = tl.where(base < 0, -base, 0)
    r0 = tl.maximum(base, 0)
    cnt = tl.minimum(NLANEC, kvlo - r0)
    rw = tl.minimum(tl.maximum(lane - shift, 0), cnt - 1)
    js = r0 + rw
    tl.store(OUT + lane, tl.load(KV + js * DC))


def main():
    kv = torch.zeros(NROWS, 1, D, dtype=torch.float16, device=DEV)
    rid = torch.arange(NROWS, dtype=torch.float32, device=DEV)[:, None] * 2.0
    dpar = (torch.arange(D, dtype=torch.float32, device=DEV)[None, :] % 2)
    kv[:, 0, :] = (rid + dpar).to(torch.float16)

    for s1v, kvlov, qlenv in ((0, 64, 5), (0, 192, 8)):
        base = s1v + (kvlov - qlenv) - 127
        s1a = torch.tensor(s1v, dtype=torch.int32, device=DEV)
        kvla = torch.tensor(kvlov, dtype=torch.int32, device=DEV)
        qlena = torch.tensor(qlenv, dtype=torch.int32, device=DEV)
        for nl in (8, 16, 32, 64, 128):
            out = torch.full((nl,), -1.0, dtype=torch.float16, device=DEV)
            exp = [max(base + i, 0) if 0 <= base + i < kvlov else 0 for i in range(nl)]
            for name, kern in (("where(vld)", _w), ("shift+minmax", _w2)):
                out.fill_(-1.0)
                if name == "where(vld)":
                    _w[(1,)](kv, s1a, kvla, qlena, out, WINL=127, NLANEC=nl, DC=D,
                             BASEV=128)
                else:
                    _w2[(1,)](kv, s1a, kvla, qlena, out, WINL=127, NLANEC=nl, DC=D)
                torch.npu.synchronize()
                rows = [int(v) // 2 for v in out.detach().cpu().tolist()]
                nd = sum(1 for a, b in zip(rows, exp) if a != b)
                first_bad = next((i for i, (a, b) in enumerate(zip(rows, exp)) if a != b),
                                 None)
                print(f"base={base:5d} NLANE={nl:3d} {name:14s} 不符={nd:3d}"
                      f" 首个不符lane={first_bad}", flush=True)
                if nd:
                    print(f"      got={rows[:16]}", flush=True)
                    print(f"      exp={exp[:16]}", flush=True)
    print("BLKDONE", flush=True)


if __name__ == "__main__":
    main()
