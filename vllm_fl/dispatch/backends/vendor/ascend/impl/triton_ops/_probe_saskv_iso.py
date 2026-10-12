# -*- coding: utf-8 -*-
"""Probe13：二分定位 —— 负 base 下是哪个表达式把逐 lane 行号压成常量。

变体：
  V1 shift_w = tl.where(BASE<0, -BASE, 0)
  V2 shift_w = -BASE                （BASEC 为 Python 常量，一定 ≥0）
  V3 不用 shift，直接用 j = BASE + lane，row = tl.maximum(j, 0)
  V4 同 V2 但 rw 用 tl.where 而不是 min/max
每个变体把 off 和计算出的 row 都 dump 出来。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_iso.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.sparse_attn_sharedkv import _kv_elem_off  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
BS = 16
MB = 4
NROWS = 2 * MB * BS
N2 = 1
D = 512
NLANE = 48
BASE = -68


@triton.jit
def _v1(KV, BT, OUTOFF, BASEV, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        N2C: tl.constexpr, DC: tl.constexpr, S2C: tl.constexpr,
        NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    shift_w = tl.where(BASEV < 0, -BASEV, 0)
    row0_o = tl.maximum(BASEV, 0)
    cnt = tl.minimum(NLANE, S2C - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    off = _kv_elem_off(KV, BT, 0, row0_o + rw, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C,
                       True)
    tl.store(OUTOFF + lane, off)


@triton.jit
def _v2(KV, BT, OUTOFF, BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, N2C: tl.constexpr, DC: tl.constexpr,
        S2C: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    shift_w = -BASEC if BASEC < 0 else 0
    row0_o = max(BASEC, 0)
    cnt = tl.minimum(NLANE, S2C - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    off = _kv_elem_off(KV, BT, 0, row0_o + rw, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C,
                       True)
    tl.store(OUTOFF + lane, off)


@triton.jit
def _v3(KV, BT, OUTOFF, BASEV, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        N2C: tl.constexpr, DC: tl.constexpr, S2C: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    j = BASEV + lane
    row = tl.maximum(j, 0)
    off = _kv_elem_off(KV, BT, 0, row, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C, True)
    tl.store(OUTOFF + lane, off)


@triton.jit
def _v4(KV, BT, OUTOFF, BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, N2C: tl.constexpr, DC: tl.constexpr,
        S2C: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    shift_w = tl.where(BASEC < 0, -BASEC, 0)
    row0_o = tl.maximum(BASEC, 0)
    cnt = tl.minimum(NLANE, S2C - row0_o)
    rw = tl.where(lane >= shift_w, lane - shift_w, 0)
    rw = tl.where(rw < cnt, rw, cnt - 1)
    off = _kv_elem_off(KV, BT, 0, row0_o + rw, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C,
                       True)
    tl.store(OUTOFF + lane, off)


def main():
    kv = torch.zeros(NROWS, N2, D, dtype=torch.float16, device=DEV)
    bt = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device=DEV)
    o1 = torch.full((NLANE,), -1, dtype=torch.int32, device=DEV)
    o2 = torch.full((NLANE,), -1, dtype=torch.int32, device=DEV)
    o3 = torch.full((NLANE,), -1, dtype=torch.int32, device=DEV)
    o4 = torch.full((NLANE,), -1, dtype=torch.int32, device=DEV)
    bv = torch.tensor(BASE, dtype=torch.int32, device=DEV)
    _v1[(1,)](kv, bt, o1, bv, BLOCK_SIZE=BS, MAX_BLOCKS=MB, N2C=N2, DC=D, S2C=MB * BS,
              NLANE=NLANE)
    _v2[(1,)](kv, bt, o2, BASEC=BASE, BLOCK_SIZE=BS, MAX_BLOCKS=MB, N2C=N2, DC=D,
              S2C=MB * BS, NLANE=NLANE)
    _v3[(1,)](kv, bt, o3, bv, BLOCK_SIZE=BS, MAX_BLOCKS=MB, N2C=N2, DC=D, S2C=MB * BS,
              NLANE=NLANE)
    _v4[(1,)](kv, bt, o4, BASEC=BASE, BLOCK_SIZE=BS, MAX_BLOCKS=MB, N2C=N2, DC=D,
              S2C=MB * BS, NLANE=NLANE)
    torch.npu.synchronize()
    # 期望：lane<68 → row0(off=8192)；lane>=68 → (lane-68) 的物理行*512
    exp = []
    for lane in range(NLANE):
        rel = max(lane - 68, 0)
        page = int(bt[rel // BS].item())
        exp.append((page * BS + rel % BS) * D)
    print("BASE =", BASE, flush=True)
    for name, o in (("V1 where(标量tensor)", o1), ("V2 py常量分支", o2),
                    ("V3 只用 max(j,0)", o3), ("V4 where 版 rw", o4)):
        g = o.cpu().tolist()
        nd = sum(1 for a, b in zip(g, exp) if a != b)
        print(f"  {name:20s} 不符={nd}/{NLANE}", flush=True)
        print(f"    got={g[:12]}", flush=True)
        print(f"    exp={exp[:12]}", flush=True)
    print("ISODONE", flush=True)


if __name__ == "__main__":
    main()
