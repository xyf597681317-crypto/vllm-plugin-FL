# -*- coding: utf-8 -*-
"""Probe25：块表读取本身的取法对比（2 维块表 + 向量下标 vs 其他取法）。

BT[b, bidx] 的四种读法，逐 bidx dump 出来比对：
  P1: tl.load(BT + b*MAX_BLOCKS + bidx)            ← 当前实现
  P2: p = BT + b*MAX_BLOCKS; tl.load(p + bidx)
  P3: 用 tl.arange(0, N) 构造 bidx（与 P1 同式，排除"下标来源"因素）
  P4: 常量最小下标 + 逐元素（对照）
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_bt.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
MB = 4
NB = 12          # 3 个 batch * 4 块
N = 8


@triton.jit
def _p1(BT, OUT, BIDXA, MAXB: tl.constexpr, NC: tl.constexpr):
    bidx = tl.arange(0, NC)
    b = tl.load(BIDXA)
    v = tl.load(BT + b * MAXB + bidx)
    tl.store(OUT + bidx, v)


@triton.jit
def _p2(BT, OUT, BIDXA, MAXB: tl.constexpr, NC: tl.constexpr):
    bidx = tl.arange(0, NC)
    b = tl.load(BIDXA)
    p = BT + b * MAXB
    v = tl.load(p + bidx)
    tl.store(OUT + bidx, v)


@triton.jit
def _p3(BT, OUT, BIDXA, MAXB: tl.constexpr, NC: tl.constexpr):
    bidx = tl.arange(0, NC) % MAXB
    b = tl.load(BIDXA)
    v = tl.load(BT + b * MAXB + bidx)
    tl.store(OUT + bidx, v)


@triton.jit
def _p4(BT, OUT, BIDXA, MAXB: tl.constexpr, NC: tl.constexpr):
    bidx = tl.arange(0, NC)
    v = tl.load(BT + bidx)
    tl.store(OUT + bidx, v)


def main():
    bt = (torch.arange(NB, dtype=torch.int32, device=DEV).reshape(3, MB) * 10 +
          torch.arange(NB, dtype=torch.int32, device=DEV).reshape(3, MB) % 7)
    # 让每个 (b, bidx) 的期望值独一无二
    bt = torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]],
                      dtype=torch.int32, device=DEV)
    BTD = bt.detach().cpu()
    print("BT =", BTD.tolist(), flush=True)
    for b in (0, 1, 2):
        ba = torch.tensor(b, dtype=torch.int32, device=DEV)
        for name, kern in (("P1 BT+b*MB+bidx", _p1), ("P2 (BT+b*MB)+bidx", _p2),
                           ("P3 同P1但下标取模", _p3), ("P4 无 batch 偏移", _p4)):
            out = torch.full((N,), -9, dtype=torch.int32, device=DEV)
            kern[(1,)](bt, out, ba, MAXB=MB, NC=N)
            torch.npu.synchronize()
            got = out.detach().cpu().tolist()
            if name.startswith("P4"):
                exp = [int(BTD[0, i % MB]) for i in range(N)]
            else:
                exp = [int(BTD[b, i % MB]) for i in range(N)]
            ok = "OK " if got == exp else "BAD"
            print(f"  {ok} b={b} {name:22s} got={got}", flush=True)
            if ok == "BAD":
                print(f"      {'':22s} exp={exp}", flush=True)
    print("BTDONE", flush=True)


if __name__ == "__main__":
    main()
