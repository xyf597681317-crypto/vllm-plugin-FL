# -*- coding: utf-8 -*-
"""Probe17：shift/cnt 为运行时标量时，逐个算子定位塌陷点。

每个变体把一个"逐 lane 向量"写到 OUT，单独看哪个算子把向量压成常量。
输入：SHIFTA/CNTA 为 1 元素 int32 设备张量（运行时标量）。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_op.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
NLANE = 16
SHIFT = 4
CNT = 8


@triton.jit
def _t1(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    sh = tl.load(SHIFTA)
    cn = tl.load(CNTA)
    tl.store(OUT + lane, lane - sh)                 # 期望 [-4..11]


@triton.jit
def _t2(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    sh = tl.load(SHIFTA)
    tl.store(OUT + lane, tl.maximum(lane - sh, 0))  # 期望 [0,0,0,0,0,1..11]


@triton.jit
def _t3(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    sh = tl.load(SHIFTA)
    cn = tl.load(CNTA)
    tl.store(OUT + lane, tl.minimum(tl.maximum(lane - sh, 0), cn - 1))  # [0,0,0,0,0,1..7]


@triton.jit
def _t4(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    cn = tl.load(CNTA)
    tl.store(OUT + lane, tl.minimum(lane, cn - 1))  # [0..7,7,...]


@triton.jit
def _t5(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    sh = tl.load(SHIFTA)
    cn = tl.load(CNTA)
    rw = tl.where(lane >= sh, lane - sh, 0)          # 期望 [0,0,0,0,0,1..11]
    tl.store(OUT + lane, tl.minimum(rw, cn - 1))


@triton.jit
def _t6(SHIFTA, CNTA, OUT, NLANE: tl.constexpr):
    """只做 where 的标量条件，看 base<0 这种运行时标量判断是否正常。"""
    lane = tl.arange(0, NLANE)
    sh = tl.load(SHIFTA)
    cond = sh > 2
    v = tl.where(cond, lane + 100, lane)
    tl.store(OUT + lane, v)


def main():
    sha = torch.tensor(SHIFT, dtype=torch.int32, device=DEV)
    cna = torch.tensor(CNT, dtype=torch.int32, device=DEV)
    out = torch.full((NLANE,), -99.0, dtype=torch.float32, device=DEV)
    exp = {
        "_t1 lane-sh": [i - SHIFT for i in range(NLANE)],
        "_t2 max(.,0)": [max(i - SHIFT, 0) for i in range(NLANE)],
        "_t3 minmax": [min(max(i - SHIFT, 0), CNT - 1) for i in range(NLANE)],
        "_t4 min(lane,cn-1)": [min(i, CNT - 1) for i in range(NLANE)],
        "_t5 where+min": [min(max(i - SHIFT, 0), CNT - 1) for i in range(NLANE)],
        "_t6 where(cond)": [i + 100 if SHIFT > 2 else i for i in range(NLANE)],
    }
    for name, kern in (("_t1 lane-sh", _t1), ("_t2 max(.,0)", _t2), ("_t3 minmax", _t3),
                       ("_t4 min(lane,cn-1)", _t4), ("_t5 where+min", _t5),
                       ("_t6 where(cond)", _t6)):
        out.fill_(-99.0)
        kern[(1,)](sha, cna, out, NLANE=NLANE)
        torch.npu.synchronize()
        g = [int(v) for v in out.detach().cpu().tolist()]
        ok = "OK " if g == exp[name] else "BAD"
        print(f"{ok} {name:20s} got={g}", flush=True)
        if ok == "BAD":
            print(f"    {'':20s} exp={exp[name]}", flush=True)
    print("OPDONE", flush=True)


if __name__ == "__main__":
    main()
