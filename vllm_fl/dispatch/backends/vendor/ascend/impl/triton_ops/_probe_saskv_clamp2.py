# -*- coding: utf-8 -*-
"""Probe5b：最小复现（修正别名 + 统一参数）—— TND+PA 下负 j 的行寻址。

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_clamp2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.sparse_attn_sharedkv import _row_of  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
BS = 16
MAXB = 4
S2 = BS * MAXB


@triton.jit
def _k_off(KV, BT, OUT, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
           N2: tl.constexpr, DD: tl.constexpr, S2C: tl.constexpr, JLO: tl.constexpr,
           BLK: tl.constexpr):
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    off = _row_of(KV, BT, 0, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2C, True, vld)
    tl.store(OUT + tl.arange(0, BLK), off)


@triton.jit
def _k_gather(KV, BT, OUT, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
              N2: tl.constexpr, DD: tl.constexpr, S2C: tl.constexpr, JLO: tl.constexpr,
              BLK: tl.constexpr):
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    off = _row_of(KV, BT, 0, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2C, True, vld)
    val = tl.load(KV + off, mask=vld, other=-7.0)
    tl.store(OUT + tl.arange(0, BLK), val)


@triton.jit
def _k_gather_shift(KV, BT, OUT, SHIFT, BLOCK_SIZE: tl.constexpr,
                    MAX_BLOCKS: tl.constexpr, N2: tl.constexpr, DD: tl.constexpr,
                    S2C: tl.constexpr, JLO: tl.constexpr, BLK: tl.constexpr):
    """修法草案：把整窗右移，使参与寻址的下标恒为非负。"""
    lane = tl.arange(0, BLK)
    j = (JLO + SHIFT) + lane
    vld = (j - SHIFT) >= 0
    bidx = tl.minimum(tl.maximum(j // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
    page = tl.load(BT + bidx, mask=vld, other=0)
    page = tl.maximum(page, 0)
    kvrow = page * BLOCK_SIZE + (j % BLOCK_SIZE)
    off = (kvrow * N2 + 0) * DD
    val = tl.load(KV + off, mask=vld, other=-7.0)
    tl.store(OUT + tl.arange(0, BLK), val)


def main():
    nrows = MAXB * BS
    bt = torch.arange(MAXB, dtype=torch.int32, device=DEV)
    kv2 = torch.zeros(nrows, 8, dtype=torch.float32, device=DEV)
    kv2[:, 0] = torch.arange(nrows, dtype=torch.float32, device=DEV)   # 每行值 = 行号
    out = torch.full((8,), 123.0, dtype=torch.float32, device=DEV)
    offb = torch.full((8,), 999, dtype=torch.int32, device=DEV)

    def run(name, kern, jlo, shift=None):
        out.fill_(123.0)
        offb.fill_(999)
        if name.startswith("off"):
            args = [kv2, bt, offb]
        elif shift is not None:
            args = [kv2, bt, out, shift]
        else:
            args = [kv2, bt, out]
        kern[(1,)](*args, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=1, DD=8, S2C=S2, JLO=jlo,
                   BLK=8)
        torch.npu.synchronize()
        tgt = offb if name.startswith("off") else out
        print(f"  {name:22s} {tgt.cpu().tolist()}", flush=True)

    print("j =", list(range(-4, 4)), " 每行值=行号；gather 期望 [-7,-7,-7,-7,0,1,2,3]",
          flush=True)
    run("off _row_of", _k_off, -4)
    run("gather _row_of", _k_gather, -4)
    run("gather shift", _k_gather_shift, -4,
        torch.tensor(4, dtype=torch.int32, device=DEV))
    print("j =", list(range(-20, -12)), " gather 期望 全 -7", flush=True)
    run("off _row_of", _k_off, -20)
    run("gather _row_of", _k_gather, -20)
    run("gather shift", _k_gather_shift, -20,
        torch.tensor(20, dtype=torch.int32, device=DEV))
    print("CLAMP2DONE", flush=True)


if __name__ == "__main__":
    main()
