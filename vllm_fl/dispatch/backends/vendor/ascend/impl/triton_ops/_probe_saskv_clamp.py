# -*- coding: utf-8 -*-
"""Probe5：最小复现 —— TND+PA 下 j<0 的 clamp 是否生效。

假设：`tl.maximum(j, 0)` 在 Ascend 后端上对 `j // BLOCK_SIZE` / `j % BLOCK_SIZE`
的传播没有生效，导致负 j 仍然参与取模/除法 → 读到别的行。

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_clamp.py
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
N2D = 1
D = 512


@triton.jit
def _k_off(BT, B, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, N2: tl.constexpr,
           DD: tl.constexpr, S2: tl.constexpr, JLO: tl.constexpr, BLK: tl.constexpr):
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    off = _row_of(BT, BT, B, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2, True, vld)
    tl.store(BT + tl.arange(0, BLK), off, mask=tl.arange(0, BLK) < BLK)


@triton.jit
def _k_off_v2(BT, B, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, N2: tl.constexpr,
              DD: tl.constexpr, S2: tl.constexpr, JLO: tl.constexpr, BLK: tl.constexpr):
    """变体：先 tl.where 夹到 0，再算行/列。"""
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    js = tl.where(j >= 0, j, 0)
    bidx = tl.minimum(js // BLOCK_SIZE, MAX_BLOCKS - 1)
    page = tl.load(BT + B * MAX_BLOCKS + bidx, mask=vld, other=0)
    page = tl.maximum(page, 0)
    kvrow = page * BLOCK_SIZE + (js % BLOCK_SIZE)
    off = (kvrow * N2 + 0) * DD
    tl.store(BT + tl.arange(0, BLK), off, mask=tl.arange(0, BLK) < BLK)


@triton.jit
def _k_gather(KV, BT, OUT, B, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
              N2: tl.constexpr, DD: tl.constexpr, S2: tl.constexpr, JLO: tl.constexpr,
              BLK: tl.constexpr):
    """变体：直接读 8 行 kv，看 clamp 后的 4 个负 lane 是否读到同一行。"""
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    off = _row_of(KV, BT, B, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2, True, vld)
    val = tl.load(KV + off, mask=vld, other=-7.0)
    tl.store(OUT + tl.arange(0, BLK), val)


def main():
    torch.manual_seed(0)
    bt = torch.arange(MAXB, dtype=torch.int32, device=DEV)
    kv = (torch.arange(MAXB * BS * N2D * 8, dtype=torch.float32, device=DEV)
          .reshape(MAXB * BS, N2D, 8))
    offbuf = torch.zeros(8, dtype=torch.int32, device=DEV)
    out = torch.zeros(8, dtype=torch.float32, device=DEV)

    print("=== A: _row_of offsets, j in [-4, 3], BLOCK_SIZE=16, bt=[0,1,2,3] ===", flush=True)
    _k_off[(1,)](offbuf, 0, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=N2D, DD=D, S2=BS * MAXB,
                 JLO=-4, BLK=8)
    torch.npu.synchronize()
    print("  j      =", list(range(-4, 4)), flush=True)
    print("  offsets=", offbuf.cpu().tolist(), flush=True)
    print("  期望（clamp 生效）: [0, 0, 0, 0, 0, 512, 1024, 1536]", flush=True)

    offbuf.zero_()
    print("=== B: 变体 where-clamp ===", flush=True)
    _k_off_v2[(1,)](offbuf, 0, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=N2D, DD=D, S2=BS * MAXB,
                    JLO=-4, BLK=8)
    torch.npu.synchronize()
    print("  offsets=", offbuf.cpu().tolist(), flush=True)

    print("=== C: gather 8 行（值 = 行号），j in [-4,3] ===", flush=True)
    # 让每行值 = 行号：kv[row,0,0]=row
    kv2 = torch.zeros(MAXB * BS, N2D, 8, dtype=torch.float32, device=DEV)
    kv2[:, 0, 0] = torch.arange(MAXB * BS, dtype=torch.float32, device=DEV)
    _k_gather[(1,)](kv2, bt, out, 0, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=N2D, DD=8,
                    S2=BS * MAXB, JLO=-4, BLK=8)
    torch.npu.synchronize()
    print("  values =", out.cpu().tolist(), flush=True)
    print("  期望（clamp 生效）: [-7, -7, -7, -7, 0, 1, 2, 3]", flush=True)

    print("=== D: 同上但 j in [-20, -13] （负数更多、含跨块） ===", flush=True)
    _k_gather[(1,)](kv2, bt, out, 0, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=N2D, DD=8,
                    S2=BS * MAXB, JLO=-20, BLK=8)
    torch.npu.synchronize()
    print("  values =", out.cpu().tolist(), flush=True)
    print("  期望: 全 -7", flush=True)
    print("CLAMPDONE", flush=True)


if __name__ == "__main__":
    main()
