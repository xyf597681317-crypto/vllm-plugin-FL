# -*- coding: utf-8 -*-
"""Probe6：验证"整窗右移使下标恒非负"能否修好 TND+PA（只改寻址，不改数学）。

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_shift.py
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
NR = MAXB * BS


@triton.jit
def _k_old(KV, BT, OUT, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
           N2: tl.constexpr, DD: tl.constexpr, S2C: tl.constexpr, JLO: tl.constexpr,
           BLK: tl.constexpr):
    j = JLO + tl.arange(0, BLK)
    vld = j >= 0
    off = _row_of(KV, BT, 0, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2C, True, vld)
    tl.store(OUT + tl.arange(0, BLK), tl.load(KV + off, mask=vld, other=-7.0))


@triton.jit
def _k_new(KV, BT, OUT, SHIFT, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
           N2: tl.constexpr, DD: tl.constexpr, S2C: tl.constexpr, JLO: tl.constexpr,
           BLK: tl.constexpr):
    lane = tl.arange(0, BLK)
    j = (JLO + SHIFT) + lane                 # 恒非负
    vld = j < SHIFT                          # 原 j = j - SHIFT < 0 ⇒ 无效
    off = _row_of(KV, BT, 0, j, 0, BLOCK_SIZE, MAX_BLOCKS, N2, DD, S2C, True, vld)
    tl.store(OUT + tl.arange(0, BLK), tl.load(KV + off, mask=vld, other=-7.0))


def main():
    bt = torch.arange(MAXB, dtype=torch.int32, device=DEV)
    kv = torch.zeros(NR, 8, dtype=torch.float32, device=DEV)
    kv[:, 0] = torch.arange(NR, dtype=torch.float32, device=DEV)
    out = torch.full((8,), 123.0, dtype=torch.float32, device=DEV)

    def run(tag, kern, jlo, shift=None):
        out.fill_(123.0)
        args = [kv, bt, out] if shift is None else [kv, bt, out, shift]
        kern[(1,)](*args, BLOCK_SIZE=BS, MAX_BLOCKS=MAXB, N2=1, DD=8, S2C=NR, JLO=jlo,
                   BLK=8)
        torch.npu.synchronize()
        print(f"  {tag:18s} {out.cpu().tolist()}", flush=True)

    for jlo, exp in ((-4, "[-7,-7,-7,-7,0,1,2,3]"), (-20, "全 -7"),
                     (12, "[12,13,14,15,16,17,18,19]")):
        print(f"JLO={jlo} 期望 {exp}", flush=True)
        run("old(clamp)", _k_old, jlo)
        sh = -jlo if jlo < 0 else 0
        run("new(shift)", _k_new, jlo,
            torch.tensor(sh, dtype=torch.int32, device=DEV))
    print("SHIFTDONE", flush=True)


if __name__ == "__main__":
    main()
