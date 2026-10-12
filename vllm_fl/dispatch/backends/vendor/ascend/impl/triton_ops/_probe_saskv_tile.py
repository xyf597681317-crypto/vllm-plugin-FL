# -*- coding: utf-8 -*-
"""Probe9：定位 2D tile load `KV + off[:,None] + d[None,:]` 在 TND+PA 下的行为。

KV[row,0,d] = row*2 + (d%2)（bit-exact），因此读出值可直接反推行号。
对比三种写法：
  A: off[:,None] + d[None,:]      （当前实现）
  B: off[:,None] + d[None,:] 但 d 用 tl.arange（同 A，验证是否 grid 轴 d 的影响）
  C: 逐行 1D load（等价语义）
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_tile.py
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
NROWS = 2 * MB * BS          # 8 页 * 16 行
N2 = 1
D = 512
NVEC = 8


@triton.jit
def _k_A(KV, BT, OUT, BIDX, BASE, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
         N2C: tl.constexpr, DC: tl.constexpr, S2C: tl.constexpr, NLANE: tl.constexpr,
         BLK: tl.constexpr):
    lane = tl.arange(0, NLANE)
    d = tl.arange(0, BLK)
    kvrow = BASE + lane
    off = _kv_elem_off(KV, BT, BIDX, kvrow, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C, True)
    t = tl.load(KV + off[:, None] + d[None, :])
    tl.store(OUT + lane[:, None] * BLK + d[None, :], t)


@triton.jit
def _k_C(KV, BT, OUT, BIDX, BASE, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
         N2C: tl.constexpr, DC: tl.constexpr, S2C: tl.constexpr, NLANE: tl.constexpr,
         BLK: tl.constexpr):
    """逐行 1D load（用标量行号），等价语义但走完全不同的 lowering。"""
    d = tl.arange(0, BLK)
    for i in tl.static_range(NLANE):
        kvrow = BASE + i
        off = _kv_elem_off(KV, BT, BIDX, kvrow, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC,
                           S2C, True)
        v = tl.load(KV + off + d)
        tl.store(OUT + i * BLK + d, v)


def main():
    kv = torch.zeros(NROWS, N2, D, dtype=torch.float16, device=DEV)
    rowid = torch.arange(NROWS, dtype=torch.float32, device=DEV)[:, None] * 2.0
    dpar = (torch.arange(D, dtype=torch.float32, device=DEV)[None, :] % 2)
    kv[:, 0, :] = (rowid + dpar).to(torch.float16)
    bt = torch.tensor([[1, 2, 3, 0], [7, 5, 6, 4]], dtype=torch.int32, device=DEV)
    out = torch.full((NVEC, D), -1.0, dtype=torch.float16, device=DEV)

    def expect_row(b, krow):
        if krow < 0 or krow >= MB * BS:
            return None
        page = int(bt[b, krow // BS].item())
        return page * BS + krow % BS

    for b in (0, 1):
        for base in (0, 5, 59, -4):
            for name, kern in (("A: off[:,None]+d[None,:]", _k_A),
                               ("C: 逐行 1D", _k_C)):
                out.fill_(-1.0)
                kern[(1,)](kv, bt, out, b, base, BLOCK_SIZE=BS, MAX_BLOCKS=MB, N2C=N2,
                           DC=D, S2C=MB * BS, NLANE=NVEC, BLK=D)
                torch.npu.synchronize()
                o = out.detach().cpu()
                errs = []
                for i in range(NVEC):
                    er = expect_row(b, base + i)
                    if er is None:
                        continue
                    e = (o[i].float() - (2.0 * er + (torch.arange(D) % 2).float()))
                    errs.append((i, er, e.abs().max().item()))
                worst = max((x[2] for x in errs), default=-1.0)
                detail = " ".join(f"L{i}(row{er})={e:.1e}" for i, er, e in errs)
                print(f"  b={b} base={base:4d} {name:24s} 最差={worst:.3e}  {detail}",
                      flush=True)
    print("TILEDONE", flush=True)


if __name__ == "__main__":
    main()
