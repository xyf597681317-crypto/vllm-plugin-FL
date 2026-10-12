# -*- coding: utf-8 -*-
"""Probe7：最小 2D KV gather 复现（严格照抄 TND+PA 用例的参数）。

- 用真实的块表 + 真实 KV cache + 真实行号布局
- 行号 j 由 base_o + lane 给出，覆盖 base_o<0（滑窗左端越界）与 base_o>=0 两种
- 与 CPU 参考逐元素对比，打印前 16 个 lane 的 (row, off, got, exp)

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_2d.py
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
MB = 4          # 每 batch 4 个块（kvolens=64）
B = 2
N2 = 1
D = 512
NVEC = 8        # 一次 8 个 lane（BLK_K=8）


@triton.jit
def _k_gather2d(KV, BT, OUT, BIDX, BASE, BLOCK_SIZE: tl.constexpr,
                MAX_BLOCKS: tl.constexpr, N2C: tl.constexpr, DC: tl.constexpr,
                S2C: tl.constexpr, NLANE: tl.constexpr, BLK: tl.constexpr):
    d = tl.arange(0, BLK)
    lane = tl.arange(0, NLANE)
    kvrow = BASE + lane
    off = _kv_elem_off(KV, BT, BIDX, kvrow, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C, True)
    row2 = tl.load(KV + off[:, None] + d[None, :])
    tl.store(OUT + lane[:, None] * BLK + d[None, :], row2)


@triton.jit
def _k_gather2d_nopa(KV, BT, OUT, BIDX, BASE, S2C: tl.constexpr, NLANE: tl.constexpr,
                     BLK: tl.constexpr):
    """对照：完全不做块表，off = (b*S2 + kvrow)*D（纯行偏移）。"""
    d = tl.arange(0, BLK)
    lane = tl.arange(0, NLANE)
    kvrow = BASE + lane
    off = (BIDX * S2C + kvrow) * 512
    row2 = tl.load(KV + off[:, None] + d[None, :])
    tl.store(OUT + lane[:, None] * BLK + d[None, :], row2)


def main():
    torch.manual_seed(81)
    kv = (torch.randn(B * MB * BS, N2, D, device=DEV) * 0.6).to(torch.float16)
    bt = torch.zeros(B, MB, dtype=torch.int32, device=DEV)
    for b in range(B):
        bt[b] = (torch.randperm(MB, device=DEV) + b * MB).to(torch.int32)
    out = torch.full((NVEC, D), 777.0, dtype=torch.float16, device=DEV)
    KD = kv.detach().cpu().double()
    BTD = bt.detach().cpu()

    def dense(b):
        dd = torch.zeros(MB * BS, N2, D, dtype=torch.float64)
        for lb in range(MB):
            dd[lb * BS:(lb + 1) * BS] = KD[int(BTD[b, lb])]
        return dd

    print("块表:", BTD.tolist(), flush=True)
    for base in (-68, -4, 0, 5, 59):
        for b in (0, 1):
            out.fill_(777.0)
            _k_gather2d[(1,)](kv, bt, out, b, base, BLOCK_SIZE=BS, MAX_BLOCKS=MB,
                              N2C=N2, DC=D, S2C=MB * BS, NLANE=NVEC, BLK=D)
            torch.npu.synchronize()
            o = out.detach().cpu().double()
            dd = dense(b)
            # 期望：row = base+lane（可能越界 → 直接对比 0..MB*BS-1 范围内的行）
            bad = []
            for ln in range(NVEC):
                row = base + ln
                if 0 <= row < MB * BS:
                    e = (o[ln] - dd[row, 0]).abs().max().item()
                    bad.append((ln, row, e))
            worst = max((x[2] for x in bad), default=-1.0)
            detail = " ".join(f"lane{ln}(row{row})={e:.2e}" for ln, row, e in bad)
            print(f"  base={base:4d} b={b}  最差={worst:.3e}  {detail}", flush=True)
    # 对照：不做块表
    print("---- 对照：不做块表（等价于 BSND 行偏移） ----", flush=True)
    for base in (-68, 0):
        out.fill_(777.0)
        _k_gather2d_nopa[(1,)](kv, bt, out, 0, base, S2C=MB * BS, NLANE=NVEC, BLK=D)
        torch.npu.synchronize()
        o = out.detach().cpu().double()
        bad = []
        for ln in range(NVEC):
            row = base + ln
            if 0 <= row < MB * BS:
                e = (o[ln] - KD[row, 0]).abs().max().item()
                bad.append((ln, row, e))
        print(f"  base={base:4d} 最差={max((x[2] for x in bad), default=-1):.3e} "
              + " ".join(f"lane{ln}(row{row})={e:.2e}" for ln, row, e in bad), flush=True)
    print("GATHER2DDONE", flush=True)


if __name__ == "__main__":
    main()
