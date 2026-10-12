# -*- coding: utf-8 -*-
"""Probe8：把 TND+PA 的最小 2D gather 拆到最小：dump page / row / off，并与主机期望比。

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_2d2.py
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
N2 = 1
D = 512
NVEC = 8


@triton.jit
def _k_diag(KV, BT, OUTP, OUTR, OUTOFF, BIDX, BASE, BLOCK_SIZE: tl.constexpr,
            MAX_BLOCKS: tl.constexpr, N2C: tl.constexpr, DC: tl.constexpr,
            S2C: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    kvrow = BASE + lane
    bidx = tl.minimum(tl.maximum(kvrow // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
    page = tl.maximum(tl.load(BT + BIDX * MAX_BLOCKS + bidx), 0)
    row = page * BLOCK_SIZE + (kvrow % BLOCK_SIZE)
    off2 = _kv_elem_off(KV, BT, BIDX, kvrow, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC, S2C, True)
    tl.store(OUTP + lane, page)
    tl.store(OUTR + lane, row)
    tl.store(OUTOFF + lane, off2)


def main():
    torch.manual_seed(81)
    kv = torch.zeros(MB * BS * 2, N2, D, dtype=torch.float16, device=DEV)
    # 每行填一个可识别的模式：KV[row,0,d] = row + d/1000
    r = torch.arange(MB * BS * 2, dtype=torch.float32, device=DEV)[:, None]
    dcol = torch.arange(D, dtype=torch.float32, device=DEV)[None, :] / 1000.0
    kv[:, 0, :] = (r + dcol).to(torch.float16)
    bt = torch.tensor([[1, 2, 3, 0], [7, 5, 6, 4]], dtype=torch.int32, device=DEV)
    outp = torch.full((NVEC,), -1, dtype=torch.int32, device=DEV)
    outr = torch.full((NVEC,), -1, dtype=torch.int32, device=DEV)
    outoff = torch.full((NVEC,), -1, dtype=torch.int32, device=DEV)

    for b in (0, 1):
        for base in (0, 5, 59):
            outp.fill_(-1)
            outr.fill_(-1)
            outoff.fill_(-1)
            _k_diag[(1,)](kv, bt, outp, outr, outoff, b, base, BLOCK_SIZE=BS,
                          MAX_BLOCKS=MB, N2C=N2, DC=D, S2C=MB * BS, NLANE=NVEC)
            torch.npu.synchronize()
            rows = [base + i for i in range(NVEC)]
            exp_page = [int(bt[b, min(r_ // BS, MB - 1)].item()) if r_ < MB * BS else -1
                        for r_ in rows]
            exp_row = [p * BS + r_ % BS if r_ < MB * BS else -1
                       for p, r_ in zip(exp_page, rows)]
            exp_off = [rr * D if rr >= 0 else -1 for rr in exp_row]
            print(f"b={b} base={base}", flush=True)
            print(f"  kvrow  期望={rows}", flush=True)
            print(f"  page   got ={outp.cpu().tolist()}", flush=True)
            print(f"  page   exp ={exp_page}", flush=True)
            print(f"  row    got ={outr.cpu().tolist()}", flush=True)
            print(f"  row    exp ={exp_row}", flush=True)
            print(f"  off    got ={outoff.cpu().tolist()}", flush=True)
            print(f"  off    exp ={exp_off}", flush=True)
    print("DIAG2DONE", flush=True)


if __name__ == "__main__":
    main()
