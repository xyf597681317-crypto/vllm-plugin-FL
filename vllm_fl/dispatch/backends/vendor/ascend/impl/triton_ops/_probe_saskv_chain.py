# -*- coding: utf-8 -*-
"""Probe15：把窗口→行号→块表→地址 全链路逐 lane dump 出来。

跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_chain.py
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
NLANE = 48
BASE = -68
DC = 512


@triton.jit
def _chain(KV, BT, D_SHIFT, D_ROW0, D_CNT, D_RW, D_ROW, D_BIDX, D_PAGE, D_OFF,
           BASEC: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
           DC: tl.constexpr, NLANE: tl.constexpr):
    lane = tl.arange(0, NLANE)
    shift_w = tl.where(BASEC < 0, -BASEC, 0)
    row0_o = tl.maximum(BASEC, 0)
    cnt = tl.minimum(NLANE, MAX_BLOCKS * BLOCK_SIZE - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    row = row0_o + rw
    bidx = tl.minimum(tl.maximum(row // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
    page = tl.maximum(tl.load(BT + bidx), 0)
    prow = page * BLOCK_SIZE + (row % BLOCK_SIZE)
    off = (prow * 1 + 0) * DC
    off2 = _kv_elem_off(KV, BT, 0, row, 0, BLOCK_SIZE, MAX_BLOCKS, 1, DC,
                        MAX_BLOCKS * BLOCK_SIZE, True)
    tl.store(D_SHIFT + lane, shift_w + lane * 0)
    tl.store(D_ROW0 + lane, row0_o + lane * 0)
    tl.store(D_CNT + lane, cnt + lane * 0)
    tl.store(D_RW + lane, rw)
    tl.store(D_ROW + lane, row)
    tl.store(D_BIDX + lane, bidx)
    tl.store(D_PAGE + lane, page)
    tl.store(D_OFF + lane, off)
    tl.store(D_OFF + NLANE + lane, off2)


def main():
    kv = torch.zeros(MB * BS, 1, DC, dtype=torch.float16, device=DEV)
    bt = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device=DEV)
    bufs = {n: torch.full((NLANE,), -9, dtype=torch.int32, device=DEV)
            for n in ("shift", "row0", "cnt", "rw", "row", "bidx", "page")}
    off = torch.full((2 * NLANE,), -9, dtype=torch.int32, device=DEV)
    _chain[(1,)](kv, bt, bufs["shift"], bufs["row0"], bufs["cnt"], bufs["rw"],
                 bufs["row"], bufs["bidx"], bufs["page"], off,
                 BASEC=BASE, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=DC, NLANE=NLANE)
    torch.npu.synchronize()
    r = {k: v.detach().cpu().tolist() for k, v in bufs.items()}
    o = off.detach().cpu().tolist()
    print(f"BASE={BASE}  NLANE={NLANE}", flush=True)
    for k in ("shift", "row0", "cnt"):
        print(f"  {k:6s}[0]={r[k][0]}", flush=True)
    print(f"  lane      ={list(range(NLANE))[:14]}", flush=True)
    print(f"  rw        ={r['rw'][:14]}", flush=True)
    print(f"  row       ={r['row'][:14]}", flush=True)
    print(f"  bidx      ={r['bidx'][:14]}", flush=True)
    print(f"  page      ={r['page'][:14]}", flush=True)
    exp_page = [int(bt[min(x // BS, MB - 1)].item()) for x in r["row"]]
    print(f"  page exp  ={exp_page[:14]}", flush=True)
    print(f"  off 手算  ={o[:14]}", flush=True)
    print(f"  off func  ={o[NLANE:NLANE + 14]}", flush=True)
    print(f"  off exp   ={[(int(bt[min(x // BS, MB - 1)].item()) * BS + x % BS) * DC for x in r['row']][:14]}",
          flush=True)
    print("CHAINDONE", flush=True)


if __name__ == "__main__":
    main()
