# -*- coding: utf-8 -*-
"""流式两段（④）崩溃定位：逐步打 marker，最后一个 marker 即崩溃点。

用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_stream.py
"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("MOE_DEV", "npu:0")
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from triton_ops.compressor import compressor  # noqa: E402


def mk(tag):
    print(f"[stream] MARK {tag}", flush=True)


def main():
    torch.npu.set_device(DEV)
    R, D, RD, H = 4, 128, 64, 64
    S1, S2 = 18, 14
    S = S1 + S2
    B = 1
    torch.manual_seed(3)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    Sr = (S + R - 1) // R
    rs = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    mk("tensors ready")

    nblk = (S + 15) // 16
    stride0 = 16 * 4 * D
    sc = torch.zeros(nblk, 16, 4 * D, dtype=torch.float32, device=DEV)
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, nblk)
    sp1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    su1 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    mk("call1 start")
    o1, sc1 = compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                         rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(),
                         sc, bt, seqused=su1, start_pos=sp1,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=1, state_cache_stride_dim0=stride0, block_size=16)
    mk("call1 launched")
    torch.npu.synchronize()
    mk(f"call1 synced  o1={tuple(o1.shape)} max|o1|={float(o1.float().abs().max())}")
    print(f"[stream] sc1 非零元素={int((sc1 != 0).sum())}/{sc1.numel()}", flush=True)

    sp2 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    su2 = torch.tensor([S2], dtype=torch.int32, device=DEV)
    n2 = (S1 + S2) // R - S1 // R
    mk("call2 start")
    o2, sc2 = compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                         rs[:, S1 // R:S1 // R + n2].contiguous(),
                         rc[:, S1 // R:S1 // R + n2].contiguous(),
                         sc1, bt, seqused=su2, start_pos=sp2,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=1, state_cache_stride_dim0=stride0, block_size=16)
    mk("call2 launched")
    torch.npu.synchronize()
    mk(f"call2 synced  o2={tuple(o2.shape)} max|o2|={float(o2.float().abs().max())}")
    print("[stream] DONE（两次调用都没崩）", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        print(f"[stream] EXC :: {str(ex)[:400]}", flush=True)
        raise
