# -*- coding: utf-8 -*-
"""_probe_kvce_13.py —— 直接测 `_to_e4m3_bits` 的平局取整（隔离量化段）。

构造精确平局：令 y = m * 2^(-3+e)，其中 q = 11.5 / 12.5 / 9.5 / 10.5 …
期望（RNE）：11.5→12(0x7C)  12.5→12(0x7C)  9.5→10(0x6A)  10.5→10(0x6A)
对照 torch：float32 → float8_e4m3fn 的逐字节
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_13.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton_ops.kv_compress_epilog import _to_e4m3_bits  # noqa: E402

print("device =", DEV, flush=True)


@triton.jit
def _k_enc(Y, OUT, N: tl.constexpr, BLK: tl.constexpr):
    i = tl.arange(0, BLK)
    m = i < N
    y = tl.load(Y + i, mask=m, other=0.0)
    tl.store(OUT + i, _to_e4m3_bits(y).to(tl.uint8), mask=m)


def main():
    # 期望字节（手工）：y = q * 2^(e-3)，取 e=3 ⇒ y = q
    # q 为 [8,16) 内的平局点
    qs = [11.5, 12.5, 9.5, 10.5, 13.5, 7.5, 15.5, 8.5, 11.4, 11.6, 0.5, 1.5]
    y = torch.tensor(qs, dtype=torch.float32, device=DEV)
    out = torch.zeros(len(qs), dtype=torch.uint8, device=DEV)
    _k_enc[(1,)](y, out, N=len(qs), BLK=len(qs))
    torch.npu.synchronize()
    got = out.detach().cpu()
    ref = y.detach().cpu().to(torch.float8_e4m3fn).view(torch.uint8)
    print(f"{'q':>8} {'kernel':>8} {'torch':>8} {'一致':>6}", flush=True)
    nbad = 0
    for i, q in enumerate(qs):
        g, r = int(got[i]), int(ref[i])
        same = "OK" if g == r else "BAD"
        if g != r:
            nbad += 1
        print(f"{q:8.4f} {g:8d} {r:8d} {same:>6}   (0x{g:02X} / 0x{r:02X})", flush=True)
    print("不一致个数 =", nbad, flush=True)
    print("KVCE13DONE", flush=True)


if __name__ == "__main__":
    main()
