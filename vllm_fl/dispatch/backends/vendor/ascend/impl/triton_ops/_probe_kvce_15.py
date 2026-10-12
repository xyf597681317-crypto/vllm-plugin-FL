# -*- coding: utf-8 -*-
"""_probe_kvce_15.py —— 全值域扫描 `_to_e4m3_bits` vs torch fp8（隔离编码器）。

覆盖 |y| ∈ [0, 448]，指数从 -9 到 8，找出所有不一致点并给出 q / frac 细节。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_15.py
"""
import os
import struct
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
def _k_enc(Y, OUT, N, BLK: tl.constexpr):
    i = tl.arange(0, BLK)
    m = i < N
    y = tl.load(Y + i, mask=m, other=0.0)
    tl.store(OUT + i, _to_e4m3_bits(y).to(tl.uint8), mask=m)


def main():
    vals = []
    # 2 的幂附近 + 每个指数段的线性扫描
    for e in range(-9, 9):
        base = 2.0 ** e
        for k in range(1, 17):
            vals.append(base * (1.0 + k / 16.0))
            vals.append(base * (1.0 + (k - 0.5) / 16.0))
    vals = [v for v in vals if 0.0 < v <= 448.0]
    vals += [0.0, 1e-8, 1e-4, 448.0]
    y = torch.tensor(vals, dtype=torch.float32, device=DEV)
    n = y.numel()
    out = torch.zeros(n, dtype=torch.uint8, device=DEV)
    _k_enc[(1,)](y, out, n, BLK=n)
    torch.npu.synchronize()
    got = out.detach().cpu()
    ref = y.detach().cpu().to(torch.float8_e4m3fn).view(torch.uint8)
    bad = []
    for i in range(n):
        g, r = int(got[i]), int(ref[i])
        if g != r:
            bad.append((float(y[i]), g, r))
    print(f"共 {n} 个样本，不一致 {len(bad)} 个", flush=True)
    for v, g, r in bad[:25]:
        ex = ((struct.unpack("<I", struct.pack("<f", v))[0] >> 23) & 0xFF) - 127
        q = v * (2.0 ** (3 - ex))
        print(f"  y={v:.9e} ex={ex} q={q:.9e} frac={q - int(q):.9e}"
              f" kernel={g}(0x{g:02X}) torch={r}(0x{r:02X})", flush=True)
    print("KVCE15DONE", flush=True)


if __name__ == "__main__":
    main()
