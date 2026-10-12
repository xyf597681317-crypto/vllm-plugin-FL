# -*- coding: utf-8 -*-
"""compressor_coff2 重构后的编译诊断：捕获 ub overflow / 编译异常原文。"""
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

torch.npu.set_device(DEV)


def try_case(tag, S, H, D, R=4, RD=64, dtype=torch.bfloat16):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(H)
    Sr = S // R
    x = (torch.randn(1, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    rs = torch.randn(1, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(1, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    try:
        out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD,
                            cmp_ratio=R, coff=2, rotary_mode=2)
        torch.npu.synchronize()
        print(f"  [{tag}] OK  max|out|={float(out.float().abs().max()):.4f}", flush=True)
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        import re
        m = re.search(r"requires\s+([0-9]+)\s*bits", msg)
        print(f"  [{tag}] EXC bits={m.group(1) if m else None} :: {msg[:600]}", flush=True)


if __name__ == "__main__":
    print(f"[cub] dev={DEV}", flush=True)
    try_case("D=128 H=64 S=24", 24, 64, 128)
    try_case("D=128 H=64 S=24 fp16", 24, 64, 128, dtype=torch.float16)
    try_case("D=512 H=256 S=24 (生产)", 24, 256, 512)
    print("[cub] done", flush=True)
