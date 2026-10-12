# -*- coding: utf-8 -*-
"""compressor_coff2 的 507015 定位：把两次启动**分开跑**（每个模式一个进程）。

用法（每个模式必须单独起进程，因为 507015 会让设备/进程状态不可信）：
    MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_split.py wrap
    MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_split.py main
    MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_split.py first

  wrap  —— 直接调 host 包装 compressor()（= 两次启动）
  main  —— 只跑 FIRST=False 那次启动（方案表 = (0,1)..(0,Sr-1)）
  first —— 只跑 FIRST=True 那次启动（BIDX=[0]）
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
from triton_ops.compressor_coff2 import _MIN_SCORE, _compressor_coff2_kernel  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "wrap"
S, H, D, R, RD = 24, 64, 128, 4, 64


def main():
    torch.npu.set_device(DEV)
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(H)
    Sr = S // R
    x = (torch.randn(1, S, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    rs = torch.randn(1, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(1, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    print(f"[split] dev={DEV} mode={MODE} S={S} H={H} D={D}", flush=True)

    if MODE == "wrap":
        out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD,
                            cmp_ratio=R, coff=2, rotary_mode=2)
        torch.npu.synchronize()
        print(f"[split] wrap OK max|out|={float(out.float().abs().max()):.4f}", flush=True)
        return

    dev = torch.device(DEV)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    BLK_D = 1 << (D - 1).bit_length()
    BLK_H = max(16, min(128, 8192 // BLK_D))
    M = max(16, 1 << (2 * R - 1).bit_length())
    out = torch.zeros(Sr, D, dtype=torch.bfloat16, device=DEV)
    nb = Sr - 1
    args = (x.contiguous(), wkv.contiguous(), wgate.contiguous(), ape, nw,
            rs.reshape(-1, RD).contiguous(), rc.reshape(-1, RD).contiguous(),
            out, i32(0), out, i32(0), i32(0), i32(S), i32(0), i32(0),
            torch.zeros(nb, dtype=torch.int32, device=dev),
            torch.arange(1, Sr, dtype=torch.int32, device=dev), 16, 0, 1)
    cm = dict(H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=RD // 2, M=M,
              EPS=1e-6, MIN_SCORE=_MIN_SCORE, HALF_MODE=False, HAS_STATE=False,
              CACHE_CYCLE=False)
    if MODE == "main":
        _compressor_coff2_kernel[(nb,)](*args, **cm, FIRST=False)
        torch.npu.synchronize()
        print(f"[split] main OK  out={out.detach().cpu().float().abs().max():.4f}", flush=True)
    elif MODE == "first":
        args_f = args[:15] + (i32(0), args[16]) + args[17:]
        _compressor_coff2_kernel[(1,)](*args_f, **cm, FIRST=True)
        torch.npu.synchronize()
        print(f"[split] first OK out={out.detach().cpu().float().abs().max():.4f}", flush=True)
    else:
        print(f"[split] unknown mode {MODE}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001
        print(f"[split] {MODE} EXC :: {str(ex)[:300]}", flush=True)
        raise
