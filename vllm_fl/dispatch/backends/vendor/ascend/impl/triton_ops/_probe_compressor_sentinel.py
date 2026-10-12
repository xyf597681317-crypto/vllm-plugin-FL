# -*- coding: utf-8 -*-
"""compressor(coff=2) 第 0 行损坏的**签名判定**：双 sentinel 法（直接调 kernel，不改算子文件）。

同一输入跑两遍，输出缓冲分别预填 F1=-7.5 / F2=+7.5（bf16 可精确表示）：
  某元素两遍仍分别等于各自填充值 ⇒ **store 完全没写（丢写）**
  两遍值相同但不等于填充   ⇒ 算了但值错
  两遍值不同               ⇒ 非确定性
用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_sentinel.py
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
from triton_ops.compressor_coff2 import _MIN_SCORE, _compressor_coff2_kernel  # noqa: E402

F1, F2 = -7.5, 7.5


def case(name, S, H, D, R=4, RD=64, dtype=torch.bfloat16, seed=0):
    torch.npu.set_device(DEV)          # ⚠️ 必须显式设置，否则 kernel 发到当前设备(默认0)而张量在 npu:N
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    Sr = S // R
    x = (torch.randn(1, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = torch.zeros(2 * D, H, dtype=torch.float32, device=DEV).to(dtype)
    ape = torch.zeros(R, 2 * D, dtype=torch.float32, device=DEV)
    nw = torch.ones(D, dtype=torch.float32, device=DEV)
    rs = torch.zeros(1, Sr, RD, dtype=torch.float32, device=DEV)
    rc = torch.ones(1, Sr, RD, dtype=torch.float32, device=DEV)
    dev = torch.device(DEV)
    i32 = lambda v: torch.tensor([v], dtype=torch.int32, device=dev)  # noqa: E731
    x_cache = x.contiguous()
    wkv_c, wgate_c = wkv.contiguous(), wgate.contiguous()
    BLK_D = 1 << (D - 1).bit_length()
    BLK_H = max(16, min(128, 8192 // BLK_D))
    M = max(16, 1 << (2 * R - 1).bit_length())
    grid = (1, Sr)
    outs = []
    for fill in (F1, F2):
        out = torch.full((Sr, D), fill, dtype=dtype, device=DEV)
        # 与 host 完全一致：常规启动（方案表 = (b,ct)，跳过 c0==0 的首块）+ 首块专用启动
        nb = Sr - 1
        bix = torch.zeros(max(nb, 1), dtype=torch.int32, device=dev)
        ctx = torch.arange(1, Sr, dtype=torch.int32, device=dev) if nb > 0 \
            else torch.zeros(1, dtype=torch.int32, device=dev)
        args = (x_cache, wkv_c, wgate_c, ape, nw, rs.reshape(-1, RD).contiguous(),
                rc.reshape(-1, RD).contiguous(), out, i32(0), out,
                i32(0), i32(0), i32(S), i32(0), i32(0), bix, ctx, 16, 0, 1)
        cm = dict(H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=RD // 2, M=M,
                  EPS=1e-6, MIN_SCORE=_MIN_SCORE, HALF_MODE=True, HAS_STATE=False,
                  CACHE_CYCLE=False)
        if nb > 0:
            _compressor_coff2_kernel[(nb,)](*args, **cm, FIRST=False)
        args_f = args[:15] + (i32(0), ctx) + args[17:]
        _compressor_coff2_kernel[(1,)](*args_f, **cm, FIRST=True)
        torch.npu.synchronize()
        outs.append(out.detach().cpu().float())
    a1, a2 = outs
    dropped = (a1 == F1) & (a2 == F2)
    print(f"\n=== {name} (S={S} Sr={Sr} D={D} H={H} M={M} BLK_H={BLK_H}) ===", flush=True)
    for r in range(Sr):
        d_both = int(dropped[r].sum())
        d1 = int((a1[r] == F1).sum())
        d2 = int((a2[r] == F2).sum())
        z1 = int((a1[r] == 0).sum())
        z2 = int((a2[r] == 0).sum())
        nd = int((a1[r] != a2[r]).sum())
        print(f"  row={r}: 未写(run1/run2)={d1}/{d2} (both={d_both}) 恰为0(run1/run2)={z1}/{z2} "
              f"两遍不同元素={nd}/{D} "
              f"run1[0:3]={[round(v, 4) for v in a1[r, :3].tolist()]} "
              f"run2[0:3]={[round(v, 4) for v in a2[r, :3].tolist()]}", flush=True)


if __name__ == "__main__":
    print(f"[sentinel] dev={DEV} F1={F1} F2={F2}", flush=True)
    case("S=8 (2 chunks)", 8, 64, 128)
    case("S=16 (4 chunks)", 16, 64, 128)
    case("S=24 (6 chunks) 生产 D=512 H=256", 24, 256, 512)
    print("\n[sentinel] done", flush=True)
