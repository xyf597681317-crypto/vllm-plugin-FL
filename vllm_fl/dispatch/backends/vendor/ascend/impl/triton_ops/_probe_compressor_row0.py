# -*- coding: utf-8 -*-
"""compressor(coff=2) 最小定位：identity-RoPE + uniform softmax 下，逐行对「解析候选」比。

构造：wgate=0、ape=0 ⇒ score≡0 ⇒ softmax 在 2r 窗口上均匀；
      cos=1、sin=0、rotary_mode=1 ⇒ RoPE 为恒等；
      norm_weight=ones ⇒ y = rmsnorm(o)。
于是第 c 行输出可由「哪几行的 kv（哪半通道）」解析算出：
  A: 当前 chunk 的 r 行、通道 [D,2D)             （= 参考 ref_A/ref_B 的语义）
  B: 当前 chunk 的 r 行、通道 [0,D)
  C: 前一个 chunk 的 r 行、通道 [0,D)
  D: 2r 行（前 chunk 通道[0,D) + 当前 chunk 通道[D,2D)）；c=0 时前块为空
  E: 下一个 chunk 的 r 行
第一个与 kernel 逐位吻合的候选就是 kernel 的真实语义。

用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_row0.py
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
from triton_ops.compressor import compressor as tri_compressor  # noqa: E402

B, S, H, D, R, RD = 1, 16, 64, 128, 4, 64
EPS = 1e-6


def rms(y):
    return y / torch.sqrt((y * y).sum() / y.numel() + EPS)


def run_S(S):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wgate = torch.zeros(2 * D, H, dtype=torch.float32, device=DEV).to(torch.bfloat16)
    ape = torch.zeros(R, 2 * D, dtype=torch.float32, device=DEV)
    nw = torch.ones(D, dtype=torch.float32, device=DEV)
    Sr = S // R
    rs = torch.zeros(B, Sr, RD, dtype=torch.float32, device=DEV)
    rc = torch.ones(B, Sr, RD, dtype=torch.float32, device=DEV)
    out, _ = tri_compressor(x, wkv, wgate, ape, nw, rs, rc,
                            rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=1)
    torch.npu.synchronize()
    o_k = out.detach().cpu().double().reshape(-1, D)
    xd = x.detach().cpu().double()
    wk = wkv.detach().cpu().double()

    def kv_rows(rows, half):
        if not rows:
            return torch.zeros(D, dtype=torch.float64)
        return sum(xd[0, p] @ wk[half * D:(half + 1) * D].t() for p in rows) / len(rows)

    print(f"\n===== S={S} (Sr={Sr}) identity-rope, uniform softmax =====")
    for c in range(Sr):
        cur = list(range(c * R, (c + 1) * R))
        prev = list(range((c - 1) * R, c * R)) if c > 0 else []
        w = (kv_rows(prev, 0) if prev else torch.zeros(D, dtype=torch.float64)) + kv_rows(cur, 1)
        exp = rms(w) if prev else rms(kv_rows(cur, 1))
        e = float((o_k[c] - exp).abs().max())
        print(f"  row={c} max|kernel|={float(o_k[c].abs().max()):.4f} "
              f"expected(2r窗口)={float(exp.abs().max()):.4f} err={e:.3e} "
              f"{'ZERO!' if float(o_k[c].abs().max()) == 0.0 else ''}")


def main():
    print(f"[row0] dev={DEV} R={R} D={D} rd={RD} rm=1(identity rope)")
    for S_ in (4, 8, 16):
        run_S(S_)
    print("\n[row0] done")


if __name__ == "__main__":
    main()
