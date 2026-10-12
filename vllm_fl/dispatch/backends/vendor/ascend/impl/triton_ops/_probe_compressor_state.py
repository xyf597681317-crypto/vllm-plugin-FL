# -*- coding: utf-8 -*-
"""`_compressor_coff2_state_kernel` 的独立验证：双 sentinel + fp64 参考逐元素比对。

只调 state kernel（不调主 kernel），复现 test_compressor_coff2 的 stream 场景：
  B=1, R=4, D=128, H=64, block_size=16, stride0=16*4*D, block_table=[0,1]
  call1: sp=0,  su=18  (CONTINUOUS → wlo=0,  whi=18)
  call2: sp=18, su=14  (CONTINUOUS → wlo=18, whi=32)

输出每个写入位置的：
  * fill1/fill2 两遍是否仍等于各自的填充值（= 完全没写）
  * 与 fp64 参考的最大绝对误差
  * 4 个子区 [kv_a | kv_b | sc_a | sc_b] 各自的误差（定位是哪一路写错）

用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_state.py
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
from triton_ops.compressor_coff2 import _compressor_coff2_state_kernel  # noqa: E402

F1, F2 = -7.5, 7.5
B, R, D, RD, H, BS = 1, 4, 128, 64, 64, 16
S1, S2 = 18, 14
S = S1 + S2
STRIDE0 = BS * 4 * D
NBLK = (S + BS - 1) // BS


def build():
    torch.manual_seed(3)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(1, S, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    return x, wkv, wgate, ape


def ref_state(x, wkv, wgate, ape, sp, wlo, whi, cache_cycle):
    """fp64 参考（与 test_compressor_coff2.ref_A 的 state 写回逐行一致）。"""
    xq = x.detach().cpu().double()
    wkvq = wkv.detach().cpu().double()
    wgq = wgate.detach().cpu().double()
    apeq = ape.detach().cpu().double()
    st = torch.zeros(NBLK, BS, 4 * D, dtype=torch.float64)
    for p in range(wlo, whi):
        tok = p - sp
        blk = 0 if cache_cycle else p // BS
        r = p % BS
        st[blk, r, 0:D] = xq[0, tok] @ wkvq[:D].t()
        st[blk, r, D:2 * D] = xq[0, tok] @ wkvq[D:].t()
        st[blk, r, 2 * D:3 * D] = xq[0, tok] @ wgq[:D].t() + apeq[p % R, :D]
        st[blk, r, 3 * D:4 * D] = xq[0, tok] @ wgq[D:].t() + apeq[p % R, D:]
    return st


def run_case(tag, x, wkv, wgate, ape, sp, wlo, whi, cache_cycle):
    dev = torch.device(DEV)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    BLK_D = 1 << (D - 1).bit_length()
    BLK_H = max(16, min(128, 8192 // BLK_D))
    BLK_D_ST = min(BLK_D, 128)
    nrows = whi - wlo
    grid = (nrows,)
    blkt = torch.arange(NBLK, dtype=torch.int32, device=dev).reshape(1, NBLK)
    x_c, wkv_c, wgate_c = x.contiguous(), wkv.contiguous(), wgate.contiguous()
    ape_c = ape.contiguous().to(torch.float32)
    rb = torch.zeros(nrows, dtype=torch.int32, device=dev)
    rq = torch.arange(nrows, dtype=torch.int32, device=dev)
    outs = []
    for fill in (F1, F2):
        st = torch.full((NBLK, BS, 4 * D), fill, dtype=torch.float32, device=DEV)
        _compressor_coff2_state_kernel[grid](
            x_c, wkv_c, wgate_c, ape_c, st, blkt,
            i32(0), i32(sp), i32(wlo), rb, rq,
            BS, STRIDE0, NBLK,
            H=H, D=D, R=R, BLK_H=BLK_H, BLK_D_ST=BLK_D_ST,
            CACHE_CYCLE=bool(cache_cycle))
        torch.npu.synchronize()
        outs.append(st.detach().cpu())
    (a1, a2) = outs
    r = ref_state(x, wkv, wgate, ape, sp, wlo, whi, cache_cycle)
    print(f"\n=== {tag}  sp={sp} wlo={wlo} whi={whi} cycle={cache_cycle} "
          f"grid={grid} nrows={nrows} ===", flush=True)
    bad = 0
    for p in range(wlo, whi):
        blk = 0 if cache_cycle else p // BS
        rr = p % BS
        e = (a1[blk, rr].double() - r[blk, rr]).abs()
        d1 = int((a1[blk, rr] == F1).sum())
        d2 = int((a2[blk, rr] == F2).sum())
        if float(e.max()) > 1e-6 or d1 > 0 or d2 > 0:
            bad += 1
            print(f"  p={p:3d} blk={blk} row={rr:2d} maxerr={float(e.max()):.3e} "
                  f"未写r1/r2={d1}/{d2} | 分区 err kvA={float(e[0:D].max()):.2e} "
                  f"kvB={float(e[D:2 * D].max()):.2e} scA={float(e[2 * D:3 * D].max()):.2e} "
                  f"scB={float(e[3 * D:4 * D].max()):.2e}", flush=True)
    print(f"  → 有问题的位置: {bad}/{max(whi - wlo, 0)}"
          f"  {'✅ 全部正确' if bad == 0 else '❌'}", flush=True)
    return bad


if __name__ == "__main__":
    torch.npu.set_device(DEV)
    print(f"[state] dev={DEV} F1={F1} F2={F2} S={S} BS={BS} NBLK={NBLK}", flush=True)
    x, wkv, wgate, ape = build()
    n = 0
    n += run_case("call1 CONTINUOUS", x, wkv, wgate, ape, 0, 0, S1, False)
    n += run_case("call2 CONTINUOUS", x, wkv, wgate, ape, S1, S1, S1 + S2, False)
    n += run_case("call1 CYCLE", x, wkv, wgate, ape, 0, 16, S1, True)
    print(f"\n[state] 合计问题位置 = {n}", flush=True)
