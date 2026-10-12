# -*- coding: utf-8 -*-
"""流式第二段（seg2）定位：复现 test_compressor_coff2 的 case_stream，并逐行对 fp64 参考。

判据：
  * 只有「窗口里含 history（head）行」的 chunk 错 ⇒ 主 kernel 的 state 读回路径；
  * 全部 chunk 都错 ⇒ 别的问题。
同时跑 CYCLE 一档，看崩溃能否在独立进程里复现。

用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_compressor_seg2.py [cont|cyc]
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

MIN_SCORE = -1.0e30
R, D, RD, H = 4, 128, 64, 64
S1, S2 = 18, 14
S = S1 + S2
BS = 16
STRIDE0 = BS * 4 * D
NBLK = (S + BS - 1) // BS


def rope_interleave(y, rc, rs, D, RD):
    """test_compressor_coff2.ref_A 的 interleave 写法（fp64）。"""
    y = y.clone()
    yr = y[D - RD:].clone()
    y[D - RD::2] = rc[0::2] * yr[0::2] - rs[0::2] * yr[1::2]
    y[D - RD + 1::2] = rc[1::2] * yr[1::2] + rs[1::2] * yr[0::2]
    return y


def ref_seg(x2, wkvq, wgq, apeq, nwq, sc_before, sp, nc, c0, rr0, rcq, rsq):
    """fp64 参考：窗口 2r + 通道分半 + 首块无历史，state 从 sc_before 读。"""
    out = []
    for ci in range(nc):
        c = c0 + ci
        kv = torch.zeros(2 * R, D, dtype=torch.float64)
        sc = torch.full((2 * R, D), MIN_SCORE, dtype=torch.float64)
        for j in range(2 * R):
            p = (c - 1) * R + j
            ho = 0 if j < R else D
            if p < 0:
                continue
            if p < sp:
                blk = p // BS
                base = blk * STRIDE0 + (p % BS) * (4 * D)
                kv[j] = sc_before[base:base + D]
                sc[j] = sc_before[base + 2 * D:base + 2 * D + D]
            else:
                kv[j] = x2[p - sp] @ wkvq[ho:ho + D].t()
                sc[j] = x2[p - sp] @ wgq[ho:ho + D].t() + apeq[j % R, ho:ho + D]
        mx = sc.max(dim=0).values
        ex = torch.exp(sc - mx)
        pj = ex / ex.sum(dim=0)
        o = (pj * kv).sum(dim=0)
        y = o / torch.sqrt((o * o).sum() / D + 1e-6) * nwq
        out.append(rope_interleave(y, rcq[rr0 + ci], rsq[rr0 + ci], D, RD))
    return torch.stack(out)


def main():
    torch.npu.set_device(DEV)
    mode = sys.argv[1] if len(sys.argv) > 1 else "cont"
    cache_mode = 2 if mode == "cyc" else 1
    torch.manual_seed(3)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(1, S, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    Sr_all = (S + R - 1) // R
    rs = torch.randn(1, Sr_all, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(1, Sr_all, RD, dtype=torch.float32, device=DEV) * 0.1
    print(f"[seg2] dev={DEV} mode={mode}", flush=True)

    if cache_mode == 1:
        sc = torch.zeros(NBLK, BS, 4 * D, dtype=torch.float32, device=DEV)
        bt = torch.arange(NBLK, dtype=torch.int32, device=DEV).reshape(1, NBLK)
        max_blk = NBLK
    else:
        sc = torch.zeros(1, BS, 4 * D, dtype=torch.float32, device=DEV)
        bt = torch.zeros(1, dtype=torch.int32, device=DEV)
        max_blk = 1
    su1 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    sp1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    o1, sc1 = compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                         rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(),
                         sc, bt, seqused=su1, start_pos=sp1,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=cache_mode, state_cache_stride_dim0=STRIDE0, block_size=BS)
    torch.npu.synchronize()
    print(f"[seg2] call1 OK max|o1|={float(o1.float().abs().max()):.4f}", flush=True)

    sp2 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    su2 = torch.tensor([S2], dtype=torch.int32, device=DEV)
    n2 = (S1 + S2) // R - S1 // R
    o2, sc2 = compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                         rs[:, S1 // R:S1 // R + n2].contiguous(),
                         rc[:, S1 // R:S1 // R + n2].contiguous(),
                         sc1, bt, seqused=su2, start_pos=sp2,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=cache_mode, state_cache_stride_dim0=STRIDE0, block_size=BS)
    torch.npu.synchronize()
    print(f"[seg2] call2 OK shape={tuple(o2.shape)}", flush=True)

    o2q = o2.detach().cpu().double().reshape(-1, D)
    x2 = x[:, S1:].contiguous().detach().cpu().double().reshape(-1, H)
    r = ref_seg(x2, wkv.detach().cpu().double(), wgate.detach().cpu().double(),
                ape.detach().cpu().double(), nw.detach().cpu().double(),
                sc1.detach().cpu().double().reshape(-1),
                S1, n2, S1 // R, S1 // R,
                rc.detach().cpu().double().reshape(-1, RD),
                rs.detach().cpu().double().reshape(-1, RD))
    print(f"[seg2] 逐行 max|err|（row=c0+ci；head 行数 = 窗口里 p_abs<sp 的行数）", flush=True)
    for ci in range(n2):
        c = S1 // R + ci
        head_n = sum(1 for j in range(2 * R) if 0 <= (c - 1) * R + j < S1)
        e = (o2q[ci] - r[ci]).abs()
        print(f"  ci={ci} c={c} head行={head_n}/8  maxerr={float(e.max()):.3e} "
              f"非rope={float(e[:D - RD].max()):.3e} rope={float(e[D - RD:].max()):.3e}", flush=True)
    torch.npu.synchronize()
    print("[seg2] DONE", flush=True)


if __name__ == "__main__":
    main()
