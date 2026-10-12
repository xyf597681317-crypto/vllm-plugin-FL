# -*- coding: utf-8 -*-
"""task-3 探针：mla_preprocess_int8.py 的 2-D 掩码 store（L65 / L133 / L172）。

A) 本文件内 3 个 in-scope kernel 的 sentinel 双填充检测（非 2 幂形状）：
   _quant_asymm_kernel  (L65)  —— y_i8 = saturate_i8(round_f16(x/s + o))
   _q_nope_quant_kernel (L133) —— q_nope_i8 = saturate_i8(round_f16(s1*scale[head]))
   _kv_cache_i8_kernel  (L172) —— rmsnorm + int8 + C0=32 NZ 写回
B) 归因检查：kv_cache_out 的两个写回者
   _kv_cache_i8_kernel（本文件） vs _kv_cache_kernel（mla_preprocess.py，**不在本任务写范围**），
   同一套 sentinel 检测，看丢写/越界写究竟出在哪一边。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_mlai8.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
F1, F2 = -99, -77

from triton_ops.mla_preprocess_int8 import (  # noqa: E402
    _quant_asymm_kernel, _q_nope_quant_kernel, _kv_cache_i8_kernel)
from triton_ops.mla_preprocess import _kv_cache_kernel  # noqa: E402


def rint_even(t):
    return torch.round(t)


# ---------------------------------------------------------------- A1: quant_asymm
def probe_quant_asymm(M, D, blk_m=16, blk_d=256):
    torch.manual_seed(21)
    x = (torch.randn(M, D, dtype=torch.float32, device=DEV) * 3.0).to(torch.float16)
    s = torch.tensor([0.037], device=DEV)
    o = torch.tensor([3], dtype=torch.int8, device=DEV)
    ref = rint_even((x.detach().cpu().double() / 0.037
                     ).to(torch.float16).double() + 3.0).clamp(-128, 127)
    outs = []
    for f in (F1, F2):
        y = torch.full((M, D), f, dtype=torch.int8, device=DEV)
        _quant_asymm_kernel[(triton.cdiv(M, blk_m),)](x, y, s, o, M, D, BLK_M=blk_m, BLK_D=blk_d)
        torch.npu.synchronize()
        outs.append(y.detach().cpu())
    dropped = (outs[0] == F1) & (outs[1] == F2)
    wrong = (~dropped) & (outs[0].long() != ref.long())
    print(f"  [quant_asymm] M={M:3d} D={D:5d} dropped={int(dropped.sum()):5d} "
          f"wrong={int(wrong.sum()):5d}/{M * D}  "
          f"{'DROP!' if dropped.any() else ('wrong' if wrong.any() else 'clean')}", flush=True)
    if dropped.any():
        r0 = int(dropped.any(dim=1).nonzero()[0])
        print("      r=%d: %s" % (r0, "".join("X" if dropped[r0, c] else "." for c in range(D))), flush=True)


# ---------------------------------------------------------------- A2: q_nope_quant
def probe_q_nope(N, HN, KVD, blk_d=256):
    torch.manual_seed(22)
    s1 = (torch.randn(N * HN, KVD, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    sc = (torch.rand(16, dtype=torch.float32, device=DEV) + 0.5)
    rows = torch.arange(N * HN)
    xd = s1.detach().cpu().double() * sc[:HN].detach().cpu().double()[rows % HN][:, None]
    ref_full = rint_even(xd.to(torch.float16).double()).clamp(-128, 127)
    outs = []
    for f in (F1, F2):
        q = torch.full((N * HN, KVD), f, dtype=torch.int8, device=DEV)
        _q_nope_quant_kernel[(N,)](s1, sc, q, N, HN, KVD, BLK_M=16, BLK_D=blk_d)
        torch.npu.synchronize()
        outs.append(q.detach().cpu())
    dropped = (outs[0] == F1) & (outs[1] == F2)
    wrong = (~dropped) & (outs[0].long() != ref_full.long())
    print(f"  [q_nope_quant] N={N} HN={HN} KVD={KVD:5d} dropped={int(dropped.sum()):5d} "
          f"wrong={int(wrong.sum()):5d}/{N * HN * KVD}  "
          f"{'DROP!' if dropped.any() else ('wrong' if wrong.any() else 'clean')}", flush=True)


# ---------------------------------------------------------- A3/B: kv cache write
def probe_kv_i8(N, KVD, RD, ns=64, blk_m=16, blk_d=256):
    """int8_nzcache 写回（本文件 L172）：sentinel 检测丢写 + 越界写。"""
    D2 = KVD + RD
    HALF = RD // 2
    nout = (ns + 127) // 128
    torch.manual_seed(23)
    s3 = (torch.randn(N, D2, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gam = torch.rand(KVD, dtype=torch.float32, device=DEV) * 0.5 + 0.5
    th = torch.rand(N, RD, dtype=torch.float32, device=DEV) * 6.28
    cos, sin = torch.cos(th), torch.sin(th)
    slot = (torch.arange(N, dtype=torch.int32, device=DEV) * 3 + 1)
    csc = torch.tensor([0.5], device=DEV)
    exp_off = set()
    for t in range(N):
        sl = int(slot[t])
        base = (sl // 128) * (128 * KVD) + (sl % 128) * 32
        for d in range(KVD):
            exp_off.add(base + (d // 32) * 4096 + (d % 32))
    size = max(nout * 128 * KVD, 8192) + 8192
    kv_fill = []
    for f in (F1, F2):
        kv = torch.full((size,), f, dtype=torch.int8, device=DEV)
        kr = torch.full((nout * 128 * RD,), -1000.0, dtype=torch.bfloat16, device=DEV)
        _kv_cache_i8_kernel[(triton.cdiv(N, blk_m),)](
            s3, gam, cos, sin, slot, csc, kv, kr, N, KVD, RD, D2,
            1.0 / KVD, 1e-6, BLK_M=blk_m, BLK_D=blk_d, HALF=HALF)
        torch.npu.synchronize()
        kv_fill.append(kv.detach().cpu())
    dd = (kv_fill[0] == F1) & (kv_fill[1] == F2)
    wrote = (kv_fill[0] != F1)
    exp_mask = torch.zeros(size, dtype=torch.bool)
    exp_mask[list(exp_off)] = True
    n_drop_in = int((dd & exp_mask).sum())
    n_drop_out = int((dd & ~exp_mask).sum())
    n_extra = int((wrote & ~exp_mask).sum())
    print(f"  [kv_i8] N={N} KVD={KVD:4d} RD={RD:3d} size={size} "
          f"drop@expected={n_drop_in} drop@other={n_drop_out} "
          f"write@non-expected={n_extra}  "
          f"{'DROP!' if n_drop_in else ('extra-write' if n_extra else 'clean')}", flush=True)


def probe_kv_orig(N, KVD, RD, D2, ns=64, split_out=True, nz=False, blk_m=16, blk_d=256):
    """mla_preprocess.py 的 _kv_cache_kernel（**不在写范围**）—— 同一套 sentinel 检测。"""
    HALF = RD // 2
    torch.manual_seed(24)
    s3 = (torch.randn(N, D2, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gam = torch.rand(KVD, dtype=torch.float32, device=DEV) * 0.5 + 0.5
    th = torch.rand(N, RD, dtype=torch.float32, device=DEV) * 6.28
    cos, sin = torch.cos(th), torch.sin(th)
    slot = (torch.arange(N, dtype=torch.int32, device=DEV) * 3 + 1)
    KLINE = KVD + RD
    if nz:
        out_size = ((ns + 127) // 128) * 128 * KVD
        ko_size = ((ns + 127) // 128) * 128 * RD
        exp = set()
        for t in range(N):
            sl = int(slot[t])
            for d in range(KVD):
                exp.add((sl // 128) * (128 * KVD) + (d // 16) * 2048 + (sl % 128) * 16 + (d % 16))
    else:
        out_size = ns * (KVD if split_out else KLINE)
        ko_size = ns * RD
        exp = set()
        for t in range(N):
            for d in range(KVD):
                exp.add(int(slot[t]) * (KVD if split_out else KLINE) + d)
    exp_mask = torch.zeros(out_size, dtype=torch.bool)
    exp_mask[list(exp)] = True
    filled = []
    for f in (-3.0, -5.0):
        kv = torch.full((out_size,), f, dtype=torch.float16, device=DEV)
        kr = torch.full((ko_size,), -1000.0, dtype=torch.float16, device=DEV)
        _kv_cache_kernel[(triton.cdiv(N, blk_m),)](
            s3, gam, cos, sin, slot, kv, kr, N, KVD, RD, D2, KLINE, 1.0 / KVD, 1e-6,
            SPLIT_OUT=split_out, NZ=nz, BLK_M=blk_m, BLK_D=blk_d, HALF=HALF)
        torch.npu.synchronize()
        filled.append(kv.detach().cpu().float())
    dd = (filled[0] == -3.0) & (filled[1] == -5.0)
    wrote = (filled[0] != -3.0)
    huge = (filled[0].abs() > 1e3) | filled[0].isnan()
    print(f"  [kv_orig] N={N} KVD={KVD} RD={RD} D2={D2} split={split_out} nz={nz} size={out_size} "
          f"drop@expected={int((dd & exp_mask).sum())} drop@other={int((dd & ~exp_mask).sum())} "
          f"write@non-expected={int((wrote & ~exp_mask).sum())} "
          f"huge_or_nan={int(huge.sum())}  "
          f"{'DROP!' if (dd & exp_mask).any() else ('extra' if (wrote & ~exp_mask).any() else 'clean')}",
          flush=True)


def _safe(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as ex:  # noqa: BLE001
        print(f"  [EXC] {fn.__name__}{a}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)


def main():
    print(f"[mlai8] dev={DEV} triton={triton.__version__}", flush=True)
    print("== A1) _quant_asymm_kernel (L65, 2-D 掩码 store, BLK_M=16/BLK_D=256) ==", flush=True)
    for M in (16, 19):
        for D in (17, 24, 33, 48, 100, 127, 129, 256, 1000, 7168):
            _safe(probe_quant_asymm, M, D)
    print("== A2) _q_nope_quant_kernel (L133) ==", flush=True)
    for KVD in (17, 33, 48, 100, 128, 512):
        _safe(probe_q_nope, 4, 4, KVD)
    print("== A3) _kv_cache_i8_kernel (L172, int8 NZ) ==", flush=True)
    for KVD, RD in ((64, 32), (17, 16), (33, 32), (100, 32), (512, 64)):
        _safe(probe_kv_i8, 6, KVD, RD)
    print("== B) 归因：mla_preprocess.py 的 _kv_cache_kernel（非写范围） ==", flush=True)
    _safe(probe_kv_orig, 6, 64, 32, 96)
    _safe(probe_kv_orig, 6, 64, 32, 96, nz=True)
    _safe(probe_kv_orig, 4, 512, 64, 576, ns=32)
    print("[mlai8] done", flush=True)


if __name__ == "__main__":
    main()
