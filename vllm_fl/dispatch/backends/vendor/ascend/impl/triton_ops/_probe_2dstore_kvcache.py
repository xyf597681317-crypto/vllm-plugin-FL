# -*- coding: utf-8 -*-
"""task-3 探针：KV 写回的 huge/NaN 值 —— 定位 + 候选 1-D 修法验证。

已知：`mla_preprocess.py::_kv_cache_kernel` 在 split=True / NZ=False 下会把
28 个（KVD=64）/ 52 个（KVD=512）**huge 或 NaN** 值写进「本应写入的 offset」
（sentinel 检测：drop@expected=0、write@non-expected=0，但 huge_or_nan>0）；
NZ=True 路径 0 个。本探针：

 1) 直接复现单测失败用例 `小配置 bf16 krope_ctkv N=6`，打印 kvc0/kvc1 里 huge/NaN 的
    offset 与真值对照（逐元素，不看均值）；
 2) 用 sentinel 跑生产 `_kv_cache_kernel`，逐 offset 打印 huge/NaN 的位置；
 3) 用**候选 1-D 改写形** `_kv_cache_1d` 跑同一套输入，验证能否消除。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_kvcache.py
"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)

from triton_ops.mla_preprocess import _kv_cache_kernel  # noqa: E402


# --------------------------------------------------------------------------
# 候选 1-D 改写形：标量行号 + [BLK_D] 列 tile + 1-D 掩码 store
# --------------------------------------------------------------------------
@triton.jit
def _kv_cache_1d(
    S3, GAMMA, COS, SIN, SLOT, KVOUT, KROPOUT,
    M, KVD, RD, D2, KLINE, AVG, EPS,
    SPLIT_OUT: tl.constexpr, NZ: tl.constexpr,
    BLK_D: tl.constexpr, HALF: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    rd0 = tl.arange(0, BLK_D)
    ssv = tl.zeros([BLK_D], dtype=tl.float32)
    for d0 in range(0, KVD, BLK_D):
        rd = d0 + rd0
        x = tl.load(S3 + row * D2 + rd, mask=rd < KVD, other=0.0).to(tl.float32)
        ssv += x * x
    inv = 1.0 / tl.sqrt(tl.sum(ssv, axis=0) * AVG + EPS)
    slot = tl.load(SLOT + row)
    for d0 in range(0, KVD, BLK_D):
        rd = d0 + rd0
        dm = rd < KVD
        x = tl.load(S3 + row * D2 + rd, mask=dm, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
        y = x * inv * g
        if NZ:
            so = (slot // 128) * (128 * KVD) + (rd // 16) * 2048 + (slot % 128) * 16 + (rd % 16)
        elif SPLIT_OUT:
            so = slot * KVD + rd
        else:
            so = slot * KLINE + rd
        tl.store(KVOUT + so, y.to(KVOUT.dtype.element_ty), mask=dm)
    h = tl.arange(0, HALF)
    hm = h < HALF
    base = row * D2 + KVD
    xa = tl.load(S3 + base + h, mask=hm, other=0.0).to(tl.float32)
    xb = tl.load(S3 + base + HALF + h, mask=hm, other=0.0).to(tl.float32)
    ca = tl.load(COS + row * RD + h, mask=hm, other=0.0).to(tl.float32)
    cb = tl.load(COS + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
    sa = tl.load(SIN + row * RD + h, mask=hm, other=0.0).to(tl.float32)
    sb = tl.load(SIN + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
    oa = ca * xa - sa * xb
    ob = sb * xa + cb * xb
    if NZ:
        base2 = (slot // 128) * (128 * RD) + (slot % 128) * 16
        soa = base2 + (h // 16) * 2048 + (h % 16)
        sob = base2 + ((HALF + h) // 16) * 2048 + ((HALF + h) % 16)
        tl.store(KROPOUT + soa, oa.to(KROPOUT.dtype.element_ty), mask=hm)
        tl.store(KROPOUT + sob, ob.to(KROPOUT.dtype.element_ty), mask=hm)
    elif SPLIT_OUT:
        tl.store(KROPOUT + slot * RD + h, oa.to(KROPOUT.dtype.element_ty), mask=hm)
        tl.store(KROPOUT + slot * RD + HALF + h, ob.to(KROPOUT.dtype.element_ty), mask=hm)
    else:
        tl.store(KVOUT + slot * KLINE + KVD + h, oa.to(KVOUT.dtype.element_ty), mask=hm)
        tl.store(KVOUT + slot * KLINE + KVD + HALF + h, ob.to(KVOUT.dtype.element_ty), mask=hm)


def dump_huge(tag, kv, exp_off, kvd):
    t = kv.detach().cpu().float()
    bad = ((t.abs() > 1e3) | t.isnan()).nonzero().flatten().tolist()
    exp = set(exp_off)
    in_exp = [i for i in bad if i in exp]
    out_exp = [i for i in bad if i not in exp]
    print(f"    {tag}: huge_or_nan={len(bad)} (in_expected={len(in_exp)}, other={len(out_exp)})"
          f" 首 8 个 offset={bad[:8]}", flush=True)
    for i in bad[:8]:
        print(f"       off={i} slot={i // kvd} d={i % kvd} val={t[i].item():.4e}  "
              f"expected_offset?={i in exp}", flush=True)


def sentinel_run(launch, kvd, rd, n, ns, split_out, nz, d2=None, kline=None, tag="prod"):
    HALF = rd // 2
    torch.manual_seed(51)
    D2 = d2 if d2 is not None else kvd + rd
    KLINE = kline if kline is not None else kvd + rd
    s3 = (torch.randn(n, D2, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gam = torch.rand(kvd, dtype=torch.float32, device=DEV) * 0.5 + 0.5
    th = torch.rand(n, rd, dtype=torch.float32, device=DEV) * 6.28
    cos, sin = torch.cos(th), torch.sin(th)
    slot = torch.arange(n, dtype=torch.int32, device=DEV) * 3 + 1
    if nz:
        out_size = ((ns + 127) // 128) * 128 * kvd
        ko_size = ((ns + 127) // 128) * 128 * rd
        exp = set()
        for t in range(n):
            sl = int(slot[t])
            for d in range(kvd):
                exp.add((sl // 128) * (128 * kvd) + (d // 16) * 2048 + (sl % 128) * 16 + (d % 16))
    else:
        out_size = ns * (kvd if split_out else KLINE)
        ko_size = ns * rd
        exp = set()
        for t in range(n):
            for d in range(kvd):
                exp.add(int(slot[t]) * (kvd if split_out else KLINE) + d)
    kv = torch.full((out_size,), -3.0, dtype=torch.float16, device=DEV)
    kr = torch.full((ko_size,), -1000.0, dtype=torch.float16, device=DEV)
    launch(s3, gam, cos, sin, slot, kv, kr, n, kvd, rd, D2, KLINE, 1.0 / kvd, 1e-6)
    torch.npu.synchronize()
    exp_mask = torch.zeros(out_size, dtype=torch.bool)
    exp_mask[list(exp)] = True
    w = kv.detach().cpu().float()
    written = (w != -3.0)
    ndrop = int((~written & exp_mask).sum())
    nextra = int((written & ~exp_mask).sum())
    print(f"  [{tag}] N={n} KVD={kvd} RD={rd} D2={D2} split={split_out} nz={nz}: "
          f"drop@expected={ndrop} write@non-expected={nextra}", flush=True)
    dump_huge(tag, kv, exp, kvd)
    return s3, gam, cos, sin, slot, D2, KLINE


def compare_1d_vs_prod():
    """同一输入下，生产 2-D kernel 与候选 1-D kernel 的结果逐元素比对。"""
    n, kvd, rd, ns = 6, 64, 32, 64
    D2, KLINE, HALF = kvd + rd, kvd + rd, rd // 2
    torch.manual_seed(52)
    s3 = (torch.randn(n, D2, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gam = torch.rand(kvd, dtype=torch.float32, device=DEV) * 0.5 + 0.5
    th = torch.rand(n, rd, dtype=torch.float32, device=DEV) * 6.28
    cos, sin = torch.cos(th), torch.sin(th)
    slot = torch.arange(n, dtype=torch.int32, device=DEV) * 3 + 1
    res = {}
    for tag in ("prod2d", "fixed1d"):
        kv = torch.full((ns, kvd), -3.0, dtype=torch.float16, device=DEV)
        kr = torch.full((ns, rd), -1000.0, dtype=torch.float16, device=DEV)
        if tag == "prod2d":
            _kv_cache_kernel[(triton.cdiv(n, 16),)](
                s3, gam, cos, sin, slot, kv, kr, n, kvd, rd, D2, KLINE, 1.0 / kvd, 1e-6,
                SPLIT_OUT=True, NZ=False, BLK_M=16, BLK_D=256, HALF=HALF)
        else:
            _kv_cache_1d[(n,)](
                s3, gam, cos, sin, slot, kv, kr, n, kvd, rd, D2, KLINE, 1.0 / kvd, 1e-6,
                SPLIT_OUT=True, NZ=False, BLK_D=256, HALF=HALF)
        torch.npu.synchronize()
        res[tag] = (kv.detach().cpu().float(), kr.detach().cpu().float())
    for i, nm in ((0, "kv"), (1, "krope")):
        a, b = res["prod2d"][i], res["fixed1d"][i]
        d = (a - b).abs()
        bad_a = int(((a.abs() > 1e3) | a.isnan()).sum())
        bad_b = int(((b.abs() > 1e3) | b.isnan()).sum())
        print(f"  [cmp] {nm}: prod2d huge/nan={bad_a}  fixed1d huge/nan={bad_b}  "
              f"max|prod-fixed|={float(d.max()):.4e}", flush=True)


def main():
    print(f"[kvcache] dev={DEV} triton={triton.__version__}", flush=True)
    n, kvd, rd, ns = 6, 64, 32, 64
    HALF = rd // 2
    print("== 1) 生产 2-D kernel（sentinel，split=True / NZ=False） ==", flush=True)

    def l_prod(s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps):
        _kv_cache_kernel[(triton.cdiv(n_, 16),)](
            s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps,
            SPLIT_OUT=True, NZ=False, BLK_M=16, BLK_D=256, HALF=rd_ // 2)

    sentinel_run(l_prod, kvd, rd, n, ns, True, False, tag="prod2d")
    print("== 2) 同类输入但 BLK_D=64（列不跨块） ==", flush=True)

    def l_prod64(s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps):
        _kv_cache_kernel[(triton.cdiv(n_, 16),)](
            s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps,
            SPLIT_OUT=True, NZ=False, BLK_M=16, BLK_D=64, HALF=rd_ // 2)

    sentinel_run(l_prod64, kvd, rd, n, ns, True, False, tag="prod2d_bd64")
    print("== 3) 候选 1-D 改写形 ==", flush=True)

    def l_fix(s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps):
        _kv_cache_1d[(n_,)](
            s3, gam, cos, sin, slot, kv, kr, n_, kvd_, rd_, d2_, kline_, avg, eps,
            SPLIT_OUT=True, NZ=False, BLK_D=256, HALF=rd_ // 2)

    sentinel_run(l_fix, kvd, rd, n, ns, True, False, tag="fixed1d")
    print("== 4) 逐元素对照 ==", flush=True)
    compare_1d_vs_prod()
    print("[kvcache] done", flush=True)


if __name__ == "__main__":
    main()
