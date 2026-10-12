# -*- coding: utf-8 -*-
"""冻结版最小复现（交给 fix-2dstore 对齐 harness）：**旧 2-D 掩码 store 丢写**。

背景：`_analysis/triton_ops/moe_init_routing_custom.py` 旧版（2-D `[BLK_D, BLK_C]` +
2-D 掩码 store）在 cols 非 2 幂时**静默丢写**，本文件把**旧 kernel 源码原样冻结**
在这里（算子文件本身已改成 1-D 写法，import 已经复现不出来了），并提供 V1/V2/V3
逐步简化的对照，用来定位触发条件。

判定用「双填充」法（不会被 kernel 合法输出误判）：
    同一 kernel 跑两遍，输出缓冲分别预填 F1=-99 / F2=-77；
    两遍仍分别等于各自填充值 ⇒ store 完全没写（dropped）。
用法（容器内）：
    MOE_DEV=npu:0 python3 -W ignore _probe_moe_2dstore.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = os.environ.get("MOE_DEV", "npu:0")
F1, F2 = -99, -77


# --------------------------------------------------------------------------
# 冻结：旧版 _rowscale_kernel / _apply_quant_kernel（2-D block + 2-D 掩码 store）
# 出处：moe_init_routing_custom.py 改动前版本（同文本也保存在
#       _analysis/handoff_1011/pr_base/vllm_fl/.../impl/triton_ops/moe_init_routing_custom.py L196/L231）
# --------------------------------------------------------------------------
@triton.jit
def rowscale_old(X, SRC, DE, SMOOTH, ESCALE, TOTAL, K, COLS,
                 BLK_D: tl.constexpr, BLK_C: tl.constexpr, HAS_SMOOTH: tl.constexpr = True):
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    e = tl.load(DE + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    amax = tl.zeros([BLK_D], dtype=tl.float32)
    for c0 in range(0, tl.cdiv(COLS, BLK_C)):
        c = c0 * BLK_C + tl.arange(0, BLK_C)
        mc = c < COLS
        m2 = md[:, None] & mc[None, :]
        xv = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
        if HAS_SMOOTH:
            sm = tl.load(SMOOTH + e[:, None] * COLS + c[None, :], mask=m2,
                         other=0.0).to(tl.float32)
            sv = xv * sm
        else:
            sv = xv
        amax = tl.maximum(amax, tl.max(tl.where(m2, tl.abs(sv), 0.0), axis=1))
    tl.store(ESCALE + d, amax / 127.0, mask=md)


@triton.jit
def apply_old(X, SRC, DE, SMOOTH, ESCALE, OUTX, TOTAL, K, COLS,
              BLK_D: tl.constexpr, BLK_C: tl.constexpr, HAS_SMOOTH: tl.constexpr = True):
    """V0：旧实现原样（2-D grid + 2-D 掩码 load/store + fp16 往返 + rint + 饱和）。"""
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    mc = c < COLS
    src = tl.load(SRC + d, mask=md, other=0)
    e = tl.load(DE + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & mc[None, :]
    xv = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    if HAS_SMOOTH:
        sm = tl.load(SMOOTH + e[:, None] * COLS + c[None, :], mask=m2,
                     other=0.0).to(tl.float32)
        num = xv * sm
    else:
        num = xv
    s = tl.load(ESCALE + d, mask=md, other=1.0)
    q = num / s[:, None]
    q16 = q.to(tl.float16)
    f = q16.to(tl.float32)
    r = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], r.to(tl.int8), mask=m2)


@triton.jit
def apply_v1(X, SRC, OUTX, TOTAL, K, COLS, BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    """V1：2-D 掩码 store，但值只有一次类型转换（去掉除法/fp16 往返/rint/饱和）。"""
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], x.to(tl.int8), mask=m2)


@triton.jit
def apply_v2(X, SRC, ESCALE, OUTX, TOTAL, K, COLS,
             BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    """V2：V1 + escale 载入 + 除法（仍无 fp16 往返/rint/饱和）。"""
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    s = tl.load(ESCALE + d, mask=md, other=1.0)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], (x / s[:, None]).to(tl.int8), mask=m2)


@triton.jit
def apply_v3(X, SRC, ESCALE, OUTX, TOTAL, K, COLS,
             BLK_D: tl.constexpr, BLK_C: tl.constexpr):
    """V3：V2 + fp16 往返 + floor/ceil/where + 饱和 = 与 V0 同值链。"""
    d = tl.program_id(0) * BLK_D + tl.arange(0, BLK_D)
    c = tl.program_id(1) * BLK_C + tl.arange(0, BLK_C)
    md = d < TOTAL
    src = tl.load(SRC + d, mask=md, other=0)
    md = md & (src >= 0)
    row = src // K
    m2 = md[:, None] & (c < COLS)[None, :]
    x = tl.load(X + row[:, None] * COLS + c[None, :], mask=m2, other=0.0).to(tl.float32)
    s = tl.load(ESCALE + d, mask=md, other=1.0)
    q = x / s[:, None]
    f = q.to(tl.float16).to(tl.float32)
    r = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
    tl.store(OUTX + d[:, None] * COLS + c[None, :], r.to(tl.int8), mask=m2)


def two_fill(launch, total, cols):
    outs = []
    for f in (F1, F2):
        o = torch.full((total, cols), f, dtype=torch.int8, device=DEV)
        launch(o)
        torch.npu.synchronize()
        outs.append(o.clone())
    return outs[0], (outs[0] == F1) & (outs[1] == F2)


def case(cols, bd, bc, k=2, N=16, seed=0):
    total = N * k
    torch.manual_seed(seed)
    x = (torch.randn(N, cols, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    src = torch.arange(total, dtype=torch.int32, device=DEV)
    de = torch.zeros(total, dtype=torch.int32, device=DEV)
    smat = torch.zeros(1, dtype=torch.float32, device=DEV)
    esc = torch.zeros(total, dtype=torch.float32, device=DEV)
    rowscale_old[(triton.cdiv(total, 8),)](x, src, de, smat, esc, total, k, cols,
                                           BLK_D=8, BLK_C=64, HAS_SMOOTH=False)
    torch.npu.synchronize()
    grid = (triton.cdiv(total, bd), triton.cdiv(cols, bc))
    print(f"\n=== cols={cols} BLK_D={bd} BLK_C={bc} grid={grid} total={total} k={k} ===",
          flush=True)
    variants = [
        ("V0 apply_old (2-D mask, 全值链)",
         lambda o: apply_old[grid](x, src, de, smat, esc, o, total, k, cols,
                                   BLK_D=bd, BLK_C=bc, HAS_SMOOTH=False)),
        ("V1 无除法/无 fp16 往返",
         lambda o: apply_v1[grid](x, src, o, total, k, cols, BLK_D=bd, BLK_C=bc)),
        ("V2 + 除法",
         lambda o: apply_v2[grid](x, src, esc, o, total, k, cols, BLK_D=bd, BLK_C=bc)),
        ("V3 + fp16 往返/rint/饱和",
         lambda o: apply_v3[grid](x, src, esc, o, total, k, cols, BLK_D=bd, BLK_C=bc)),
    ]
    for tag, fn in variants:
        try:
            got, dropped = two_fill(fn, total, cols)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{tag}] EXC {type(ex).__name__}: {str(ex)[:150]}", flush=True)
            continue
        print(f"  [{tag:32s}] dropped={dropped.sum().item()}", flush=True)
        if dropped.sum().item():
            for i in range(total):
                if dropped[i].any():
                    m = "".join("X" if dropped[i, j] else "." for j in range(cols))
                    print(f"      d={i:3d} {m}", flush=True)


if __name__ == "__main__":
    print(f"[2dstore-repro] dev={DEV} triton={triton.__version__}", flush=True)
    case(33, 8, 64)
    case(33, 2, 64)
    case(48, 8, 64)
    case(64, 8, 64)      # 2 幂对照：预期 dropped=0
    print("\n[2dstore-repro] done", flush=True)
