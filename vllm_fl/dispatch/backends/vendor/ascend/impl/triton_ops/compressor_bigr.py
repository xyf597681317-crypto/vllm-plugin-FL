"""compressor 的 `coff=1` + `cmp_ratio > 16`（长窗）路径 —— 行方向分块的 online softmax。

================================================================================
为什么需要这个模块
================================================================================
`compressor_tiling.cpp:958-964` 明确允许三个组合：
    {r=4,   coff=2, D=512}
    {r=4,   coff=2, D=128}
    {r=128, coff=1, D=512}     ← 本模块
模型侧 `vllm_ascend/models/deepseek_v4/model.py:548`
`self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)`，
config 的 `compress_ratios` 逐层在 `4 / 128` 之间交替（见
`tests/ut/test_platform.py:40  compress_ratios=[0, 0] + [4, 128] * 20 + [4]`）
⇒ **r=128 是生产配置，不是"合法但无人用"**。

原 `compressor.py` 的 coff=1 kernel 把**整块 r 行**放进寄存器
（`[M, BLK_D]` fp32 累加器 ×2，`M=16` 固定）；r=128、D=512 时
`128*512*4 = 256KB/个`，远超 UB 192KB ⇒ 必须**行方向分块**。

================================================================================
语义（与 coff=1 完全一致，只是实现方式不同）
================================================================================
对绝对 chunk `c`（覆盖绝对位置 `[c*r, (c+1)*r)`）：

    kv_j[d]    = (x[p_j] @ wkv^T)[d]
    score_j[d] = (x[p_j] @ wgate^T)[d] + ape[j % r, d]
    o[d]       = sum_j softmax_j(score[:, d]) * kv_j[d]        # softmax 在 r 行上按列
    y          = RMSNorm_D(o) * norm_weight
    out        = RoPE(y)                                       # 行基址 = 输出 chunk 行号
    p_j = c*r + j；p_j < sp 的行从 state_cache 读回；p_j < 0 的行不存在。

**online / streaming softmax**（数学上与一次性 softmax 等价，仅浮点结合顺序不同）：

    run_max = -big ; run_sum = 0 ; run_acc = 0
    for each row-block of BM rows:
        blk_max = max(score_blk, axis=0)
        new_max = max(run_max, blk_max)
        alpha   = exp(run_max - new_max)                    # 首次 run_max=-big ⇒ alpha=0
        e       = where(good, exp(score_blk - new_max), 0)  # 无效行显式置 0
        run_acc = run_acc * alpha + sum(e * kv_blk, axis=0)
        run_sum = run_sum * alpha + sum(e,          axis=0)
        run_max = new_max
    o = run_acc / run_sum

⚠️ **两个易错点**（本实现显式处理）：
1. `alpha` 的 `exp(run_max - new_max)`：首次 `run_max = -1e30`、`new_max` 有限
   ⇒ `exp(-1e30) = 0` ✓；若该 block 全无效则 `new_max = -1e30`、`alpha = exp(0) = 1`，
   但此时 `e ≡ 0` ⇒ `run_acc/run_sum` 不变 ✓。
2. `e` 必须用 `tl.where(good, ..., 0.0)` 显式置 0 —— 否则无效行在
   `score = -1e30` 与 `new_max = -1e30` 相等时会得到 `exp(0) = 1`，
   凭空增加 `run_sum` 而不增加 `run_acc`，**结果会错**。

================================================================================
后端约束
================================================================================
* store 索引 = `tl.arange`（单位步长）或"标量 + 常量"（坑 1）。
* **不用 `tl.reshape` / `tl.join`**（坑 2）；RoPE 复用已验证的逐列标量展开。
* `BLK_H` 自适应控制 UB（坑 3）：`[BM, BLK_D]` fp32 累加器 ×2 + 权重 tile ×2。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["compressor_bigr"]

_MIN_SCORE = -1.0e30


@triton.jit
def _compressor_bigr_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, BM: tl.constexpr, HALF: tl.constexpr,
    EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HALF_MODE: tl.constexpr, HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    b = tl.program_id(0)
    ct = tl.program_id(1)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    nc = tl.load(NC + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    ok = ct < nc
    c = c0 + ct
    sc = ct

    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)
    m = tl.arange(0, BM)

    run_max = tl.zeros((BLK_D,), dtype=tl.float32) + MIN_SCORE
    run_sum = tl.zeros((BLK_D,), dtype=tl.float32)
    run_acc = tl.zeros((BLK_D,), dtype=tl.float32)

    for m0 in range(0, R, BM):
        mm = m0 + m
        p_abs = c * R + mm
        in_chunk = mm < R
        real = in_chunk & (p_abs >= sp) & (p_abs < sp + su) & ok
        head = in_chunk & (p_abs >= 0) & (p_abs < sp) & ok
        have = real | head
        tok = p_abs - sp
        xrow = xbase + tl.where(tok >= 0, tok, 0)

        acc_kv = tl.zeros((BM, BLK_D), dtype=tl.float32)
        acc_sc = tl.zeros((BM, BLK_D), dtype=tl.float32)
        if HAS_STATE:
            # ⚠️ 必须用「先初始化累加器、再让 dot 累加」的写法：
            #    BiSheng 不支持 `tl.where(mask, <vector load>, <dot 结果>)` 这种 select
            #    （报 "Unsupported copy from cc to cbuf"）。real 与 head 互斥 ⇒ 语义等价。
            p_safe = tl.where(p_abs >= 0, p_abs, 0)
            blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
            blk_id = tl.load(BLKT + blk_idx, mask=head, other=0)
            soff = blk_id[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
            sm = head[:, None] & dmask[None, :]
            acc_kv = tl.load(STATE + soff, mask=sm, other=0.0)
            acc_sc = tl.load(STATE + soff + D, mask=sm, other=0.0)

        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xm = real[:, None] & hmask[None, :]
            xv = tl.load(X + xrow[:, None] * H + hh[None, :], mask=xm, other=0.0)
            wm = hmask[:, None] & dmask[None, :]
            acc_kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            acc_sc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))

        j = mm % R
        # 等价于 tl.where(real, acc_sc + apev, acc_sc)，但避免在 cube 结果上做 select
        apev = tl.load(APE + j[:, None] * D + d[None, :],
                       mask=real[:, None] & dmask[None, :], other=0.0)
        acc_sc = acc_sc + apev

        good = have[:, None] & dmask[None, :] & ok
        s = tl.where(good, acc_sc, MIN_SCORE)
        blk_max = tl.max(s, axis=0)
        new_max = tl.maximum(run_max, blk_max)
        alpha = tl.exp(run_max - new_max)
        e = tl.where(good, tl.exp(s - new_max[None, :]), 0.0)
        run_acc = run_acc * alpha + tl.sum(e * acc_kv, axis=0)
        run_sum = run_sum * alpha + tl.sum(e, axis=0)
        run_max = new_max

    o = run_acc / tl.where(run_sum > 0.0, run_sum, 1.0)

    # --- RMSNorm over D ---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    ROFF: tl.constexpr = D - RD
    if HALF_MODE:
        hr: tl.constexpr = RD // 2
        for i in range(0, hr):
            va = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vb = tl.sum(tl.where(d == ROFF + hr + i, y, 0.0), axis=0)
            ca = tl.load(RCOS + orow * RD + i)
            sa = tl.load(RSIN + orow * RD + i)
            cb = tl.load(RCOS + orow * RD + hr + i)
            sb = tl.load(RSIN + orow * RD + hr + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ca * va - sa * vb).to(CMPKV.dtype.element_ty),
                     mask=ok)
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty), mask=ok)
    else:
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)


@triton.jit
def _compressor_bigr_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D_ST: tl.constexpr,
    P: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    whi = tl.load(WHI + b)
    xbase = tl.load(X_BASE + b)

    rows = t * P + tl.arange(0, P)
    p_abs = wlo + rows
    ok = p_abs < whi
    tok = p_abs - sp

    h = tl.arange(0, BLK_H)
    p_safe = tl.where(p_abs >= 0, p_abs, 0)
    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx, mask=ok, other=0)
    base = blk_id[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D)
    j = p_abs % R

    for d0 in range(0, D, BLK_D_ST):
        d = d0 + tl.arange(0, BLK_D_ST)
        dmask = d < D
        sm = ok[:, None] & dmask[None, :]
        kv = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        scc = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xm = ok[:, None] & hmask[None, :]
            xv = tl.load(X + (xbase + tl.where(tok >= 0, tok, 0))[:, None] * H + hh[None, :],
                         mask=xm, other=0.0)
            wm = hmask[:, None] & dmask[None, :]
            kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            scc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
        scc = scc + tl.load(APE + j[:, None] * D + d[None, :], mask=sm, other=0.0)
        tl.store(STATE + base + d[None, :], kv, mask=sm)
        tl.store(STATE + base + D + d[None, :], scc, mask=sm)


def _as_int32(t, device):
    if t is None:
        return None
    return t.to(device=device, dtype=torch.int32).contiguous()


# ===========================================================================
# 三段式（架构对齐 vllm-ascend 官方 Triton 实现
#          `ops/triton/compressor/compressor_triton.py`：
#          `_proj_kernel` / `_pool_kernel` / `_cache_update_kernel`）
#
# 动机：单 kernel 里「`tl.dot` + `load(块表)→当地址→再 load` + 展开的行块循环」
#       会让 `bishengir-compile` SIGSEGV（见 `后端崩溃最小复现_r128.md`）。
#       官方写法的关键是：**`tl.dot` 只出现在 `_proj_kernel`；池化 kernel 是纯向量**，
#       连「段内/段前」的选择都是 `tl.where(is_seg, <vector load>, <vector load>)`
#       —— 两个操作数都是 vector 载入，天然不碰 `cc→cbuf` 缺陷（坑 26）。
# ===========================================================================
@triton.jit
def _bigr_proj_kernel(
    X, WKV, WGATE, APE, KVWS, SCWS, X_BASE, SP, SU,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, PT: tl.constexpr,
):
    """K1：纯投影 + ape。`kv = x@wkv^T`、`score = x@wgate^T + ape[p % r]` → fp32 workspace。"""
    b = tl.program_id(0)
    t0 = tl.program_id(1) * PT
    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    xbase = tl.load(X_BASE + b)

    t = t0 + tl.arange(0, PT)
    ok = t < su
    p = sp + t
    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)

    kv = tl.zeros((PT, BLK_D), dtype=tl.float32)
    sc = tl.zeros((PT, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = ok[:, None] & hmask[None, :]
        xv = tl.load(X + (xbase + t)[:, None] * H + hh[None, :], mask=xm, other=0.0)
        wm = hmask[:, None] & dmask[None, :]
        kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
        sc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))

    apev = tl.load(APE + (p % R)[:, None] * D + d[None, :],
                   mask=ok[:, None] & dmask[None, :], other=0.0)
    sc = sc + apev

    row = (xbase + t)[:, None] * D + d[None, :]
    sm = ok[:, None] & dmask[None, :]
    tl.store(KVWS + row, kv, mask=sm)
    tl.store(SCWS + row, sc, mask=sm)


@triton.jit
def _bigr_pool_kernel(
    KVWS, SCWS, STATE, BLKT, NORMW, RSIN, RCOS, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_D: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
    MIN_SCORE: tl.constexpr, HALF_MODE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    """K2：纯向量 online softmax + RMSNorm + RoPE（**无 `tl.dot`**）。"""
    b = tl.program_id(0)
    ct = tl.program_id(1)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    nc = tl.load(NC + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    ok = ct < nc
    c = c0 + ct
    sc_idx = ct

    d = tl.arange(0, BLK_D)
    dmask = d < D
    m = tl.arange(0, BM)

    run_max = tl.zeros((BLK_D,), dtype=tl.float32) + MIN_SCORE
    run_sum = tl.zeros((BLK_D,), dtype=tl.float32)
    run_acc = tl.zeros((BLK_D,), dtype=tl.float32)

    su_safe = tl.maximum(su - 1, 0)
    for m0 in range(0, R, BM):
        mm = m0 + m
        p = c * R + mm
        in_c = mm < R
        is_seg = in_c & (p >= sp) & (p < sp + su) & ok        # 段内 → 读 workspace
        is_cache = in_c & (p >= 0) & (p < sp) & ok            # 段前 → 读 state cache
        have = is_seg | is_cache

        # workspace 行号夹回 [0, su)：mask 之前后端也会先生成 DMA 地址
        # （官方注释："Ascend may form a DMA address for a masked lane before
        #   applying the load mask"）
        seg_off = tl.minimum(tl.maximum(p - sp, 0), su_safe)
        wrow = (xbase + seg_off)[:, None] * D + d[None, :]
        wm2 = is_seg[:, None] & dmask[None, :]

        p_safe = tl.where(p >= 0, p, 0)
        blk = tl.load(BLKT + tl.where(CACHE_CYCLE, b, b * MAX_BLK + p_safe // BLK_SIZE),
                      mask=is_cache, other=0)
        soff = blk[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
        cm2 = is_cache[:, None] & dmask[None, :]

        # ⚠️ 两个 mask 互斥 ⇒ 「两次带掩码的载入相加」与「select」语义完全等价，
        #    但**同时存活的 [BM, BLK_D] tile 从 4 个降到 2 个**（UB 192KB 很紧），
        #    也避免了「vector 载入 vs tl.dot 结果 select」那个会崩编译器的写法（坑 26）。
        kv = tl.load(KVWS + wrow, mask=wm2, other=0.0) \
            + tl.load(STATE + soff, mask=cm2, other=0.0)
        s = tl.load(SCWS + wrow, mask=wm2, other=0.0) \
            + tl.load(STATE + soff + D, mask=cm2, other=0.0)

        good = have[:, None] & dmask[None, :] & ok
        s = tl.where(good, s, MIN_SCORE)
        new_max = tl.maximum(run_max, tl.max(s, axis=0))
        alpha = tl.exp(run_max - new_max)
        e = tl.where(good, tl.exp(s - new_max[None, :]), 0.0)
        run_acc = run_acc * alpha + tl.sum(e * kv, axis=0)
        run_sum = run_sum * alpha + tl.sum(e, axis=0)
        run_max = new_max

    o = run_acc / tl.where(run_sum > 0.0, run_sum, 1.0)
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc_idx
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    ROFF: tl.constexpr = D - RD
    if HALF_MODE:
        hr: tl.constexpr = RD // 2
        for i in range(0, hr):
            va = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vb = tl.sum(tl.where(d == ROFF + hr + i, y, 0.0), axis=0)
            ca = tl.load(RCOS + orow * RD + i)
            sa = tl.load(RSIN + orow * RD + i)
            cb = tl.load(RCOS + orow * RD + hr + i)
            sb = tl.load(RSIN + orow * RD + hr + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ca * va - sa * vb).to(CMPKV.dtype.element_ty),
                     mask=ok)
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty), mask=ok)
    else:
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)


@triton.jit
def _bigr_cache_kernel(
    KVWS, SCWS, STATE, BLKT, X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    D: tl.constexpr, BLK_D: tl.constexpr, P: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    """K3：workspace → state_cache 的**纯拷贝**（无 dot）。

    必须在 K2 之后 launch（对齐官方"先读后写"约定；K2 读 `p < sp`，K3 写 `p >= wlo >= sp`，本不冲突）。
    """
    b = tl.program_id(0)
    t = tl.program_id(1)
    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    whi = tl.load(WHI + b)
    xbase = tl.load(X_BASE + b)

    p = wlo + t * P + tl.arange(0, P)
    ok = p < whi
    d = tl.arange(0, BLK_D)
    dmask = d < D

    seg_off = tl.minimum(tl.maximum(p - sp, 0), tl.maximum(whi - sp - 1, 0))
    wrow = (xbase + seg_off)[:, None] * D + d[None, :]
    sm = ok[:, None] & dmask[None, :]
    kv = tl.load(KVWS + wrow, mask=sm, other=0.0)
    scc = tl.load(SCWS + wrow, mask=sm, other=0.0)

    p_safe = tl.where(p >= 0, p, 0)
    blk = tl.load(BLKT + tl.where(CACHE_CYCLE, b, b * MAX_BLK + p_safe // BLK_SIZE),
                  mask=ok, other=0)
    soff = blk[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
    tl.store(STATE + soff, kv, mask=sm)
    tl.store(STATE + soff + D, scc, mask=sm)


def compressor_bigr(
    x, wkv, wgate, ape, norm_weight, rope_sin, rope_cos,
    state_cache=None, state_block_table=None, cu_seqlens=None, seqused=None,
    start_pos=None, rope_head_dim=64, cmp_ratio=128, coff=1, norm_eps=1e-6,
    rotary_mode=2, cache_mode=1, state_cache_stride_dim0=0, block_size=None,
    out_rows=None,
):
    """AscendC `compressor` 的 `coff == 1` 且 `cmp_ratio > 16` 路径（online softmax）。"""
    if int(coff) != 1:
        raise ValueError(f"compressor_bigr 只处理 coff==1，收到 coff={coff}")

    dev = x.device
    R = int(cmp_ratio)
    D = int(norm_weight.numel())
    RD = int(rope_head_dim)
    H = int(x.shape[-1])

    if R <= 16 or (R & (R - 1)) != 0:
        raise ValueError(f"compressor_bigr 要求 cmp_ratio 是 2 的幂且 > 16，实际 {R}")
    if wkv.shape[0] != D or wgate.shape[0] != D:
        raise ValueError(f"wkv/wgate 行数必须 == coff*headDim == headDim == {D}，"
                         f"实际 {wkv.shape[0]}/{wgate.shape[0]}")
    if wkv.shape[1] != H or wgate.shape[1] != H:
        raise ValueError(f"wkv/wgate 列数必须 == hidden == {H}")
    if tuple(ape.shape) != (R, D):
        raise ValueError(f"ape.shape={tuple(ape.shape)} 必须等于 [cmpRatio, coff*headDim]=[{R}, {D}]")
    if state_cache is not None and state_cache.shape[-1] != 2 * D:
        raise ValueError(f"stateCache.shape[-1]={state_cache.shape[-1]} 必须等于 2*coff*headDim={2 * D}")
    if RD % 2 != 0:
        raise ValueError(f"rope_head_dim 必须是偶数，实际 {RD}")
    HALF = RD // 2

    BLK_D = next_pow2(D)
    BM = 16
    BLK_H = max(16, min(128, 8192 // BLK_D))

    is_bs = x.dim() == 3
    if is_bs:
        B, S = int(x.shape[0]), int(x.shape[1])
        Sr = int(rope_sin.shape[1])
        x_base = [b_ * S for b_ in range(B)]
        lens = [S] * B
        out_shape = (B, Sr, D)
    else:
        T = int(x.shape[0])
        if cu_seqlens is None:
            raise ValueError("TH 布局（x 为 [T,H]）需要提供 cu_seqlens")
        B = int(cu_seqlens.numel()) - 1
        csl = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        lens = [int(csl[b_ + 1] - csl[b_]) for b_ in range(B)]
        x_base = [int(csl[b_]) for b_ in range(B)]
        Sr = int(rope_sin.shape[0]) if out_rows is None else int(out_rows)
        out_shape = (Sr, D)

    sp_l, su_l = [], []
    for b_ in range(B):
        sp = int(start_pos[b_].item()) if start_pos is not None else 0
        su = int(seqused[b_].item()) if seqused is not None else lens[b_]
        sp_l.append(sp)
        su_l.append(su)
    c0_l = [sp_l[b_] // R for b_ in range(B)]
    nc_l = [(sp_l[b_] + su_l[b_]) // R - c0_l[b_] for b_ in range(B)]
    ob_l = []
    acc = 0
    for b_ in range(B):
        if is_bs:
            ob_l.append(b_ * Sr)
        else:
            ob_l.append(acc)
            acc += nc_l[b_]
    max_nc = max(nc_l) if nc_l else 0

    out = torch.zeros(out_shape, dtype=x.dtype, device=dev)
    x_c = x.contiguous()
    wkv_c = wkv.contiguous()
    wgate_c = wgate.contiguous()
    ape_c = ape.contiguous().to(torch.float32)
    nw_c = norm_weight.contiguous().to(torch.float32)
    rs_c = rope_sin.contiguous().to(torch.float32)
    rc_c = rope_cos.contiguous().to(torch.float32)
    if rs_c.dim() == 3:
        rs_c = rs_c.reshape(-1, RD)
        rc_c = rc_c.reshape(-1, RD)

    has_state = state_cache is not None
    if has_state:
        sc_in = state_cache.to(torch.float32)
        sc_out = sc_in.clone()
        if block_size is None:
            block_size = int(sc_in.shape[1]) if sc_in.dim() >= 2 else int(sc_in.numel() // (2 * D))
    else:
        sc_out = None
        block_size = 0
    if has_state and state_block_table is None:
        raise ValueError("使用 state_cache 时必须提供 state_block_table")

    blkt = _as_int32(state_block_table, dev) if has_state else torch.zeros(1, dtype=torch.int32, device=dev)
    max_blk = int(state_block_table.shape[1]) if (state_block_table is not None
                                                  and state_block_table.dim() == 2) else 1

    dev_arrays = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    x_base_t = dev_arrays(x_base)
    sp_t = dev_arrays(sp_l)
    su_t = dev_arrays(su_l)
    c0_t = dev_arrays(c0_l)
    nc_t = dev_arrays(nc_l)
    ob_t = dev_arrays(ob_l)

    cache_cycle = int(cache_mode) == 2
    half_mode = int(rotary_mode) == 1

    if has_state:
        # ---- 三段式：K1 投影(带 dot) → K2 纯向量池化 → K3 纯拷贝写 state ----
        n_tok = B * int(x.shape[1]) if is_bs else int(x.shape[0])
        PT = 16
        kvws = torch.empty((n_tok, D), dtype=torch.float32, device=dev)
        scws = torch.empty((n_tok, D), dtype=torch.float32, device=dev)
        max_su = max(su_l) if su_l else 0
        if max_su > 0:
            _bigr_proj_kernel[(B, ceil_div(max_su, PT))](
                x_c, wkv_c, wgate_c, ape_c, kvws, scws, x_base_t, sp_t, su_t,
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D=BLK_D, PT=PT)
        if max_nc > 0:
            # pool kernel 无 tl.dot ⇒ BM 不必受 dot 的 M>=16 限制；
            # D=512 时用 BM=8 把 [BM, BLK_D] tile 减半，避免 UB overflow
            # （实测 BM=16/BLK_D=512 需要 345KB > UB 192KB）。
            BM_POOL = 8 if BLK_D >= 512 else max(16, BM)
            _bigr_pool_kernel[(B, max_nc)](
                kvws, scws, sc_out, blkt, nw_c, rs_c, rc_c, out,
                x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                D=D, R=R, RD=RD, BLK_D=BLK_D, BM=BM_POOL, EPS=float(norm_eps),
                MIN_SCORE=_MIN_SCORE, HALF_MODE=bool(half_mode), CACHE_CYCLE=bool(cache_cycle))

        wlo_l, whi_l = [], []
        for b_ in range(B):
            hi = sp_l[b_] + su_l[b_]
            lo = sp_l[b_]
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            wlo_l.append(lo)
            whi_l.append(max(hi, lo))
        max_w = max((whi_l[b_] - wlo_l[b_]) for b_ in range(B)) if B else 0
        if max_w > 0:
            # [P, BLK_D] tile + int64 地址向量的 UB 占用必须留在 192KB 以内。
            # 生产 shape（D=512 ⇒ BLK_D=512）下旧的 P=16 需要 1847296 bit ≈ 225KB，
            # BiShengIR 直接报 `error: ub overflow, requires 1847296 bits`
            # （compressor_bigr.py:421 的 _bigr_cache_kernel），所以这里按 BLK_D
            # 反比收缩 P（与上面 pool kernel 的 BM_POOL 同一个处理思路）。
            P = max(8, min(32, 2048 // BLK_D))
            _bigr_cache_kernel[(B, ceil_div(max_w, P))](
                kvws, scws, sc_out, blkt,
                x_base_t, sp_t, dev_arrays(wlo_l), dev_arrays(whi_l),
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                D=D, BLK_D=BLK_D, P=P, CACHE_CYCLE=bool(cache_cycle))
    elif max_nc > 0:
        _compressor_bigr_kernel[(B, max_nc)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, BM=BM, HALF=HALF,
            EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=False, CACHE_CYCLE=bool(cache_cycle),
        )

    return out, sc_out
