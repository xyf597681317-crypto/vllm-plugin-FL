"""hc_pre —— AscendC → Triton

来源：csrc/ascend/moe/hc_pre/
接口（op_host/hc_pre_def.cpp L24-63）：
    Input : x        [bs, hc_mult, d] 或 [b, s, hc_mult, d]   bf16
            hc_fn    [hc_mix, hc_mult*d]                      fp32
            hc_scale [3]                                      fp32
            hc_base  [hc_mix]                                 fp32
    Output: y         [bs, d]                        bf16
            post      [bs, hc_mult]                  fp32
            comb_frag [bs, hc_mult, hc_mult]         fp32
    Attr  : hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6
    hc_mix = 2*hc_mult + hc_mult^2（tiling.cpp L116-132 约束 hc_fn.dim1 == d*hc_mult、
    hc_base 长度 == hc_mix；L278 k = hcMult*d）

**与 hc_pre_sinkhorn / hc_pre_inv_rms 的关系**（独立实现，未 import）：
    hc_pre = hc_pre_inv_rms(x; norm_eps)          → rsqrt[bs]
           + cube(x @ hc_fn^T)                    → mixes[bs, hc_mix]
           + hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x; hc_eps, iters)
    依据：cube 算 mixes（hc_pre_cube_compute.h，k=hcMult*d）；向量侧先算 inv_rms
    （hc_pre_m_k_split_core.h L366-368：Adds(normEps) → Sqrt），再把 rsqrt 乘到 mixes 上
    （hc_pre_base.h L428-439 ProcessPre / L519-527 ProcessPost / comb 段）。

公式（op_kernel/hc_pre_base.h）：
    K = hc_mult * d
    rsqrt[i]    = 1 / sqrt( Σ_k x[i,k]^2 / K + norm_eps )
    mixes[i, n] = Σ_k x_flat[i,k] * hc_fn[n,k]
    pre[i,j]    = sigmoid( mixes[i,j]    * rsqrt[i] * scale[0] + base[j] ) + hc_eps
    y[i,:]      = Σ_j pre[i,j] * x[i,j,:]（fp32 累加，回写 bf16）
    post[i,j]   = 2 * sigmoid( mixes[i,hm+j] * rsqrt[i] * scale[1] + base[hm+j] )
    A[i,j,k]    = mixes[i,2hm+j*hm+k] * rsqrt[i] * scale[2] + base[2hm+j*hm+k]
    A           = softmax_k(A) + hc_eps ;  A = A / (Σ_j A + hc_eps)
    iters-1 次：A = A/(Σ_k A + hc_eps) ; A = A/(Σ_j A + hc_eps)
    comb_frag   = A

Ascend 后端实测坑（本文件据此设计）：
  1. `tl.dot` 直接崩（BiShengIR/LLVM crash，fp32/bf16 都崩）→ 用一维广播归约。
  2. **动态 `while` 循环 + 多行 2-D 偏移会被算错**（探针实测：16 行里 15 行错、行 0 对；
     把循环换成 `tl.static_range` 后 0 错）→ 本文件所有循环都用 static_range，
     循环次数由 host 侧按 shape 算好当 constexpr 传入。
  3. 2-D 掩码（`rmask[:,None] & km[None,:]`）不可靠 → host 侧零填充消除掩码。
  4. 循环里不用循环变量做指针偏移（列号用 program_id(1) 传）。

语义假设：
  * hc_mult 仅支持 2 的幂（comb 矩阵需稠密，避免 2-D 掩码）。
  * y 回写 bf16（round-to-nearest）；post/comb_frag 保持 fp32。
  * 4D 输入按前两维展平成 bs 处理。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre"]


@triton.jit
def _hc_pre_rsqrt_kernel(
    X, RSQRT,
    BS, K, KS,
    NORM_EPS,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    KCH: tl.constexpr,
):
    """inv_rms：rsqrt[i] = 1/sqrt(Σ_k x^2 / K + norm_eps)。X 已零填充（行 stride=KS=KCH*BLK_K）。"""
    pid = tl.program_id(0)
    rows = pid * BLK_M + tl.arange(0, BLK_M)
    ss = tl.zeros([BLK_M], dtype=tl.float32)
    for i in tl.static_range(KCH):
        kk = i * BLK_K + tl.arange(0, BLK_K)
        xv = tl.load(X + rows[:, None] * KS + kk[None, :]).to(tl.float32)
        ss += tl.sum(xv * xv, axis=1)
    tl.store(RSQRT + rows, 1.0 / tl.sqrt(ss / K + NORM_EPS), mask=rows < BS)


@triton.jit
def _hc_pre_mixes_kernel(
    X, FN, MIXES,
    BS, K, KS, HMIX,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    KCH: tl.constexpr,
    KCH_P2: tl.constexpr,
):
    """mixes[:, n] = Σ_k x_flat[:, k] * hc_fn[n, k]（program_id(1) 给列号 n）。

    X/FN 的行 stride 都是 KS = KCH*BLK_K（已补零）——**必须用 KS 而不是 K**，
    否则最后一块会跨行读到邻行数据（实测 d 非 BLK_K 倍数时全错）。

    ⚠️ **unroll 次数必须是 2 的幂**（`KCH_P2 = next_pow2(KCH)`，超出 `KCH` 的迭代跳过）：
    实测 Ascend 后端在「静态 unroll 次数不是 2 的幂」时会算错 —— 同一 kernel，
    `hm*d = 4*7168 = 28672` ⇒ `KCH=28` 时 **mixes 误差 5.388**（并直接污染 y/post/comb），
    而 `KCH ∈ {20, 24, 32}`（d=5120/6144/8192）误差都是 4e-7。
    这正是生产 `d=7168` 出错、而 `d=4096` 正常的根因。
    """
    n = tl.program_id(1)
    rows = tl.program_id(0) * BLK_M + tl.arange(0, BLK_M)
    accn = tl.zeros([BLK_M], dtype=tl.float32)
    for i in range(KCH_P2):
        if i < KCH:
            kk = i * BLK_K + tl.arange(0, BLK_K)
            a = tl.load(X + rows[:, None] * KS + kk[None, :]).to(tl.float32)
            bv = tl.load(FN + n * KS + kk)
            accn += tl.sum(a * bv[None, :], axis=1)
    tl.store(MIXES + rows * HMIX + n, accn, mask=rows < BS)


@triton.jit
def _hc_pre_sinkhorn_kernel(
    MIXES, RSQRT, SCALE, BASE, X, Y, POST, COMB,
    BS, HMIX, HC_EPS,
    HM: tl.constexpr,
    ITERS: tl.constexpr,
    BLK_D: tl.constexpr,
    DCH: tl.constexpr,
):
    # d 维已由 host 侧补零到 DCH*BLK_D → kernel 内无任何掩码
    row = tl.program_id(0)
    if row >= BS:
        return
    rq = tl.load(RSQRT + row)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    jj = tl.arange(0, HM)
    D = DCH * BLK_D

    # --- pre 与 y = Σ_j pre[j] * x[row,j,:] ---
    for i in tl.static_range(DCH):
        dd = i * BLK_D + tl.arange(0, BLK_D)
        yacc = tl.zeros([BLK_D], dtype=tl.float32)
        for j in tl.static_range(HM):
            mj = tl.load(MIXES + row * HMIX + j)
            bj = tl.load(BASE + j)
            pj = 1.0 / (1.0 + tl.exp(0.0 - (mj * rq * s0 + bj))) + HC_EPS
            xv = tl.load(X + (row * HM + j) * D + dd).to(tl.float32)
            yacc += pj * xv
        tl.store(Y + row * D + dd, yacc.to(Y.dtype.element_ty))

    # --- post = 2*sigmoid(mix*rsqrt*scale1 + base1) ---
    b1 = tl.load(BASE + HM + jj)
    m1 = tl.load(MIXES + row * HMIX + (HM + jj))
    post = 2.0 / (1.0 + tl.exp(0.0 - (m1 * rq * s1 + b1)))
    tl.store(POST + row * HM + jj, post)

    # --- comb_frag：softmax 后交替行列归一（Sinkhorn）---
    a = tl.load(MIXES + row * HMIX + (2 * HM + jj[:, None] * HM + jj[None, :]))
    b2 = tl.load(BASE + 2 * HM + jj[:, None] * HM + jj[None, :])
    a = a * rq * s2 + b2
    e = tl.exp(a - tl.max(a, axis=1)[:, None])
    A = e / tl.sum(e, axis=1)[:, None] + HC_EPS
    A = A / (tl.sum(A, axis=0)[None, :] + HC_EPS)
    for _ in tl.static_range(ITERS - 1):
        A = A / (tl.sum(A, axis=1)[:, None] + HC_EPS)
        A = A / (tl.sum(A, axis=0)[None, :] + HC_EPS)
    tl.store(COMB + row * (HM * HM) + jj[:, None] * HM + jj[None, :], A)


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
):
    """等价于 AscendC 的 HcPre。

    Args:
        x: [bs, hc_mult, d] bf16（或 [b, s, hc_mult, d]，前两维展平）
        hc_fn: [hc_mix, hc_mult*d] fp32
        hc_scale: [3] fp32
        hc_base: [hc_mix] fp32
        hc_mult / hc_sinkhorn_iters / hc_eps / norm_eps: 属性

    Returns:
        (y [bs,d] bf16, post [bs,hc_mult] fp32, comb_frag [bs,hc_mult,hc_mult] fp32)
    """
    assert x.dim() in (3, 4), f"x 应为 3D/4D，实际 {tuple(x.shape)}"
    assert x.dtype == torch.bfloat16, f"x 应为 bf16，实际 {x.dtype}"
    hm = int(hc_mult)
    assert hm in (1, 2, 4, 8, 16), \
        f"hc_mult={hm} 本实现仅支持 2 的幂（comb 需稠密矩阵以规避后端 2-D 掩码问题）"
    lead = tuple(int(v) for v in x.shape[:-2])
    bs = 1
    for v in lead:
        bs *= v
    d = int(x.shape[-1])
    assert int(x.shape[-2]) == hm, f"x 的 hc 维 {x.shape[-2]} != hc_mult={hm}"
    hmix = 2 * hm + hm * hm
    assert tuple(hc_fn.shape) == (hmix, hm * d), \
        f"hc_fn 应为 [{hmix},{hm * d}]，实际 {tuple(hc_fn.shape)}"
    assert hc_scale.numel() == 3 and hc_base.numel() == hmix
    K = hm * d

    dev = x.device
    xc = x.contiguous().view(bs, hm, d)
    fn = hc_fn.contiguous().float()
    mixes = torch.empty((bs, hmix), dtype=torch.float32, device=dev)
    rsqrt = torch.empty((bs,), dtype=torch.float32, device=dev)
    y = torch.empty((bs, d), dtype=x.dtype, device=dev)
    post = torch.empty((bs, hm), dtype=torch.float32, device=dev)
    comb = torch.empty((bs, hm, hm), dtype=torch.float32, device=dev)

    blk_m = 16
    blk_k = 64
    # `_hc_pre_rsqrt_kernel` / `_hc_pre_mixes_kernel` unroll the K loop at compile
    # time (`tl.static_range(KCH)`), so KCH drives the build time directly.  DeepSeek-V4
    # uses hm=4, d=4096 -> K=16384, i.e. KCH=256 with BLK_K=64, which made the Ascend
    # compiler run for tens of minutes on the first call.  Grow BLK_K until the loop
    # needs at most 32 blocks; small shapes (the tested ones) keep BLK_K=64 untouched.
    while ceil_div(K, blk_k) > 32:
        blk_k *= 2
    kch = ceil_div(K, blk_k)
    # host 侧零填充行与 K（消除 kernel 内 2-D 掩码）；循环次数 kch 当 constexpr 传入
    bs_pad = ceil_div(bs, blk_m) * blk_m
    k_pad = kch * blk_k
    if bs_pad != bs or k_pad != K:
        xpad = torch.zeros((bs_pad, k_pad), dtype=x.dtype, device=dev)
        xpad[:bs, :K] = xc.view(bs, K)
        fnpad = torch.zeros((hmix, k_pad), dtype=torch.float32, device=dev)
        fnpad[:, :K] = fn
    else:
        xpad, fnpad = xc.view(bs, K), fn
    _hc_pre_rsqrt_kernel[(bs_pad // blk_m,)](
        xpad, rsqrt, bs, K, k_pad, float(norm_eps),
        BLK_M=blk_m, BLK_K=blk_k, KCH=kch,
    )
    _hc_pre_mixes_kernel[(bs_pad // blk_m, hmix)](
        xpad, fnpad, mixes, bs, K, k_pad, hmix,
        BLK_M=blk_m, BLK_K=blk_k, KCH=kch, KCH_P2=next_pow2(kch),
    )

    # Ascend UB=192KB：一行一 program + d 分块；d 维 host 侧补零到 DCH*BLK_D（kernel 内零掩码）
    blk_d = 128 if d > 128 else max(2, next_pow2(d))
    # Same reasoning as BLK_K above: the d loop and its inner HM loop are unrolled,
    # so cap DCH at 8 blocks (=32 unrolled iterations for hm=4).
    while ceil_div(d, blk_d) > 8:
        blk_d *= 2
    dch = ceil_div(d, blk_d)
    d_pad = dch * blk_d
    if d_pad != d:
        xpad2 = torch.zeros((bs, hm, d_pad), dtype=x.dtype, device=dev)
        xpad2[:, :, :d] = xc
        ypad = torch.empty((bs, d_pad), dtype=x.dtype, device=dev)
    else:
        xpad2, ypad = xc, y
    _hc_pre_sinkhorn_kernel[(bs,)](
        mixes, rsqrt, hc_scale.contiguous().view(3), hc_base.contiguous().view(-1),
        xpad2, ypad, post, comb,
        bs, hmix, float(hc_eps),
        HM=hm,
        ITERS=int(hc_sinkhorn_iters),
        BLK_D=blk_d,
        DCH=dch,
    )
    if d_pad != d:
        y[:, :] = ypad[:, :d].reshape(bs, d)
    return (y.view(*lead, d), post.view(*lead, hm), comb.view(*lead, hm, hm))
