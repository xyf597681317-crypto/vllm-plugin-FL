"""hc_pre_sinkhorn —— AscendC → Triton

来源：csrc/ascend/moe/hc_pre_sinkhorn/
接口（op_host/hc_pre_sinkhorn_def.cpp L24-67 + _proto.cpp L22-90 + _tiling.cpp L81-112）：
    Input : mixes    [B, S, hc_mix] 或 [B*S, hc_mix]  fp32
            rsqrt    [B*S]                            fp32
            hc_scale [3]                              fp32
            hc_base  [hc_mix]                         fp32（tiling L106-109：长度必须等于 hc_mix）
            x        [B, S, hc_mult, d] 或 [B*S, hc_mult, d]  bf16
    Output: y        [B, S, d] / [B*S, d]            同 x 的 dtype
            post     [B, S, hc_mult] / [B*S, hc_mult]        fp32
            comb_frag[B, S, hc_mult, hc_mult]                fp32
    Attr  : hc_mult = 4, hc_sinkhorn_iters = 20, hc_eps = 1e-6

    布局：mixes 每行 = [pre 段 hc_mult | post 段 hc_mult | comb 段 hc_mult*hc_mult]
          ==> hc_mix = 2*hc_mult + hc_mult^2（hc_base 长度也是 hc_mix）
          （perf 内核用 CopyInWithOuterFor 逐行读 comb 段，hm=1 时两种解释一致；
            这里采用与 infershape/hc_base 长度约束自洽的行主序 [hc_mult, hc_mult] 块）

公式（op_kernel/hc_pre_sinkhorn_perf.h L95-202 Process + hc_pre_sinkhorn_base.h）：
    hm = hc_mult, eps = hc_eps
    # 1) pre（mixes 第 0 段 hm 个）：
    #    L120-121 ProcessPre -> L387-399：sigmoid(mix*rsqrt*scale[0] + base[0:hm]) + eps
    pre[i,j] = sigmoid(mixes[i, j] * rsqrt[i] * hc_scale[0] + hc_base[j]) + eps
    # 2) y（L132-141 ProcessY -> L462-472）：
    y[i,:]   = sum_j pre[i,j] * x[i,j,:]                # 累加在 fp32，CAST_RINT 回 bf16
    # 3) post（mixes 第 1 段；L145-154 -> L475-488 ProcessPost）：
    post[i,j] = 2 * sigmoid(mixes[i, hm+j] * rsqrt[i] * hc_scale[1] + hc_base[hm+j])
    # 4) comb_frag（mixes 第 2 段 [hm,hm]；L158-192）：
    A[i,j,k] = mixes[i, 2hm + j*hm + k] * rsqrt[i] * hc_scale[2] + hc_base[2hm + j*hm + k]
    A = softmax_over_k(A) + eps                          # L174 SoftmaxFP32Perf（末维 softmax）
    A = A / (sum_j A[i,j,k] + eps)                       # L176-180 按 dim1 归一
    for it in range(hc_sinkhorn_iters - 1):              # L181-192
        A = A / (sum_k A[i,j,k] + eps)                   #   L182-186 按 dim2 归一
        A = A / (sum_j A[i,j,k] + eps)                   #   L187-191 按 dim1 归一
    comb_frag = A

    - 所有中间量都在 fp32；y 输出取整回 x 的 dtype
    - ReduceSumARAPerf（base.h L402-441）= 对 dim1 求和；LastDimReduceSumPerf（L499-505）= 对末维求和
    - 对齐填充 lanes 不参与求和（dim1/dim2 分别取真实 hm），本实现用 mask 保持一致
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre_sinkhorn"]


@triton.jit
def _hc_pre_sinkhorn_kernel(
    MIX, RSQRT, SCALE, BASE, X, Y, POST, COMB,
    ROWS, HMIX, HM, D, EPS,
    ITERS: tl.constexpr,
    ITERS_P2: tl.constexpr,
    BLK_R: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_HM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS

    rs = tl.load(RSQRT + rows, mask=rmask, other=1.0)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)

    jj = tl.arange(0, BLK_HM)
    jmask = jj < HM
    m2d = rmask[:, None] & jmask[None, :]

    # ---------- 1) pre + 2) y = sum_j pre[i,j] * x[i,j,:] ----------
    # pre[i,j] = sigmoid(mixes[i,j]*rsqrt[i]*scale[0] + base[j]) + eps
    dcur = 0
    while dcur < D:
        doff = dcur + tl.arange(0, BLK_D)
        dmask = doff < D
        acc = tl.zeros((BLK_R, BLK_D), dtype=tl.float32)
        j = 0
        while j < HM:
            mj = tl.load(MIX + rows * HMIX + j, mask=rmask, other=0.0)
            bj = tl.load(BASE + j)
            pj = 1.0 / (1.0 + tl.exp(-(mj * rs * s0 + bj))) + EPS
            xv = tl.load(
                X + (rows[:, None] * HM + j) * D + doff[None, :],
                mask=rmask[:, None] & dmask[None, :], other=0.0,
            ).to(tl.float32)
            acc += pj[:, None] * xv
            j += 1
        tl.store(Y + rows[:, None] * D + doff[None, :], acc.to(Y.dtype.element_ty),
                 mask=rmask[:, None] & dmask[None, :])
        dcur += BLK_D

    # ---------- 3) post ----------
    b1 = tl.load(BASE + HM + jj, mask=jmask, other=0.0)
    m1 = tl.load(MIX + rows[:, None] * HMIX + (HM + jj)[None, :], mask=m2d, other=0.0)
    postv = 2.0 / (1.0 + tl.exp(-(m1 * rs[:, None] * s1 + b1[None, :])))
    tl.store(POST + rows[:, None] * HM + jj[None, :], postv, mask=m2d)

    # ---------- 4) comb_frag (sinkhorn) ----------
    m3 = rmask[:, None, None] & jmask[None, :, None] & jmask[None, None, :]
    off3 = (2 * HM + jj[None, :, None] * HM + jj[None, None, :])
    m2 = tl.load(MIX + rows[:, None, None] * HMIX + off3, mask=m3, other=0.0)
    b2 = tl.load(BASE + 2 * HM + jj[:, None] * HM + jj[None, :],
                 mask=(jj[:, None] < HM) & (jj[None, :] < HM), other=0.0)
    a = m2 * rs[:, None, None] * s2 + b2[None, :, :]
    a = tl.where(m3, a, -1.0e30)

    mx = tl.max(a, axis=2)                                   # [BLK_R, BLK_HM]
    e = tl.exp(a - mx[:, :, None])
    e = tl.where(m3, e, 0.0)
    ssum = tl.sum(e, axis=2)                                 # softmax 分母（末维）
    A = e / ssum[:, :, None] + EPS

    A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + EPS)

    # 静态展开：Ascend 后端对「动态循环里携带 3D tensor + masked reduce」会编译失败。
    # ⚠️ 展开次数必须取 **2 的幂**（`ITERS_P2 = next_pow2(ITERS-1)`），
    #    多余迭代用 `if _it < ITERS - 1:` 跳过。实测同族证据：`hc_pre` 的 mixes kernel
    #    在 unroll 次数 = 28（非 2 的幂）时误差 5.388，而 20/24/32 都是 4e-7。
    #    sinkhorn 默认 ITERS=20 ⇒ 原先是 19 次（非 2 的幂）。
    for _it in range(ITERS_P2):
        if _it < ITERS - 1:
            A = A / (tl.sum(tl.where(jmask[None, None, :], A, 0.0), axis=2)[:, :, None] + EPS)
            A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + EPS)

    tl.store(COMB + rows[:, None, None] * (HM * HM) + jj[None, :, None] * HM + jj[None, None, :],
             A, mask=m3)


def hc_pre_sinkhorn(
    mixes: torch.Tensor,
    rsqrt: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    x: torch.Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 HcPreSinkhorn。

    Args:
        mixes: [B, S, 3*hc_mult] 或 [B*S, 3*hc_mult] fp32
        rsqrt: [B*S] fp32
        hc_scale: [3] fp32
        hc_base: [3*hc_mult + hc_mult*hc_mult] fp32
        x: [B, S, hc_mult, d] 或 [B*S, hc_mult, d] bf16
        属性：hc_mult / hc_sinkhorn_iters / hc_eps

    Returns:
        (y 同 x dtype [B,S,d]，post fp32 [B,S,hc_mult]，comb_frag fp32 [B,S,hc_mult,hc_mult])
    """
    assert mixes.dtype == torch.float32, f"mixes 必须是 fp32，实际 {mixes.dtype}"
    assert x.dtype == torch.bfloat16, f"x 必须是 bf16，实际 {x.dtype}"

    hm = int(hc_mult)
    lead = tuple(int(v) for v in mixes.shape[:-1])
    hmix = int(mixes.shape[-1])
    want_mix = 2 * hm + hm * hm
    assert hmix == want_mix, f"mixes 末维 {hmix} != 2*hc_mult + hc_mult^2 = {want_mix}"
    assert x.dim() == len(lead) + 2, "x 维度与 mixes 不匹配"
    assert tuple(int(v) for v in x.shape[:-2]) == lead, \
        f"x 前导维 {tuple(x.shape[:-2])} != mixes 前导维 {lead}"
    assert int(x.shape[-2]) == hm, f"x 的 hc 维 {x.shape[-2]} != hc_mult={hm}"
    d = int(x.shape[-1])
    rows = 1
    for v in lead:
        rows *= v

    assert int(rsqrt.numel()) == rows, f"rsqrt 长度 {rsqrt.numel()} != rows={rows}"
    assert int(hc_scale.numel()) == 3, f"hc_scale 长度 {hc_scale.numel()} != 3"
    assert int(hc_base.numel()) == hmix, \
        f"hc_base 长度 {hc_base.numel()} != hc_mix={hmix}（tiling L106-109 要求相等）"

    mix = mixes.contiguous().view(rows, hmix)
    rq = rsqrt.contiguous().view(rows).to(torch.float32)
    sc = hc_scale.contiguous().view(3).to(torch.float32)
    bs = hc_base.contiguous().view(-1).to(torch.float32)
    xv = x.contiguous().view(rows, hm, d)

    y = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    post = torch.empty((rows, hm), dtype=torch.float32, device=x.device)
    comb = torch.empty((rows, hm, hm), dtype=torch.float32, device=x.device)

    BLK_R = 4
    BLK_D = min(next_pow2(d), 256)
    BLK_HM = max(2, next_pow2(hm))
    iters = int(hc_sinkhorn_iters)
    grid = (ceil_div(rows, BLK_R),)
    _hc_pre_sinkhorn_kernel[grid](
        mix, rq, sc, bs, xv, y, post, comb,
        rows, hmix, hm, d, float(hc_eps),
        ITERS=iters,
        ITERS_P2=next_pow2(max(1, iters - 1)),
        BLK_R=BLK_R, BLK_D=BLK_D, BLK_HM=BLK_HM,
    )
    return (y.view(*lead, d), post.view(*lead, hm), comb.view(*lead, hm, hm))
