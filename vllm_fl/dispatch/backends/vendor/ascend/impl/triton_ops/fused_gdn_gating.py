"""fused_gdn_gating —— AscendC → Triton

来源：csrc/ascend/attention/fused_gdn_gating/
接口（op_host/fused_gdn_gating_def.cpp）：
    Input : a_log [H], a [B,H], b [B,H], dt_bias [H]
    Output: g [B,H] (fp32), beta_output [B,H] (同 a 的 dtype)
    Attr  : beta = 1.0, threshold = 20.0

公式（op_kernel/fused_gdn_gating.h L12-13 注释 + L244-311 实现）：

    x   = a.to(fp32) + dt_bias
    bx  = beta * x
    sp  = (1/beta) * log(1 + exp(min(bx, threshold)))
    # 仅当 bx <= threshold 时取 sp，否则取 x
    softplus = where(bx <= threshold, sp, x)
    g    = -exp(a_log) * softplus
    beta_output = sigmoid(b.to(fp32)) -> cast 回 InDtype
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["fused_gdn_gating"]


@triton.jit
def _fused_gdn_gating_kernel(
    A_LOG, A, B, DT_BIAS, G, BETA_OUT,
    H, n_rows,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLK_H: tl.constexpr,
    BLK_R: tl.constexpr,
    BETA_IS_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    h_off = tl.arange(0, BLK_H)
    hmask = h_off < H

    # 每个 program 处理 BLK_R 行（行 = batch），行内按 head 向量化
    rows = pid * BLK_R + tl.arange(0, BLK_R)

    a_log = tl.load(A_LOG + h_off, mask=hmask, other=0.0).to(tl.float32)
    dt_bias = tl.load(DT_BIAS + h_off, mask=hmask, other=0.0).to(tl.float32)
    neg_exp = -tl.exp(a_log)
    inv_beta = 1.0 / BETA

    # [BLK_R, BLK_H]
    offs = rows[:, None] * H + h_off[None, :]
    m = (rows[:, None] < n_rows) & hmask[None, :]

    a_v = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    b_v = tl.load(B + offs, mask=m, other=0.0).to(tl.float32)

    x = a_v + dt_bias[None, :]
    bx = BETA * x
    sp = tl.log(1.0 + tl.exp(tl.minimum(bx, THRESHOLD))) * inv_beta
    softplus = tl.where(bx <= THRESHOLD, sp, x)
    g = softplus * neg_exp[None, :]
    tl.store(G + offs, g, mask=m)

    beta_out = 1.0 / (1.0 + tl.exp(-b_v))
    if BETA_IS_FP32:
        tl.store(BETA_OUT + offs, beta_out, mask=m)
    else:
        tl.store(BETA_OUT + offs, beta_out.to(BETA_OUT.dtype.element_ty), mask=m)


def fused_gdn_gating(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 fused_gdn_gating。

    Args:
        a_log:  [H]     fp32
        a:      [B, H]  bf16/fp16
        b:      [B, H]  bf16/fp16
        dt_bias:[H]     bf16/fp16/fp32
        beta, threshold: 标量属性

    Returns:
        (g [B,H] fp32, beta_output [B,H] 同 a 的 dtype)
    """
    assert a.shape == b.shape, f"a/b 形状不一致: {a.shape} vs {b.shape}"
    assert a.dim() == 2, f"期望 a 是 [B,H]，实际 {tuple(a.shape)}"
    B, H = a.shape
    assert a_log.numel() == H, f"a_log 长度 {a_log.numel()} != H={H}"
    assert dt_bias.numel() == H, f"dt_bias 长度 {dt_bias.numel()} != H={H}"

    g = torch.empty((B, H), dtype=torch.float32, device=a.device)
    beta_out = torch.empty((B, H), dtype=a.dtype, device=a.device)

    BLK_H = next_pow2(H)
    BLK_R = 4
    grid = (ceil_div(B, BLK_R),)

    _fused_gdn_gating_kernel[grid](
        a_log.contiguous(), a.contiguous(), b.contiguous(), dt_bias.contiguous(),
        g, beta_out,
        H, B,
        BETA=float(beta),
        THRESHOLD=float(threshold),
        BLK_H=BLK_H,
        BLK_R=BLK_R,
        BETA_IS_FP32=bool(a.dtype == torch.float32),
    )
    return g, beta_out
