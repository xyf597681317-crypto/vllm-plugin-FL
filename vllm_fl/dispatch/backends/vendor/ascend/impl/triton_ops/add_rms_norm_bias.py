"""add_rms_norm_bias —— AscendC/native → Triton

来源：`_repos/vllm-ascend/csrc/moe/add_rms_norm_bias/`
调用点 schema（`csrc/torch_binding.cpp` L3250-3257）：
    npu_add_rms_norm_bias(Tensor x1, Tensor x2, Tensor gamma, Tensor? beta=None,
                          float epsilon=1e-6) -> (Tensor y, Tensor rstd, Tensor x)
接口（op_host/add_rms_norm_bias_def.cpp L22-64）：
    Input : x1 [.., N], x2 [.., N], gamma [N], beta [N] (OPTIONAL)
    Output: y [.., N]（同 x1 dtype）、rstd fp32 [.., 1]、x [.., N]（= x1+x2，同 dtype）
    Attr  : epsilon = 1e-6

**bias 加在哪里（Lead 特别要求核实，已读内核确认：加在 norm 之后）**：
    op_kernel/add_rms_norm_bias_merge_n.h L226-275 ComputeY：
        ComputeRstd(xLocal, rstdLocal, ...)          # xLocal = x1+x2，先算 rstd
        y = x * rstd                                 # L229-234（Brcb 广播 rstd）
        y = gamma * y                                # L245
        if (!nullptrBeta) Add(y, beta, y)            # L256-257 / L266-274  ← **norm 之后**
    op_kernel/add_rms_norm_bias_split_d.h L258-334 同序：
        Muls(x_fp32, x_fp32, rstdValue) → Mul(y, gamma, y) → Add(y, beta, y)   # L270/275/278
    即  y = (x1 + x2) * rstd * gamma + beta       （**不是** (x1+x2+beta)*rstd*gamma）
    rstd（op_kernel/..._merge_n.h L195-221）：
        rstd = 1 / sqrt( Σ_j x[i,j]^2 / N + epsilon )   # 先 Adds(epsilon) 再 Sqrt 再取倒数
    第三个输出 x = x1 + x2（L226-… 里 xLocal 就是 x1+x2，并由主流程拷出）

语义假设：
  * gamma/beta 广播在最后一维；beta 为 None 时跳过（tiling 的 nullptr_beta）；
  * x1/x2/gamma/beta 同 dtype（def 里三种 dtype 组合一致）；
  * rstd 是 fp32，且只依赖 x1+x2（不含 beta）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["add_rms_norm_bias"]


@triton.jit
def _add_rms_norm_bias_kernel(
    X1, X2, GAMMA, BETA, Y, RSTD, XOUT,
    ROWS, N, EPS,
    HAS_BETA: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLK_N: tl.constexpr,
    NCH: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= ROWS:
        return
    base = row * N

    # 趟 1：x = x1 + x2 → 写出第三个输出，并累积 Σ x²
    ss = tl.zeros([BLK_N], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        a = tl.load(X1 + base + cc, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X2 + base + cc, mask=m, other=0.0).to(tl.float32)
        xv = a + b
        if OUT_FP32:
            tl.store(XOUT + base + cc, xv, mask=m)
        else:
            tl.store(XOUT + base + cc, xv.to(XOUT.dtype.element_ty), mask=m)
        ss += xv * xv
    rstd = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / N + EPS)
    tl.store(RSTD + row, rstd)

    # 趟 2：y = x * rstd * gamma (+ beta) —— beta 在 norm 之后（见模块 docstring）
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        a = tl.load(X1 + base + cc, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X2 + base + cc, mask=m, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + cc, mask=m, other=0.0).to(tl.float32)
        y = (a + b) * rstd * g
        if HAS_BETA:
            bt = tl.load(BETA + cc, mask=m, other=0.0).to(tl.float32)
            y = y + bt
        if OUT_FP32:
            tl.store(Y + base + cc, y, mask=m)
        else:
            tl.store(Y + base + cc, y.to(Y.dtype.element_ty), mask=m)


def add_rms_norm_bias(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None = None,
    epsilon: float = 1e-6,
):
    """等价于 vllm-ascend 的 npu_add_rms_norm_bias。

    Args:
        x1, x2: [.., N] 同 dtype
        gamma: [N] 同 dtype
        beta: [N] 可选，同 dtype（**加在 norm 之后**）
        epsilon: 属性，默认 1e-6

    Returns:
        (y 同 x1 形状/dtype, rstd fp32 [.., 1], x = x1 + x2 同 x1)
    """
    assert x1.shape == x2.shape, f"x1/x2 形状不一致: {tuple(x1.shape)} vs {tuple(x2.shape)}"
    assert x1.dtype == x2.dtype == gamma.dtype, "x1/x2/gamma dtype 必须一致"
    N = int(x1.shape[-1])
    rows = 1
    for v in x1.shape[:-1]:
        rows *= int(v)
    assert gamma.numel() == N, f"gamma 长度 {gamma.numel()} != N={N}"
    if beta is not None:
        assert beta.numel() == N and beta.dtype == x1.dtype, "beta 形状/dtype 不匹配"

    dev = x1.device
    a = x1.contiguous()
    b = x2.contiguous()
    g = gamma.contiguous()
    y = torch.empty_like(a)
    xout = torch.empty_like(a)
    rstd = torch.empty(tuple(x1.shape[:-1]) + (1,), dtype=torch.float32, device=dev)

    blk_n = min(next_pow2(N), 1024)
    _add_rms_norm_bias_kernel[(rows,)](
        a.view(rows, N), b.view(rows, N), g,
        beta.contiguous() if beta is not None else g,
        y.view(rows, N), rstd.view(rows), xout.view(rows, N),
        rows, N, float(epsilon),
        HAS_BETA=beta is not None,
        OUT_FP32=(a.dtype == torch.float32),
        BLK_N=blk_n, NCH=ceil_div(N, blk_n),
    )
    return y, rstd, xout
