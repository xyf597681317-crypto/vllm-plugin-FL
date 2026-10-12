"""gemma_rms_norm —— AscendC/native → Triton

来源：`_repos/vllm-ascend/csrc/torch_binding.cpp` L2881-2887（C++ 层实现，无独立 op_kernel）
调用点 schema：
    npu_gemma_rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor y, Tensor rstd)
公式参考（`_repos/FlagGems/src/flag_gems/ops/gemma_rms_norm.py` L40-62 逐行对应）：
    cdtype = fp32（x 为 fp16/bf16 时）
    x    = load(x).to(cdtype)                                   # L53
    var  = Σ x^2 / N                                            # L55
    rrms = 1 / sqrt(var + eps)                                  # L56
    w    = load(gamma).to(fp32)                                 # L60
    y    = (x * rrms * (1.0 + w)).to(x.dtype)                   # L61 ← Gemma 特有：weight + 1
    rstd = rrms                                                 # L62（fp32）

要点：
  * gamma 是 **1.0 + gamma**（Gemma 的 unit-offset 约定），且加法在 fp32 里做；
  * 第二个输出是 **rrms**（不是 rstd 的倒数形式），即 1/sqrt(mean(x²)+eps)；
  * x 为 fp16/bf16 时中间量全走 fp32，输出回写原 dtype。

Ascend 后端规避（hc_pre 实测踩坑，见 hc_pre.py docstring）：
  * 动态 `while` 循环 + 多行 2-D 偏移会被算错 → 本文件一行一 program（行号是标量）、
    分块循环用 `tl.static_range`（块数 host 侧算好当 constexpr 传）；
  * 不用 `tl.dot`；不用 2-D 掩码（只用一维掩码）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["gemma_rms_norm"]


@triton.jit
def _gemma_rms_norm_kernel(
    X, W, Y, RSTD,
    ROWS, N,
    EPS,
    BLK_N: tl.constexpr,
    NCH: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= ROWS:
        return
    base = row * N

    # 趟 1：Σ x² / N
    ss = tl.zeros([BLK_N], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        xv = tl.load(X + base + cc, mask=cc < N, other=0.0).to(tl.float32)
        ss += xv * xv
    var = tl.sum(ss, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + EPS)
    tl.store(RSTD + row, rrms)

    # 趟 2：y = x * rrms * (1 + gamma)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        xv = tl.load(X + base + cc, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cc, mask=m, other=0.0).to(tl.float32)
        y = xv * rrms * (1.0 + w)
        if OUT_FP32:
            tl.store(Y + base + cc, y, mask=m)
        else:
            tl.store(Y + base + cc, y.to(Y.dtype.element_ty), mask=m)


def gemma_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
):
    """等价于 vllm-ascend 的 npu_gemma_rms_norm。

    Args:
        x: [..., N]（内部按最后两维展平成 [rows, N]）
        gamma: [N]（Gemma 约定：实际权重是 1 + gamma）
        epsilon: 属性，默认 1e-6

    Returns:
        (y 同 x 形状与 dtype, rstd fp32 [.., 1])
    """
    assert x.dim() >= 2, f"x 至少 2D，实际 {tuple(x.shape)}"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"x dtype {x.dtype} 不支持"
    N = int(x.shape[-1])
    rows = 1
    for v in x.shape[:-1]:
        rows *= int(v)
    assert gamma.numel() == N, f"gamma 长度 {gamma.numel()} != N={N}"
    dev = x.device
    xc = x.contiguous()
    g = gamma.contiguous()
    y = torch.empty_like(xc)
    rstd = torch.empty(tuple(x.shape[:-1]) + (1,), dtype=torch.float32, device=dev)

    blk_n = min(next_pow2(N), 1024)
    nch = ceil_div(N, blk_n)
    _gemma_rms_norm_kernel[(rows,)](
        xc.view(rows, N), g, y.view(rows, N), rstd.view(rows),
        rows, N, float(epsilon),
        BLK_N=blk_n, NCH=nch,
        OUT_FP32=(xc.dtype == torch.float32),
    )
    return y, rstd
