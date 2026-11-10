# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of RMS normalization (CANN `RmsNorm` / `AddRmsNormBias`).

Two kernel shapes are used, picked by the host from the layout:

* `_rms_norm_2d_kernel` / `_add_rms_norm_2d_kernel` treat `BT` consecutive rows
  as one flat 1-D tile, reshape it to `[BT, d]` *inside* the kernel and reduce
  along the row axis.  The load and the store stay flat 1-D accesses at a scalar
  base, so the whole tile is one dense transaction instead of `BT` serialised
  ones; measured 0.153 ms for `rms_norm` and 0.180 ms for `add_rms_norm` at
  T=4096, d=2048 on the 105 host, against 0.406 / 0.479 ms for the row loop.
* `_rms_norm_row_kernel` / `_add_rms_norm_row_kernel` handle every other layout
  (non-power-of-two width, or a row count that is not a multiple of `BT`) with
  one row per step and a column-chunk loop.

`add_rms_norm` is a single kernel: it writes the residual sum and the
normalized value in one pass, where the previous version needed an add launch
followed by a normalization launch that re-read the sum.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Column tile of the row kernels.
_BLK_N = 2048
# Rows per program in the 2-D kernels. _TILE_LANES caps BT*d so the fp32 tiles
# stay inside UB: 8192 lanes is 32 KiB per live value.
_BT_2D = 4
_TILE_LANES = 8192
# Rows per program in the row kernels (the guard covers an indivisible tail).
_BT_ROW = 4
# The Ascend backend rejects a coreDim of exactly 65536.
_MAX_GRID = 65535
# Below this width the kernel is dominated by masking overhead, and no
# production model hidden size is that narrow.
_MIN_D = 64
# The kernels index with int32 element offsets; a wider span goes back to
# torch_npu instead of wrapping around.
_INT32_LIMIT = 2**31 - 1


def _next_pow2(n: int) -> int:
    """Smallest power of two that is not smaller than `n`."""
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _rms_norm_2d_kernel(x_ptr, w_ptr, y_ptr, M, eps, D: tl.constexpr,
                            BT: tl.constexpr, N: tl.constexpr):
        """
        `BT` rows as one flat tile, reshaped to `[BT, D]` for the row reduction.

        The reduction runs along axis 1, i.e. over the `BT` rows at once, which
        is what keeps the vector unit busy while the row loop variant waits on
        one reduction at a time.  `D` and `BT` are constexpr, so `N = BT * D`
        is a compile-time tile width and no mask is needed.

        The fp32 sum-of-squares order differs from the native operator in the
        last bit for some shapes (measured at most 1 bf16 ULP on T=4096,
        d=2048); the row kernel matches it bit for bit and is used whenever the
        host cannot split the rows exactly.
        """
        pid = tl.program_id(0)
        row0 = pid * BT
        offs = tl.arange(0, N)
        value = tl.load(x_ptr + row0 * D + offs).to(tl.float32)
        tile = tl.reshape(value, (BT, D))
        rstd = tl.rsqrt(tl.sum(tile * tile, axis=1) / D + eps)
        weight = tl.load(w_ptr + tl.arange(0, D)).to(tl.float32)
        out = tile * rstd[:, None] * weight[None, :]
        tl.store(y_ptr + row0 * D + offs, tl.reshape(out, (N,)).to(y_ptr.dtype.element_ty))

    @triton.jit
    def _add_rms_norm_2d_kernel(x_ptr, r_ptr, w_ptr, y_ptr, s_ptr, M, eps,
                               D: tl.constexpr, BT: tl.constexpr, N: tl.constexpr):
        """Fused `sum = x + residual`, `out = rms_norm(sum)` with `BT` rows per tile."""
        pid = tl.program_id(0)
        row0 = pid * BT
        offs = tl.arange(0, N)
        lhs = tl.load(x_ptr + row0 * D + offs).to(tl.float32)
        rhs = tl.load(r_ptr + row0 * D + offs).to(tl.float32)
        summed = (lhs + rhs).to(s_ptr.dtype.element_ty)
        tl.store(s_ptr + row0 * D + offs, summed)
        # Normalize the value the residual output actually holds, so the two
        # outputs of this kernel stay consistent with the native operator,
        # which rounds the sum to the compute dtype before the reduction.
        tile = tl.reshape(summed.to(tl.float32), (BT, D))
        rstd = tl.rsqrt(tl.sum(tile * tile, axis=1) / D + eps)
        weight = tl.load(w_ptr + tl.arange(0, D)).to(tl.float32)
        out = tile * rstd[:, None] * weight[None, :]
        tl.store(y_ptr + row0 * D + offs, tl.reshape(out, (N,)).to(y_ptr.dtype.element_ty))

    @triton.jit
    def _rms_norm_row_kernel(x_ptr, w_ptr, y_ptr, M, D, eps, BLK: tl.constexpr,
                             NCHUNK: tl.constexpr, BT: tl.constexpr,
                             EVEN: tl.constexpr):
        """One row per step; a single chunk keeps the row in registers."""
        pid = tl.program_id(0)
        col = tl.arange(0, BLK)
        for j in tl.static_range(BT):
            row = pid * BT + j
            if row < M:
                row_base = row * D
                if NCHUNK == 1:
                    if EVEN:
                        value = tl.load(x_ptr + row_base + col).to(tl.float32)
                    else:
                        mask = col < D
                        value = tl.load(x_ptr + row_base + col, mask=mask,
                                        other=0.0).to(tl.float32)
                    rstd = tl.rsqrt(tl.sum(value * value, axis=0) / D + eps)
                    if EVEN:
                        weight = tl.load(w_ptr + col).to(tl.float32)
                        tl.store(y_ptr + row_base + col,
                                 (value * rstd * weight).to(y_ptr.dtype.element_ty))
                    else:
                        weight = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
                        tl.store(y_ptr + row_base + col,
                                 (value * rstd * weight).to(y_ptr.dtype.element_ty),
                                 mask=mask)
                else:
                    acc = tl.zeros([BLK], dtype=tl.float32)
                    for c in tl.static_range(NCHUNK):
                        offs = c * BLK + col
                        if EVEN:
                            value = tl.load(x_ptr + row_base + offs).to(tl.float32)
                        else:
                            mask = offs < D
                            value = tl.load(x_ptr + row_base + offs, mask=mask,
                                            other=0.0).to(tl.float32)
                        acc += value * value
                    rstd = tl.rsqrt(tl.sum(acc, axis=0) / D + eps)
                    for c in tl.static_range(NCHUNK):
                        offs = c * BLK + col
                        if EVEN:
                            value = tl.load(x_ptr + row_base + offs).to(tl.float32)
                            weight = tl.load(w_ptr + offs).to(tl.float32)
                            tl.store(y_ptr + row_base + offs,
                                     (value * rstd * weight).to(y_ptr.dtype.element_ty))
                        else:
                            mask = offs < D
                            value = tl.load(x_ptr + row_base + offs, mask=mask,
                                            other=0.0).to(tl.float32)
                            weight = tl.load(w_ptr + offs, mask=mask,
                                             other=0.0).to(tl.float32)
                            tl.store(y_ptr + row_base + offs,
                                     (value * rstd * weight).to(y_ptr.dtype.element_ty),
                                     mask=mask)

    @triton.jit
    def _add_rms_norm_row_kernel(x_ptr, r_ptr, w_ptr, y_ptr, s_ptr, M, D, eps,
                                 BLK: tl.constexpr, NCHUNK: tl.constexpr,
                                 BT: tl.constexpr, EVEN: tl.constexpr):
        """One row per step: add, store the sum, normalize, store the result."""
        pid = tl.program_id(0)
        col = tl.arange(0, BLK)
        for j in tl.static_range(BT):
            row = pid * BT + j
            if row < M:
                row_base = row * D
                acc = tl.zeros([BLK], dtype=tl.float32)
                for c in tl.static_range(NCHUNK):
                    offs = c * BLK + col
                    if EVEN:
                        lhs = tl.load(x_ptr + row_base + offs).to(tl.float32)
                        rhs = tl.load(r_ptr + row_base + offs).to(tl.float32)
                        summed = (lhs + rhs).to(s_ptr.dtype.element_ty)
                        tl.store(s_ptr + row_base + offs, summed)
                    else:
                        mask = offs < D
                        lhs = tl.load(x_ptr + row_base + offs, mask=mask,
                                      other=0.0).to(tl.float32)
                        rhs = tl.load(r_ptr + row_base + offs, mask=mask,
                                      other=0.0).to(tl.float32)
                        summed = (lhs + rhs).to(s_ptr.dtype.element_ty)
                        tl.store(s_ptr + row_base + offs, summed, mask=mask)
                    acc += summed.to(tl.float32) * summed.to(tl.float32)
                rstd = tl.rsqrt(tl.sum(acc, axis=0) / D + eps)
                for c in tl.static_range(NCHUNK):
                    offs = c * BLK + col
                    if EVEN:
                        value = tl.load(s_ptr + row_base + offs).to(tl.float32)
                        weight = tl.load(w_ptr + offs).to(tl.float32)
                        tl.store(y_ptr + row_base + offs,
                                 (value * rstd * weight).to(y_ptr.dtype.element_ty))
                    else:
                        mask = offs < D
                        value = tl.load(s_ptr + row_base + offs, mask=mask,
                                        other=0.0).to(tl.float32)
                        weight = tl.load(w_ptr + offs, mask=mask,
                                         other=0.0).to(tl.float32)
                        tl.store(y_ptr + row_base + offs,
                                 (value * rstd * weight).to(y_ptr.dtype.element_ty),
                                 mask=mask)


def _is_supported_tensor(x: torch.Tensor) -> bool:
    """Layout gate shared by both entry points: bf16/fp16, non-empty, contiguous."""
    if x is None or x.dim() < 1:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.numel() == 0:
        return False
    if x.stride(-1) != 1 or not x.is_contiguous():
        return False
    return True


def _pick_2d(d: int, rows: int) -> int:
    """Rows per program for the 2-D kernels, or 0 when the layout cannot use them."""
    if d < _MIN_D or (d & (d - 1)) != 0:
        return 0
    bt = min(_BT_2D, _TILE_LANES // d)
    if bt < 1:
        return 0
    if rows % bt != 0:
        return 0
    return bt


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    RMS normalization through the Triton kernels.

    Args:
        x: Input tensor of shape [..., d], bf16/fp16, contiguous.
        weight: Scale vector of shape [d] with the dtype of `x`.
        eps: Epsilon of the variance term, a Python scalar.
        residual: Optional tensor with the shape and dtype of `x`.

    Returns:
        The normalized tensor, or `(normalized, sum)` when `residual` is given,
        or `None` when the layout is outside the supported set and the caller
        must keep the torch_npu operator.
    """
    if not HAS_TRITON or not _is_supported_tensor(x):
        return None
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        return None
    if weight is None or weight.dim() != 1 or weight.dtype != x.dtype:
        return None
    if weight.numel() != x.shape[-1] or weight.stride(-1) != 1 or not weight.is_contiguous():
        return None
    if residual is not None:
        if residual.shape != x.shape or residual.dtype != x.dtype:
            return None
        if residual.stride(-1) != 1 or not residual.is_contiguous():
            return None

    d = x.shape[-1]
    if d < _MIN_D or x.numel() > _INT32_LIMIT:
        return None
    rows = x.numel() // d
    out = torch.empty_like(x)
    summed = None
    if residual is not None:
        summed = torch.empty_like(x)
    eps = float(eps)

    bt = _pick_2d(d, rows)
    if bt:
        grid = (rows // bt,)
        if residual is None:
            _rms_norm_2d_kernel[grid](
                x, weight, out, rows, eps, D=d, BT=bt, N=bt * d
            )
        else:
            _add_rms_norm_2d_kernel[grid](
                x, residual, weight, out, summed, rows, eps, D=d, BT=bt, N=bt * d
            )
        return out if residual is None else (out, summed)

    block = min(_BLK_N, _next_pow2(d))
    if block < 2:
        return None
    nchunk = triton.cdiv(d, block)
    grid = (min(triton.cdiv(rows, _BT_ROW), _MAX_GRID),)
    if residual is None:
        _rms_norm_row_kernel[grid](
            x, weight, out, rows, d, eps, BLK=block, NCHUNK=nchunk,
            BT=_BT_ROW, EVEN=(d % block) == 0,
        )
    else:
        _add_rms_norm_row_kernel[grid](
            x, residual, weight, out, summed, rows, d, eps, BLK=block,
            NCHUNK=nchunk, BT=_BT_ROW, EVEN=(d % block) == 0,
        )
    return out if residual is None else (out, summed)
