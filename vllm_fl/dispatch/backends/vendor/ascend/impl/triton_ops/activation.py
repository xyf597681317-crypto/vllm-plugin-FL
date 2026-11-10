# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of the fused SiLU-and-multiply activation (CANN `SwiGlu`).

`silu_and_mul(x)` consumes a tensor whose last axis holds the concatenation
`[gate, up]` of width `2 * d` and returns `silu(gate) * up` of width `d`.
"""

from __future__ import annotations

from typing import Optional

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Column tile. A chunk of _BLK_N fp32 lanes costs _BLK_N * 32 bits of UB; with
# 2048 lanes and the four live values of the kernel body the peak stays around
# 3.3e5 bits, well below the 1.57e6 bit UB budget of an A2-class part even
# after the default multi-buffering factor of 2.
_BLK_N = 2048
# Rows per program. Measured on the 105 host with T=4096, d=2048: 0.253 ms at
# one row per program, 0.183 at two, 0.142 at four, 0.146 at eight, 0.156 at
# sixteen, so four rows is the flat part of the curve.
_BT = 4
# The Ascend backend rejects a coreDim of exactly 65536; cap the grid and grow
# the in-kernel row group instead.
_MAX_GRID = 65535
# Below this column width the kernel is dominated by masking overhead, and no
# production model shape is that narrow.
_MIN_D = 64
# The kernel indexes with int32 element offsets; a wider span goes back to
# torch_npu rather than wrapping around.
_INT32_LIMIT = 2**31 - 1


def _next_pow2(n: int) -> int:
    """Smallest power of two that is not smaller than `n`."""
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _silu_and_mul_kernel(x_ptr, out_ptr, M, D, BLK: tl.constexpr,
                             NCHUNK: tl.constexpr, BT: tl.constexpr,
                             EVEN: tl.constexpr):
        """
        `BT` rows per program, one 1-D column tile per row.

        Every load and store is a 1-D vector at a *scalar* base address: a 2-D
        store whose per-row base is a vector is lowered to one scalar store per
        lane on this backend (measured 118x slower on the same traffic).
        Column offsets are int32 (the host gates the element span), and the
        masks are dropped when `D` is an exact multiple of the tile, because a
        mask that is always true still costs a compare and a select per lane.

        The sigmoid is the plain `1 / (1 + exp(-x))` form. For a bf16/fp16
        input `exp` overflows to +inf for `x < -88`, and the reciprocal then
        gives the correct 0, so the two-branch overflow guard is not needed; it
        is also measurably slower (two divisions instead of one) and on the
        production shape it is the plain form that reproduces the native
        operator bit for bit.
        """
        pid = tl.program_id(0)
        col = tl.arange(0, BLK)
        for j in tl.static_range(BT):
            row = pid * BT + j
            if row < M:
                x_base = row * (2 * D)
                o_base = row * D
                for c in tl.static_range(NCHUNK):
                    offs = c * BLK + col
                    if EVEN:
                        gate = tl.load(x_ptr + x_base + offs).to(tl.float32)
                        up = tl.load(x_ptr + x_base + D + offs).to(tl.float32)
                        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
                        tl.store(out_ptr + o_base + offs,
                                 value.to(out_ptr.dtype.element_ty))
                    else:
                        mask = offs < D
                        gate = tl.load(x_ptr + x_base + offs, mask=mask,
                                       other=0.0).to(tl.float32)
                        up = tl.load(x_ptr + x_base + D + offs, mask=mask,
                                     other=0.0).to(tl.float32)
                        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
                        tl.store(out_ptr + o_base + offs,
                                 value.to(out_ptr.dtype.element_ty), mask=mask)


def triton_silu_and_mul(x: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Fused SiLU-and-multiply through the Triton kernel.

    Args:
        x: Tensor of shape [..., 2 * d], bf16/fp16, contiguous.

    Returns:
        Tensor of shape [..., d] with the dtype of `x`, or `None` when the
        layout is outside the supported set and the caller must keep the
        torch_npu operator.
    """
    if not HAS_TRITON or x is None:
        return None
    if x.dim() < 1 or x.dtype not in (torch.bfloat16, torch.float16):
        return None
    if x.numel() == 0:
        return None
    last = x.shape[-1]
    if last < 2 * _MIN_D or last % 2 != 0:
        return None
    if x.stride(-1) != 1 or not x.is_contiguous():
        return None
    if x.numel() > _INT32_LIMIT:
        return None

    d = last // 2
    block = min(_BLK_N, _next_pow2(d))
    if block < 2:
        return None

    rows = x.numel() // last
    bt = _BT
    while (rows + bt - 1) // bt > _MAX_GRID:
        bt *= 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    # Metadata-only view: the contiguity check above guarantees it aliases `x`.
    flat_x = x.view(rows, last)
    flat_out = out.view(rows, d)
    _silu_and_mul_kernel[(triton.cdiv(rows, bt),)](
        flat_x,
        flat_out,
        rows,
        d,
        BLK=block,
        NCHUNK=triton.cdiv(d, block),
        BT=bt,
        EVEN=(d % block) == 0,
    )
    return out
