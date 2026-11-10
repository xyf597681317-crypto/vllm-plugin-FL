# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of the CANN `MoeTokenUnpermute` operator.

Contract, measured on the 106 host from the operator doc and from the single
vendor call site (`impl/fused_moe.py::_torch_fused_experts_impl`):

    permuted_tokens : [tokens * topk, hidden], bf16/fp16
    sorted_indices  : [tokens * topk], int32 (the native operator rejects int64)
    probs           : [tokens, topk], fp32 router weights
    output          : [tokens, hidden], dtype of `permuted_tokens`

    out[t, h] = sum_k permuted_tokens[sorted_indices[t * topk + k], h] * probs[t, k]

`padded_mode=False` and `restore_shape=None` are the only variants the vendor
path uses, so this module implements exactly that case and returns `None` for
every other one; the caller then keeps `torch_npu.npu_moe_token_unpermute`.
"""

from __future__ import annotations

from typing import Optional

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Column tile: one hidden row is never wider than this in the supported models,
# and 2048 fp32 lanes keep the accumulator tile at 65536 bits.
_BLK_CAP = 2048
# Tokens per program. Measured on the 105 host with T=4096, topk=8, hidden=2048:
# 0.439 ms at one token per program, 0.248 at two, 0.173 at four, 0.194 at
# eight, so four tokens is the optimum.
_BT = 4
# The Ascend backend rejects a coreDim of exactly 65536.
_MAX_GRID = 65535
# Narrower rows are dominated by masking overhead and no production hidden size
# is that small; those shapes fall back to the native operator.
_MIN_H = 64


def _next_pow2(n: int) -> int:
    """Smallest power of two that is not smaller than `n`."""
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _moe_token_unpermute_kernel(
        perm_ptr,
        idx_ptr,
        prob_ptr,
        out_ptr,
        num_tokens,
        num_rows,
        H,
        TOPK: tl.constexpr,
        BLK: tl.constexpr,
        BT: tl.constexpr,
        EVEN: tl.constexpr,
    ):
        """
        Weighted scatter-reduce of the permuted expert rows back to token rows.

        One program owns a column block of `BT` tokens: it gathers the `topk`
        source rows of each token with a 1-D load at a scalar row base and
        accumulates `value * prob` in fp32.

        The bounds check on the source row is a *scalar* select, not a vector
        mask: masking the load with `(row >= 0) & (row < num_rows)` makes the
        mask depend on a scalar load, which forces a compare and a select per
        lane, and it measured 0.404 ms against 0.173 ms for the branchless
        clamp below.  Clamping the row keeps the address in range and zeroing
        the weight keeps the contribution exact, so the out-of-range case costs
        nothing in the fast path.
        """
        pid_h = tl.program_id(0)
        pid_t = tl.program_id(1)
        cols = pid_h * BLK + tl.arange(0, BLK)
        for j in tl.static_range(BT):
            token = pid_t * BT + j
            if token < num_tokens:
                acc = tl.zeros([BLK], dtype=tl.float32)
                for k in tl.static_range(TOPK):
                    flat = token * TOPK + k
                    row = tl.load(idx_ptr + flat)
                    weight = tl.load(prob_ptr + flat)
                    in_range = (row >= 0) & (row < num_rows)
                    row = tl.where(in_range, row, 0)
                    weight = tl.where(in_range, weight, 0.0)
                    if EVEN:
                        value = tl.load(perm_ptr + row * H + cols)
                    else:
                        value = tl.load(perm_ptr + row * H + cols, mask=cols < H,
                                        other=0.0)
                    acc += value.to(tl.float32) * weight
                if EVEN:
                    tl.store(out_ptr + token * H + cols,
                             acc.to(out_ptr.dtype.element_ty))
                else:
                    tl.store(out_ptr + token * H + cols,
                             acc.to(out_ptr.dtype.element_ty), mask=cols < H)


def triton_moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Unpermute the expert output through the Triton kernel.

    Args:
        permuted_tokens: `[tokens * topk, hidden]` bf16/fp16, contiguous.
        sorted_indices: `[tokens * topk]` int32/int64 source row of each route.
        probs: `[tokens, topk]` fp32 router weights.

    Returns:
        The `[tokens, hidden]` unpermuted tensor, or `None` when the layout is
        outside the supported set and the caller must keep the torch_npu
        operator.
    """
    if not HAS_TRITON:
        return None
    if not isinstance(permuted_tokens, torch.Tensor) or permuted_tokens.dim() != 2:
        return None
    if permuted_tokens.dtype not in (torch.bfloat16, torch.float16):
        return None
    if permuted_tokens.numel() == 0 or not permuted_tokens.is_contiguous():
        return None
    if not isinstance(sorted_indices, torch.Tensor) or sorted_indices.dim() != 1:
        return None
    if sorted_indices.dtype not in (torch.int32, torch.int64):
        return None
    if not sorted_indices.is_contiguous():
        return None
    if not isinstance(probs, torch.Tensor) or probs.dim() != 2:
        return None
    if probs.dtype != torch.float32 or probs.numel() == 0 or not probs.is_contiguous():
        return None
    tokens, topk = probs.shape
    rows, hidden = permuted_tokens.shape
    if rows != tokens * topk or sorted_indices.numel() != rows:
        return None
    if hidden < _MIN_H or topk < 1:
        return None
    if hidden > _BLK_CAP * _MAX_GRID:
        return None

    block = min(_BLK_CAP, _next_pow2(hidden))
    out = torch.empty((tokens, hidden), dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    grid = (triton.cdiv(hidden, block), min(triton.cdiv(tokens, _BT), _MAX_GRID))
    _moe_token_unpermute_kernel[grid](
        permuted_tokens,
        sorted_indices,
        probs,
        out,
        tokens,
        rows,
        hidden,
        TOPK=topk,
        BLK=block,
        BT=_BT,
        EVEN=(hidden % block) == 0,
    )
    return out
