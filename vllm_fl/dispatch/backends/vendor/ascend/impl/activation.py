# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend activation operator implementations.

The fused Triton kernel is preferred for bf16/fp16 inputs that are contiguous
and at least `_MIN_D` columns wide; every other layout keeps the original
torch_npu operator, so the numerical contract is unchanged.
"""

from __future__ import annotations

import torch


def silu_and_mul_ascend(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using Ascend NPU.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from .triton_ops.activation import triton_silu_and_mul

    fused = triton_silu_and_mul(x)
    if fused is not None:
        return fused

    import torch_npu

    return torch_npu.npu_swiglu(x)
