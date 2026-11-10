# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend normalization operator implementations.

The fused Triton kernels are preferred for bf16/fp16 inputs with a contiguous
last axis, a matching 1-D weight and at least `_MIN_D` columns; every other
layout keeps the original torch_npu operator, so the numerical contract is
unchanged.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    weight = obj.weight
    epsilon = obj.variance_epsilon

    from .triton_ops.normalization import triton_rms_norm

    fused = triton_rms_norm(x, weight, epsilon, residual)
    if fused is not None:
        return fused

    import torch_npu

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x
