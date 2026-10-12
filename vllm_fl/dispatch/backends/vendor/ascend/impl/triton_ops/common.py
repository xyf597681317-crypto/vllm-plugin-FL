"""AscendC → Triton 迁移：公共工具。

来源：vllm-fl 分支 feature/v024-ascend-platform-runtime 的 csrc/ascend/ 下 25 个 AscendC 算子。

本包内每个算子一个模块，函数命名与原算子一致。
所有实现只用标准 `triton` / `triton.language`，不依赖 vllm / triton_ascend / cann.extension。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["npu_device", "ceil_div", "next_pow2", "IS_NPU"]

IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()


def npu_device(idx: int = 0) -> str:
    """返回当前可用的 device 字符串。"""
    return f"npu:{idx}" if IS_NPU else "cpu"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())


def grid_1d(n: int, block: int) -> tuple[int]:
    return (ceil_div(n, block),)
