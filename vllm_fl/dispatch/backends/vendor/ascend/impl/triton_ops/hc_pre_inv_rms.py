"""hc_pre_inv_rms —— AscendC → Triton

来源：csrc/ascend/moe/hc_pre_inv_rms/
接口（op_host/hc_pre_inv_rms_def.cpp L24-37 + infershape L31-54）：
    Input : x   [b, s, hc, d] (fp16/fp32/bf16)  或  [b*s, hc, d]
    Output: y   [b, s, 1] / [b*s, 1]   fp32
    Attr  : epsilon = 1e-6 (OPTIONAL, float)

shape 规则（op_host/hc_pre_inv_rms_tiling.cpp L103-123 / L326-347）：
    x 4D: A = dim0 * dim1,  R = dim2 * dim3
    x 3D: A = dim0,         R = dim1 * dim2
    R == 28672 时走 large_d 分支（数学公式相同）

公式（op_kernel/hc_pre_inv_rms_full_load.h L152-195 ComputeB16 / L198-236 ComputeB32）：
    s   = sum_r x[a, r]^2            # fp32
    mean = s * (1.0f / R)            # L181-183 Muls(y, y, meanCof)
    mean = mean + epsilon            # L185 Adds
    y[a] = 1.0 / sqrt(mean)          # L187 Duplicate(1.0) + L189 Sqrt + L191 Div
    即  y[a] = 1 / sqrt(mean_r(x[a,:]^2) + eps)

注意：epsilon 加在「均值」之后（不是加在每个平方项上）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre_inv_rms"]


@triton.jit
def _hc_pre_inv_rms_kernel(
    X, Y, A, R, EPS,
    BLK_A: tl.constexpr,
    BLK_R: tl.constexpr,
    NCH: tl.constexpr,
    NCH_P2: tl.constexpr,
):
    """逐行 1-D 累加 + **constexpr 循环次数**。

    ⚠️ 两个必须坚持的写法（实测）：
    1. **循环次数必须是 constexpr，且**取 2 的幂（`NCH_P2 = next_pow2(NCH)`），
       真实块数之外的迭代用 `if c < NCH:` 跳过。原因：Ascend 后端对
       「**unroll 次数不是 2 的幂**」的静态循环会出错 —— `hc_pre` 的 mixes kernel
       在 `KCH=28` 上误差 5.39，而 `KCH=20/24/32` 都是 4e-7（同族证据）。
    2. 不要用 `while r < R`（runtime bound）配 `[BLK_A, BLK_R]` 二维掩码 ——
       与 BACKEND_NOTES 坑 1 同族。这里改成**逐行 1-D**。
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLK_R)
    for i in range(BLK_A):
        row = pid * BLK_A + i
        if row < A:
            acc = tl.zeros((BLK_R,), dtype=tl.float32)
            for c in range(NCH_P2):
                if c < NCH:
                    col = c * BLK_R + cols
                    x = tl.load(X + row * R + col, mask=col < R, other=0.0).to(tl.float32)
                    acc += x * x
            s = tl.sum(acc, axis=0)
            y = 1.0 / tl.sqrt(s / R + EPS)
            tl.store(Y + row, y)


def hc_pre_inv_rms(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """等价于 AscendC 的 HcPreInvRms。

    Args:
        x: 3D [b*s, hc, d] 或 4D [b, s, hc, d]，dtype ∈ {fp16, bf16, fp32}
        epsilon: 属性，默认 1e-6

    Returns:
        y: fp32，shape = x.shape[:-2] + (1,)（4D → [b,s,1]；3D → [b*s,1]）
    """
    assert x.dim() in (3, 4), f"x 必须是 3D 或 4D，实际 {tuple(x.shape)}"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"x dtype {x.dtype} 不支持"

    xc = x.contiguous()
    A = 1
    for d in xc.shape[:-2]:
        A *= int(d)
    R = int(xc.shape[-2]) * int(xc.shape[-1])
    assert A > 0 and R > 0, f"A={A} R={R} 非法"

    x2 = xc.view(A, R)
    y = torch.empty((A,), dtype=torch.float32, device=xc.device)

    # Ascend UB = 192KB：tile 取小一些（block 内多次循环），否则 bishengir 报 ub overflow
    BLK_A = 4
    BLK_R = min(next_pow2(R), 512)
    nch = ceil_div(R, BLK_R)
    grid = (ceil_div(A, BLK_A),)
    _hc_pre_inv_rms_kernel[grid](
        x2, y, A, R, float(epsilon),
        BLK_A=BLK_A, BLK_R=BLK_R,
        NCH=nch, NCH_P2=next_pow2(nch),
    )
    return y.view(*xc.shape[:-2], 1)
