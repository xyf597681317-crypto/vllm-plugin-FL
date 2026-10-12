"""hc_post —— AscendC → Triton

来源：csrc/ascend/moe/hc_post/
接口（op_host/hc_post_def.cpp L24-48 + hc_post_proto.cpp L29-59 + hc_post_tiling.h L69-79）：
    Input : x        [B, D]        fp16/bf16/fp32          (T1)
            residual [B, H, D]    fp16/bf16/fp32          (T1，与 x 同 dtype)
            post     [B, H]       fp16/bf16/fp32          (T2，可与 T1 不同)
            comb     [B, H, H]    fp16/bf16/fp32          (T2，与 post 同 dtype)
    Output: y        [B, H, D]    dtype = x 的 dtype (InferDtype L52-59)
    Attr  : 无

    （op_host/hc_post_tiling.cpp 里 hcParam = H，dParam = D，batchSize = B）

公式（op_kernel/hc_post_d_split.h L299-359 DoCompute，910B 路径；
      op_kernel/hc_post_float32.h L204-278 DoMulAndAdd，950 regbase 路径，两者一致）：

    910B: out[h,d] = post[h] * x[d]
          for k in range(H):  sumTemp[h,d] = comb[k,h] * residual[k,d]
                              out[h,d]    += sumTemp[h,d]
    950 : sumReg = residual[0]*comb[0,h] ; MulAddDst(..., residual[3]*comb[3,h])
                  += residual[1]*comb[1,h] ; += residual[2]*comb[2,h] ; += x*post[h]
    ==> y[b,h,d] = post[b,h] * x[b,d] + sum_k comb[b,k,h] * residual[b,k,d]
        （即 y = post[:,:,None]*x[:,None,:] + comb^T @ residual，注意 comb 是 [k,h] 转置使用）

精度：累加在 fp32 中进行（内核先 Cast 到 fp32 再 Mul/Add），最后 CAST_RINT 回 T1。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_post"]


@triton.jit
def _hc_post_kernel(
    X, RESIDUAL, POST, COMB, Y,
    D, H,
    BLK_H: tl.constexpr,
    BLK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    h = tl.arange(0, BLK_H)
    hmask = h < H
    d = pid_d * BLK_D + tl.arange(0, BLK_D)
    dmask = d < D

    post = tl.load(POST + pid_b * H + h, mask=hmask, other=0.0).to(tl.float32)      # [BLK_H]
    x = tl.load(X + pid_b * D + d, mask=dmask, other=0.0).to(tl.float32)            # [BLK_D]

    acc = post[:, None] * x[None, :]                                                # [BLK_H, BLK_D]

    base = pid_b * H * H
    rbase = pid_b * H * D
    for k in range(0, H):
        comb_k = tl.load(COMB + base + k * H + h, mask=hmask, other=0.0).to(tl.float32)
        res_k = tl.load(RESIDUAL + rbase + k * D + d, mask=dmask, other=0.0).to(tl.float32)
        acc += comb_k[:, None] * res_k[None, :]

    m = hmask[:, None] & dmask[None, :]
    tl.store(Y + rbase + h[:, None] * D + d[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    dim: int | None = None,
) -> torch.Tensor:
    """等价于 AscendC 的 HcPost（超连接 post 组合）。

    支持两种等价形态（前导维可任意多，内部展平成 B）：
        L0/op_host 形态   : x [B, D]        residual [B, H, D]     post [B, H]     comb [B, H, H]
        真实调用点形态     : x [b, s, d]     residual [b, s, hc, d] post [b, s, hc] comb [b, s, hc, hc]
        （调用点 vllm_ascend/models/deepseek_v41/model.py L866 / glm5next/model.py L523 /
          vllm_fl/models/deepseek_v4_ascend.py L975 都是
          `npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))`；
          torch binding check_hc_post_shape_and_dtype（vllm-ascend csrc/torch_binding.cpp L1402-1428）
          明确要求 x/post 为 3D、residual/comb 为 4D，输出沿用 residual 的形状 L1396）

    Args:
        x:        [..., D]      T1
        residual: [..., H, D]   T1
        post:     [..., H]      T2
        comb:     [..., H, H]   T2   （comb[..., k, h]）
        dim:      **本算子没有 dim 参数**。审计报告里的 `dim` 是扫描器误报：调用点原文是
                  `npu_hc_post(x.unsqueeze(dim=0), ...)`，`dim=0` 属于 `unsqueeze`。
                  binding schema（L3425-3432）与 op_host def 都只有 4 个 tensor。
                  这里保留该关键字仅为给出明确错误（dim=0 与「前导维全部展平」语义一致，视为无操作）。

    Returns:
        y: residual 的形状 [..., H, D]，dtype = x.dtype
    """
    if dim is not None:
        assert int(dim) == 0, (
            f"dim={dim} 不被支持：hc_post 没有 dim 参数（binding schema 只有 x/residual/post/comb 四个 tensor）。"
            "审计里的 dim 来自调用点 `x.unsqueeze(dim=0)`，是扫描器误报。仅 dim=0（首维为 batch）无操作通过。")

    assert x.dim() >= 2, f"x 至少 2 维 [..., D]，实际 {tuple(x.shape)}"
    lead = tuple(int(v) for v in x.shape[:-1])
    D = int(x.shape[-1])
    assert residual.dim() == x.dim() + 1, \
        f"residual 应为 x 的前导维 + (H,D)，实际 x{tuple(x.shape)} residual{tuple(residual.shape)}"
    H = int(residual.shape[-2])
    assert tuple(int(v) for v in residual.shape) == lead + (H, D), \
        f"residual shape {tuple(residual.shape)} != {lead + (H, D)}"
    assert tuple(int(v) for v in post.shape) == lead + (H,), \
        f"post shape {tuple(post.shape)} != {lead + (H,)}"
    assert tuple(int(v) for v in comb.shape) == lead + (H, H), \
        f"comb shape {tuple(comb.shape)} != {lead + (H, H)}"

    B = 1
    for v in lead:
        B *= v
    xc = x.contiguous().view(B, D)
    rc = residual.contiguous().view(B, H, D)
    pc = post.contiguous().view(B, H)
    cc = comb.contiguous().view(B, H, H)
    y = torch.empty((B, H, D), dtype=xc.dtype, device=xc.device)

    BLK_H = max(2, next_pow2(H))
    # 限制 tile 元素数，避免 bishengir ub overflow（UB=192KB）
    BLK_D = min(next_pow2(D), max(16, 4096 // BLK_H))
    grid = (B, ceil_div(D, BLK_D))
    _hc_post_kernel[grid](
        xc, rc, pc, cc, y,
        D, H,
        BLK_H=BLK_H, BLK_D=BLK_D,
    )
    return y.view(*lead, H, D)
