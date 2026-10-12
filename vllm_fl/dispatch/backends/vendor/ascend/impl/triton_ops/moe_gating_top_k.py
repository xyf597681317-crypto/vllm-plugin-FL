"""moe_gating_top_k —— AscendC → Triton

来源：csrc/ascend/moe/moe_gating_top_k/
接口（op_host/moe_gating_top_k_def.cpp L22-57 + infershape L115-142）：
    Input : x    [rows, expertNum]  fp16/fp32/bf16
            bias [expertNum]        (OPTIONAL) 同 dtype
    Output: y         [rows, k]         同 x dtype
            expert_idx[rows, k]         int32
            out       [rows, expertNum] fp32
    Attr  : k, k_group=1, group_count=1, group_select_mode=0, renorm=0,
            norm_type=0, out_flag=false, routed_scaling_factor=1.0, eps=1e-20

公式（op_kernel/moe_gating_top_k_without_group.h + _generalized.h）：
  ComputeX（without_group L115-175 / generalized L153-226）：
      norm_type==0（softmax）: m = max_e x ; s = exp(x-m) / sum(exp(x-m))     # L135-155
      norm_type==1（sigmoid）: s = sigmoid(x)                                  # L128-131
      routing = s + bias（有 bias 时；否则 = s）                               # L158-162
      out = s（fp32，**不含 bias**，仅 out_flag 时写出）                       # L178-188
  分组（generalized L243-368，仅 group_count>1 且 k_group<group_count 时走）：
      组内按 routing 降序排序                                                     # L243-262
      group_select_mode==1（SUM）：组得分 = 组内 top-2 routing 之和              # L285-306
      group_select_mode==0（MAX）：组得分 = 组内 top-1 routing（最大值）         # L308-321
      取组得分 top-k_group 的组（并按组号降序排列）                              # L343-368
      专家候选集 = 被选中组的专家并集                                             # L371-515
  选择与输出（without_group L191-272 / generalized L371-556）：
      在候选集内按 routing 取 top-k（降序）                                       # L200-215 / L513
      y = s[topk_idx]（注意：取的是**不含 bias** 的 s）                           # L229-230 / L527
      need_renorm = (norm_type==1) || (norm_type==0 && renorm==1)                # L232-233 / L529-530
      if need_renorm: y = y / (sum(y) + eps)                                     # L238-248 / L535-545
      y = y * routed_scaling_factor                                              # L251 / L548
      y 非 fp32 时按 RINT 转回 T                                                 # L253-256 / L550-553
      expert_idx = topk 专家号（generalized 的编码解码见 L559-575，等价于全局专家号）

走哪条内核（op_host/..._tiling.cpp L551-563）：
      expertNum==256 && group_count==8 && k_group==4 && k<=32 && bias &&
      group_select_mode==1 && renorm==0 && norm_type==1 && !out_flag
          -> MoeGatingTopKEKFullload（TILING_KEY 0，性能变体，数学同上）
      group_count==1 || group_count==expertNum || k_group==group_count -> without_group（1）
      其余 -> generalized（2）

Triton 实现说明 / 语义假设：
  1. 三个内核模板（fullload / without_group / generalized）只影响切分与排序实现，
     数学统一为「softmax|sigmoid(+bias) 分组筛选后取 top-k」，本实现按统一公式写。
  2. 平局（分数完全相等）时的取号：Ascend 的 Sort/MrgSort 行为未在源码中明确，
     本实现取「更小的专家号」。测试数据刻意避免平局（并打印最小间隔校验）。
  3. out_flag=False 时 Ascend 不写 out（内容未定义），本实现返回 None。
  4. k 必须 <= expertNum，且各输出行独立。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import os as _os

from .common import ceil_div, next_pow2

#: UB-budget knobs (see the comment above ``blk_r``): the [BLK_R, BLK_E] tile has to
#: stay inside the 192KB unified buffer. 实测（910_93，生产组合 E=256 k=8 g=8 kg=4
#: SUM+sigmoid+bias、bf16，即 Qwen3.6-35B-A3B 的调用形态；见
#: `_probe_mgating_ub.py`）：
#:   BLK_R=8 -> "ub overflow, requires 4315136 bits" (2.74x)
#:   BLK_R=4 -> "ub overflow, requires 2180096 bits" (1.39x)   ← 2026-10-11 打挂模型的就是它
#:   BLK_R=2 -> 编译通过
#:   BLK_R=1 -> 编译通过；BLK_E 一路到 4096 仍通过，BLK_E=8192 才 1975040 bits (1.26x)
#: 由 BLK_R=4/8 两点线性外推：生产 shape 下 BLK_R=1 的 UB ≈ 2180096-3*533760 ≈ 578816 bits
#: （≈37% of 1,572,864，约 2.7x 余量）。因此默认 BLK_R=1；可用环境变量在大 UB 的
#: 版本上重新调参。
BLK_R_CAP = int(_os.environ.get("MOE_GATING_BLK_R_CAP", "1"))
ROW_BUDGET = int(_os.environ.get("MOE_GATING_ROW_BUDGET", "1024"))

__all__ = ["moe_gating_top_k"]


@triton.jit
def _moe_gating_top_k_kernel(
    X, BIAS, Y, IDX, OUT,
    ROWS, E, K,
    EPS, ROUTED_SCALE,
    GROUP_COUNT: tl.constexpr,
    PER_GROUP: tl.constexpr,
    K_GROUP: tl.constexpr,
    GROUP_SUM: tl.constexpr,
    NEED_RENORM: tl.constexpr,
    SIGMOID: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLK_R: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_G: tl.constexpr,
    BLK_KG: tl.constexpr,
    NEG_BIG,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS
    cols = tl.arange(0, BLK_E)
    cmask = cols < E

    x = tl.load(X + rows[:, None] * E + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0)
    x = x.to(tl.float32)

    # --- 归一化得到 s（padded 列用 -BIG，保证不参与 max/sum）---
    if SIGMOID:
        s = 1.0 / (1.0 + tl.exp(-x))
    else:
        xm = tl.where(cmask[None, :], x, NEG_BIG)
        mx = tl.max(xm, axis=1)
        ex = tl.where(cmask[None, :], tl.exp(xm - mx[:, None]), 0.0)
        s = ex / tl.sum(ex, axis=1)[:, None]

    if HAS_BIAS:
        bias = tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)
        routing = s + bias[None, :]
    else:
        routing = s
    routing = tl.where(cmask[None, :], routing, NEG_BIG)

    # --- 分组筛选 ---
    gmask = routing > NEG_BIG * 0.5  # 非 padding 列
    if GROUP_COUNT > 1:
        gar = tl.arange(0, BLK_G)
        gof = cols // PER_GROUP
        gscore = tl.zeros([BLK_R, BLK_G], dtype=tl.float32)
        for g in range(GROUP_COUNT):
            v = tl.where(gof[None, :] == g, routing, NEG_BIG)
            m1 = tl.max(v, axis=1)
            i1 = tl.min(tl.where(v == m1[:, None], cols[None, :], 2147483647), axis=1)
            gs = m1
            if GROUP_SUM:
                v2 = tl.where(cols[None, :] == i1[:, None], NEG_BIG, v)
                m2 = tl.max(v2, axis=1)
                gs = m1 + m2
            gscore = tl.where(gar[None, :] == g, gs[:, None], gscore)

        # 取组得分 top-K_GROUP，并把「被选中组」直接折叠成专家掩码
        # （刻意不用 tl.sum(tl.where(...)) 从 int32 向量取标量：Ascend 后端该写法有坑）
        acc = tl.zeros([BLK_R, BLK_E], dtype=tl.int1)
        cur = gscore
        for j in range(K_GROUP):
            ang = tl.arange(0, BLK_G)
            gm = tl.max(cur, axis=1)
            gi = tl.min(tl.where(cur == gm[:, None], ang[None, :], 2147483647), axis=1)
            acc = acc | (gof[None, :] == gi[:, None])
            cur = tl.where(ang[None, :] == gi[:, None], NEG_BIG, cur)
        gmask = acc

    routing_m = tl.where(gmask, routing, NEG_BIG)

    # --- top-k（逐次取最大，平局取小号）---
    kar = tl.arange(0, BLK_K)
    y2d = tl.zeros([BLK_R, BLK_K], dtype=tl.float32)
    i2d = tl.zeros([BLK_R, BLK_K], dtype=tl.int32)
    ytot = tl.zeros([BLK_R], dtype=tl.float32)
    cur2 = routing_m
    for j in range(K):
        mm = tl.max(cur2, axis=1)
        ii = tl.min(tl.where(cur2 == mm[:, None], cols[None, :], 2147483647), axis=1)
        sv = tl.sum(tl.where(cols[None, :] == ii[:, None], s, 0.0), axis=1)
        y2d = tl.where(kar[None, :] == j, sv[:, None], y2d)
        i2d = tl.where(kar[None, :] == j, ii[:, None], i2d)
        ytot += sv
        cur2 = tl.where(cols[None, :] == ii[:, None], NEG_BIG, cur2)

    if NEED_RENORM:
        y2d = y2d / (ytot[:, None] + EPS)
    y2d = y2d * ROUTED_SCALE

    kmask = (kar[None, :] < K) & rmask[:, None]
    if OUT_FP32:
        tl.store(Y + rows[:, None] * K + kar[None, :], y2d, mask=kmask)
    else:
        tl.store(Y + rows[:, None] * K + kar[None, :], y2d.to(Y.dtype.element_ty), mask=kmask)
    tl.store(IDX + rows[:, None] * K + kar[None, :], i2d, mask=kmask)

    if OUT_FLAG:
        tl.store(OUT + rows[:, None] * E + cols[None, :], s, mask=rmask[:, None] & cmask[None, :])


def moe_gating_top_k(
    x: torch.Tensor,
    k: int,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    renorm: int = 0,
    norm_type: int = 0,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    bias: torch.Tensor | None = None,
    bias_opt: torch.Tensor | None = None,
):
    """等价于 AscendC 的 MoeGatingTopK。

    Args:
        x: [rows, expertNum]，fp16/fp32/bf16
        bias / bias_opt: [expertNum] 可选，同 x dtype（只影响 top-k 选择，不进 y/out）；
                         `bias_opt` 是调用点参数名（npu_moe_gating_top_k 的 `Tensor? bias_opt`），
                         两者都给时以 `bias_opt` 为准
        k / k_group / group_count / group_select_mode / renorm / norm_type: 属性
        out_flag: 是否写第三个输出 out
        routed_scaling_factor / eps: 属性

    Returns:
        (y [rows,k] 同 x dtype, expert_idx [rows,k] int32, out [rows,E] fp32 或 None)
    """
    if bias_opt is not None:
        bias = bias_opt
    assert x.dim() == 2, f"x 应为 [rows, expertNum]，实际 {tuple(x.shape)}"
    rows, E = x.shape
    assert 0 < k <= E, f"k={k} 非法（expertNum={E}）"
    assert norm_type in (0, 1), "norm_type 只支持 0(softmax)/1(sigmoid)"
    assert group_select_mode in (0, 1), "group_select_mode 只支持 0(MAX)/1(SUM)"
    assert 1 <= k_group <= group_count <= E, "需满足 1 <= k_group <= group_count <= expertNum"
    assert E % group_count == 0, "expertNum 必须能被 group_count 整除"
    if bias is not None:
        assert bias.numel() == E and bias.dtype == x.dtype

    per_group = E // group_count
    if group_select_mode == 1:
        assert per_group >= 2, "group_select_mode=SUM 需要每组至少 2 个专家"

    need_renorm = (norm_type == 1) or (norm_type == 0 and renorm == 1)

    dev = x.device
    x = x.contiguous()
    bias_t = bias.contiguous() if bias is not None else x
    y = torch.empty((rows, k), dtype=x.dtype, device=dev)
    idx = torch.empty((rows, k), dtype=torch.int32, device=dev)
    out = torch.empty((rows, E), dtype=torch.float32, device=dev) if out_flag else None

    blk_e = next_pow2(E)
    # [BLK_R, BLK_E] tile 及其掩码/索引向量必须塞进 192KB UB：见文件头 BLK_R_CAP 处的
    # 实测表（生产 shape E=256 下 BLK_R=4 需 2180096 bits = 1.39x 超限，BLK_R=1/2 通过）。
    blk_r = max(1, min(BLK_R_CAP, ROW_BUDGET // blk_e))
    _moe_gating_top_k_kernel[(ceil_div(rows, blk_r),)](
        x, bias_t, y, idx, out if out_flag else x,
        rows, E, k,
        float(eps), float(routed_scaling_factor),
        GROUP_COUNT=group_count,
        PER_GROUP=per_group,
        K_GROUP=k_group,
        GROUP_SUM=(group_select_mode == 1),
        NEED_RENORM=need_renorm,
        SIGMOID=(norm_type == 1),
        HAS_BIAS=(bias is not None),
        OUT_FLAG=out_flag,
        OUT_FP32=(x.dtype == torch.float32),
        BLK_R=blk_r,
        BLK_E=blk_e,
        BLK_K=next_pow2(k),
        BLK_G=next_pow2(group_count),
        BLK_KG=next_pow2(max(k, k_group)),
        NEG_BIG=-3.0e38,
    )
    return y, idx, out
