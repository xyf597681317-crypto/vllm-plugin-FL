"""moe_gating_top_k_hash —— AscendC → Triton

来源：csrc/ascend/moe/moe_gating_top_k_hash/

接口（op_host/moe_gating_top_k_hash_def.cpp L22-131）：
    Input : x        [N, E]     fp32 / fp16 / bf16       (REQUIRED)
            bias     [E]        与 x 同 dtype              (OPTIONAL)
            input_ids[N]        int32 / int64             (OPTIONAL, hash 用)
            tid2eid  [V, >=k]   int32 / int64             (OPTIONAL, hash 用)
    Output: y        [N, k]     同 x dtype
            expert_idx [N, k]   int32
            out      [N, E]     fp32                      (仅 out_flag=true 时写)
    Attr  : k(int), k_group(int=1), group_count(int=1), group_select_mode(int=0),
            renorm(int=0), norm_type(int=0), out_flag(bool=false),
            routed_scaling_factor(float=1.0), eps(float=1e-20)

norm_type 取值（op_kernel/common.h L41-43）：
    0 = SOFTMAX, 1 = SIGMOID, 2 = SOFTPLUS

--------------------------------------------------------------------------
公式出处
--------------------------------------------------------------------------
[1] 不分组主体：op_kernel/moe_gating_top_k_hash_without_group.h
    L126-196  ComputeX     —— 归一化 (+bias)
    L212-238  SelectTopKExpertIdx   —— 非 hash：按 (xNorm+bias) 降序取 top-k 专家号
    L294-313  SelectExpertIdxByHash —— hash：专家号 = tid2eid[input_ids[row], :k]
    L241-278  SelectTopKExpertScore —— y = xNorm[expert_idx]; sigmoid/softplus 再归一; *rsf
    L199-209  CopuOutXNorm —— out = xNorm（**不含 bias**）
    L366-383  Process      —— 每行：归一化 → (hash? 查表 : topk) → 取分 → 写回

[2] 分组主体（group_count>1）：op_kernel/moe_gating_top_k_hash_generalized.h
    L152-224  ComputeX     —— 同上，按 perGroupExpertCount 分组 + 组内尾部补 -inf
    L242-260  SortInGroup  —— 每组内降序排序
    L263-367  SelectTopKGroupIndex —— 组得分 = 组内 top2 之和(mode=1) 或 top1(mode=0)，
                                      组得分降序取前 k_group 个组
    L370-514  SelectTopKExpertIdx  —— 把选中组的组内有序表归并，取全局 top-k
    L517-554  SelectTopKExpertScore—— 同 [1]
    L557-574  CumputeActualTopKExpertId —— 还原全局专家号

[3] 高性能特化（tiling_key=0）：op_kernel/moe_gating_top_k_hash_e_k_fullload.h
    L125-144 / L147-157 / L160-232 / L235-273 / L276-317
    —— 当 E=256,G=8,k_group=4,k<=32,加bias,group_select_mode=SUM,renorm=0,
       norm_type=SIGMOID,out_flag=false 时启用（op_host/..._tiling.cpp L576-580）。
       与 [2] 数学等价（组得分=组内 top2 之和；Sort32 宽度 32 恰好等于 P），
       区别只是它假定 E%32==0、P==32，无需补 -inf。

tiling 约束（op_host/moe_gating_top_k_hash_tiling.cpp L206-271）：
    E <= 2048；E % group_count == 0；softplus 仅支持 group_count==1；
    renorm 只支持 0；group_select_mode ∈ {0:max, 1:sum}；k <= k_group*perGroupCount。

--------------------------------------------------------------------------
精度说明（重要）
--------------------------------------------------------------------------
* 归一化在 **fp32** 中做（AscendC 内核把 T Cast 到 float 后再算），fp64 参考对拍。
* y 的取值来自 **xNorm（未加 bias）**，bias 只参与 top-k 的选择（[1] L180 vs L250）。
* sigmoid / softplus 之后有一次 **除以 (sum(y)+eps)** 的重归一化，softmax 没有
  （[1] L253-268，[2] L529-544）。
* 最后乘 routed_scaling_factor，再 CAST_RINT 回 T 写出。
* 平局（相等得分）的专家顺序：AscendC 用硬件 Sort32，未定义；本实现取
  **得分相同时专家号小者优先**（确定性），测试中已用可辨识数据覆盖。
* AscendC 用 `FLOAT32_NEG_INF` 给 padding 填 -inf；手册明确禁止在 Ascend 上用
  `float("-inf")` 字面量，故本实现用有限大负数 `_NEG` 作为哨兵。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import next_pow2

__all__ = ["moe_gating_top_k_hash"]

NORM_SOFTMAX = 0
NORM_SIGMOID = 1
NORM_SOFTPLUS = 2

GROUP_SELECT_MODE_MAX = 0
GROUP_SELECT_MODE_SUM = 1

# 哨兵：不用 -inf 字面量（Ascend 上会退化），用有限大负数
_NEG = -1.0e30
# int32 索引哨兵
_IDX_BIG = 1 << 30


@triton.jit
def _normalize_flat(XV, EMASK, NORM_TYPE: tl.constexpr, NEG: tl.constexpr):
    """一行 [BLK] 的归一化（fp32），对应 ComputeX。"""
    if NORM_TYPE == 0:  # softmax
        mx = tl.max(tl.where(EMASK, XV, NEG), axis=0)
        ev = tl.where(EMASK, tl.exp(XV - mx), 0.0)
        sm = tl.sum(ev, axis=0)
        return ev * (1.0 / sm)
    elif NORM_TYPE == 1:  # sigmoid
        return 1.0 / (1.0 + tl.exp(-XV))
    else:  # softplus: sqrt(ln(1+exp(x)))
        return tl.sqrt(tl.log(1.0 + tl.exp(XV)))


@triton.jit
def _apply_norm_to_selected(XSEL, MX, SM, NORM_TYPE: tl.constexpr):
    """对已 gather 出来的 k 个原始值施加同一归一化。

    softmax 复用整行的 max/sum（MX/SM 仅 softmax 时有效）。
    """
    if NORM_TYPE == 0:
        return tl.exp(XSEL - MX) * (1.0 / SM)
    elif NORM_TYPE == 1:
        return 1.0 / (1.0 + tl.exp(-XSEL))
    else:
        return tl.sqrt(tl.log(1.0 + tl.exp(XSEL)))


@triton.jit
def _gating_flat_kernel(
    X, BIAS, INPUT_IDS, TID2EID, Y, EXPERT_IDX, OUT,
    E, K, RSF, EPS,
    NORM_TYPE: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HASH: tl.constexpr,
    K_C: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
    E_BIG: tl.constexpr,
):
    """不分组（group_count==1）路径：对应 without_group.h L366-383 的逐行循环。"""
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E
    kmask = k_off < K

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    xn = _normalize_flat(xv, emask, NORM_TYPE, NEG)

    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)

    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    if HASH:
        # without_group.h L294-313: expertIdx = tid2eid[inputIds[row], :k]
        key = tl.load(INPUT_IDS + row).to(tl.int64)
        sel = tl.load(TID2EID + key * K + k_off.to(tl.int64), mask=kmask, other=0).to(tl.int32)
    else:
        # without_group.h L212-238 + L179-183: 按 (xNorm + bias) 降序取 top-k
        scores = xn
        if HAS_BIAS:
            scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
        scores = tl.where(emask, scores, NEG)
        # 重复 argmax（同级取最小专家号）
        sel = tl.zeros([BLK_K], dtype=tl.int32)
        for i in tl.static_range(K_C):
            mxi = tl.max(scores, axis=0)
            idx = tl.min(tl.where(scores == mxi, e_off, E_BIG), axis=0)
            sel = tl.where(k_off == i, idx, sel)
            scores = tl.where(e_off == idx, NEG, scores)

    # y = xNorm[expertIdx]（注意取的是未加 bias 的 xNorm）
    xsel = tl.load(X + row * E + sel, mask=kmask, other=0.0).to(tl.float32)
    yv = _apply_norm_to_selected(xsel, mx, sm, NORM_TYPE)

    # without_group.h L253-268: sigmoid / softplus 额外做一次 sum 重归一化
    if NORM_TYPE == 1 or NORM_TYPE == 2:
        s = tl.sum(tl.where(kmask, yv, 0.0), axis=0) + EPS
        yv = yv / s
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=kmask)
    tl.store(EXPERT_IDX + row * K + k_off, sel, mask=kmask)


@triton.jit
def _gating_group_kernel(
    X, BIAS, Y, EXPERT_IDX, OUT,
    E, K, RSF, EPS, P,
    NORM_TYPE: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SELECT_MODE: tl.constexpr,
    K_GROUP: tl.constexpr,
    K_C: tl.constexpr,
    G_C: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_G: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
    E_BIG: tl.constexpr,
):
    """分组路径：对应 generalized.h L645-662（EKFullload 同数学）。

    实现说明：刻意全部使用 **1-D 张量 + 编译期组循环** 表达。Ascend Triton
    后端对 2-D 归约（如 `tl.max(x, axis=1)`）会触发 LLVM 断言
    `PointerUnion::get: Invalid accessor called`（实测崩溃），故规避所有 2-D reduce。
    代价是组得分计算为 O(G*E)；G 以 constexpr 传入，建议 G <= 128。
    """
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    g_off = tl.arange(0, BLK_G)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:  # softmax（对整行所有专家归一化）
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)
        xn = ev * (1.0 / sm)
    elif NORM_TYPE == 1:
        xn = 1.0 / (1.0 + tl.exp(-xv))
    else:
        xn = tl.sqrt(tl.log(1.0 + tl.exp(xv)))

    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    scores = xn
    if HAS_BIAS:
        scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
    scores = tl.where(emask, scores, NEG)

    # --- 组得分：组内 top2 之和 (mode=1) 或 top1 (mode=0)  [generalized.h L284-321]
    gscore = tl.full([BLK_G], NEG, dtype=tl.float32)
    # ⚠️ 这里必须是**运行期**循环：`tl.static_range(G_C)` 会把组循环完全展开
    # （生产 G 可达 128），每份展开体都要一份 UB 缓冲，实测
    # `ub overflow, requires 7013888 bits`（192KB 的 4.5 倍）。
    # 换成 `range` 后生成 scf.for，UB 只按单次迭代计。
    for g in range(0, G_C):
        gm = (e_off >= g * P) & (e_off < (g + 1) * P)
        sv = tl.where(gm, scores, NEG)
        v1 = tl.max(sv, axis=0)
        i1 = tl.min(tl.where(sv == v1, e_off, E_BIG), axis=0)
        v2 = tl.max(tl.where(e_off == i1, NEG, sv), axis=0)
        if GROUP_SELECT_MODE == 1:
            gsv = v1 + v2
        else:
            gsv = v1
        gscore = tl.where(g_off == g, gsv, gscore)

    # --- 组得分降序取前 K_GROUP 个组，同时把选中组的全部专家并入候选集
    #     [generalized.h L343-366 选组 + L370-514 归并选中组的有序表]
    #     注：归并 k_group 个组的有序表后取 top-k，等价于在这些组覆盖的专家集合上取全局 top-k。
    #     这里把「选组」和「收集候选」融合在同一个编译期循环里，避免用 tl.sum 从 int32
    #     向量里取标量（实测该写法被错误地折叠成最后一个元素）。
    cand = tl.full([BLK_E], NEG, dtype=tl.float32)
    gs = gscore
    # 同样用运行期循环：`tl.static_range` 展开会让每份展开体各占一份 UB
    # （实测 K_GROUP/K_C 展开后仍报 `ub overflow, requires 2819584 bits`）。
    for i in range(0, K_GROUP):
        gmx = tl.max(gs, axis=0)
        gi = tl.min(tl.where(gs == gmx, g_off, BLK_G), axis=0)
        cand = tl.where((e_off >= gi * P) & (e_off < (gi + 1) * P), scores, cand)
        gs = tl.where(g_off == gi, NEG, gs)

    # --- 归并取全局 top-k  [generalized.h L370-514]
    sel = tl.zeros([BLK_K], dtype=tl.int32)
    for i in range(0, K_C):
        mxi = tl.max(cand, axis=0)
        idx = tl.min(tl.where(cand == mxi, e_off, E_BIG), axis=0)
        sel = tl.where(k_off == i, idx, sel)
        cand = tl.where(e_off == idx, NEG, cand)

    # --- y = xNorm[sel]，sigmoid 再重归一化  [generalized.h L517-554]
    xsel = tl.load(X + row * E + sel, mask=k_off < K, other=0.0).to(tl.float32)
    yv = _apply_norm_to_selected(xsel, mx, sm, NORM_TYPE)
    if NORM_TYPE == 1:
        s = tl.sum(tl.where(k_off < K, yv, 0.0), axis=0) + EPS
        yv = yv / s
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=k_off < K)
    tl.store(EXPERT_IDX + row * K + k_off, sel, mask=k_off < K)


def moe_gating_top_k_hash(
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    k: int = 1,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = GROUP_SELECT_MODE_MAX,
    renorm: int = 0,
    norm_type: int = NORM_SOFTMAX,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 MoeGatingTopKHash。

    Args:
        x:        [N, E]  fp32/fp16/bf16
        bias:     [E] 或 None，与 x 同 dtype
        input_ids:[N] int32/int64，hash 模式必需
        tid2eid:  [V, >=k] int32/int64，hash 模式必需
        其余为算子属性。

    Returns:
        (y [N,k] 同 x dtype, expert_idx [N,k] int32, out [N,E] fp32)
        out_flag=False 时 out 为全 0（AscendC 内核不写该输出）。
    """
    assert x.dim() == 2, f"x 期望 [N,E]，实际 {tuple(x.shape)}"
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), x.dtype
    assert renorm == 0, "AscendC tiling 只支持 renorm=0"
    N, E = x.shape
    assert 0 < k <= E, f"k={k} 需满足 0<k<=E={E}"
    assert group_count >= 1 and E % group_count == 0, "E 必须能被 group_count 整除"
    assert 1 <= k_group <= group_count, f"k_group={k_group} 需在 [1, {group_count}]"
    if norm_type == NORM_SOFTPLUS:
        assert group_count == 1, "softplus 仅支持 group_count==1"
    assert norm_type in (NORM_SOFTMAX, NORM_SIGMOID, NORM_SOFTPLUS), norm_type
    assert group_select_mode in (GROUP_SELECT_MODE_MAX, GROUP_SELECT_MODE_SUM), group_select_mode

    P = E // group_count
    HASH = input_ids is not None
    assert HASH == (tid2eid is not None), "input_ids / tid2eid 必须同时给出或同时为 None"

    device = x.device
    x = x.contiguous()
    y = torch.empty((N, k), dtype=x.dtype, device=device)
    expert_idx = torch.empty((N, k), dtype=torch.int32, device=device)
    # out_flag=False 时 AscendC 不写该输出；这里置 0 保证确定性
    out = torch.zeros((N, E), dtype=torch.float32, device=device)

    has_bias = bias is not None
    if has_bias:
        assert bias.numel() == E, f"bias 长度 {bias.numel()} != E={E}"
        bias_t = bias.contiguous()
    else:
        bias_t = x  # 占位指针；HAS_BIAS=False 时不会被 load

    if HASH:
        input_ids_t = input_ids.contiguous()
        tid2eid_t = tid2eid.contiguous()
        assert tid2eid_t.shape[-1] >= k, f"tid2eid 最后一维 {tid2eid_t.shape[-1]} < k={k}"
    else:
        input_ids_t = tid2eid_t = x

    if group_count == 1:
        _gating_flat_kernel[(N,)](
            x, bias_t, input_ids_t, tid2eid_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps),
            NORM_TYPE=int(norm_type),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(has_bias),
            HASH=bool(HASH),
            K_C=int(k),
            BLK_E=max(next_pow2(E), 2),
            BLK_K=max(next_pow2(k), 2),
            NEG=_NEG,
            E_BIG=_IDX_BIG,
        )
    else:
        assert group_count <= 128, "分组路径用编译期组循环，group_count 需 <= 128"
        _gating_group_kernel[(N,)](
            x, bias_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps), P,
            NORM_TYPE=int(norm_type),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(has_bias),
            GROUP_SELECT_MODE=int(group_select_mode),
            K_GROUP=int(k_group),
            K_C=int(k),
            G_C=int(group_count),
            BLK_E=max(next_pow2(E), 2),
            BLK_G=max(next_pow2(group_count), 2),
            BLK_K=max(next_pow2(k), 2),
            NEG=_NEG,
            E_BIG=_IDX_BIG,
        )
    return y, expert_idx, out
