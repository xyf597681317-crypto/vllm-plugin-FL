# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""moe_gating_top_k -- AscendC to Triton port.

Source: csrc/ascend/moe/moe_gating_top_k/
(built into the OPP payload as the op file moe_gating_top_k / moe_gating_top_k_apt).

Interface (op_host/moe_gating_top_k_def.cpp L22-57, op_host/moe_gating_top_k_infershape.cpp
L115-128 and L135-144; torch signature in moe_gating_top_k_torch_adpt.h L19-31 and L49-51):
    Input : x        [rows, expertNum]  fp16 / fp32 / bf16   (REQUIRED, 2-D)
            bias     [expertNum]        same dtype as x      (OPTIONAL)
    Output: y         [rows, k]          same dtype as x
            expert_idx[rows, k]          int32
            out       [rows, expertNum]  fp32
    Attr  : k (int, required), k_group (int, 1), group_count (int, 1),
            group_select_mode (int, 0), renorm (int, 0), norm_type (int, 0),
            out_flag (bool, false), routed_scaling_factor (float, 1.0), eps (float, 1e-20)

norm_type (op_kernel/common.h L41-42):
    0 = softmax, 1 = sigmoid.  There is no softplus in this operator (unlike the sibling
    moe_gating_top_k_hash, whose common.h also defines NORM_TYPE_SOFTPLUS).

----------------------------------------------------------------------------------------
Formula sources
----------------------------------------------------------------------------------------
[1] Ungrouped body: op_kernel/moe_gating_top_k_without_group.h
    L115-175  ComputeX              -- cast T -> fp32, then
                                       norm_type==1: xNorm = sigmoid(x)              (L128-131)
                                       norm_type==0: xNorm = exp(x - max) / sum(exp) (L132-157)
                                       routing = xNorm + bias (bias present) or xNorm (L158-162)
                                       tail padding of the last Sort32 repeat = -inf   (L164-173)
    L178-188  CopuOutXNorm          -- out = xNorm, **bias is NOT added** (fp32 copy)
    L191-217  SelectTopKExpertIdx   -- Sort<(value, expertId)> descending over the whole row
                                       and keep the first k expert ids
    L220-259  SelectTopKExpertScore -- y = xNorm[expertIdx] (again without bias, L229)
                                       needRenorm = (normType==1) || (normType==0 && renorm==1)
                                                                                     (L232-233)
                                       if needRenorm: y = y / (sum(y) + eps)          (L234-249)
                                       y = y * routedScalingFactor                    (L251)
                                       cast fp32 -> T with CAST_RINT                   (L253-256)
    L262-272  CopyOut               -- y and expertIdx written per row
    L323-336  Process               -- per row loop; `out` only written when outFlag_ (L329-331)

[2] Grouped body: op_kernel/moe_gating_top_k_generalized.h
    L153-226  ComputeX              -- as [1], but the -inf padding is written at the tail of
                                       every group (L166-176)
    L264-368  SelectTopKGroupIndex  -- group score = sum of the top-2 routing values inside the
                                       group when groupSelectMode==1 (L285-306), or the top-1
                                       (max) when groupSelectMode==0 (L307-322); the groups are
                                       then sorted by descending group score and the first
                                       k_group group indices are kept (L344-367)
    L371-515  SelectTopKExpertIdx   -- merge the sorted tables of the selected groups and take
                                       the global top-k over the union
    L518-556  SelectTopKExpertScore -- identical to [1] L220-259, including needRenorm (L529-530)
    L559-575  CumputeActualTopKExpertId -- decode the merged position back to a global expert id
    L651-665  Process               -- per row loop; `out` only written when outFlag_

[3] High-performance specialisation (tiling key TILING_KEY_EXPERTNUM_GROUPNUM_ALIGN_HIGH_PERF):
    op_kernel/moe_gating_top_k_e_k_fullload.h (L125-160 / L160-235 / L235-276 / L276-...)
    selected by op_host/moe_gating_top_k_tiling.cpp L551-558 when
        expertNum==256 && group_count==8 && k_group==4 && k<=32 && bias present &&
        group_select_mode==1 (SUM) && renorm==0 && norm_type==1 (sigmoid) && !out_flag
    It relies on E%32==0 and P==32 and therefore needs no -inf padding, but its mathematics
    is exactly [2] (group score = top-2 sum).

Kernel selection (op_host/moe_gating_top_k_tiling.cpp L559-563):
    group_count==1 || group_count==expertNum || k_group==group_count -> ungrouped body [1]
    otherwise                                                       -> grouped body  [2]
In the first case every expert is a candidate, so no group score is computed at all.

Tiling constraints (op_host/moe_gating_top_k_tiling.cpp L190-249):
    norm_type in {0,1}; group_select_mode in {0,1}; renorm in {0,1};
    expertNum % group_count == 0; perGroupExpertCount >= 1 (and >= 2 for SUM);
    k <= k_group * perGroupExpertCount; k_group <= group_count.

----------------------------------------------------------------------------------------
Precision notes
----------------------------------------------------------------------------------------
* All normalisation and selection arithmetic is done in **fp32** (ComputeX casts T to fp32
  first), and y is rounded back to T with CAST_RINT (round to nearest, ties to even), which is
  what ``.to(Y.dtype.element_ty)`` does in Triton.
* ``out`` is xNorm, i.e. the normalised value **without bias**; bias only participates in the
  top-k selection ([1] L159 vs L229).
* sigmoid always renormalises; softmax renormalises only when ``renorm == 1`` ([1] L232-233).
* Tie handling: AscendC sorts with the hardware Sort32 primitive, whose order for equal scores
  is not specified by the sources.  This implementation is deterministic and prefers the
  **smaller expert id** on equal scores.
* The AscendC sources use ``FLOAT32_NEG_INF`` for padding; the Ascend backend manual discourages
  the ``float("-inf")`` literal, so a finite large negative sentinel is used instead.  The
  sentinel is never stored to an output.

----------------------------------------------------------------------------------------
Call-site compatibility
----------------------------------------------------------------------------------------
vllm-fl calls this operator at
  * dispatch/backends/vendor/ascend/device/device_op.py L147-159
  * dispatch/backends/vendor/ascend/impl/device_operator.py L138-152
always as ``topk_weights, topk_ids, out = torch.ops._C_ascend.moe_gating_top_k(x, k=...,
k_group=..., group_count=..., group_select_mode=..., renorm=..., norm_type=..., out_flag=...,
routed_scaling_factor=..., eps=..., bias_opt=...)``, i.e. three return values.

The torch schema is ``(Tensor x, int k, int k_group, int group_count, int group_select_mode,
int renorm, int norm_type, bool out_flag, float routed_scaling_factor, float eps,
Tensor? bias_opt=None) -> (Tensor y, Tensor expert_idx, Tensor out)`` (csrc/ascend/
torch_binding.cpp L1188-1191).  This implementation keeps ``bias`` (the historical Triton-side
positional name) as the 11th parameter and additionally accepts the schema name ``bias_opt`` as
a keyword; ``bias_opt`` wins when both are given.  The operator always returns three tensors,
exactly like moe_gating_top_k_torch_adpt.h L49-51/L70, so the arity matches the call sites even
when ``out_flag`` is false (in that case the AscendC kernel leaves ``out`` unwritten; a zero
tensor is returned here so the behaviour is deterministic).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import next_pow2

__all__ = ["moe_gating_top_k"]

NORM_TYPE_SOFTMAX = 0
NORM_TYPE_SIGMOID = 1

GROUP_SELECT_MODE_MAX = 0
GROUP_SELECT_MODE_SUM = 1

# Finite large negative sentinel (no float("-inf") literal: it degrades on Ascend).
_NEG = -1.0e30
# int32 index sentinel used to pick the smallest expert id among equal scores.
_IDX_BIG = 1 << 30


@triton.jit
def _normalize_row(XV, EMASK, NORM_TYPE: tl.constexpr, NEG: tl.constexpr):
    """ComputeX normalisation of one [BLK_E] row in fp32 ([1] L128-157).

    norm_type==1 -> sigmoid(x)                  [1] L128-131
    norm_type==0 -> exp(x - max) / sum(exp(..)) [1] L132-157
    """
    if NORM_TYPE == 0:  # softmax
        mx = tl.max(tl.where(EMASK, XV, NEG), axis=0)
        ev = tl.where(EMASK, tl.exp(XV - mx), 0.0)
        sm = tl.sum(ev, axis=0)
        return ev * (1.0 / sm)
    else:  # sigmoid
        return 1.0 / (1.0 + tl.exp(-XV))


@triton.jit
def _renormalize_from_raw(XSEL, MX, SM, NORM_TYPE: tl.constexpr):
    """Re-apply the row normalisation to the k raw values gathered from x ([1] L229)."""
    if NORM_TYPE == 0:  # softmax
        return tl.exp(XSEL - MX) * (1.0 / SM)
    else:
        return 1.0 / (1.0 + tl.exp(-XSEL))


@triton.jit
def _apply_renorm(YV, KMASK, EPS, RENORM: tl.constexpr):
    """needRenorm = (normType==1) || (normType==0 && renorm==1)  ([1] L232-233, [2] L529-530)."""
    if RENORM:
        s = tl.sum(tl.where(KMASK, YV, 0.0), axis=0) + EPS
        return YV / s
    return YV


@triton.jit
def _moe_gating_flat_kernel(
    X, BIAS, Y, IDX, OUT,
    E, K, RSF, EPS,
    NORM_TYPE: tl.constexpr,
    RENORM: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    K_C: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
    E_BIG: tl.constexpr,
):
    """Ungrouped body: one program per row, mirroring [1] L323-336.

    Used when the tiling selects TILING_KEY_WITHOUT_GROUP, i.e. group_count == 1,
    group_count == expertNum or k_group == group_count (op_host tiling L559-560).

    Everything is expressed with 1-D tensors: the Ascend Triton backend crashes on 2-D
    reductions such as ``tl.max(x, axis=1)``, so no 2-D reduce is used here.
    """
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E
    kmask = k_off < K

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    xn = _normalize_row(xv, emask, NORM_TYPE, NEG)
    # The softmax statistics are recomputed so that the gathered values can be re-normalised
    # without a second row reduction (same shape as the sibling moe_gating_top_k_hash port).
    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:  # softmax
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)

    # CopuOutXNorm: out = xNorm (bias is not added), only when out_flag ([1] L178-188, L329-331)
    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    # routing = xNorm + bias ([1] L158-162); the tail is padded so it can never be selected
    scores = xn
    if HAS_BIAS:
        scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
    scores = tl.where(emask, scores, NEG)

    # SelectTopKExpertIdx: top-k of `routing` by descending order ([1] L191-217).
    # Implemented as repeated argmax; on equal scores the smaller expert id wins.
    sel = tl.zeros([BLK_K], dtype=tl.int32)
    for i in tl.static_range(K_C):
        mxi = tl.max(scores, axis=0)
        idx = tl.min(tl.where(scores == mxi, e_off, E_BIG), axis=0)
        sel = tl.where(k_off == i, idx, sel)
        scores = tl.where(e_off == idx, NEG, scores)

    # SelectTopKExpertScore ([1] L220-259): y is gathered from xNorm WITHOUT bias
    xsel = tl.load(X + row * E + sel, mask=kmask, other=0.0).to(tl.float32)
    yv = _renormalize_from_raw(xsel, mx, sm, NORM_TYPE)
    yv = _apply_renorm(yv, kmask, EPS, RENORM)
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=kmask)
    tl.store(IDX + row * K + k_off, sel, mask=kmask)


@triton.jit
def _moe_gating_group_kernel(
    X, BIAS, Y, IDX, OUT,
    E, K, RSF, EPS, P,
    NORM_TYPE: tl.constexpr,
    RENORM: tl.constexpr,
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
    """Grouped body (TILING_KEY_GENERALIZED), mirroring [2].

    group score   = sum of the top-2 (mode SUM) or the top-1 (mode MAX) inside the group
                    [2] L285-322
    group select  = first k_group groups by descending group score           [2] L344-367
    expert select = global top-k over the union of the selected groups       [2] L371-515
    score/renorm  = same as the ungrouped body                               [2] L518-556

    Implementation note: exactly like the sibling moe_gating_top_k_hash port, everything is
    1-D and the group loop is a **runtime** loop.  ``tl.static_range`` would fully unroll it
    and every unrolled body needs its own UB buffer, which overflows on Ascend
    ("ub overflow, requires ... bits"); with ``range`` the loop becomes an scf.for and UB is
    accounted per single iteration.
    """
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    g_off = tl.arange(0, BLK_G)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E
    kmask = k_off < K

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    xn = _normalize_row(xv, emask, NORM_TYPE, NEG)
    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:  # softmax
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)

    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    scores = xn
    if HAS_BIAS:
        scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
    scores = tl.where(emask, scores, NEG)

    # --- group score: top-2 sum (SUM) or top-1 (MAX) inside each group [2] L285-322
    gscore = tl.full([BLK_G], NEG, dtype=tl.float32)
    for g in range(0, G_C):
        gm = (e_off >= g * P) & (e_off < (g + 1) * P)
        sv = tl.where(gm, scores, NEG)
        v1 = tl.max(sv, axis=0)
        i1 = tl.min(tl.where(sv == v1, e_off, E_BIG), axis=0)
        gsv = v1
        if GROUP_SELECT_MODE == 1:  # sum of the group top-2
            v2 = tl.max(tl.where(e_off == i1, NEG, sv), axis=0)
            gsv = v1 + v2
        gscore = tl.where(g_off == g, gsv, gscore)

    # --- keep the first k_group groups by descending group score and collect their experts
    #     [2] L344-367 group selection + L371-515 merging of the selected tables.
    #     Merging the per-group sorted tables and taking the top-k is the same as taking the
    #     global top-k over the union of the selected groups, so both steps are fused here.
    cand = tl.full([BLK_E], NEG, dtype=tl.float32)
    gs = gscore
    for i in range(0, K_GROUP):
        gmx = tl.max(gs, axis=0)
        gi = tl.min(tl.where(gs == gmx, g_off, BLK_G), axis=0)
        cand = tl.where((e_off >= gi * P) & (e_off < (gi + 1) * P), scores, cand)
        gs = tl.where(g_off == gi, NEG, gs)

    # --- global top-k over the candidate set
    sel = tl.zeros([BLK_K], dtype=tl.int32)
    for i in range(0, K_C):
        mxi = tl.max(cand, axis=0)
        idx = tl.min(tl.where(cand == mxi, e_off, E_BIG), axis=0)
        sel = tl.where(k_off == i, idx, sel)
        cand = tl.where(e_off == idx, NEG, cand)

    # --- y = xNorm[sel] (no bias), renormalise, scale [2] L518-556
    xsel = tl.load(X + row * E + sel, mask=kmask, other=0.0).to(tl.float32)
    yv = _renormalize_from_raw(xsel, mx, sm, NORM_TYPE)
    yv = _apply_renorm(yv, kmask, EPS, RENORM)
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=kmask)
    tl.store(IDX + row * K + k_off, sel, mask=kmask)


def moe_gating_top_k(
    x: torch.Tensor,
    k: int,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = GROUP_SELECT_MODE_MAX,
    renorm: int = 0,
    norm_type: int = NORM_TYPE_SOFTMAX,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    bias: torch.Tensor | None = None,
    bias_opt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton equivalent of the AscendC MoeGatingTopK operator.

    Args:
        x: [rows, expertNum], fp16 / fp32 / bf16.
        k: number of selected experts per row (attribute, required).
        k_group: number of selected groups (attribute).
        group_count: number of expert groups (attribute).
        group_select_mode: 0 = group score is the group max, 1 = sum of the group top-2.
        renorm: 0 / 1; only affects the softmax path.
        norm_type: 0 = softmax, 1 = sigmoid.
        out_flag: whether the third output is written.
        routed_scaling_factor: scale applied to y after renormalisation.
        eps: added to the renormalisation denominator.
        bias: [expertNum] optional, same dtype as x; only affects the top-k selection.
        bias_opt: alias of ``bias`` using the torch-schema parameter name
            (moe_gating_top_k_torch_adpt.h L30).  Takes precedence when both are given.

    Returns:
        (y [rows, k] with the dtype of x,
         expert_idx [rows, k] int32,
         out [rows, expertNum] fp32; zero-filled when out_flag is false because the AscendC
         kernel does not write it in that case - see [1] L329-331).
    """
    if bias_opt is not None:
        bias = bias_opt

    assert x.dim() == 2, f"x must be [rows, expertNum], got {tuple(x.shape)}"
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), x.dtype
    rows, E = x.shape
    assert 0 < k <= E, f"k={k} must satisfy 0 < k <= expertNum={E}"
    assert norm_type in (NORM_TYPE_SOFTMAX, NORM_TYPE_SIGMOID), \
        "norm_type must be 0 (softmax) or 1 (sigmoid); this operator has no softplus"
    assert group_select_mode in (GROUP_SELECT_MODE_MAX, GROUP_SELECT_MODE_SUM), group_select_mode
    assert renorm in (0, 1), "tiling accepts renorm 0 or 1 (op_host tiling L217-219)"
    assert group_count >= 1 and E % group_count == 0, "expertNum must be divisible by group_count"
    assert 1 <= k_group <= group_count, f"k_group={k_group} must be in [1, {group_count}]"

    per_group = E // group_count
    assert per_group >= 1, "perGroupExpertCount must be >= 1 (op_host tiling L229-231)"
    if group_select_mode == GROUP_SELECT_MODE_SUM:
        assert per_group >= 2, "group_select_mode=1 needs at least two experts per group"
    assert k <= k_group * per_group, "tiling requires k <= k_group * perGroupExpertCount"

    if bias is not None:
        assert bias.numel() == E, f"bias length {bias.numel()} != expertNum={E}"
        assert bias.dtype == x.dtype, "bias must have the same dtype as x"

    # Tiling kernel selection (op_host/moe_gating_top_k_tiling.cpp L559-563): with
    # group_count == 1, group_count == expertNum or k_group == group_count the ungrouped body
    # is used and no group score is computed.
    ungrouped = group_count == 1 or group_count == E or k_group == group_count

    # needRenorm = (normType==1) || (normType==0 && renorm==1)   ([1] L232-233)
    need_renorm = (norm_type == NORM_TYPE_SIGMOID) or (
        norm_type == NORM_TYPE_SOFTMAX and renorm == 1
    )

    device = x.device
    x = x.contiguous()
    bias_t = bias.contiguous() if bias is not None else x
    y = torch.empty((rows, k), dtype=x.dtype, device=device)
    expert_idx = torch.empty((rows, k), dtype=torch.int32, device=device)
    # The AscendC host adapter always allocates this output (torch_adpt.h L51) and the kernel
    # writes it only when out_flag is set; zero it here so the result is deterministic.
    out = torch.zeros((rows, E), dtype=torch.float32, device=device)

    if ungrouped:
        _moe_gating_flat_kernel[(rows,)](
            x, bias_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps),
            NORM_TYPE=int(norm_type),
            RENORM=bool(need_renorm),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(bias is not None),
            K_C=int(k),
            BLK_E=max(next_pow2(E), 2),
            BLK_K=max(next_pow2(k), 2),
            NEG=_NEG,
            E_BIG=_IDX_BIG,
        )
    else:
        assert group_count <= 128, \
            "the grouped body needs group_count as a constexpr (it drives the group loop), <= 128"
        _moe_gating_group_kernel[(rows,)](
            x, bias_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps), per_group,
            NORM_TYPE=int(norm_type),
            RENORM=bool(need_renorm),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(bias is not None),
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
