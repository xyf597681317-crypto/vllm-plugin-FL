# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementations of the two CANN MoE routing operators.

`MoeGatingTopK` (softmax + top-k + optional renormalize over the expert axis) and
`MoeInitRoutingCustom` (`npu_moe_init_routing_v2` in count mode).  Both contracts
below were measured on the 105 host against the native operator, using a 4-token
/ topk=2 / 8-expert case with duplicate expert ids:

    topk_softmax
        gating_output   : [num_tokens, num_experts], bf16/fp16/fp32, contiguous
        topk_weights    : [num_tokens, topk], fp32, preallocated by the caller
        topk_ids        : [num_tokens, topk], int32, preallocated by the caller
        renormalize     : Python bool; divide by the sum of the selected weights

        weights = softmax(gating_output.float(), dim=-1), then the `topk`
        largest entries per row, descending.  Exact ties are resolved by the
        lowest expert index; `torch.topk` leaves ties implementation defined, so
        the equality cases are reported rather than claimed to match.

    moe_init_routing (count mode, full active range)
        hidden_states   : [num_tokens, hidden], bf16/fp16, contiguous
        topk_ids        : [num_tokens, topk], int32, ids in [0, num_experts)
        routed          : [num_tokens * topk, hidden], dtype of hidden_states,
                          route rows grouped by expert id, stable inside an
                          expert (ascending route index t * topk + k)
        row_indices     : [num_tokens * topk], int32, `row_indices[r]` is the
                          row of `routed` that holds route `r`; that is the
                          inverse of the grouping permutation, which is exactly
                          what `npu_moe_token_unpermute` consumes
        expert_tokens   : [num_experts], int64, number of routes per expert

    The native operator drops routes outside `active_expert_range` and marks
    them -1 in `row_indices`; this module implements only the full-range case
    and returns `None` for a sharded range, so the caller keeps the native
    operator for expert-parallel layouts.

Performance model (measured on the 105 host, T=4096, H=2048, topk=8, E=128):

* The 134 MB `routed` copy is the bandwidth floor of this operator: 0.18 ms
  measured for the row scatter, ~0.21 ms for the whole native operator.
* Two store shapes are pathological on this backend and the sort is built to
  avoid both.  A *2-D* store whose per-row base is a vector is lowered to one
  scalar store per lane (11.4 ms for the KV-cache copy where the 1-D form costs
  0.097 ms), and a *masked* store from many programs into one small buffer is
  not race safe: two programs writing the same store granule lose one of the
  updates, which left holes in `row_indices` and produced out-of-range rows (a
  device trap) in the grouped copy.  Both kernels below therefore write with
  1-D addresses, one writer per lane, dense stores.
* `tl.cumsum` is a slow primitive here when it runs over a long axis: a scan
  over `num_experts * num_routes` lane costs 1.4-2.0 ms and does not get faster
  when the chunk is split (measured 0.48-0.51 ns per lane, i.e. one lane per
  cycle per core).  The scan is therefore always blocked, and the two axes are
  treated differently: the chunk axis keeps a short scan inside a
  `[chunk-slot, expert-block]` tile (contiguous rows), while the expert axis is
  a single 256-lane scan.  The stable counting sort is split into: (a) a
  per-chunk histogram, (b) an exclusive prefix over chunks per expert block,
  (c) an exclusive prefix over experts, and (d) a program that owns
  `_CHUNK_PER_PROG` chunks and computes the within-chunk rank as an all-pairs
  equality matrix with a strictly-lower triangular mask (`CHUNK^2` plain vector
  lanes: 0.13 ms for the whole route axis) and writes `row_indices` with one
  dense store.  The grouped rows are then produced by a scatter whose
  destination is one scalar row index per route, so no grouped-order index
  array is ever built.
* A kernel launch costs ~36 us of *device* dispatch on this host (measured with
  20 back-to-back launches and a single synchronize), so with five launches
  ~0.18 ms of the ~0.46 ms total is dispatch.  That is why the pipeline favours
  fewer, fatter kernels and why the per-kernel work is batched into fewer
  programs (see `_CHUNK_PER_PROG` and `_PREFIX_NB`); it is also the reason a
  further merge of the histogram into the prefix kernel, which needs a
  device-wide barrier, was not attempted.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# The Ascend backend rejects a coreDim of exactly 65536.
_MAX_GRID = 65535
# Cap on the per-chunk expert-offset scratch (int32 elements): 16 MB.
_MAX_OFFSETS = 1 << 22
# Routes per histogram / rank chunk.  The rank is an all-pairs equality matrix,
# so its cost grows with CHUNK^2 while the chunk-prefix and histogram cost grows
# with the chunk count; 64 is the measured optimum (rank 0.22 ms + prefix
# 0.19 ms at the production shape).
_CHUNK = 64
# Chunks per rank program. A single trivial kernel costs ~36 us of device
# dispatch on this host, and one chunk program was measured at 0.375 us, i.e.
# the rank kernel is program-count bound rather than lane bound: folding four
# chunks into one program measured 0.192 ms -> 0.132 ms at the production
# shape, while eight chunks regressed (0.161 ms).  The histogram and the chunk
# prefix were swept the same way and prefer one chunk / one expert per program.
_CHUNK_PER_PROG = 4
# Chunk-prefix tile: `NCP * NB` int32 lanes of UB (32 KiB) and the widest expert
# block the tile allows. 16 experts per program measured 0.035 ms at the
# production shape against 0.125 ms for the per-expert form.
_PREFIX_LANES = 8192
_PREFIX_NB = 32
# Routes per gather/scatter program in the row copy.
_ROW_BT = 16
# The row copy aims for at least this many programs before it trades BT away.
_ROW_MIN_GRID = 96
# Hidden columns per gather program; 2048 fp32 lanes keep the tile at 65536 bits.
_BLK_H = 2048
# One top-k program holds BT tokens x next_pow2(experts) fp32 lanes in UB.
_TOPK_LANES = 2048
# Narrower rows are dominated by masking overhead and no production hidden size
# is that small; those shapes fall back to the native operator.
_MIN_H = 64
_MAX_EXPERTS = 4096
# `topk` is a constexpr unrolled loop; the production value is 8.
_MAX_TOPK = 64
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _next_pow2(n: int) -> int:
    """Smallest power of two that is not smaller than `n`."""
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _topk_softmax_kernel(
        logits_ptr,
        weight_ptr,
        id_ptr,
        num_rows,
        num_experts,
        TOPK: tl.constexpr,
        BLK: tl.constexpr,
        BT: tl.constexpr,
        RENORM: tl.constexpr,
        TILE_STORE: tl.constexpr,
    ):
        """
        `BT` token rows per program, one `[BT, BLK]` fp32 tile.

        The tile holds several rows so every reduction along axis 1 runs on BT
        rows at once; the per-row variant spends the same instruction count on
        one row.  The `TOPK` selections are argmax scans with the already
        selected lanes masked to -inf, which keeps a deterministic lowest-index
        tie-break.

        `TILE_STORE` (topk a power of two) accumulates the k-th weight/index into
        a `[BT, TOPK]` tile and writes it as one contiguous 1-D vector;
        otherwise each k is stored as its own 1-D vector.
        """
        pid = tl.program_id(0)
        rows = pid * BT + tl.arange(0, BT)
        row_mask = rows < num_rows
        cols = tl.arange(0, BLK)
        col_mask = cols < num_experts
        logits = tl.load(
            logits_ptr + rows.to(tl.int64)[:, None] * num_experts + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=float("-inf"),
        ).to(tl.float32)
        exps = tl.where(
            col_mask[None, :],
            tl.exp(logits - tl.max(logits, axis=1)[:, None]),
            0.0,
        )
        denom = tl.sum(exps, axis=1)
        if TILE_STORE:
            kcols = tl.arange(0, TOPK)
            w_acc = tl.zeros((BT, TOPK), dtype=tl.float32)
            i_acc = tl.zeros((BT, TOPK), dtype=tl.int32)
        for k in tl.static_range(TOPK):
            index = tl.argmax(logits, axis=1).to(tl.int32)
            prob = tl.sum(tl.where(cols[None, :] == index[:, None], exps, 0.0), axis=1) / denom
            logits = tl.where(cols[None, :] == index[:, None], float("-inf"), logits)
            if TILE_STORE:
                w_acc = tl.where(kcols[None, :] == k, prob[:, None], w_acc)
                i_acc = tl.where(kcols[None, :] == k, index[:, None], i_acc)
            else:
                tl.store(weight_ptr + rows * TOPK + k, prob, mask=row_mask)
                tl.store(id_ptr + rows * TOPK + k, index, mask=row_mask)
        if TILE_STORE:
            if RENORM:
                w_acc = w_acc / tl.sum(w_acc, axis=1)[:, None]
            off = pid * (BT * TOPK) + tl.arange(0, BT * TOPK)
            limit = num_rows * TOPK
            m = off < limit
            tl.store(weight_ptr + off, tl.reshape(w_acc, (BT * TOPK,)), mask=m)
            tl.store(id_ptr + off, tl.reshape(i_acc, (BT * TOPK,)), mask=m)

    @triton.jit
    def _routing_hist_kernel(
        id_ptr,
        cnt_ptr,
        num_routes,
        BINS: tl.constexpr,
        E: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        """Per-chunk histogram of the expert ids, one program per chunk.

        Lanes past `num_routes` are filled with `E`, an index that is outside
        the real expert range but still inside the padded bin count, and bins
        `>= E` are never read back.  That keeps a partial last chunk exact
        whatever the backend does with an out-of-range lane (drop, clamp or
        wrap): all three land in a bin that is discarded.
        """
        pid = tl.program_id(0)
        cols = pid * CHUNK + tl.arange(0, CHUNK)
        ids = tl.load(id_ptr + cols, mask=cols < num_routes, other=E)
        hist = tl.histogram(ids, BINS)
        tl.store(cnt_ptr + pid * BINS + tl.arange(0, BINS), hist)

    @triton.jit
    def _routing_chunk_prefix_kernel(
        cnt_ptr,
        tot_ptr,
        num_chunks,
        BINS: tl.constexpr,
        NCP: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Exclusive prefix over the chunk axis, in place, one expert block per program.

        On return `cnt[c, e]` is the number of routes of expert `e` in chunks
        `< c`, and `tot[e]` is the total count of expert `e`.

        A program owns a *block* of `NB` experts and all `NCP >= num_chunks`
        chunk slots, so its loads and stores are `[chunk, expert-block]` tiles
        whose rows are contiguous runs of `NB` int32.  The Round-1 form (one
        program per expert, blocked scan over chunks) reads a single column of
        that matrix, i.e. one 4-byte transaction per lane: 131k transactions,
        measured 0.125 ms.  This form reads 64 B per row, measured 0.035 ms for
        the same data.  The scan runs along axis 0 (the chunk axis) and the
        totals fall out of the same tile, so no second pass is needed.
        """
        pid = tl.program_id(0)
        c = tl.arange(0, NCP)
        e = pid * NB + tl.arange(0, NB)
        off = c[:, None] * BINS + e[None, :]
        m = (c < num_chunks)[:, None] & (e < BINS)[None, :]
        cnt = tl.load(cnt_ptr + off, mask=m, other=0)
        tl.store(cnt_ptr + off, tl.cumsum(cnt, axis=0) - cnt, mask=m)
        tl.store(tot_ptr + e, tl.sum(cnt, axis=0), mask=e < BINS)

    @triton.jit
    def _routing_expert_prefix_kernel(
        tot_ptr, start_ptr, tok_ptr, num_experts, BINS: tl.constexpr
    ):
        """`start[e]` = number of routes of experts smaller than `e`.

        The totals are also converted to the int64 `expert_tokens` contract here
        so the caller's dtype needs no extra device copy.
        """
        e = tl.arange(0, BINS)
        totals = tl.load(tot_ptr + e, mask=e < num_experts, other=0)
        starts = tl.cumsum(totals, axis=0) - totals
        tl.store(start_ptr + e, starts, mask=e < num_experts)
        tl.store(tok_ptr + e, totals.to(tl.int64), mask=e < num_experts)

    @triton.jit
    def _routing_assign_kernel(
        id_ptr,
        off_ptr,
        start_ptr,
        row_ptr,
        num_routes,
        BINS: tl.constexpr,
        CHUNK: tl.constexpr,
        NCP: tl.constexpr,
    ):
        """One program per `NCP` chunks: the within-chunk rank of every route.

        `row_indices[route] = start[expert] + chunk_offset[chunk, expert] + rank`
        with `rank` = the number of earlier routes of the same expert inside the
        chunk.  The rank is an all-pairs equality matrix with a strictly-lower
        triangular mask (`CHUNK^2` plain vector lanes) instead of a cumsum per
        expert (`num_experts * num_routes` scan lanes, measured 0.48-0.51 ns per
        lane and not parallelizable further): 0.22 ms vs 1.99 ms at the
        production shape.

        The chunk is written by a single program with one dense store.  That
        single-writer structure is required, not an optimization: a masked
        store from many programs into one small buffer is not race safe here.
        The measured failure mode is a lost update whenever two programs touch
        the same store granule, which left holes in `row_indices` and produced
        out-of-range rows (a device trap) in the grouped copy.
        """
        pid = tl.program_id(0)
        i = tl.arange(0, CHUNK)
        lower = i[None, :] < i[:, None]
        for g in tl.static_range(NCP):
            chunk = pid * NCP + g
            cols = chunk * CHUNK + i
            valid = cols < num_routes
            ids = tl.load(id_ptr + cols, mask=valid, other=-1)
            eq = (ids[:, None] == ids[None, :]) & lower & valid[:, None]
            rank = tl.sum(eq.to(tl.int32), axis=1)
            base = tl.load(start_ptr + ids, mask=valid, other=0) + tl.load(
                off_ptr + chunk * BINS + ids, mask=valid, other=0
            )
            tl.store(row_ptr + cols, base + rank, mask=valid)

    @triton.jit
    def _routing_scatter_kernel(
        x_ptr,
        row_ptr,
        out_ptr,
        num_routes,
        H,
        TOPK: tl.constexpr,
        BT: tl.constexpr,
        BLK: tl.constexpr,
    ):
        """
        `routed[row_indices[r]] = hidden_states[r // topk]`.

        The destination row is one scalar per route, so each store is a 1-D
        vector at a scalar base and the grouped buffer never needs the grouped
        order materialized as an index array.  A hidden row wider than `BLK` is
        covered by grid axis 0 (one program per column chunk).  Element offsets
        are int64 because the route axis of a grouped buffer can exceed the
        int32 element range.
        """
        pid_h = tl.program_id(0)
        pid_r = tl.program_id(1)
        cols = pid_h * BLK + tl.arange(0, BLK)
        col_mask = cols < H
        for j in tl.static_range(BT):
            route = pid_r * BT + j
            if route < num_routes:
                row = tl.load(row_ptr + route)
                token = route // TOPK
                value = tl.load(x_ptr + token.to(tl.int64) * H + cols, mask=col_mask, other=0.0)
                tl.store(out_ptr + row.to(tl.int64) * H + cols, value, mask=col_mask)


def triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> bool:
    """
    Write `topk_weights` / `topk_ids` with the Triton `MoeGatingTopK` kernel.

    Args:
        topk_weights: `[num_tokens, topk]` fp32 output buffer, written in place.
        topk_ids: `[num_tokens, topk]` int32 output buffer, written in place.
        gating_output: `[num_tokens, num_experts]` bf16/fp16/fp32 router logits.
        renormalize: Python bool; divide by the sum of the selected weights.

    Returns:
        `True` when the kernel wrote both buffers, `False` when the layout is
        outside the supported set and the caller must use its reference path.
    """
    if not HAS_TRITON:
        return False
    if not isinstance(gating_output, torch.Tensor) or gating_output.dim() != 2:
        return False
    if gating_output.dtype not in _DTYPES or not gating_output.is_contiguous():
        return False
    if not isinstance(topk_weights, torch.Tensor) or topk_weights.dim() != 2:
        return False
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        return False
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.shape != topk_weights.shape:
        return False
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        return False
    num_tokens, num_experts = gating_output.shape
    if topk_weights.shape[0] != num_tokens:
        return False
    topk = topk_weights.shape[1]
    if topk < 1 or topk > num_experts or topk > _MAX_TOPK:
        return False
    if num_experts < 1 or num_experts > _MAX_EXPERTS:
        return False
    if num_tokens == 0:
        return True
    block = _next_pow2(num_experts)
    tokens_per_block = max(1, min(_TOPK_LANES // block, num_tokens))
    grid = (triton.cdiv(num_tokens, tokens_per_block),)
    _topk_softmax_kernel[grid](
        gating_output,
        topk_weights,
        topk_ids,
        num_tokens,
        num_experts,
        TOPK=topk,
        BLK=block,
        BT=tokens_per_block,
        RENORM=bool(renormalize),
        TILE_STORE=_next_pow2(topk) == topk,
    )
    return True


def triton_moe_init_routing(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    first_expert: int,
    local_experts: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run the Triton `MoeInitRoutingCustom` (count mode) equivalent.

    Args:
        hidden_states: `[num_tokens, hidden]` bf16/fp16, contiguous.
        topk_ids: `[num_tokens, topk]` int32 expert ids, all `< num_experts`.
        num_experts: global expert count of the routing space.
        first_expert: first expert of the local contiguous range.
        local_experts: number of locally active experts.

    Returns:
        `(routed, row_indices, expert_tokens)` with the native dtypes, or `None`
        for a sharded expert range (expert parallel), where the native operator
        keeps its -1 sentinel and truncated-range semantics.
    """
    if not HAS_TRITON:
        return None
    if first_expert != 0 or local_experts != num_experts:
        return None
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return None
    if hidden_states.dtype not in _DTYPES or not hidden_states.is_contiguous():
        return None
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.dim() != 2:
        return None
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        return None
    num_tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    if topk_ids.shape[0] != num_tokens or topk < 1:
        return None
    if hidden < _MIN_H or hidden > _BLK_H * _MAX_GRID:
        return None
    if num_experts < 1 or num_experts > _MAX_EXPERTS:
        return None
    num_routes = num_tokens * topk
    if num_routes == 0:
        return None
    num_chunks = triton.cdiv(num_routes, _CHUNK)
    # One spare bin absorbs the padding lanes of a partial last chunk; it must
    # stay a power of two for `tl.histogram`.
    bins = _next_pow2(num_experts + 1)
    if num_chunks > _MAX_GRID or num_chunks * bins > _MAX_OFFSETS:
        return None

    device = hidden_states.device
    offsets = torch.empty(num_chunks * bins, dtype=torch.int32, device=device)
    totals = torch.empty(bins, dtype=torch.int32, device=device)
    starts = torch.empty(bins, dtype=torch.int32, device=device)
    expert_tokens = torch.empty(num_experts, dtype=torch.int64, device=device)
    row_indices = torch.empty(num_routes, dtype=torch.int32, device=device)
    # The prefix tile spans every chunk slot (padded up to a power of two) and a
    # block of experts whose width keeps the tile inside _PREFIX_LANES.
    prefix_ncp = _next_pow2(num_chunks)
    prefix_nb = max(1, min(_PREFIX_NB, _PREFIX_LANES // prefix_ncp))
    _routing_hist_kernel[(num_chunks,)](
        topk_ids, offsets, num_routes, BINS=bins, E=num_experts, CHUNK=_CHUNK
    )
    _routing_chunk_prefix_kernel[(triton.cdiv(bins, prefix_nb),)](
        offsets, totals, num_chunks, BINS=bins, NCP=prefix_ncp, NB=prefix_nb
    )
    _routing_expert_prefix_kernel[(1,)](
        totals, starts, expert_tokens, num_experts, BINS=bins
    )
    _routing_assign_kernel[(triton.cdiv(num_chunks, _CHUNK_PER_PROG),)](
        topk_ids, offsets, starts, row_indices, num_routes, BINS=bins,
        CHUNK=_CHUNK, NCP=_CHUNK_PER_PROG
    )
    routed = torch.empty((num_routes, hidden), dtype=hidden_states.dtype, device=device)
    block = min(_BLK_H, _next_pow2(hidden))
    tokens_per_block = min(_ROW_BT, max(1, triton.cdiv(num_routes, _ROW_MIN_GRID)))
    grid = (
        triton.cdiv(hidden, block),
        triton.cdiv(num_routes, tokens_per_block),
    )
    _routing_scatter_kernel[grid](
        hidden_states,
        row_indices,
        routed,
        num_routes,
        hidden,
        TOPK=topk,
        BT=tokens_per_block,
        BLK=block,
    )
    return routed, row_indices, expert_tokens
