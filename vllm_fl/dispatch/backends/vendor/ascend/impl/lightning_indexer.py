# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""lightning_indexer -- pure Triton implementation of `npu_lightning_indexer`

## The operator
DeepSeek-V3.2 "lightning indexer": for every query token and every key head, pick the
`sparse_count` key positions with the largest

    score[s1,n2,s2] = sum_g relu( query[s1, n2*gs+g, :] . key[s2, n2, :] ) * weights[s1,n2*gs+g]

with `gs = N1 // N2`.  The selection is returned as int32; slots short of `sparse_count`
valid positions are filled with -1 (`INVALID_IDX` in the AscendC sources).

## Provenance of the AscendC original (verified 2026-10-14; do not delete)
`npu_lightning_indexer` is **not** implemented in vllm-ascend's own `csrc/`; it is a **CANN
built-in** operator reached through torch_npu:
  - CANN 9.0.1 ships the kernel as
    `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/lightning_indexer/`
    (`lightning_indexer.cpp`, `lightning_indexer_kernel.h`, `lightning_indexer_service_cube.h`,
    `lightning_indexer_service_vector.h`, `lightning_indexer_vector.h`, ...).  Its host stub
    `lightning_indexer.cpp` L37-57 dispatches `LIPreload<half, half, int32_t, ...>` for fp16 and
    `LIPreload<bfloat16_t, bfloat16_t, int32_t, ...>` otherwise, i.e. query/key are **fp16 or
    bf16** and the index output is int32.  `lightning_indexer_service_cube.h` L384 sets
    `reluPre = 1` (ReLU fused into the Cube QK path) and the kernel applies no dequant scale.
  - inside vllm-ascend, `grep -rn "OP_ADD(" csrc/` lists only `LightningIndexerQuant`,
    `FusedLightningIndexerManage`, `FusedQuantLightningIndexerManage` and
    `VllmAscendQuantLightningIndexerV2`; there is no plain `LightningIndexer` op.
    `csrc/attention/lightning_indexer_v2/` holds a single kernel header
    (`op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h`, 277 lines) with no
    op_host/def/tiling, so nothing there can be built into an operator either.
  - the legacy private vendor namespace of the AscendC extension carries no lightning indexer
    entry in the target container (it resolves to a single shim op), so the branch that used to
    call it cannot run there; the only vendor path left is `torch_npu.npu_lightning_indexer`,
    which is enabled for `glm_moe_dsa` alone (`vllm_ascend/device/device_op.py` L628-640).
  => this module is the Triton replacement for the operator and never dispatches to a vendor
     kernel; only standard `torch` / `triton` / `triton.language` are used.

## Call-site style (`vllm_ascend/device/device_op.py` L628-653)
The production path keeps two equivalent branches and picks between them:
    topk_indices, _ = npu_lightning_indexer(       # vendor branch, glm_moe_dsa only
        query=q_li, key=kv_cache[idx], weights=weights,
        actual_seq_lengths_query=..., actual_seq_lengths_key=...,
        block_table=attn_metadata.block_table,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=2048, sparse_mode=3)
    topk_indices, _ = lightning_indexer(           # this module
        query=q_li, key=kv_cache[idx], weights=weights,
        actual_seq_lengths_query=..., actual_seq_lengths_key=...,
        block_table=attn_metadata.block_table,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=2048, sparse_mode=3)
Both branches pass *identical* arguments, so `npu_lightning_indexer` and `lightning_indexer`
are two spellings of one operator (`npu_` is the torch_npu prefix).  Both names are exported
here and both resolve to the function below.

## Relation to the quantized sibling
`LightningIndexerQuant` (`csrc/attention/lightning_indexer_quant/`) is the int8 member of the
same family: the same QK / ReLU / weighted-sum / top-K pipeline plus `query_dequant_scale` and
`key_dequant_scale`.  Its Triton migration is `lightning_indexer_quant.py` in this package, and
the int8 path below reuses it with identity scales -- `x * 1.0 == x` is exact in fp16, so both
kernels then compute the same scores.

## Outputs (two slots)
  * `sparse_indices` -- int32; TND -> [T, N2, K], BSND -> [B, S1, N2, K];
  * `sparse_values`  -- empty.  The AscendC operator declares a second output (`sparseValues`,
    key dtype) but only fills it when its tiling asks for it: the copy-out is guarded by
    `constInfo_.returnValue` in `lightning_indexer_service_vector.h`, and the vendor entry point
    returns an empty tensor for the call-site arguments used here.  This module keeps that
    convention (empty tensor of the key dtype on the fp16/bf16 path, empty fp32 on the int8
    path, like the sibling `vllm_quant_lightning_indexer(return_values=False)`); the only
    production call site discards the slot (`topk_indices, _ = ...`).

**Zero tolerance for the index output**: `sparse_indices` must match element-wise exactly.
**Tie-breaking**: the order of equal scores is undefined in the AscendC sort; this
implementation picks the **smaller index** (the rule used by `lightning_indexer_quant` too).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import next_pow2
from .lightning_indexer_quant import lightning_indexer_quant

__all__ = ["lightning_indexer", "npu_lightning_indexer"]

_NEG = -1.0e30          # sentinel written for positions that are out of range


@triton.jit
def _li_score_kernel(
    Q, K, W, SC, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS,
    BLOCK_SIZE, MAX_BLOCKS, S2,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    HD: tl.constexpr,
    BLK_S2: tl.constexpr,
    BLK_D: tl.constexpr,
):
    """One program per (token, key head) row: score[s2] = sum_g relu(q_g . k) * w_g."""
    row = tl.program_id(0)
    n2 = row % N2
    t = row // N2
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)

    h0 = n2 * GS
    d = tl.arange(0, BLK_D)
    dm = d < HD
    q_base = (t * N1 + h0) * HD
    w_base = t * N1 + h0

    kvlen = tl.load(KVLEN + b)
    if CAUSAL:
        # sparse_mode=3: s2 <= s1 + (kvLen - qLen)
        j_hi = s1 + (kvlen - tl.load(QLEN + b))
    else:
        j_hi = S2 - 1

    for c in range(0, tl.cdiv(S2, BLK_S2)):
        s2 = c * BLK_S2 + tl.arange(0, BLK_S2)
        s2m = s2 < S2
        vld = s2m & (s2 <= j_hi) & (s2 < kvlen)
        if PA:
            page = tl.load(BT + b * MAX_BLOCKS + s2 // BLOCK_SIZE, mask=vld, other=0)
            kvrow = page * BLOCK_SIZE + (s2 % BLOCK_SIZE)
        else:
            kvrow = b * S2 + s2
        k_off = (kvrow * N2 + n2)[:, None] * HD + d[None, :]
        kblk = tl.load(K + k_off, mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)

        acc = tl.zeros([BLK_S2], dtype=tl.float32)
        for g in range(0, GS):
            qg = tl.load(Q + q_base + g * HD + d, mask=dm, other=0.0).to(tl.float32)
            wq = tl.load(W + w_base + g).to(tl.float32)
            qk = tl.sum(kblk * qg[None, :], axis=1)
            acc += tl.where(qk > 0.0, qk, 0.0) * wq          # weighted accumulation of relu(QK)
        sc = tl.where(vld, acc, NEG)
        tl.store(SC + row * S2 + s2, sc, mask=s2m)


@triton.jit
def _li_topk_kernel(
    SC, IDX, S2,
    K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
):
    """Per-row top-K by repeated max; ties keep the smaller index; short rows stay invalid."""
    pid = tl.program_id(0)
    off = tl.arange(0, BLK_ALL)
    s = tl.load(SC + pid * S2 + off, mask=off < S2, other=NEG)
    ar = tl.arange(0, BLK_K)
    oi = tl.full([BLK_K], -1, tl.int32)                  # INVALID_IDX = -1
    for k in range(K):
        mx = tl.max(s, axis=0)
        lane = tl.min(tl.where(s >= mx, off, BLK_ALL))   # on ties, the smaller index wins
        oi = tl.where(ar == k, tl.where(mx > NEG, lane, -1), oi)
        s = tl.where(off == lane, NEG, s)
    tl.store(IDX + pid * K + ar, oi, mask=ar < K)


def _lens(seq, B, name):
    """Per-batch lengths from either the lengths [B] or the cu_seqlens [B+1] spelling."""
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B:
        return s.to(torch.int32)
    if s.numel() == B + 1:
        return (s[1:] - s[:-1]).to(torch.int32)
    raise ValueError(f"{name} length must be B={B} or B+1={B + 1}, got {s.numel()}")


def _plan(
    query, key, weights,
    actual_seq_lengths_query, actual_seq_lengths_key, block_table,
    layout_query, layout_key, sparse_count, sparse_mode,
):
    """Validate the call-site arguments and build the index tensors used by the kernels."""
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"unsupported layout_query={layout_query!r} (BSND/TND)")
    if layout_key not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"unsupported layout_key={layout_key!r} (BSND/PA_BSND)")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode must be 0 or 3, got {sparse_mode}")
    pa = layout_key in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (cu_seqlens)")
        T, N1, D = query.shape
        cu = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu.numel() < 2 or int(cu[-1]) != T:
            raise ValueError(f"actual_seq_lengths_query must be cu_seqlens (length B+1, "
                             f"last entry {T})")
        B = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64)
                  - cu[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    if pa:
        if block_table is None:
            raise ValueError("layout_key=PA_BSND requires block_table")
        _, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        _, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key D {Dk} does not match query D {D}")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    if weights.numel() != T * N1:
        raise ValueError(f"weights must have T*N1={T * N1} elements, got {weights.numel()}")

    kvlen_b = _lens(actual_seq_lengths_key, B, "actual_seq_lengths_key")
    if kvlen_b is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)
    return {
        "T": T, "B": B, "S1": S1, "N1": N1, "N2": N2, "D": D, "S2": S2, "tnd": tnd, "pa": pa,
        "block_size": block_size, "max_blocks": max_blocks,
        "gs": N1 // N2, "k": int(sparse_count),
        "tok_b": tok_b, "tok_s1": tok_s1, "kvlen_b": kvlen_b, "qlen_b": qlen_b,
        # kept verbatim: the quantized sibling accepts the lengths in either spelling
        "asq": actual_seq_lengths_query, "ask": actual_seq_lengths_key,
        "block_table": block_table,
    }


def _int8_path(query, key, weights, p, layout_query, layout_key, sparse_mode):
    """int8 inputs: the quantized kernel with identity dequant scales (bit-identical)."""
    dev = query.device
    qs = torch.ones(p["T"], p["N1"], dtype=torch.float16, device=dev)
    if p["pa"]:
        ks = torch.ones(key.shape[0], key.shape[1], dtype=torch.float16, device=key.device)
    else:
        ks = torch.ones(p["B"] * key.shape[1], dtype=torch.float16, device=key.device)
        ks = ks.view(p["B"], key.shape[1])
    sparse_indices = lightning_indexer_quant(
        query=query, key=key, weights=weights,
        query_dequant_scale=qs, key_dequant_scale=ks,
        actual_seq_lengths_query=p["asq"],
        actual_seq_lengths_key=p["ask"],
        block_table=p["block_table"],
        query_quant_mode=0, key_quant_mode=0,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=p["k"], sparse_mode=sparse_mode)
    # The reused quantized kernel returns indices only; the call site discards this slot.
    return sparse_indices, torch.empty(0, dtype=torch.float32, device=dev)


def _native_path(query, key, weights, p, sparse_mode):
    """fp16/bf16 inputs: the native Triton score + top-K kernels."""
    dev = query.device
    T, N1, D, N2 = p["T"], p["N1"], p["D"], p["N2"]
    gs, S2, k = p["gs"], p["S2"], p["k"]

    q2 = query.contiguous().reshape(T, N1, D)
    w2 = weights.contiguous().reshape(T, N1)
    kv = key.contiguous()
    bt = p["block_table"].contiguous().to(torch.int32) if p["pa"] else kv

    n_rows = T * N2
    scores = torch.empty((n_rows, S2), dtype=torch.float32, device=dev)
    _li_score_kernel[(n_rows,)](
        q2, kv, w2, scores,
        p["tok_b"].to(dev), p["tok_s1"].to(dev), p["kvlen_b"].to(dev), p["qlen_b"].to(dev), bt,
        N1, N2, gs,
        p["block_size"], p["max_blocks"], S2,
        PA=bool(p["pa"]), CAUSAL=bool(int(sparse_mode) == 3), NEG=_NEG,
        HD=D, BLK_S2=min(64, next_pow2(S2)), BLK_D=next_pow2(D),
    )
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=dev)
    _li_topk_kernel[(n_rows,)](
        scores, idx, S2,
        K=k, BLK_ALL=next_pow2(S2), BLK_K=next_pow2(k), NEG=_NEG,
    )
    # The values slot stays empty, exactly like the vendor op observed here (its `sparseValues`
    # output is only materialised when the tiling asks for it).
    vals = torch.empty(0, dtype=key.dtype, device=dev)
    if p["tnd"]:
        return idx.view(T, N2, k), vals
    return idx.view(p["B"], p["S1"], N2, k), vals


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of `npu_lightning_indexer` (see the module docstring).

    Args:
        query: fp16/bf16 (or int8)  TND [T,N1,D] / BSND [B,S1,N1,D]
        key:   fp16/bf16 (or int8)  PA_BSND [numBlocks, blockSize, N2, D] / BSND [B,S2,N2,D]
        weights: per q head, fp16/bf16/fp32, [T,N1] (TND) or [B,S1,N1]
        actual_seq_lengths_query/key, block_table, layout_*, sparse_count, sparse_mode:
            same names and meaning as the AscendC operator (TND needs cu_seqlens;
            PA_BSND needs block_table)
        query_quant_mode/key_quant_mode: kept for signature compatibility only; the
            non-quantized operator has no quant modes (only 0 is accepted)

    Returns:
        A **two-element** tuple `(sparse_indices, sparse_values)`, matching the AscendC /
        torch_npu return convention and the production call site
        `topk_indices, _ = npu_lightning_indexer(...)` (`device_op.py` L628-653):

          * sparse_indices: int32, TND -> [T,N2,K]; BSND -> [B,S1,N2,K]
          * sparse_values: empty (see the "Outputs" section of the module docstring);
            the call site discards the slot.
    """
    if int(query_quant_mode) != 0 or int(key_quant_mode) != 0:
        raise NotImplementedError(
            "the non-quantized lightning_indexer has no quant_mode semantics (only 0 is "
            "accepted); use lightning_indexer_quant for the quantized variant")
    if query.dtype != key.dtype:
        raise ValueError(f"query/key must share one dtype, got {query.dtype} and {key.dtype}")
    if query.dtype not in (torch.int8, torch.float16, torch.bfloat16):
        raise ValueError(f"query/key must be int8, fp16 or bf16, got {query.dtype}")
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be fp16, bf16 or fp32, got {weights.dtype}")

    p = _plan(
        query, key, weights,
        actual_seq_lengths_query, actual_seq_lengths_key, block_table,
        layout_query, layout_key, sparse_count, sparse_mode)
    if p["k"] <= 0:
        raise ValueError("sparse_count must be > 0")

    if query.dtype == torch.int8:
        return _int8_path(query, key, weights, p, layout_query, layout_key, sparse_mode)
    return _native_path(query, key, weights, p, sparse_mode)


# Call-site name alias (both names are exported; see the module docstring).
npu_lightning_indexer = lightning_indexer
