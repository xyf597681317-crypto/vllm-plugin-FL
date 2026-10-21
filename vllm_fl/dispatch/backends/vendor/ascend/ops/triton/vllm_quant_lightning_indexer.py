# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""vllm_quant_lightning_indexer -- AscendC -> Triton

Source: csrc/ascend/attention/vllm_quant_lightning_indexer/
Interface (op_host/vllm_quant_lightning_indexer_def.cpp L23-81, arch32/910B configuration):
    Input : query               int8   [B, S1, N1, 128]   (layout_query="BSND")
            key                 int8   [B, S2, N2, 128]   (layout_key="BSND")
            weights             fp16   [B, S1, N1]
            query_dequant_scale fp16   [B, S1, N1]
            key_dequant_scale   fp16   [B, S2]            (per key position)
            actual_seq_lengths_query/key, block_table, metadata  (optional)
    Output: sparse_indices int32 [B, S1, N2, sparse_count]   <- InferShape L64-75
            sparse_values  fp32  [B, S1, N2, sparse_count]   <- only when return_values=True;
                                                                otherwise an empty tensor (L76-83)
    Attr  : query/key_quant_mode=0 (per-token-head), sparse_count (default 2048),
            sparse_mode=3 (lower triangle only), pre_tokens/next_tokens=INT64_MAX, cmp_ratio=1,
            return_values=False, stride, scale_stride

Constant constraints (op_host/vllm_quant_lightning_indexer_tiling.cpp):
    - head_dim must be 128 (L549 HEAD_DIM_LIMIT)
    - gSize = n1Size_ / n2Size_ must be G_SIZE_LIMIT (L483-484)
    - the N1 dimension of weights must equal n1Size_ (L720), i.e. it is indexed by **q head**
    - sparse_mode in {0, 3} (L249), pre/next_tokens must be INT64_MAX (L252-257)

Formula source (op_kernel/arch35/vf/quant_lightning_indexer_vector1.h):
    MulWeightAndReduceSum L329-353 (fp32 version) and L206-280:
        regW    = weights * qScale                      # L243 / L331 per (s1,qhead)
        for g in 0..gSize-1:  Relu is folded into the cube path   # L341 comment
            MulAddDst(sum, qk[g], w_brc[g])             # L181-185 WeightedAccum: Relu, then MulAdd
        sum = sum * kScale                              # L352-353
    The same formula in arch32: op_kernel/arch32/quant_lightning_indexer_service_vector.h
        L265 Mul(weights, weights, qScale) ; L360 Mul(mm1Res, mm1Res, kScale)

    => core computation (**semantically equivalent version**):
        QK[s1, n2, g, s2] = sum_d query[s1, n2*gSize+g, d] * key[s2, n2, d]   (int8 dot, exact in int32)
        wq[s1, g']        = weights[s1, g'] * query_dequant_scale[s1, g']     (g' = n2*gSize+g)
        score[s1, n2, s2] = key_dequant_scale[s2] * sum_g relu(QK[g, s2]) * wq[g]
        sparse_mode=3 (lower triangle): score = -inf  when s2 > s1 + (S2 - S1)
                                  (tiling kernel L327: cuRealAcSeq = actS2SizeOrig - actS1Size
                                   + cuS1BeginIdx)
        top-k: take the sparse_count key positions with the largest score in descending order;
               pad with -1 when there are fewer
               (INVALID_IDX = -1, see arch32/quant_lightning_indexer_common.h L79)

    Note on sort precision: AscendC's topk converts scores to **bfloat16**, then to sortable keys,
    and sorts them (L273-277 FloatToSortableKey<bfloat16_t>), so ordering compares at bf16
    precision. This implementation compares in fp32; for data where "the best K scores are clearly
    separated" both agree (the tests print those gaps as evidence).

**Simplifications** (relative to the full AscendC implementation):
    1. only layout_query/layout_key = "BSND" (no TND / PA_BSND paged KV)
    2. no cmp_ratio != 1 compression and no actual_seq_lengths variable length (treated as full BSND)
    3. no multi-core Cube/Vector pipelining and no fusion; the score matrix is written to GM once
       (two-pass kernel: score -> topk)
    4. topk is implemented as repeated argmax (O(K) rounds); on ties the **smaller index** wins
       (AscendC leaves the tie order undefined)
    5. the int8 query/key dot product uses fp32 broadcast multiply-add (exact: the int8 product sum
       stays below 2^24); the Cube int8 matmul is not used
    6. sparse_values emits 0.0 in invalid slots (index -1); AscendC leaves that value undefined
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

#: Largest per-chunk width used by the top-k kernel.  Chosen so the phase-1 working set
#: (~12 B/elem) stays well under the 192 KB UB: 8192 * 12 = 96 KB, leaving room for the
#: candidate buffer and the K-sized outputs.
UB_CHUNK = 4096

__all__ = ["vllm_quant_lightning_indexer"]

_NEG = -1.0e30


@triton.jit
def _qli_score_kernel(
    Q, K, W, QS, KS, SC, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS,
    BLOCK_SIZE, MAX_BLOCKS, S2, CMP_RATIO,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    HD: tl.constexpr,
    BLK_S2: tl.constexpr,
    BLK_D: tl.constexpr,
):
    row = tl.program_id(0)              # one row = one (token, n2)
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
        # matches AscendC: cuRealAcSeq = (count + (actS2Orig - actS1 + s1)) / cmpRatio
        j_hi = (s1 + 1 + (kvlen - tl.load(QLEN + b))) // CMP_RATIO - 1
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
            wq = tl.load(W + w_base + g).to(tl.float32) * tl.load(QS + w_base + g).to(tl.float32)
            qk = tl.sum(kblk * qg[None, :], axis=1)          # int8 dot product (exact in fp32)
            acc += tl.where(qk > 0.0, qk, 0.0) * wq          # ReLU + weighted accumulation
        ksc = tl.load(KS + kvrow, mask=vld, other=0.0).to(tl.float32)
        sc = tl.where(vld, acc * ksc, NEG)
        tl.store(SC + row * S2 + s2, sc, mask=s2m)


@triton.jit
def _qli_topk_kernel(
    SC, CIDX, CVAL, S2,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLK_K: tl.constexpr,
    WRITE_VAL: tl.constexpr,
    NEG: tl.constexpr,
):
    """PHASE 1 of a two-KERNEL top-K: local top-K per chunk -> candidates in HBM.

    T-116 (r16): this used to be one kernel that also ran phase 2, so the CHUNK-sized
    working set (`lc`/`off`/`s` + `tl.where` temporaries) and the COFF-sized candidate
    pool (`coff`/`cval`/`cidx`) were live at the same time: 144 KB static, but the device
    measured 208,960 B (> 196,608 B UB) and FULL capture died at compile time
    (`requires 1671680 bits`).  Splitting the two phases into two kernels removes the
    co-residency; each kernel is then comfortably under UB.

    Equivalence: the global top-K under the total order (value DESC, index ASC) is the
    top-K of the union of the per-chunk top-Ks (a chunk's (K+1)-th best can never be
    global).  Chunks are contiguous ascending index ranges and phase 1 breaks ties by the
    smallest local index, so the candidate list is ordered by (value DESC, global index
    ASC) -- which is exactly what phase 2 needs.
    """
    pid = tl.program_id(0)

    lc = tl.arange(0, CHUNK)
    ar = tl.arange(0, BLK_K)
    for c in range(NCHUNK):
        base = c * CHUNK
        off = base + lc
        s = tl.load(SC + pid * S2 + off, mask=off < S2, other=NEG)
        oi = tl.full([BLK_K], -1, tl.int32)
        ov = tl.zeros([BLK_K], dtype=tl.float32)
        for k in range(K):
            mx = tl.max(s, axis=0)
            lane = tl.min(tl.where(s >= mx, lc, CHUNK))   # on ties take the smaller index
            sel = tl.where(mx > NEG, lane + base, -1)
            oi = tl.where(ar == k, sel, oi)
            if WRITE_VAL:
                ov = tl.where(ar == k, tl.where(mx > NEG, mx, 0.0), ov)
            s = tl.where(lc == lane, NEG, s)
        m = ar < K
        tl.store(CIDX + (pid * NCHUNK + c) * K + ar, oi, mask=m)
        if WRITE_VAL:
            tl.store(CVAL + (pid * NCHUNK + c) * K + ar, ov, mask=m)


@triton.jit
def _qli_topk_merge_kernel(
    CVAL, CIDX, IDX, VAL,
    K: tl.constexpr,
    NCAND: tl.constexpr,     # nchunk * K : how many candidates this row has
    COFF: tl.constexpr,      # next_pow2(NCAND) : the register-tile width
    BLK_K: tl.constexpr,
    WRITE_VAL: tl.constexpr,
    NEG: tl.constexpr,
):
    """PHASE 2 of a two-KERNEL top-K: global top-K over the row's `NCAND` candidates.

    Same selection rule as the single-pass form: take the maximum value, and among equals
    the SMALLEST index.  Candidate slots are ordered by (chunk ASC, rank ASC), and chunks
    are contiguous ascending index ranges with rank = descending value / ascending index,
    so "smallest slot among maxima" IS "smallest global index among maxima".
    """
    pid = tl.program_id(0)
    coff = tl.arange(0, COFF)
    pm = coff < NCAND
    cval = tl.load(CVAL + pid * NCAND + coff, mask=pm, other=NEG)
    cidx = tl.load(CIDX + pid * NCAND + coff, mask=pm, other=-1)

    ar = tl.arange(0, BLK_K)
    oi = tl.full([BLK_K], -1, tl.int32)
    ov = tl.zeros([BLK_K], dtype=tl.float32)
    for k in range(K):
        mx = tl.max(cval, axis=0)
        lane = tl.min(tl.where(cval >= mx, coff, COFF))
        sel = tl.where(mx > NEG, tl.sum(tl.where(coff == lane, cidx, 0), axis=0), -1)
        oi = tl.where(ar == k, sel, oi)
        if WRITE_VAL:
            ov = tl.where(ar == k, tl.where(mx > NEG, mx, 0.0), ov)
        cval = tl.where(coff == lane, NEG, cval)

    m = ar < K
    tl.store(IDX + pid * K + ar, oi, mask=m)
    if WRITE_VAL:
        tl.store(VAL + pid * K + ar, ov, mask=m)


def _is_capturing() -> bool:
    """True when an ACL graph is being captured on the current stream.

    Guarded: older torch_npu builds / non-NPU hosts do not expose the probe, and raising
    from a capability check would be worse than treating it as "not capturing".
    """
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Capture-safe host-length cache (T-114 step 3)
#
# WHY: the score kernel needs `tok_b / tok_s1 / kvlen_b / qlen_b` on the DEVICE.  Building
# them per call meant four `.to(dev)` (host->device) at launch time; inside an ACL-graph
# capture an H2D copy aborts the capture.  The four device buffers are therefore built ONCE
# (eager / warmup), keyed by (num_tokens, device), and only their CONTENT is refreshed --
# `kvlen_b`'s ADDRESS is never replaced, because the already-captured graph reads it.
#
# APPEND-ONLY: entries are never evicted (evicting would free an address a captured graph may
# still hold).  Growth is bounded by the number of distinct graph sizes.
# ---------------------------------------------------------------------------
_LI_META = {}


def _li_key(num_tokens, dev):
    return (int(num_tokens), str(dev))


def _li_is_capturing():
    """True while a graph capture is in progress.  Conservative: unknown => False."""
    npu = getattr(torch, "npu", None)
    fn = getattr(npu, "is_current_stream_capturing", None) if npu is not None else None
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:                                            # noqa: BLE001
        return False


def refresh_indexer_kvlen(num_tokens, cpu_kv):
    """Copy HOST-side per-sequence KV lengths into the cached `kvlen_b` buffer, in place.

    `cpu_kv` is HOST data (Python list / tuple / numpy array / CPU tensor).  We never call
    `.cpu()` here -- the values are already on the host, and reaching for the device would
    re-introduce exactly the synchronisation this change exists to remove.

    In-place `copy_` ONLY: `kvlen_b`'s storage address must stay constant, because the
    captured graph reads that address on every replay.

    Returns how many cached entries were refreshed (0 = nothing to do / not built yet).
    """
    if cpu_kv is None:
        return 0
    host = torch.as_tensor(cpu_kv)                 # host -> host, no device traffic
    if host.numel() == 0:
        return 0
    n = 0
    for (nt, _dev), meta in list(_LI_META.items()):
        if int(nt) != int(num_tokens):
            continue
        buf = meta.get("kvlen_b")
        if not torch.is_tensor(buf):
            continue
        if host.numel() == buf.numel():
            host_i32 = host.to(torch.int32)
        elif host.numel() == buf.numel() + 1:
            # cumulative (B+1) form, the sibling convention accepted by the builder
            host_i32 = (host[1:] - host[:-1]).to(torch.int32)
        else:
            # A different length would force a resize => the address would move, which is
            # forbidden; leave the buffer untouched and let the caller notice.
            continue
        buf.copy_(host_i32)                        # in place; address unchanged
        n += 1
    return n


def _li_get_or_build_meta(num_tokens, dev, tok_b, tok_s1, kvlen_b, qlen_b):
    """Return the cached DEVICE buffers for (num_tokens, dev), building them once.

    MISS (eager / warmup): build the four device buffers once and remember them.
    HIT  inside a capture : touch nothing -- an H2D here would abort the capture; the
                            content of `kvlen_b` is refreshed by `prepare_replay`
                            (graph_runtime) before the graph is replayed.
    HIT  in eager         : refresh the CONTENT in place.  This keeps the pre-existing
                            semantics (a repeated num_tokens with new sequence lengths must
                            not serve stale lengths) while still never moving an address.
    """
    key = _li_key(num_tokens, dev)
    meta = _LI_META.get(key)
    if meta is None:
        meta = {
            "tok_b": tok_b.to(dev),
            "tok_s1": tok_s1.to(dev),
            "kvlen_b": kvlen_b.to(dev),
            "qlen_b": qlen_b.to(dev),
        }
        _LI_META[key] = meta
        return meta
    if not _li_is_capturing():
        for name, src in (("tok_b", tok_b), ("tok_s1", tok_s1),
                          ("kvlen_b", kvlen_b), ("qlen_b", qlen_b)):
            buf = meta[name]
            src_i32 = src.to(torch.int32) if src.dtype != torch.int32 else src
            if src_i32.numel() == buf.numel():
                buf.copy_(src_i32.to(buf.device))
    return meta


def _host_lengths(mirror, src, name):
    """Host copy of a sequence-length tensor, using the caller-supplied mirror when given.

    Inside ACL graph capture a device->host copy is rejected (107027), so the caller must
    pass a host mirror.  A missing mirror does NOT silently fall back while capturing:
    the silent fallback is exactly what kept these transfers hidden until capture aborted.

    Branches:
      1) mirror present            -> use it (Tensor on host, or a plain Python list)
      2) mirror None, src on host  -> use src
      3) mirror None, src on device:
           a) not capturing -> .cpu()  (prefill carries no host mirror at all -- see
              AscendDSAPrefillMetadata, which has no _cpu field -- so raising here would
              break eager prefill)
           b) capturing     -> raise
    """
    if mirror is not None:
        if isinstance(mirror, torch.Tensor):
            if mirror.device.type != "cpu":
                return mirror.detach().to(torch.int64).flatten().cpu()
            return mirror.detach().to(torch.int64).flatten()
        return torch.as_tensor(mirror, dtype=torch.int64).flatten()
    if src is None:
        return None
    if src.device.type == "cpu":
        return src.detach().to(torch.int64).flatten()
    if _is_capturing():
        raise RuntimeError(
            "%s: required host mirror missing (cpu_actual_seq_lengths_*); pass it from "
            "indexer_scale_decode_metadata (query_start_loc_cpu / seq_lens_list). A "
            "device->host copy is not allowed during ACL graph capture (107027)." % (name,))
    return src.detach().to(torch.int64).cpu().flatten()



def vllm_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    return_value: bool = False,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    pre_tokens: int | None = None,
    next_tokens: int | None = None,
    cmp_ratio: int = 1,
    return_values: bool | None = None,
    out_dtype: torch.dtype = torch.int32,
    cpu_actual_seq_lengths_query: torch.Tensor | None = None,
    cpu_actual_seq_lengths_key: torch.Tensor | None = None,
):
    """Equivalent of the AscendC VllmQuantLightningIndexer (BSND, semantically equivalent core).

    Call-site signature (vllm_fl/attention/ascend/dsa_v1.py L2658-2678):
        npu_vllm_quant_lightning_indexer(
            query=..., key=..., weights=..., query_dequant_scale=..., key_dequant_scale=...,
            actual_seq_lengths_query=..., actual_seq_lengths_key=..., block_table=...,
            metadata=..., query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND", sparse_count=...,
            sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
            cmp_ratio=4, return_value=False)
    Note: the real call site uses **TND + PA_BSND + cmp_ratio=4**, while this core version only
    implements BSND + cmp_ratio=1; unsupported values raise NotImplementedError explicitly and are
    never silently ignored.

    Args:
        query:               int8 [B, S1, N1, 128]
        key:                 int8 [B, S2, N2, 128]
        weights:             fp16/bf16/fp32 [B, S1, N1]
        query_dequant_scale: same as weights
        key_dequant_scale:   [B, S2] or [S2]
        sparse_count:        top-k count K
        sparse_mode:         0 (all keys) or 3 (lower triangle, the AscendC default)
        return_value:        whether to return sparse_values (fp32 scores); False -> empty tensor
        layout_query/layout_key: only "BSND" is supported
        actual_seq_lengths_query/key, block_table, metadata: only None is supported
        query_quant_mode/key_quant_mode: only 0 (per-token-head) is supported
        pre_tokens/next_tokens: only None or INT64_MAX is supported
        cmp_ratio: only 1 is supported
        return_values: alias of return_value (backward compatibility)

    Returns:
        (sparse_indices [B,S1,N2,K] int32, sparse_values [B,S1,N2,K] fp32 or torch.empty(0))
    """
    if return_values is not None:
        return_value = return_values
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_query={layout_query!r} is not implemented (supported: BSND/TND)")
    if layout_key not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_key={layout_key!r} is not implemented (supported: BSND/PA_BSND)")
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError(f"metadata must be an int32 descriptor tensor, got {metadata.dtype}")
    for nm, v in (("query_quant_mode", query_quant_mode), ("key_quant_mode", key_quant_mode)):
        if int(v) != 0:
            raise NotImplementedError(f"{nm}={v} is not implemented (tiling only accepts 0: per-token-head)")
    for nm, v in (("pre_tokens", pre_tokens), ("next_tokens", next_tokens)):
        if v is not None and int(v) != (1 << 63) - 1:
            raise NotImplementedError(f"{nm}={v} is not implemented (tiling only accepts INT64_MAX)")
    return_values = bool(return_value)
    pa = layout_key in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    # ---------------- layout normalization (the host only does layout bookkeeping; the core
    # ---------------- math lives in Triton) ----------------
    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (cu_seqlens)")
        T, N1, D = query.shape
        cu_q = _host_lengths(cpu_actual_seq_lengths_query, actual_seq_lengths_query,
                              "actual_seq_lengths_query")
        if cu_q.numel() == 0:
            raise ValueError("actual_seq_lengths_query must not be empty for layout_query=TND")
        # Both conventions must be accepted:
        #   * AscendC / unit tests: ``[0, a1, a1+a2, ...]`` (length B+1, leading 0)
        #   * production call site: ``dsa_v1.py`` L2653/L2664 passes ``query_start_loc[1:]``,
        #                     i.e. ``[a1, a1+a2, ...]`` (length B, **no** leading 0)
        # For single-sequence decode production passes a single element (``[1]``); the previous
        # implementation raised ValueError here, the worker died inside the collective and the
        # other ranks reported 507035.
        if int(cu_q[0]) != 0:
            cu_q = torch.cat([torch.zeros(1, dtype=torch.int64), cu_q])
        if int(cu_q[-1]) != T:
            raise ValueError(
                f"with TND, actual_seq_lengths_query must be cu_seqlens (length B or B+1, last "
                f"entry {T}), got last entry {int(cu_q[-1])}"
            )
        B = cu_q.numel() - 1
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
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
        num_blocks, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        Bk, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key D does not match query: {Dk} vs {D}")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key must be int8")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    gs = N1 // N2
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode only supports 0 or 3, got {sparse_mode}")
    # weights/query_dequant_scale are per q head: BSND passes [B,S1,N1], TND passes [T,N1];
    # both are flattened to [T,N1]
    if weights.numel() != T * N1 or query_dequant_scale.numel() != T * N1:
        raise ValueError(
            f"weights/query_dequant_scale numel must be T*N1={T * N1} "
            f"(BSND->[B,S1,N1], TND->[T,N1]), got {tuple(weights.shape)} / "
            f"{tuple(query_dequant_scale.shape)}")
    k = int(sparse_count)
    if k <= 0:
        raise ValueError("sparse_count must be > 0")

    sq_kv = actual_seq_lengths_key
    if sq_kv is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        s = _host_lengths(cpu_actual_seq_lengths_key, sq_kv, "actual_seq_lengths_key")
        if s.numel() == B:
            kvlen_b = s.to(torch.int32)
        elif s.numel() == B + 1:
            kvlen_b = (s[1:] - s[:-1]).to(torch.int32)
        else:
            raise ValueError(f"actual_seq_lengths_key length must be B={B} or B+1, got {s.numel()}")
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dev = query.device
    q2 = query.contiguous().reshape(T, N1, D)
    w2 = weights.contiguous().reshape(T, N1)
    qs2 = query_dequant_scale.contiguous().reshape(T, N1)
    kv = key.contiguous()
    bt = block_table.contiguous().to(torch.int32) if pa else key.contiguous()
    ks = key_dequant_scale.contiguous()
    if pa:
        # Under PA, key_dequant_scale is laid out by (physical block, in-block offset)
        # (GetKeyScale: kScaleGm[blockId*scaleStride+off]). The production call site carries an
        # extra trailing singleton dimension (e.g. (21321, 32, 1)) -- squeeze it away; the
        # addressing below only looks at [numBlocks, blockSize], so the extra dimension has no
        # semantic effect.
        while ks.dim() > 2 and ks.shape[-1] == 1:
            ks = ks.squeeze(-1)
        if ks.dim() != 2 or ks.shape[1] != block_size:
            raise ValueError(f"under PA, key_dequant_scale should be [numBlocks, blockSize]="
                             f"{(num_blocks, block_size)} (a trailing singleton may be omitted), "
                             f"got {tuple(key_dequant_scale.shape)}")
        ks = ks.contiguous()
    else:
        if ks.dim() == 1:
            ks = ks.unsqueeze(0).expand(B, -1).contiguous()
        if tuple(ks.shape) != (B, S2):
            raise ValueError(f"key_dequant_scale must be [B,S2] or [S2], got {tuple(key_dequant_scale.shape)}")

    n_rows = T * N2
    blk_d = next_pow2(D)
    # BLK_S2 is limited by UB (192KB): the fp32 intermediate of BLK_S2*BLK_D must fit
    # (128x128 overflows UB)
    blk_s2 = min(64, next_pow2(S2))
    scores = torch.empty((n_rows, S2), dtype=torch.float32, device=dev)
    # T-114 step 3: device buffers come from the append-only cache, so the
    # captured graph contains no host->device copy of the length tensors.
    li_meta = _li_get_or_build_meta(T, dev, tok_b, tok_s1, kvlen_b, qlen_b)
    _qli_score_kernel[(n_rows,)](
        q2, kv, w2, qs2, ks, scores,
        li_meta["tok_b"], li_meta["tok_s1"], li_meta["kvlen_b"], li_meta["qlen_b"], bt,
        N1, N2, gs,
        block_size, max_blocks, S2, int(cmp_ratio),
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        HD=D, BLK_S2=blk_s2, BLK_D=blk_d,
    )
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=dev)
    blk_k = next_pow2(k)
    # T-116 (r16): the two phases of the top-K now run in SEPARATE kernels, because one
    # kernel holding both working sets needed 208,960 B > the 196,608 B UB and killed FULL
    # capture (r15: `requires 1671680 bits`).  Phase 1 (per chunk) writes its candidates to
    # HBM; phase 2 (per row) selects the global top-K from them with the same rule.
    # NOTE: after the split the CHUNK/COFF trade-off is gone -- COFF no longer co-resides
    # with the CHUNK arrays -- so if a device run still overflows, lowering UB_CHUNK is now
    # a MONOTONE lever (before the split it was neutral: CHUNK down => COFF up).
    chunk = min(UB_CHUNK, next_pow2(S2))
    nchunk = ceil_div(S2, chunk)
    # ---- SILENT-WRONG-RESULT GUARD (r16, lead item 3) --------------------------------
    # `NCHUNK` is a tl.constexpr, i.e. baked into any captured graph: phase 1 scans
    # exactly [0, NCHUNK*CHUNK).  That is sound ONLY while every runtime kvlen <= S2.
    # Why it holds: in PA mode S2 = block_size * max_blocks and
    #     self.max_blocks = ceil(max_model_len / block_size)        (dsa_v1.py:435)
    # i.e. S2 is the KV *capacity* of the model, not the request length, so kvlen <= S2
    # for every request by construction (the score kernel then only masks positions
    # s2 < kvlen).  A graph replayed with a larger S2 is impossible: S2 is a captured
    # scalar argument and the replay reuses the captured args.
    # We still assert it LOUDLY instead of trusting the caller: a tail that is never
    # scanned produces a *wrong top-K with no error at all*, which is exactly the
    # "it ran != it is correct" failure mode.
    # T-131 (r28): S2 is the COMPRESSED key capacity -- the score kernel maps the raw
    # kvlen through `// CMP_RATIO` before it meets `s2` (j_hi below).  Comparing a raw
    # length with a compressed capacity rejects perfectly serviceable long prompts
    # (observed: S2=5120, cmp_ratio=4 => raw capacity 20480 = max_model_len, yet a
    # kvlen of 8192 was refused).  Compare like with like, and keep the message loud.
    max_kvlen = int(kvlen_b.max()) if kvlen_b.numel() else 0
    max_ckvlen = (max_kvlen + int(cmp_ratio) - 1) // int(cmp_ratio)
    if kvlen_b.numel() and max_ckvlen > int(S2):
        raise ValueError(
            "vllm_quant_lightning_indexer: actual_seq_lengths_key exceeds the compressed "
            "key capacity S2=%d (max kvlen=%d -> %d compressed positions, cmp_ratio=%d); "
            "the top-K scan (NCHUNK=%d x CHUNK=%d) would silently drop the tail -- "
            "refusing to produce a wrong result"
            % (S2, max_kvlen, max_ckvlen, int(cmp_ratio), nchunk, chunk))
    cand = nchunk * k
    coff = next_pow2(cand)
    # Scratch cost: n_rows * nchunk * k * (4+4) B  (fp32 value + int32 index).
    #   decode (this is the FULL-capture path: dsa_v1.py:2486 _forward_decode ->
    #   indexer.py:550): n_rows = num decode tokens (small) => a few hundred KB.
    #   prefill: n_rows = T*N2 can reach ~16384 => up to ~268 MB at nchunk=2, k=2048.
    #   If prefill memory ever becomes a problem, drop `cand_val` entirely and re-gather
    #   the candidate values in phase 2 with tl.load(SC + pid*S2 + cidx): the values live
    #   in `scores` already, so only an int32 index buffer is needed (halves the scratch).
    cand_val = torch.empty((n_rows, nchunk, k), dtype=torch.float32, device=dev)
    cand_idx = torch.empty((n_rows, nchunk, k), dtype=torch.int32, device=dev)
    _qli_topk_kernel[(n_rows,)](
        scores, cand_idx, cand_val, S2,
        K=k, CHUNK=chunk, NCHUNK=nchunk, BLK_K=blk_k,
        WRITE_VAL=bool(return_values), NEG=_NEG,
    )
    vals = torch.empty((n_rows, k), dtype=torch.float32, device=dev)
    _qli_topk_merge_kernel[(n_rows,)](
        cand_val, cand_idx, idx, vals,
        K=k, NCAND=cand, COFF=coff, BLK_K=blk_k,
        WRITE_VAL=bool(return_values), NEG=_NEG,
    )
    if return_values:
        out_v = vals.view(n_rows, k).to(torch.float32)
    else:
        out_v = torch.empty(0, dtype=torch.float32, device=dev)
    if tnd:
        return idx.view(T, N2, k), out_v.view(T, N2, k) if return_values else out_v
    return idx.view(B, S1, N2, k), (out_v.view(B, S1, N2, k) if return_values else out_v)
