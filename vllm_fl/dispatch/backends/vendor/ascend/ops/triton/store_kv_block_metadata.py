# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""store_kv_block_metadata -- AICPU (AscendC package) to Triton port.

Source: csrc/attention/store_kv_block_metadata/ of the vllm-ascend tree
(checked out in the container at /vllm-workspace/vllm-ascend/csrc/attention/
store_kv_block_metadata/ and in the vendor tree's csrc/attention/store_kv_block_metadata/).
Note that this operator is *not* part of the "csrc/ascend" closure that vllm-fl builds
itself: csrc/ascend/build_opp.sh does not list it and FL's own _C_ascend.so does not contain
the string ``store_kv_block_metadata``.  The runtime registration comes from the vendor
extension (vllm-ascend csrc/torch_binding.cpp L3630-3634); the FL call site reaches exactly
that registration.

Interface (vendor csrc/torch_binding.cpp L3630-3633):
    store_kv_block_metadata(Tensor slot_mapping_npu, Tensor group_len, Tensor group_key_idx,
                            Tensor group_key_cache_idx, int block_size=0) -> ()
    Input : slot_mapping [numTokens] int32   (1-D, read only)
    In/Out: group_len           [capacity] int32
            group_key_idx       [capacity] int32
            group_key_cache_idx [capacity] int32
            (all three are pre-allocated by the caller, written in place; op_kernel_aicpu/
             store_kv_block_metadata_aicpu.h L47-53 fixes that argument order and
             op_host/store_kv_block_metadata_infershape.cpp L23-32 declares no output at all)
    Attr  : block_size (int, must be > 0 - torch_adpt L37 and aicpu L50-53)

Formula (op_kernel_aicpu/store_kv_block_metadata_aicpu.cpp L57-126, GenMetaData):
    L84-90   walk slot_mapping left to right; a negative slot is dirty and is skipped
    L92      blockId = cacheSlot / blockSize                 (C truncating division)
    L95-96   group_key_idx[g] = position of the run start, group_key_cache_idx[g] = that slot
    L99-105  extend the run while
                 position + 1 < numTokens
                 && slot_mapping[pos + 1] / blockSize == blockId
                 && slot_mapping[pos + 1] == slot_mapping[pos] + 1
    L107     group_len[g] = run length
    L109-110 advance to the first token not in the run and start the next group
    L113-123 zero fill group_len/group_key_idx/group_key_cache_idx from the group count up to
             the capacity (memset with 0, deliberately not -1: store_kv_block reads group_len
             as uint32_t and a negative filler would bypass its ``groupLen <= 0`` guard)

The three outputs are the "contiguous run" description consumed by the companion operator
``store_kv_block`` (csrc/attention/store_kv_block/), which turns every group into a single
contiguous row-block copy:
    keyCache[group_key_cache_idx[g] : +group_len[g]] = keyIn[group_key_idx[g] : +group_len[g]]

Call site (the only one):
    vllm_fl/attention/ascend/sfa_v1.py L441-448, guarded by
    ``get_ascend_config().c8_enable_reshape_optim``, which is hard-coded to False in
    vllm_fl/configs/ascend.py L359, so the branch is currently inactive by default.

Reference environment (vllm-ascend vendor extension, Ascend 910C):
    * ``torch.ops._C_ascend.store_kv_block_metadata`` exists after loading the vendor
      extension: schema ``(Tensor slot_mapping_npu, Tensor group_len, Tensor group_key_idx,
      Tensor group_key_cache_idx, int block_size=0) -> ()``.  The FL extension does not
      provide it (string count 0 in vllm_fl/_C_ascend*.so).
    * Running it fails here with
      ``RuntimeError: aclnnStoreKvBlockMetadata ... not in libopapi.so`` because this
      container's OPP payload (built from csrc/ascend/build_opp.sh) does not ship the
      StoreKvBlockMetadata AICPU binary; the kernel source above is therefore the only
      executable specification available.

----------------------------------------------------------------------------------------
Implementation notes
----------------------------------------------------------------------------------------
1. The AICPU source is a serial loop; it is ported here as one **parallel** pass over the
   token axis.  A position opens a group when it holds a non-negative slot and does not
   continue the previous token's run:

       slot            = load(slot_mapping)
       valid           = slot >= 0
       brk   = valid == 0                          # dirty slot
             | previous token invalid               # index 0 uses a negative sentinel
             | slot != previous_slot + 1
             | slot / block_size != previous_slot / block_size
       start = valid & brk

   `brk` marks the boundary that ends a run; a `brk` lane at a *dirty* position is not a
   group start, and it also terminates the run that precedes it - exactly the semantics of
   the C loop, which skips the dirty token (L87-89) and would then start a fresh group
   (L95).

2. Run length.  ``run = cumsum(brk)`` is the 1-based index of the run each lane belongs to,
   and the position of the r-th run start is obtained with a single masked scatter store
       POS[run] = index  where brk
   The masked lanes carry the distinct values 1..R (every `brk` lane increments the cumsum
   exactly once), so the scatter has no write conflict and is deterministic.  With POS
   pre-filled with numTokens, the run that starts at POS[r] ends at POS[r + 1], hence
       group_len   = POS[r + 1] - POS[r]        (L107 groupEndIdx - idxSlotmap)
       group_key_idx = POS[r]                   (L95)
       group_key_cache_idx = slot_mapping[POS[r]]  (L96)
   The group ordinal of a start is the exclusive prefix count of starts,
   ``grp = cumsum(start) - start``, which reproduces the ``idxGroups`` counter (L110).

3. The unused tail ``[group_count, capacity)`` is zero filled exactly like the memsets at
   L113-123, but on the host: the wrapper zeroes ``group_len`` and the ``[0, capacity)``
   prefix of ``group_key_idx`` / ``group_key_cache_idx`` before the launch, so the kernel only
   ever writes entries it actually computed.  This keeps the kernel free of a scalar
   reduction (``tl.sum`` -> 0-d tensor) whose result would have to be broadcast back into the
   store masks; that pattern is documented as unreliable on the Ascend backend in the sibling
   moe_gating_top_k_hash port.  The group writes are additionally masked with
   ``group_count < capacity`` so that a caller which under-allocates cannot be written out of
   bounds (the C code would corrupt memory in that case).

4. Division by ``block_size``: the C code truncates towards zero, Triton's ``//`` semantics
   for negative operands are version dependent.  This is irrelevant here: the block
   comparison is only decisive when both operands are non-negative (otherwise the ``valid``
   terms already force ``brk`` to true), and for non-negative operands truncation and floor
   division agree.

5. The whole scan runs in a **single program** (one launch, one token vector), so it needs
   ``max(numTokens, capacity)`` to fit the UB budget as a power of two; the wrapper asserts
   4096.  A grid-parallel variant would need a two-pass prefix scan and is not required by
   the only call site (see the ``c8_enable_reshape_optim`` note above).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import next_pow2

__all__ = ["store_kv_block_metadata"]

#: The single-program scan keeps several int32 vectors of length BLK alive at once; the
#: Ascend unified buffer is 192KB and a 1-D i32 vector of 4096 elements is 16KB, so 4096 is
#: a safe power-of-two bound (the sibling operators report UB exhaustion at 8192 elements).
_MAX_TOKENS = 4096


@triton.jit
def _store_kv_block_metadata_kernel(
    SM, GL, GK, GKC, POS,
    N, CAP, BLOCK_SIZE,
    BLK: tl.constexpr,
):
    """Single-program port of GenMetaData (aicpu L57-126).

    SM   : slot_mapping            [N]   int32
    GL   : group_len               [CAP] int32 (in/out)
    GK   : group_key_idx           [CAP] int32 (in/out)
    GKC  : group_key_cache_idx     [CAP] int32 (in/out)
    POS  : scratch                 [N+2] int32, pre-filled with N
    """
    idx = tl.arange(0, BLK)
    live = idx < N

    # ---- L86: read the slot; L87-89: a negative slot is dirty and is skipped ----
    slot = tl.load(SM + idx, mask=live, other=-1).to(tl.int32)
    # Unmasked gather at a clamped index (a masked gather trips a FLIR assertion on Ascend);
    # the sentinel for "there is no previous token" is negative so that it reads as invalid.
    prev_idx = tl.minimum(tl.maximum(idx - 1, 0), N - 1)
    slot_prev = tl.where(idx >= 1, tl.load(SM + prev_idx).to(tl.int32), -2)

    valid = slot >= 0
    valid_prev = slot_prev >= 0
    blk = slot // BLOCK_SIZE                # L92 blockId = cacheSlot / blockSize
    blk_prev = slot_prev // BLOCK_SIZE

    # ---- L99-105: a run continues only while the next slot is valid, consecutive and
    #      inside the same block.  `brk` is the negation of "continues the previous run".
    brk = live & (
        (valid == 0)
        | (valid_prev == 0)
        | (slot != slot_prev + 1)
        | (blk != blk_prev)
    )
    start = valid & brk                     # L95: this lane opens a group

    # ---- POS[r] = position of the r-th run boundary (r = 1..R); masked lanes carry
    #      distinct r values, so the scatter store is conflict free.
    run = tl.cumsum(brk.to(tl.int32), axis=0)
    tl.store(POS + run, idx, mask=brk)

    # ---- L110: idxGroups is the exclusive prefix count of group starts
    st = start.to(tl.int32)
    grp = tl.cumsum(st, axis=0) - st

    # ---- L95/L96/L107: emit the three group arrays
    r = tl.minimum(tl.maximum(run, 0), N)
    begin = tl.load(POS + r)
    end = tl.load(POS + r + 1)
    wr = start & live & (grp < CAP)
    tl.store(GL + grp, end - begin, mask=wr)
    tl.store(GK + grp, begin, mask=wr)
    tl.store(GKC + grp, slot, mask=wr)


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int32 else t.to(torch.int32)


def store_kv_block_metadata(
    slot_mapping_npu: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int = 0,
) -> None:
    """Equivalent of the AscendC/AICPU StoreKvBlockMetadata operator.

    Args:
        slot_mapping_npu: [numTokens] int32, the per-token KV cache slot; negative entries
            are dirty slots and are skipped.
        group_len: [capacity] int32, in/out; receives the length of every contiguous run.
        group_key_idx: [capacity] int32, in/out; receives the source position of every run.
        group_key_cache_idx: [capacity] int32, in/out; receives the destination cache slot.
        block_size: number of slots per KV block, must be > 0.

    Returns:
        None.  All three group arrays are written in place, matching the schema
        ``(...) -> ()``; the caller detects the effective group count by scanning for the
        first zero ``group_len`` entry (torch_adpt L26-28).
    """
    assert slot_mapping_npu.numel() > 0, "slot_mapping must not be empty"
    assert int(block_size) > 0, f"block_size must be positive, got {block_size}"
    assert group_len.dim() == 1, f"group_len must be 1-D, got {tuple(group_len.shape)}"
    assert group_key_idx.dim() == 1, f"group_key_idx must be 1-D, got {tuple(group_key_idx.shape)}"
    assert group_key_cache_idx.dim() == 1, \
        f"group_key_cache_idx must be 1-D, got {tuple(group_key_cache_idx.shape)}"

    num_tokens = slot_mapping_npu.numel()
    capacity = group_len.numel()
    assert group_key_idx.numel() >= capacity, "group_key_idx is shorter than group_len"
    assert group_key_cache_idx.numel() >= capacity, \
        "group_key_cache_idx is shorter than group_len"

    blk = next_pow2(max(num_tokens, capacity))
    assert blk <= _MAX_TOKENS, (
        f"the single-program scan needs max(numTokens, capacity) <= {_MAX_TOKENS}, "
        f"got numTokens={num_tokens}, capacity={capacity}"
    )

    device = slot_mapping_npu.device
    slot = _as_int32(slot_mapping_npu.contiguous().reshape(-1))
    gl = _as_int32(group_len.contiguous().reshape(-1))
    gk = _as_int32(group_key_idx.contiguous().reshape(-1))
    gkc = _as_int32(group_key_cache_idx.contiguous().reshape(-1))

    # L113-123 zero fill: every entry the kernel does not write must read back as 0 (and not
    # as a negative filler, which store_kv_block would read as a huge uint32 group length).
    gl[:capacity].zero_()
    gk[:capacity].zero_()
    gkc[:capacity].zero_()

    # POS[r] defaults to numTokens, which is the "run ends at the end of the tensor" case.
    pos = torch.full((num_tokens + 2,), num_tokens, dtype=torch.int32, device=device)

    _store_kv_block_metadata_kernel[(1,)](
        slot, gl, gk, gkc, pos,
        num_tokens, capacity, int(block_size),
        BLK=blk,
    )

    # Preserve the caller's dtypes: the operator schema does not fix them, the AICPU kernel
    # simply reinterprets the buffers as int32.
    for src, dst in ((gl, group_len), (gk, group_key_idx), (gkc, group_key_cache_idx)):
        if dst.dtype != torch.int32:
            dst.copy_(src.to(dst.dtype).reshape(dst.shape))
    return None
