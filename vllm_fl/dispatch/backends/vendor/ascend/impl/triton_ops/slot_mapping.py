# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of the paged KV-cache slot-mapping build (CANN
`_compute_slot_mapping_kernel`).

The kernel was originally defined inline in
`vllm_fl/worker/common_attention_metadata.py`; it lives here so that the whole
set of FlagOS-provided Triton kernels has one verifiable location
(`vllm_fl/dispatch/backends/vendor/ascend/impl/triton_ops/*.py`).  The function
name, signature, decorator and numerical behaviour are unchanged: the worker
imports `_compute_slot_mapping_graph_kernel` from this module and keeps its
call site exactly as before.

Contract (unchanged): one launch over `(num_groups, num_reqs + 1)` programs.

    max_num_tokens      : int, padded token capacity of the batch
    query_start_loc_ptr : [num_reqs + 1] int32, prefix sums of per-request tokens
    positions_ptr       : [max_num_tokens] int64, position of each token
    block_table_ptrs    : [num_groups] uint64, raw pointers to the block tables
    block_table_strides : [num_groups] int64, row stride of each block table
    block_sizes         : [num_groups] int32, KV block size of each group
    slot_mapping_ptrs   : [num_groups] uint64, raw pointers to the slot mappings

    For every token of a request the kernel writes
    `slot = block_number * block_size + local_block_offset` into that group's
    slot mapping, writes `PAD_ID` for tokens that belong to another context
    parallel rank, writes `PAD_ID` for the padding tail beyond the last request,
    and clears the block-table rows of graph-padded requests to `NULL_BLOCK_ID`.
    The padding boundary is read from `query_start_loc` on device so one captured
    graph can replay with different request lengths without a host-derived
    argument.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton


# Backported from vLLM's multi-group BlockTables slot-mapping kernel. The
# padding boundary is read from query_start_loc on device so one captured graph
# can replay with different request lengths without a host-derived argument.
@triton.jit(do_not_specialize=["max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mapping_ptrs,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    block_table_ptr = tl.load(block_table_ptrs + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    block_table_ptr = tl.multiple_of(block_table_ptr, 16)
    block_table_stride = tl.load(block_table_strides + group_idx)
    block_size = tl.load(block_sizes + group_idx)
    slot_mapping_ptr = tl.load(slot_mapping_ptrs + group_idx).to(
        tl.pointer_type(tl.int64)
    )
    slot_mapping_ptr = tl.multiple_of(slot_mapping_ptr, 16)

    if req_idx == tl.num_programs(1) - 1:
        actual_num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        for i in range(actual_num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # Padded request rows are not refreshed by BlockTable.commit_block_table().
    # Clear them in the existing per-group producer instead of launching one
    # eager fill per cache group. query_start_loc is a fixed-address device
    # buffer, so the same captured shape can replay with a different actual
    # request count.
    # A graph-padded row has no scheduled query tokens. Do not use seq_len as
    # the predicate: valid scheduler rows can transiently carry seq_len == 0.
    if start_idx == end_idx:
        row_offset = req_idx * block_table_stride
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + row_offset + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
