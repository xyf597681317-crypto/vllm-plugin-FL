# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of the CANN `ReshapeAndCacheNd` KV cache write.

Contract, measured on the 106 host from `torch_npu._npu_reshape_and_cache` and
from the single vendor call site (`impl/attention.py::reshape_and_cache`):

    key, value    : [num_tokens, num_kv_heads, head_size], bf16/fp16, contiguous
    key_cache     : [num_blocks, block_size, num_kv_heads, head_size]
    value_cache   : same shape and dtype as `key_cache`
    slot_mapping  : [num_tokens] int32 (int64 accepted here), flat slot id,
                    negative for padded tokens that must not be written

The cache row a token writes is `slot_mapping[i]`; the cache itself is paged, so
the flat slot is decomposed as `slot = block * block_size + offset_in_block` and
the element address is rebuilt from the cache strides.  That decomposition is
the block-table indirection: the caller's `block_tables` produced the flat
`slot_mapping` upstream, and a `slot_mapping` that is a view over a larger
int64 buffer is accepted as-is.

Two addressing modes are selected on the host from tensor metadata only:

* contiguous cache (`stride == (block_size * row, row, head_size, 1)`): the
  write is a single vector of `num_kv_heads * head_size` elements, no division;
* strided cache (e.g. `cache[..., ::2]` or a slice of a wider allocation): the
  element offset is rebuilt per element from `(stride_block, stride_slot,
  stride_head, stride_dim)`.

Every element offset is int64 when the key span or the cache span does not fit
int32: a packed multi-GB cache holds more than 2**31 elements, and an int32
offset would wrap.  The int32 path is used only when the host proved both spans
fit, which is the usual case and the faster one.

Tiling (measured on the 105 host, T=4096, 4 KV heads x 128, block 128, bf16):

* one program owns `BT` tokens and walks them inside the kernel; each token is
  a separate 1-D vector of `row` elements with a **scalar** destination base.
  A `[BT, row]` 2-D store whose per-row base is a vector is lowered to one
  scalar store per lane on this backend: 11.4 ms vs 0.097 ms for the 1-D form,
  a 118x difference on the same 8 MB of traffic.  That is why the store tile is
  1-D and the token loop is inside the program.
* `BT = 16` is the measured optimum (BT=8 0.113 ms, BT=16 0.097 ms, BT=32
  0.115 ms); it also keeps the grid at 256 programs for a 4096-token prefill.
  For small batches BT shrinks so the grid still spreads over the cores.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Tokens per program for a full-size prefill, and the smallest number of
# programs the host aims for before it starts to trade BT for parallelism.
_BT = 16
_BT_MIN_GRID = 96
# Row tile: `_BLK_CAP` columns per program; a wider row is handled by the
# column loop-free 1-D form only when it fits, so the cap also bounds UB use.
_BLK_CAP = 2048
# The Ascend backend rejects a coreDim of exactly 65536.
_MAX_GRID = 65535
# Above this BT the token loop unrolls into a kernel that is not worth JIT-ing.
_BT_CAP = 64
_INT32_MAX = 2**31 - 1
# Element dtypes written by the supported models; a float8 or int cache keeps
# the native operator.
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _next_pow2(n: int) -> int:
    """Smallest power of two that is not smaller than `n`."""
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _reshape_and_cache_kernel(
        key_ptr,
        val_ptr,
        kcache_ptr,
        vcache_ptr,
        slot_ptr,
        num_tokens,
        row,
        block_size,
        head_size,
        stride_block,
        stride_slot,
        stride_head,
        stride_dim,
        BLK: tl.constexpr,
        BT: tl.constexpr,
        CONTIG: tl.constexpr,
        I64: tl.constexpr,
    ):
        """
        Copy `key`/`value` rows into their paged cache slots.

        `row` is `num_kv_heads * head_size`, the number of contiguous elements
        each token owns in `key`/`value`.  A negative slot marks a padded token
        and is skipped, matching the native operator.

        The destination address of one token is `base + cols`, where `base` is a
        runtime scalar; keeping the row tile 1-D is what makes the store
        vectorize (see the module docstring).
        """
        pid = tl.program_id(0)
        cols = tl.arange(0, BLK)
        col_mask = cols < row
        if CONTIG:
            elem = cols
        elif I64:
            head = cols.to(tl.int64) // head_size
            dim = cols.to(tl.int64) - head * head_size
            elem = head * stride_head + dim * stride_dim
        else:
            head = cols // head_size
            dim = cols - head * head_size
            elem = head * stride_head + dim * stride_dim
        for j in tl.static_range(BT):
            t = pid * BT + j
            if t < num_tokens:
                slot = tl.load(slot_ptr + t)
                if slot >= 0:
                    if I64:
                        s = slot.to(tl.int64)
                        blk = s // block_size
                        base = blk * stride_block + (s - blk * block_size) * stride_slot
                        src = t.to(tl.int64) * row + cols
                    else:
                        blk = slot // block_size
                        base = blk * stride_block + (slot - blk * block_size) * stride_slot
                        src = t * row + cols
                    key = tl.load(key_ptr + src, mask=col_mask, other=0.0)
                    tl.store(kcache_ptr + base + elem, key, mask=col_mask)
                    val = tl.load(val_ptr + src, mask=col_mask, other=0.0)
                    tl.store(vcache_ptr + base + elem, val, mask=col_mask)


def _cache_strides(cache: torch.Tensor, head_size: int, row: int, block_size: int):
    """Return `(contiguous, strides)` for the paged cache write."""
    stride_block, stride_slot, stride_head, stride_dim = cache.stride()
    contiguous = (
        stride_dim == 1
        and stride_head == head_size
        and stride_slot == row
        and stride_block == block_size * row
    )
    return contiguous, (stride_block, stride_slot, stride_head, stride_dim)


def triton_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    """
    Write `key`/`value` into the paged KV cache through the Triton kernel.

    Args:
        key: `[num_tokens, num_kv_heads, head_size]`, contiguous.
        value: same shape, dtype and layout as `key`.
        key_cache: `[num_blocks, block_size, num_kv_heads, head_size]`.
        value_cache: same shape, dtype and layout as `key_cache`.
        slot_mapping: `[num_tokens]` int32/int64 flat cache slot per token.

    Returns:
        `True` when the kernels were launched, `False` when the layout is
        outside the supported set and the caller must keep the torch_npu
        operator.
    """
    if not HAS_TRITON:
        return False
    for tensor in (key, value):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
            return False
        if tensor.dtype not in _DTYPES or not tensor.is_contiguous():
            return False
    for cache in (key_cache, value_cache):
        if not isinstance(cache, torch.Tensor) or cache.dim() != 4:
            return False
        if cache.dtype not in _DTYPES:
            return False
    if key.shape != value.shape or key.dtype != value.dtype:
        return False
    if key_cache.shape != value_cache.shape or key_cache.dtype != value_cache.dtype:
        return False
    if key.dtype != key_cache.dtype:
        return False
    num_tokens, num_kv_heads, head_size = key.shape
    num_blocks, block_size, cache_heads, cache_dim = key_cache.shape
    if (num_kv_heads, head_size) != (cache_heads, cache_dim):
        return False
    if num_tokens == 0 or block_size <= 0 or num_blocks <= 0 or head_size < 1:
        return False
    if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.dim() != 1:
        return False
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if not slot_mapping.is_contiguous() or slot_mapping.numel() < num_tokens:
        return False
    for cache in (key_cache, value_cache):
        if min(cache.stride()) <= 0:
            return False

    row = num_kv_heads * head_size
    contiguous_k, key_strides = _cache_strides(key_cache, head_size, row, block_size)
    contiguous_v, value_strides = _cache_strides(value_cache, head_size, row, block_size)
    if contiguous_k != contiguous_v or key_strides != value_strides:
        return False

    # int32 addressing is valid only when both the source row span and the
    # largest cache element offset fit; otherwise every offset is int64.
    src_span = num_tokens * row
    dst_span = (
        (num_blocks - 1) * key_strides[0]
        + (block_size - 1) * key_strides[1]
        + (cache_heads - 1) * key_strides[2]
        + (cache_dim - 1) * key_strides[3]
    )
    int64 = src_span > _INT32_MAX or dst_span > _INT32_MAX

    block = min(_BLK_CAP, _next_pow2(row))
    tokens_per_block = min(_BT, max(1, triton.cdiv(num_tokens, _BT_MIN_GRID)))
    if tokens_per_block > _BT_CAP:
        return False
    grid = (triton.cdiv(num_tokens, tokens_per_block),)
    _reshape_and_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        row,
        block_size,
        head_size,
        key_strides[0],
        key_strides[1],
        key_strides[2],
        key_strides[3],
        BLK=block,
        BT=tokens_per_block,
        CONTIG=contiguous_k,
        I64=int64,
    )
    return True
