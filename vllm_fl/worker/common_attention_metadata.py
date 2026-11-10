# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_fl.compilation.graph import Graph
from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_ops.slot_mapping import (
    _compute_slot_mapping_graph_kernel,
)

logger = init_logger(__name__)

_LAYOUT_ATTR = "_vllm_fl_common_attention_metadata_layout"


@dataclass(frozen=True)
class _CommonAttentionMetadataLayout:
    """Fixed-address tensors used by the multi-group Triton launch."""

    block_table_ptrs: torch.Tensor
    block_table_strides: torch.Tensor
    block_sizes: torch.Tensor
    slot_mapping_ptrs: torch.Tensor
    num_groups: int
    max_num_batched_tokens: int
    total_cp_world_size: int
    total_cp_rank: int
    cp_kv_cache_interleave_size: int


def common_attention_metadata_enabled() -> bool:
    """Keep the original producer on platforms without kernel validation.

    The pointer-table Triton kernel is currently validated on NVIDIA. Other
    platforms can opt in for validation; setting 0 restores the old producer
    (not merely eager execution of the new kernel).
    """
    value = os.environ.get("VLLM_FL_COMMON_ATTENTION_METADATA")
    return current_platform.is_cuda() if value is None else value == "1"


def supports_accelerator_graph() -> bool:
    """Return whether the active platform exposes the plugin graph API."""
    return callable(getattr(Graph, "graph", None)) and callable(
        getattr(current_platform.torch_device_fn, "graph", None)
    )


@triton.jit
def _compute_num_computed_tokens_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    num_computed_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        num_computed_tokens_ptr + offsets,
        seq_len - (query_end - query_start),
        mask=mask,
    )


def _make_ptr_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    # uint64 covers every possible device address. The tensors are persistent,
    # so these raw pointers remain valid across graph capture and replay.
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.uint64,
        device=tensors[0].device,
    )


def _create_common_attention_metadata_layout(
    block_table: Any,
) -> _CommonAttentionMetadataLayout:
    tables = block_table.block_tables
    if not tables:
        raise ValueError("Common attention metadata requires a KV cache group")

    max_num_batched_tokens = tables[0].max_num_batched_tokens
    total_cp_world_size = tables[0].pcp_world_size * tables[0].dcp_world_size
    total_cp_rank = tables[0].pcp_rank * tables[0].dcp_world_size + tables[0].dcp_rank
    cp_kv_cache_interleave_size = tables[0].cp_kv_cache_interleave_size
    device = tables[0].block_table.gpu.device

    for table in tables[1:]:
        table_cp_world_size = table.pcp_world_size * table.dcp_world_size
        table_cp_rank = table.pcp_rank * table.dcp_world_size + table.dcp_rank
        if (
            table.block_table.gpu.device != device
            or table.slot_mapping.gpu.device != device
        ):
            raise ValueError("All KV cache groups must be on the same device")
        if table.max_num_batched_tokens != max_num_batched_tokens:
            raise ValueError(
                "All KV cache groups must use the same max_num_batched_tokens"
            )
        if (
            table_cp_world_size != total_cp_world_size
            or table_cp_rank != total_cp_rank
            or table.cp_kv_cache_interleave_size != cp_kv_cache_interleave_size
        ):
            raise ValueError(
                "All KV cache groups must use the same context-parallel layout"
            )

    return _CommonAttentionMetadataLayout(
        block_table_ptrs=_make_ptr_tensor([table.block_table.gpu for table in tables]),
        block_table_strides=torch.tensor(
            [table.block_table.gpu.stride(0) for table in tables],
            dtype=torch.int64,
            device=device,
        ),
        block_sizes=torch.tensor(
            [table.block_size for table in tables],
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping_ptrs=_make_ptr_tensor(
            [table.slot_mapping.gpu for table in tables]
        ),
        num_groups=len(tables),
        max_num_batched_tokens=max_num_batched_tokens,
        total_cp_world_size=total_cp_world_size,
        total_cp_rank=total_cp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )


def _get_common_attention_metadata_layout(
    block_table: Any,
) -> _CommonAttentionMetadataLayout:
    layout = getattr(block_table, _LAYOUT_ATTR, None)
    if layout is None:
        layout = _create_common_attention_metadata_layout(block_table)
        setattr(block_table, _LAYOUT_ATTR, layout)
    return layout


def compute_common_attention_metadata(
    block_table: Any,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Populate fixed-address metadata buffers consumed by attention backends."""
    if block_table.block_tables:
        layout = _get_common_attention_metadata_layout(block_table)
        _compute_slot_mapping_graph_kernel[(layout.num_groups, num_reqs + 1)](
            layout.max_num_batched_tokens,
            query_start_loc,
            positions,
            layout.block_table_ptrs,
            layout.block_table_strides,
            layout.block_sizes,
            layout.slot_mapping_ptrs,
            TOTAL_CP_WORLD_SIZE=layout.total_cp_world_size,
            TOTAL_CP_RANK=layout.total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=layout.cp_kv_cache_interleave_size,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
    _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
        query_start_loc,
        seq_lens,
        num_computed_tokens,
        num_reqs=num_reqs,
        BLOCK_SIZE=256,
    )


class CommonAttentionMetadataGraphRunner:
    """Capture and replay common attention metadata on accelerator graphs."""

    def __init__(self) -> None:
        self.graphs: dict[tuple[int, int], Any] = {}
        self.graph_pool = current_platform.get_global_graph_pool()
        self._missing_graph_keys: set[tuple[int, int]] = set()
        self._graph_capture_supported = supports_accelerator_graph()
        self._warned_graph_unavailable = False

    def clear(self) -> None:
        self.graphs.clear()
        self._missing_graph_keys.clear()

    def run(
        self,
        block_table: Any,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        *,
        use_graph: bool,
        capture: bool,
        compute: Callable[
            [
                Any,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            None,
        ] = compute_common_attention_metadata,
    ) -> bool:
        if use_graph and not self._graph_capture_supported:
            if not self._warned_graph_unavailable:
                logger.warning(
                    "Accelerator graph capture is unavailable on %s; falling "
                    "back to eager common attention metadata generation.",
                    current_platform.device_type,
                )
                self._warned_graph_unavailable = True
            use_graph = False

        if not use_graph:
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        key = (id(block_table), num_reqs)
        graph = self.graphs.get(key)
        if capture:
            if graph is not None:
                graph.replay()
                return True

            # Materialize the pointer/stride tensors before entering capture.
            # Allocating them inside the graph would make replay unsafe.
            if (
                compute is compute_common_attention_metadata
                and block_table.block_tables
            ):
                _get_common_attention_metadata_layout(block_table)

            # Compile outside capture even when model graph warmups are zero.
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            graph = Graph.graph()
            with current_platform.torch_device_fn.graph(graph, pool=self.graph_pool):
                compute(
                    block_table,
                    num_reqs,
                    query_start_loc,
                    positions,
                    seq_lens,
                    num_computed_tokens,
                )
            self.graphs[key] = graph
            # Capture records work; callers immediately consume these outputs.
            graph.replay()
            return True

        if graph is None:
            if key not in self._missing_graph_keys:
                logger.warning(
                    "Common attention metadata graph for %d requests was not "
                    "captured; falling back to eager execution.",
                    num_reqs,
                )
                self._missing_graph_keys.add(key)
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        graph.replay()
        return True
