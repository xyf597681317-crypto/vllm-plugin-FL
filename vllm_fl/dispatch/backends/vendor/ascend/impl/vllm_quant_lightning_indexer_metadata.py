# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""npu_vllm_quant_lightning_indexer_metadata -- AscendC (AICPU) ported to Python.

Source
------
csrc/ascend/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/
    vllm_quant_lightning_indexer_metadata_aicpu.cpp   L20-907 (Compute L20,
        Prepare L32, ParamsCheck L67, CheckSingleParam L72, CheckExistence L146,
        CheckConsistency L192, CheckFeature L207, ProcessSocVersion L230,
        ParamsInit L242, GetS1SeqSize L269, GetS2SeqSize L286, CalcSplitInfo L300,
        CalcPreTokenLeftUp L318, CalcNextTokenLeftUp L328, CalcCost L346,
        CalcCostTable L367, CalcS2TokenRange L378, CalcBatchCache L412,
        CalcS1GCache L426, CalcBatchCost L493, CalcCostInfo L524, UpdateCursor L543,
        AssignByBatch L591, AssignByRow L626, AssignByBlock L652, ForceAssign L677,
        IsNeedRecordFDInfo L700, RecordFDInfo L719, AssignBlockToCore L741,
        CalcSplitPlan L781, SplitFD L812, BalanceSchedule L837, GenMetaData L863,
        REGISTER_CPU_KERNEL L905)
    vllm_quant_lightning_indexer_metadata_aicpu.h     L26-310 (constants L27,
        SparseMode L35, ValidSocVersion L44, SplitResult L100, SplitInfo L120,
        CostInfo L135, BatchCache L160, S1GCache L170, CoreCache L182,
        AssignContext L189, member defaults L264-298)
csrc/ascend/attention/vllm_quant_lightning_indexer/op_kernel/
    vllm_quant_lightning_indexer_metadata.h           L23-77 (QLI_META_SIZE=1024,
        AIC_CORE_NUM=36, AIV_CORE_NUM=72, LI_METADATA_SIZE=8, LD_METADATA_SIZE=8,
        LI_* field indices L33-40, LD_* field indices L43-49, QliMetaData L70-73)

The kernel runs on the **AICPU** (op JSON: engine ``DNN_VM_AICPU``, kernelSo
``libtransformer_aicpu_kernels.so``), i.e. it is a plain serial integer program -- no vector
or cube instruction is involved.  A Triton device kernel would therefore buy nothing: the
payload is a bin-packing walk in which every step depends on the previous one, and the whole
result is 1024 int32 values.  This module is a literal transliteration of that host program;
``torch`` is used only to read the length vectors and to materialise the output tensor.

Interface
---------
Registered torch schema (csrc/ascend/torch_binding.cpp L1245; the PrivateUse1 impl is
``vllm_fl_native::lightning_indexer_metadata`` L259-276, which calls
``aclnnVllmQuantLightningIndexerMetadata`` L272)::

    npu_vllm_quant_lightning_indexer_metadata(
        int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode,
        int key_quant_mode, Tensor? actual_seq_lengths_query=None,
        Tensor? actual_seq_lengths_key=None, int batch_size=0, int max_seqlen_q=0,
        int max_seqlen_k=0, str layout_query="BSND", str layout_key="BSND",
        int sparse_count=2048, int sparse_mode=3,
        int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807,
        int cmp_ratio=1, str device="npu") -> Tensor

Output: ``int32[1024]`` (op_host/..._infershape.cpp L21-33) on the device of the inputs.

The three call sites all use the same keyword set::

    vllm_fl/attention/ascend/dsa_v1.py:870                  (prefill)
    vllm_fl/attention/ascend/dsa_v1.py:1101                 (decode)
    vllm_fl/attention/ascend/context_parallel/dsa_cp.py:888 (context parallel)

    actual_seq_lengths_query=<int32 [B], exclusive prefix sum of the query lengths>
    actual_seq_lengths_key=<int32 [B], per-request KV length>
    num_heads_q=64, num_heads_k=1, head_dim=128, query_quant_mode=0, key_quant_mode=0,
    batch_size=B, max_seqlen_q=..., max_seqlen_k=..., layout_query="TND",
    layout_key="PA_BSND", sparse_count=512, sparse_mode=3,
    pre_tokens=next_tokens=(1<<63)-1, cmp_ratio=4, device="npu:0"

``actual_seq_lengths_query`` is the *exclusive* prefix-sum form produced by
``query_start_loc[1:]``; ``GetS1SeqSize`` (L275-283) differentiates it, so the two
conventions agree.

Metadata layout (``optiling::detail::QliMetaData``, QLI metadata header L70-73)
------------------------------------------------------------------------------
``LIMetadata[36][8]`` occupies elements ``[0, 288)`` and ``LDMetadata[72][8]`` occupies
``[288, 864)``; the last 160 elements of the 1024-element buffer are unused.  Field order
follows LI_*_INDEX (L33-40) / LD_*_INDEX (L43-49)::

    LI[core] = [core_enable, bN2_start, m_start, s2_start, bN2_end, m_end, s2_end,
                first_ld_data_workspace_idx]        # [start, end) block intervals
    LD[core] = [core_enable, bN2_idx, m_idx, workspace_idx, workspace_num, m_start,
                m_num, <never written>]

Algorithm
---------
1. ``ParamsInit`` (L242-267): ``groupSize = num_heads_q / num_heads_k``,
   ``batchSize = min(queryBatch, kvBatch)``, ``mBaseSize = 256``,
   ``s2BaseSize = 2048`` on 910B and ``128`` on Ascend950, and ``sparse_mode == 3`` sets
   ``attentionMode = 1``.
2. ``CalcSplitInfo`` (L300-316): per batch, the number of ``mBaseSize`` rows
   (``s1Size * groupSize`` rounded up), the number of ``s2BaseSize`` blocks
   (``s2Size / cmpRatio`` rounded up) and both tail sizes.
3. ``CalcCost`` (L346-365) is the hardware cost model.  On the 910B branch (the SoC string
   does **not** contain ``Ascend950``) it is
   ``(basicM / groupSize) * ((basicS2 <= 1024 ? 1536 : 2560) + basicS2) / 2``; the Ascend950
   branch is ``6 * ceil(basicM/16) + 10 * ceil(basicS2/64)``.  The container reports
   ``Ascend910_9382``, so the 910B branch is live here; the native kernel accepting
   ``cmp_ratio = 8`` (L215-220, rejected by the Ascend950 rule at L222-225) confirms it.
4. ``CalcS1GCache`` (L426-491) masks each row to the causal window: with ``sparse_mode = 3``
   the row covers ``[0, s1LastToken + (s2Size - s1Size)]`` (clipped, then divided by
   ``cmpRatio``), which is the number of ``s2BaseSize`` blocks the row costs.
5. ``CalcBatchCost``/``CalcCostInfo`` (L493-541) accumulate cost and block count per batch
   and remember the global ``maxS1GCost``.
6. ``CalcSplitPlan`` (L781-810) hands whole batches/rows to the AIC cores greedily: each
   core's budget is ``unassignedCost / remainingCores`` raised to at least ``maxS1GCost``
   (L747-750), consumed by ``AssignByBatch`` (L591) and then ``AssignByRow`` (L626).  The
   recorded cut points ``bN2End``/``gS1End``/``s2End`` are the *exclusive* ends of each
   core's slice.
7. ``GenMetaData`` (L863-901) turns the cut points into the table: core ``i`` owns
   ``[end(i-1), end(i))`` and is enabled iff ``i < usedCoreNum``.

Branches that are **not** covered
---------------------------------
* ``supportFd_`` is a fixed ``false`` member (aicpu.h L272) that no attribute ever writes, so
  the flash-decode sub-path is dead in the native operator too: ``AssignByBlock`` (L652),
  ``ForceAssign`` (L677), ``UpdateCursor`` (L543), ``IsNeedRecordFDInfo`` (L700),
  ``RecordFDInfo`` (L719) and ``SplitFD`` (L812) never run, ``curKvSplitPart`` stays 1 and
  ``LDMetadata`` is therefore **always all zero** (``fdUsedVecNum == 0``, so every AIV core
  only receives ``LD[i][0] = 0``).  Those routines are transliterated for completeness but
  sit behind ``_SUPPORT_FD`` and are unreachable, exactly as in the C++.
* ``isS1G_`` is a fixed ``true`` member (aicpu.h L297), so the ``else`` branch of
  ``CalcS2TokenRange`` (L394-404) is unreachable.
* ``SparseMode`` values other than ``DEFAULT_MASK`` (0) and ``RIGHT_DOWN_CAUSAL`` (3) are
  rejected by ``CheckSingleParam`` (L128-132); the ``BAND`` branches of
  ``CalcPreTokenLeftUp`` (L322-324) and ``CalcNextTokenLeftUp`` (L339-340) are therefore
  unreachable as well (they are still transliterated).
* The Ascend950 cost model and ``s2BaseSize = 128`` are implemented but not exercised on
  this container (SoC ``Ascend910_9382``, i.e. 910_93).
* ``CheckSingleParam`` (L100-108) hard-requires ``num_heads_q == 64`` and ``num_heads_k == 1``
  and the native operator fails otherwise, so this port raises as well instead of inventing a
  schedule for unsupported head counts.
* ``head_dim``, ``query_quant_mode``, ``key_quant_mode`` and ``sparse_count`` are accepted for
  signature compatibility and are **not** used by the metadata kernel.
* The native host allocates the output with ``at::empty`` (torch_binding.cpp L181), so every
  element ``GenMetaData`` does not write is *indeterminate* natively: the trailing 7 fields of
  a disabled LI core, all of ``LI[aic:]``, all of ``LD[aiv:]``, ``LD[i][7]`` (never written)
  and the unused tail ``[864, 1024)``.  No consumer reads them (an AscendC core reads field 0
  first and stops when it is 0), and this port writes a deterministic **0** there.
* ``GetS1SeqSize``/``GetS2SeqSize`` reinterpret the input buffers as ``int32``; this port
  reads the logical tensor values instead, which is identical for the int32 tensors that all
  three call sites pass.
"""

from __future__ import annotations

import torch

__all__ = ["npu_vllm_quant_lightning_indexer_metadata"]

# ---------------------------------------------------------------------------
# Constants (vllm_quant_lightning_indexer_metadata.h L24-49, aicpu.h L27)
# ---------------------------------------------------------------------------
_AIC_CORE_NUM = 36           # AIC_CORE_NUM  (QLI metadata header L24) -- LI table capacity
_AIV_CORE_NUM = 72           # AIV_CORE_NUM  (QLI metadata header L25) -- LD table capacity
_QLI_META_SIZE = 1024        # QLI_META_SIZE (QLI metadata header L26)
_LI_METADATA_SIZE = 8        # QLI metadata header L29
_LD_METADATA_SIZE = 8        # QLI metadata header L30
_LI_BLOCK = _LI_METADATA_SIZE * _AIC_CORE_NUM          # 288 -- offset of the LD table

_LI_CORE_ENABLE = 0                                     # L33
_LI_BN2_START = 1                                       # L34
_LI_M_START = 2                                         # L35
_LI_S2_START = 3                                        # L36
_LI_BN2_END = 4                                         # L37
_LI_M_END = 5                                           # L38
_LI_S2_END = 6                                          # L39
_LI_FIRST_LD_DATA_WORKSPACE_IDX = 7                     # L40

_LD_CORE_ENABLE = 0                                     # L43
_LD_BN2_IDX = 1                                         # L44
_LD_M_IDX = 2                                           # L45
_LD_WORKSPACE_IDX = 3                                   # L46
_LD_WORKSPACE_NUM = 4                                   # L47
_LD_M_START = 5                                         # L48
_LD_M_NUM = 6                                           # L49

_FA_TOLERANCE_RATIO = 2      # aicpu.h L27

_NORMAL_BLOCK = 0            # enum BlockType (aicpu.h L29-33)
_TAIL_BLOCK = 1

_SPARSE_MODE_DEFAULT_MASK = 0
_SPARSE_MODE_ALL_MASK = 1
_SPARSE_MODE_LEFT_UP_CAUSAL = 2
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3
_SPARSE_MODE_BAND = 4

_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1
_INT32_MIN = -(1 << 31)

_M_BASE_SIZE = 256           # aicpu.h L295
_S2_BASE_SIZE_910B = 2048    # aicpu.cpp L260
_S2_BASE_SIZE_950 = 128      # aicpu.cpp L262 and L264

#: aicpu.h L272 -- ``supportFd_`` is a fixed ``false`` member and no attribute writes it, so
#: every ``if (supportFd_ ...)`` branch below is dead in the native operator as well.
_SUPPORT_FD = False
#: aicpu.h L297 -- ``isS1G_`` is a fixed ``true`` member.
_IS_S1G = True

#: Fallback AIC/AIV core counts (aicpu.h L274-275 use the same 24/48); the live values come
#: from the platform in op_api/aclnn_vllm_quant_lightning_indexer_metadata.cpp L96-97.
_DEFAULT_AIC_CORE_NUM = 24
_DEFAULT_AIV_CORE_NUM = 48


def _c_div(a: int, b: int) -> int:
    """C/C++ integer division: truncation towards zero (Python floors instead)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _clip(value: int, low: int, high: int) -> int:
    """aicpu.h L56-66 ``Clip``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _within_tolerance(limit: int, tolerance: int, value: int) -> bool:
    """aicpu.h L68-72 ``IsWithinTolerance``: ``limit + tolerance >= value``."""
    return limit + tolerance >= value


def _read_int32_tensor(tensor: torch.Tensor, name: str) -> list:
    """Read a tensor the way the AICPU reads it (``const int32_t*``, linear, dim0 = count)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.numel() == 0:
        return []
    values = tensor.detach().reshape(-1).to(torch.int64).cpu().tolist()
    out = []
    for value in values:
        value = int(value)
        if value < _INT32_MIN or value > _INT32_MAX:
            raise ValueError(f"{name} contains {value}, outside the int32 range the AICPU reads")
        out.append(value)
    return out


def _dim0(tensor: torch.Tensor, fallback: int) -> int:
    """``TensorShape::GetDimSize(0)`` (aicpu.cpp L176 / L186)."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return fallback
    if tensor.dim() == 0:
        return tensor.numel() or fallback
    return int(tensor.shape[0])


def _resolve_device(actual_seq_lengths_query, actual_seq_lengths_key, device) -> torch.device:
    """Output device: the input tensors win, then the ``device`` string (schema default).

    The native host allocates the output from the input options (torch_binding.cpp L169-182).
    """
    for tensor in (actual_seq_lengths_query, actual_seq_lengths_key):
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    resolved = device if isinstance(device, torch.device) else torch.device(str(device))
    if resolved.type == "cpu" or resolved.index is not None:
        return resolved
    index = 0
    if resolved.type == "npu":
        try:
            import torch_npu  # noqa: F401

            index = int(torch.npu.current_device())
        except Exception:  # pragma: no cover - torch_npu not installed
            index = 0
    return torch.device(resolved.type, index)


class _SplitInfo:
    """aicpu.h L120-132."""

    __slots__ = ("s1GBaseNum", "s2BaseNum", "s1GTailSize", "s2TailSize", "isKvSeqAllZero")

    def __init__(self, batch_size: int) -> None:
        self.s1GBaseNum = [0] * batch_size
        self.s2BaseNum = [0] * batch_size
        self.s1GTailSize = [0] * batch_size
        self.s2TailSize = [0] * batch_size
        self.isKvSeqAllZero = True


class _CostInfo:
    """aicpu.h L135-147."""

    __slots__ = ("bN2CostOfEachBatch", "bN2BlockOfEachBatch",
                 "bN2LastBlockCostOfEachBatch", "totalBlockNum", "totalCost", "maxS1GCost")

    def __init__(self, batch_size: int) -> None:
        self.bN2CostOfEachBatch = [0] * batch_size
        self.bN2BlockOfEachBatch = [0] * batch_size
        self.bN2LastBlockCostOfEachBatch = [0] * batch_size
        self.totalBlockNum = 0
        self.totalCost = 0
        self.maxS1GCost = 0


class _BatchCache:
    """aicpu.h L160-167."""

    __slots__ = ("bIdx", "s1Size", "s2Size", "preTokenLeftUp", "nextTokenLeftUp", "typeCost")

    def __init__(self) -> None:
        self.bIdx = 0
        self.s1Size = 0
        self.s2Size = 0
        self.preTokenLeftUp = 0
        self.nextTokenLeftUp = 0
        # BlockCost<int64_t> typeCost[BLOCK_MAX_TYPE][BLOCK_MAX_TYPE] (aicpu.h L54)
        self.typeCost = [[0, 0], [0, 0]]


class _S1GCache:
    """aicpu.h L170-179."""

    __slots__ = ("bIdx", "s1GIdx", "s2Start", "s2End", "s1GCost", "s1GLastBlockCost",
                 "s1GBlock", "s1GNormalBlockCost")

    def __init__(self) -> None:
        self.bIdx = 0
        self.s1GIdx = 0
        self.s2Start = 0
        self.s2End = 0
        self.s1GCost = 0
        self.s1GLastBlockCost = 0
        self.s1GBlock = 0
        self.s1GNormalBlockCost = 0


class _CoreCache:
    """aicpu.h L182-186."""

    __slots__ = ("costLimit", "cost", "block")

    def __init__(self) -> None:
        self.costLimit = 0
        self.cost = 0
        self.block = 0


class _AssignContext:
    """aicpu.h L189-205."""

    __slots__ = ("curBIdx", "curBN2Idx", "curS1GIdx", "curS2Idx", "curCoreIdx",
                 "unassignedCost", "curKvSplitPart", "preFdDataNum", "bN2Cost", "bN2Block",
                 "isFinished", "batchCache", "s1GCache", "coreCache")

    def __init__(self) -> None:
        self.curBIdx = 0
        self.curBN2Idx = 0
        self.curS1GIdx = 0
        self.curS2Idx = 0
        self.curCoreIdx = 0
        self.unassignedCost = 0
        self.curKvSplitPart = 1
        self.preFdDataNum = 0
        self.bN2Cost = 0
        self.bN2Block = 0
        self.isFinished = False
        self.batchCache = _BatchCache()
        self.s1GCache = _S1GCache()
        self.coreCache = _CoreCache()


class _FlashDecodeResult:
    """aicpu.h L75-97 (the FD half of ``SplitResult``)."""

    __slots__ = ("fdUsedVecNum", "fdBN2Idx", "fdMIdx", "fdWorkspaceIdx", "fdS2SplitNum",
                 "fdMSize", "fdIdx", "fdMStart", "fdMNum")

    def __init__(self, aic_num: int, aiv_num: int) -> None:
        self.fdUsedVecNum = 0
        self.fdBN2Idx = [0] * aic_num
        self.fdMIdx = [0] * aic_num
        self.fdWorkspaceIdx = [0] * aic_num
        self.fdS2SplitNum = [0] * aic_num
        self.fdMSize = [0] * aic_num
        self.fdIdx = [0] * aiv_num
        self.fdMStart = [0] * aiv_num
        self.fdMNum = [0] * aiv_num


class _SplitResult:
    """aicpu.h L100-117."""

    __slots__ = ("usedCoreNum", "bN2End", "gS1End", "s2End", "firstFdDataWorkspaceIdx",
                 "maxCost", "numOfFdHead", "maxS2SplitNum", "fdRes")

    def __init__(self, aic_num: int, aiv_num: int) -> None:
        self.usedCoreNum = 0
        self.bN2End = [0] * aic_num
        self.gS1End = [0] * aic_num
        self.s2End = [0] * aic_num
        self.firstFdDataWorkspaceIdx = [0] * aic_num
        self.maxCost = 0
        self.numOfFdHead = 0
        self.maxS2SplitNum = 0
        self.fdRes = _FlashDecodeResult(aic_num, aiv_num)


class _Scheduler:
    """Transliteration of ``VllmQuantLightningIndexerMetadataCpuKernel`` (aicpu.cpp L206-901).

    Every method keeps the name, the argument order and the control flow of its C++
    counterpart so the two can be diffed method by method.
    """

    def __init__(self, *, actual_seq_lengths_query, actual_seq_lengths_key, batch_size,
                 max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, sparse_mode,
                 pre_tokens, next_tokens, cmp_ratio, layout_query, layout_key, soc_version,
                 aic_core_num, aiv_core_num) -> None:
        self.soc_version = soc_version
        self.valid_910b = "Ascend950" not in soc_version         # ProcessSocVersion L230-240
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.sparse_mode = sparse_mode
        self.cmp_ratio = cmp_ratio
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.aic_core_num = aic_core_num
        self.aiv_core_num = aiv_core_num
        self.m_base_size = _M_BASE_SIZE

        self.s1_ptr = (None if actual_seq_lengths_query is None
                       else _read_int32_tensor(actual_seq_lengths_query,
                                               "actual_seq_lengths_query"))
        self.s2_ptr = (None if actual_seq_lengths_key is None
                       else _read_int32_tensor(actual_seq_lengths_key,
                                               "actual_seq_lengths_key"))

        # GetQueryBatchSize L172-180 / GetKvBatchSize L182-190
        self.q_batch_size = _dim0(actual_seq_lengths_query, batch_size)
        self.kv_batch_size = _dim0(actual_seq_lengths_key, batch_size)

        # ParamsInit L242-267
        self.is_act_q_batch_plus = self.q_batch_size > self.kv_batch_size
        if sparse_mode == _SPARSE_MODE_RIGHT_DOWN_CAUSAL:
            self.attention_mode = 1
            self.pre_token = _INT64_MAX
        elif sparse_mode == _SPARSE_MODE_DEFAULT_MASK:
            self.attention_mode = 0
            self.pre_token = pre_tokens
        elif sparse_mode == _SPARSE_MODE_BAND:              # unreachable (CheckSingleParam)
            self.attention_mode = 1
            self.pre_token = pre_tokens
        else:
            self.attention_mode = 0
            self.pre_token = pre_tokens
        self.next_token = next_tokens
        self.group_size = self.num_heads_q // self.num_heads_k
        self.batch_size = min(self.q_batch_size, self.kv_batch_size)
        self.s2_base_size = _S2_BASE_SIZE_910B if self.valid_910b else _S2_BASE_SIZE_950

    # ------------------------------------------------------------------ length inputs
    def get_s1_seq_size(self, b_idx: int) -> int:
        """aicpu.cpp L269-284."""
        if self.s1_ptr is None:
            return self.max_seqlen_q
        if self.layout_query == "TND":
            if self.is_act_q_batch_plus:
                return self.s1_ptr[b_idx + 1] - self.s1_ptr[b_idx]
            if b_idx == 0:
                return self.s1_ptr[b_idx]
            return self.s1_ptr[b_idx] - self.s1_ptr[b_idx - 1]
        return self.s1_ptr[b_idx]

    def get_s2_seq_size(self, b_idx: int) -> int:
        """aicpu.cpp L286-298."""
        if self.s2_ptr is None:
            return self.max_seqlen_k * self.cmp_ratio
        if self.layout_key == "TND":
            if b_idx == 0:
                return self.s2_ptr[b_idx]
            return self.s2_ptr[b_idx] - self.s2_ptr[b_idx - 1]
        return self.s2_ptr[b_idx]

    # ------------------------------------------------------------------ cost model
    def calc_cost(self, basic_m: int, basic_s2: int) -> int:
        """aicpu.cpp L346-365: 910B branch L351-357, Ascend950 branch L358-364."""
        if self.valid_910b:
            align_basic_m = basic_m // self.group_size
            align_basic_s2 = (1536 + basic_s2) // 2 if basic_s2 <= 1024 \
                else (2560 + basic_s2) // 2
            return align_basic_m * align_basic_s2
        align_basic_m = (basic_m + 15) >> 4
        align_basic_s2 = (basic_s2 + 63) >> 6
        return 6 * align_basic_m + 10 * align_basic_s2

    def calc_cost_table(self, s1_normal: int, s2_normal: int, s1g_tail: int, s2_tail: int):
        """aicpu.cpp L367-376."""
        type_cost = [[0, 0], [0, 0]]
        type_cost[_NORMAL_BLOCK][_NORMAL_BLOCK] = self.calc_cost(s1_normal, s2_normal)
        type_cost[_TAIL_BLOCK][_NORMAL_BLOCK] = (
            0 if s1g_tail == 0 else self.calc_cost(s1g_tail, s2_normal))
        type_cost[_NORMAL_BLOCK][_TAIL_BLOCK] = (
            0 if s2_tail == 0 else self.calc_cost(s1_normal, s2_tail))
        type_cost[_TAIL_BLOCK][_TAIL_BLOCK] = (
            0 if (s1g_tail == 0 or s2_tail == 0) else self.calc_cost(s1g_tail, s2_tail))
        return type_cost

    # ------------------------------------------------------------------ mask window
    def calc_pre_token_left_up(self, s1_size: int, s2_size: int) -> int:
        """aicpu.cpp L318-326 (BAND is unreachable, see CheckSingleParam L128-132)."""
        if self.sparse_mode == _SPARSE_MODE_BAND:
            return s1_size - s2_size + self.pre_token
        return self.pre_token

    def calc_next_token_left_up(self, s1_size: int, s2_size: int) -> int:
        """aicpu.cpp L328-344."""
        if self.sparse_mode in (_SPARSE_MODE_DEFAULT_MASK, _SPARSE_MODE_ALL_MASK,
                                _SPARSE_MODE_LEFT_UP_CAUSAL):
            return self.next_token
        if self.sparse_mode == _SPARSE_MODE_RIGHT_DOWN_CAUSAL:
            return s2_size - s1_size
        if self.sparse_mode == _SPARSE_MODE_BAND:
            return s2_size - s1_size + self.next_token
        return self.next_token

    def calc_s2_token_range(self, s1g_idx: int, batch_cache: _BatchCache):
        """aicpu.cpp L378-410."""
        if not self.attention_mode:
            return 0, batch_cache.s2Size
        s1g_first_token = s1g_idx * self.m_base_size
        s1g_last_token = min(s1g_first_token + self.m_base_size,
                             batch_cache.s1Size * self.group_size) - 1
        if _IS_S1G:
            s1_first_token = _c_div(s1g_first_token, self.group_size)
            s1_last_token = _c_div(s1g_last_token, self.group_size)
        else:                                       # pragma: no cover - isS1G_ is fixed true
            if _c_div(s1g_first_token, batch_cache.s1Size) == \
                    _c_div(s1g_last_token, batch_cache.s1Size):
                s1_first_token = s1g_first_token % batch_cache.s1Size
                s1_last_token = s1g_last_token % batch_cache.s1Size
            else:
                s1_first_token = 0
                s1_last_token = batch_cache.s1Size
        s2_first_token = s1_first_token - batch_cache.preTokenLeftUp
        s2_last_token = s1_last_token + batch_cache.nextTokenLeftUp
        return s2_first_token, s2_last_token

    # ------------------------------------------------------------------ caches
    def calc_batch_cache(self, b_idx: int, split_info: _SplitInfo,
                         batch_cache: _BatchCache) -> None:
        """aicpu.cpp L412-424."""
        batch_cache.bIdx = b_idx
        batch_cache.s1Size = self.get_s1_seq_size(b_idx)
        batch_cache.s2Size = self.get_s2_seq_size(b_idx)
        batch_cache.preTokenLeftUp = self.calc_pre_token_left_up(batch_cache.s1Size,
                                                                 batch_cache.s2Size)
        batch_cache.nextTokenLeftUp = self.calc_next_token_left_up(batch_cache.s1Size,
                                                                   batch_cache.s2Size)
        batch_cache.typeCost = self.calc_cost_table(self.m_base_size, self.s2_base_size,
                                                    split_info.s1GTailSize[b_idx],
                                                    split_info.s2TailSize[b_idx])

    def calc_s1g_cache(self, s1g_idx: int, split_info: _SplitInfo, batch_cache: _BatchCache,
                       s1g_cache: _S1GCache) -> None:
        """aicpu.cpp L426-491."""
        s1g_cache.bIdx = batch_cache.bIdx
        s1g_cache.s1GIdx = s1g_idx

        if split_info.s1GBaseNum[batch_cache.bIdx] == 0 or \
                split_info.s2BaseNum[batch_cache.bIdx] == 0:
            s1g_cache.s1GBlock = 0
            s1g_cache.s1GCost = 0
            s1g_cache.s1GLastBlockCost = 0
            s1g_cache.s1GNormalBlockCost = 0
            return

        s2_first_token, s2_last_token = self.calc_s2_token_range(s1g_idx, batch_cache)
        s2_first_token = _clip(s2_first_token, 0, batch_cache.s2Size - 1)   # L446
        s2_last_token = _clip(s2_last_token, 0, batch_cache.s2Size - 1)     # L447

        s2_cmp_length = (s2_last_token - s2_first_token + 1) // self.cmp_ratio
        if s2_cmp_length == 0:
            s1g_cache.s2Start = 0
            s1g_cache.s2End = 0
        else:
            s1g_cache.s2Start = 0
            s1g_cache.s2End = (s2_cmp_length + self.s2_base_size - 1) // self.s2_base_size

        if s1g_cache.s2Start >= s1g_cache.s2End:
            s1g_cache.s1GBlock = 0
            s1g_cache.s1GCost = 0
            s1g_cache.s1GLastBlockCost = 0
            s1g_cache.s1GNormalBlockCost = 0
            return

        s2_tail_size = s2_cmp_length % self.s2_base_size

        s1g_cache.s1GBlock = s1g_cache.s2End - s1g_cache.s2Start
        cur_tail_s2_num = 1 if s2_tail_size != 0 else 0
        cur_normal_s2_num = s1g_cache.s1GBlock - cur_tail_s2_num

        type_cost = self.calc_cost_table(self.m_base_size, self.s2_base_size,
                                         split_info.s1GTailSize[batch_cache.bIdx], s2_tail_size)

        if s1g_idx == (split_info.s1GBaseNum[batch_cache.bIdx] - 1) and \
                split_info.s1GTailSize[batch_cache.bIdx] != 0:
            s1g_cache.s1GCost = (type_cost[_TAIL_BLOCK][_NORMAL_BLOCK] * cur_normal_s2_num +
                                 type_cost[_TAIL_BLOCK][_TAIL_BLOCK] * cur_tail_s2_num)
            s1g_cache.s1GLastBlockCost = (type_cost[_TAIL_BLOCK][_TAIL_BLOCK]
                                          if cur_tail_s2_num > 0
                                          else type_cost[_TAIL_BLOCK][_NORMAL_BLOCK])
            s1g_cache.s1GNormalBlockCost = type_cost[_TAIL_BLOCK][_NORMAL_BLOCK]
        else:
            s1g_cache.s1GCost = (type_cost[_NORMAL_BLOCK][_NORMAL_BLOCK] * cur_normal_s2_num +
                                 type_cost[_NORMAL_BLOCK][_TAIL_BLOCK] * cur_tail_s2_num)
            s1g_cache.s1GLastBlockCost = (type_cost[_NORMAL_BLOCK][_TAIL_BLOCK]
                                          if cur_tail_s2_num > 0
                                          else type_cost[_NORMAL_BLOCK][_NORMAL_BLOCK])
            s1g_cache.s1GNormalBlockCost = type_cost[_NORMAL_BLOCK][_NORMAL_BLOCK]

    # ------------------------------------------------------------------ preprocessing
    def calc_split_info(self, split_info: _SplitInfo) -> None:
        """aicpu.cpp L300-316."""
        for b_idx in range(self.batch_size):
            s1_size = self.get_s1_seq_size(b_idx)
            s2_size = self.get_s2_seq_size(b_idx) // self.cmp_ratio
            split_info.s1GBaseNum[b_idx] = (
                s1_size * self.group_size + (self.m_base_size - 1)) // self.m_base_size
            split_info.s1GTailSize[b_idx] = (s1_size * self.group_size) % self.m_base_size
            split_info.s2BaseNum[b_idx] = (
                s2_size + self.s2_base_size - 1) // self.s2_base_size
            split_info.s2TailSize[b_idx] = s2_size % self.s2_base_size
            if split_info.s1GBaseNum[b_idx] != 0 and split_info.s2BaseNum[b_idx] != 0:
                split_info.isKvSeqAllZero = False

    def calc_batch_cost(self, b_idx: int, split_info: _SplitInfo,
                        cost_info: _CostInfo) -> None:
        """aicpu.cpp L493-522."""
        cost_info.bN2CostOfEachBatch[b_idx] = 0
        cost_info.bN2BlockOfEachBatch[b_idx] = 0
        cost_info.bN2LastBlockCostOfEachBatch[b_idx] = 0

        if self.get_s1_seq_size(b_idx) == 0 or self.get_s2_seq_size(b_idx) == 0:
            return

        batch_cache = _BatchCache()
        s1g_cache = _S1GCache()
        self.calc_batch_cache(b_idx, split_info, batch_cache)
        for s1g_idx in range(split_info.s1GBaseNum[b_idx]):
            self.calc_s1g_cache(s1g_idx, split_info, batch_cache, s1g_cache)
            cost_info.bN2CostOfEachBatch[b_idx] += s1g_cache.s1GCost
            cost_info.bN2BlockOfEachBatch[b_idx] += s1g_cache.s1GBlock
            if s1g_cache.s1GCost > cost_info.maxS1GCost:
                cost_info.maxS1GCost = s1g_cache.s1GCost
            if s1g_cache.s1GBlock > 0:
                cost_info.bN2LastBlockCostOfEachBatch[b_idx] = s1g_cache.s1GLastBlockCost

    def calc_cost_info(self, split_info: _SplitInfo, cost_info: _CostInfo) -> None:
        """aicpu.cpp L524-541."""
        if split_info.isKvSeqAllZero:
            cost_info.totalCost = 0
            cost_info.totalBlockNum = 0
            return
        for b_idx in range(self.batch_size):
            self.calc_batch_cost(b_idx, split_info, cost_info)
            cost_info.totalCost += cost_info.bN2CostOfEachBatch[b_idx] * self.num_heads_k
            cost_info.totalBlockNum += (cost_info.bN2BlockOfEachBatch[b_idx] *
                                        self.num_heads_k)

    # ------------------------------------------------------------------ assignment
    def update_cursor(self, split_info: _SplitInfo, cost_info: _CostInfo,
                      ac: _AssignContext) -> None:
        """aicpu.cpp L543-589 -- unreachable while ``_SUPPORT_FD`` is False."""
        update_s1g = False
        update_batch = False

        if ac.curS2Idx >= ac.s1GCache.s2End:        # the interval is right-open (L552)
            ac.curS2Idx = 0
            ac.curS1GIdx += 1
            update_s1g = True

        if ac.curS1GIdx >= split_info.s1GBaseNum[ac.curBIdx]:
            ac.curS1GIdx = 0
            ac.curBN2Idx += 1

        if ac.curBN2Idx == self.batch_size * self.num_heads_k:
            ac.curS1GIdx = 0
            ac.curS2Idx = 0
            ac.isFinished = True
            return

        if ac.curBN2Idx // self.num_heads_k != ac.curBIdx:
            ac.curBIdx = ac.curBN2Idx // self.num_heads_k
            ac.curS1GIdx = 0
            update_batch = True
            update_s1g = True

        if update_batch:
            self.calc_batch_cache(ac.curBIdx, split_info, ac.batchCache)
            ac.bN2Cost = cost_info.bN2CostOfEachBatch[ac.curBIdx]
            ac.bN2Block = cost_info.bN2BlockOfEachBatch[ac.curBIdx]
        if update_s1g:
            self.calc_s1g_cache(ac.curS1GIdx, split_info, ac.batchCache, ac.s1GCache)
            ac.curS2Idx = ac.s1GCache.s2Start

    def assign_by_batch(self, split_info: _SplitInfo, cost_info: _CostInfo,
                        ac: _AssignContext) -> None:
        """aicpu.cpp L591-624 (always assigns at least one batch: ``bN2Cost == 0``)."""
        if ac.isFinished:
            return
        while (ac.bN2Cost == 0 or
               _within_tolerance(ac.coreCache.costLimit,
                                 cost_info.bN2LastBlockCostOfEachBatch[ac.curBIdx] //
                                 _FA_TOLERANCE_RATIO,
                                 ac.coreCache.cost + ac.bN2Cost)):
            ac.coreCache.cost += ac.bN2Cost
            ac.coreCache.block += ac.bN2Block
            ac.curBN2Idx += 1

            if ac.curBN2Idx == self.batch_size * self.num_heads_k:   # to the end
                ac.curS1GIdx = 0
                ac.curS2Idx = 0
                ac.isFinished = True
                return

            if ac.curBN2Idx // self.num_heads_k != ac.curBIdx:       # next batch
                ac.curBIdx = ac.curBN2Idx // self.num_heads_k
                self.calc_batch_cache(ac.curBIdx, split_info, ac.batchCache)

            ac.bN2Cost = cost_info.bN2CostOfEachBatch[ac.curBIdx]
            ac.bN2Block = cost_info.bN2BlockOfEachBatch[ac.curBIdx]
            ac.curS1GIdx = 0
            self.calc_s1g_cache(ac.curS1GIdx, split_info, ac.batchCache, ac.s1GCache)
            ac.curS2Idx = ac.s1GCache.s2Start

    def assign_by_row(self, split_info: _SplitInfo, ac: _AssignContext) -> None:
        """aicpu.cpp L626-650 (skips empty rows; see the guard below)."""
        if ac.isFinished:
            return

        while _within_tolerance(ac.coreCache.costLimit,
                                ac.s1GCache.s1GLastBlockCost // _FA_TOLERANCE_RATIO,
                                ac.coreCache.cost + ac.s1GCache.s1GCost):
            ac.coreCache.cost += ac.s1GCache.s1GCost
            ac.coreCache.block += ac.s1GCache.s1GBlock

            # the current batch just handed a whole row over: update what is left of it
            ac.bN2Cost = (ac.bN2Cost - ac.s1GCache.s1GCost
                          if ac.bN2Cost > ac.s1GCache.s1GCost else 0)
            ac.bN2Block = (ac.bN2Block - ac.s1GCache.s1GBlock
                           if ac.bN2Block > ac.s1GCache.s1GBlock else 0)

            # do { curS1GIdx++; CalcS1GCache(...) } while (s1GBlock == 0) -- L644-647.
            # The loop is expected to stop at the first row with blocks; with
            # sparse_mode == 3 every row at or past s1GBaseNum[b] is non-empty, so the guard
            # below can only fire if that invariant is broken (the C++ would spin forever).
            row_limit = split_info.s1GBaseNum[ac.batchCache.bIdx] + 4
            while True:
                ac.curS1GIdx += 1
                self.calc_s1g_cache(ac.curS1GIdx, split_info, ac.batchCache, ac.s1GCache)
                if ac.s1GCache.s1GBlock != 0:
                    break
                if ac.curS1GIdx > row_limit:
                    raise RuntimeError(
                        "no block-bearing S1G row after "
                        f"batch {ac.batchCache.bIdx}, row {ac.curS1GIdx}")
            ac.curS2Idx = ac.s1GCache.s2Start

    def assign_by_block(self, ac: _AssignContext) -> None:
        """aicpu.cpp L652-675 -- unreachable while ``_SUPPORT_FD`` is False."""
        if ac.isFinished:
            return

        cur_cost = ac.s1GCache.s1GNormalBlockCost
        if ac.curS2Idx == (ac.s1GCache.s2End - 1):
            cur_cost = ac.s1GCache.s1GLastBlockCost

        while _within_tolerance(ac.coreCache.costLimit, cur_cost // _FA_TOLERANCE_RATIO,
                                ac.coreCache.cost + cur_cost):
            ac.coreCache.cost += cur_cost
            ac.coreCache.block += 1
            ac.curS2Idx += 1
            ac.bN2Cost = ac.bN2Cost - cur_cost
            ac.s1GCache.s1GCost = ac.s1GCache.s1GCost - cur_cost
            ac.bN2Block -= 1
            ac.s1GCache.s1GBlock -= 1

    def force_assign(self, split_info: _SplitInfo, cost_info: _CostInfo,
                     ac: _AssignContext) -> None:
        """aicpu.cpp L677-698 -- unreachable while ``_SUPPORT_FD`` is False."""
        if ac.isFinished:
            return

        cur_cost = ac.s1GCache.s1GNormalBlockCost
        if ac.curS2Idx == (ac.s1GCache.s2End - 1):
            cur_cost = ac.s1GCache.s1GLastBlockCost

        ac.coreCache.cost += cur_cost
        ac.coreCache.block += 1
        ac.curS2Idx += 1
        ac.bN2Cost = ac.bN2Cost - cur_cost
        ac.bN2Block -= 1
        ac.s1GCache.s1GCost = ac.s1GCache.s1GCost - cur_cost
        ac.s1GCache.s1GBlock -= 1
        self.update_cursor(split_info, cost_info, ac)

    def is_need_record_fd_info(self, ac: _AssignContext, res: _SplitResult) -> bool:
        """aicpu.cpp L700-717 -- unreachable while ``_SUPPORT_FD`` is False."""
        if ac.curCoreIdx == 0:
            return False
        if ac.curKvSplitPart <= 1:
            return False
        if (ac.curBN2Idx == res.bN2End[ac.curCoreIdx - 1] and
                ac.curS1GIdx == res.gS1End[ac.curCoreIdx - 1]):
            return False
        return True

    def record_fd_info(self, split_info: _SplitInfo, ac: _AssignContext,
                       res: _SplitResult) -> None:
        """aicpu.cpp L719-739 -- unreachable while ``_SUPPORT_FD`` is False."""
        split_b_idx = res.bN2End[ac.curCoreIdx - 1] // self.num_heads_k
        split_s1g_idx = res.gS1End[ac.curCoreIdx - 1]
        s1_size = self.get_s1_seq_size(split_b_idx)
        cur_fd_s1g_size = (
            s1_size * self.group_size - split_s1g_idx * self.m_base_size
            if split_s1g_idx == split_info.s1GBaseNum[split_b_idx] - 1 else self.m_base_size)
        res.maxS2SplitNum = max(res.maxS2SplitNum, ac.curKvSplitPart)
        res.fdRes.fdBN2Idx[res.numOfFdHead] = res.bN2End[ac.curCoreIdx - 1]
        res.fdRes.fdMIdx[res.numOfFdHead] = res.gS1End[ac.curCoreIdx - 1]
        res.fdRes.fdWorkspaceIdx[res.numOfFdHead] = ac.preFdDataNum
        res.fdRes.fdS2SplitNum[res.numOfFdHead] = ac.curKvSplitPart
        res.fdRes.fdMSize[res.numOfFdHead] = cur_fd_s1g_size // self.group_size
        res.numOfFdHead += 1

    def assign_block_to_core(self, core_num: int, split_info: _SplitInfo, cost_info: _CostInfo,
                             ac: _AssignContext, res: _SplitResult) -> None:
        """aicpu.cpp L741-779."""
        res.firstFdDataWorkspaceIdx[ac.curCoreIdx] = (ac.preFdDataNum + ac.curKvSplitPart - 1)
        ac.coreCache = _CoreCache()
        ac.coreCache.costLimit = ac.unassignedCost // (core_num - ac.curCoreIdx)
        if not _SUPPORT_FD and cost_info.maxS1GCost > ac.coreCache.costLimit:
            ac.coreCache.costLimit = cost_info.maxS1GCost

        self.assign_by_batch(split_info, cost_info, ac)         # 1. whole batches
        self.assign_by_row(split_info, ac)                      # 2. whole rows
        if _SUPPORT_FD:                                         # pragma: no cover - dead
            self.assign_by_block(ac)                            # 3. blocks
            if ac.coreCache.block == 0:                         # 4. force at least one block
                self.force_assign(split_info, cost_info, ac)

        res.bN2End[ac.curCoreIdx] = ac.curBN2Idx
        res.gS1End[ac.curCoreIdx] = ac.curS1GIdx
        res.s2End[ac.curCoreIdx] = ac.curS2Idx
        res.maxCost = max(res.maxCost, ac.coreCache.cost)
        ac.unassignedCost -= ac.coreCache.cost

        if _SUPPORT_FD:                                         # pragma: no cover - dead
            if self.is_need_record_fd_info(ac, res):
                self.record_fd_info(split_info, ac, res)
                ac.preFdDataNum += ac.curKvSplitPart
                ac.curKvSplitPart = 1
            if ac.curS2Idx > ac.s1GCache.s2Start and ac.curS2Idx <= ac.s1GCache.s2End:
                ac.curKvSplitPart += 1

    def calc_split_plan(self, core_num: int, cost_limit: int, split_info: _SplitInfo,
                        cost_info: _CostInfo, res: _SplitResult) -> None:
        """aicpu.cpp L781-810."""
        if core_num == 0:
            return
        res.maxCost = 0
        res.usedCoreNum = 0
        ac = _AssignContext()
        ac.curBIdx = 0
        ac.curS1GIdx = 0
        ac.unassignedCost = cost_info.totalCost
        ac.bN2Cost = cost_info.bN2CostOfEachBatch[ac.curBIdx]
        ac.bN2Block = cost_info.bN2BlockOfEachBatch[ac.curBIdx]
        self.calc_batch_cache(ac.curBIdx, split_info, ac.batchCache)
        self.calc_s1g_cache(ac.curS1GIdx, split_info, ac.batchCache, ac.s1GCache)
        ac.curS2Idx = ac.s1GCache.s2Start
        for i in range(core_num):
            if res.maxCost > cost_limit:                        # L800-802
                return
            if ac.isFinished or ac.unassignedCost <= 0:         # L803-805
                break
            ac.curCoreIdx = i
            self.assign_block_to_core(core_num, split_info, cost_info, ac, res)
        res.usedCoreNum = ac.curCoreIdx + 1

    def split_fd(self, res: _SplitResult) -> None:
        """aicpu.cpp L812-835 -- unreachable while ``_SUPPORT_FD`` is False."""
        total_fd_load = 0
        for i in range(res.numOfFdHead):
            total_fd_load += res.fdRes.fdS2SplitNum[i] * res.fdRes.fdMSize[i]
        average_load = (total_fd_load + self.aiv_core_num - 1) // self.aiv_core_num
        cur_core_index = 0
        for i in range(res.numOfFdHead):
            cur_fd_vector_num = (res.fdRes.fdS2SplitNum[i] * res.fdRes.fdMSize[i] //
                                 average_load)
            cur_ave_m_size = ((res.fdRes.fdMSize[i] + cur_fd_vector_num - 1) //
                              cur_fd_vector_num)
            cur_fd_vector_num = ((res.fdRes.fdMSize[i] + cur_ave_m_size - 1) //
                                 cur_ave_m_size)
            for vid in range(cur_fd_vector_num):
                res.fdRes.fdIdx[cur_core_index] = i
                res.fdRes.fdMStart[cur_core_index] = vid * cur_ave_m_size
                res.fdRes.fdMNum[cur_core_index] = (
                    cur_ave_m_size if vid < cur_fd_vector_num - 1
                    else res.fdRes.fdMSize[i] - vid * cur_ave_m_size)
                cur_core_index += 1
        res.fdRes.fdUsedVecNum = cur_core_index

    # ------------------------------------------------------------------ main
    def balance_schedule(self, res: _SplitResult) -> None:
        """aicpu.cpp L837-861."""
        split_info = _SplitInfo(self.batch_size)
        cost_info = _CostInfo(self.batch_size)
        self.calc_split_info(split_info)                        # 1. basic blocks
        if split_info.isKvSeqAllZero:                           # all-empty case L843-849
            res.usedCoreNum = 1
            res.bN2End[0] = self.batch_size * self.num_heads_k
            res.gS1End[0] = 0
            res.s2End[0] = 0
            return
        self.calc_cost_info(split_info, cost_info)

        res.maxCost = _INT64_MAX
        res.usedCoreNum = 1
        self.calc_split_plan(self.aic_core_num, res.maxCost, split_info, cost_info, res)
        if _SUPPORT_FD and res.numOfFdHead > 0:                 # pragma: no cover - dead
            self.split_fd(res)                                  # 3. balance the FD tasks
        res.usedCoreNum = max(res.usedCoreNum, 1)               # at least one core

    def gen_metadata(self, res: _SplitResult) -> list:
        """aicpu.cpp L863-901 -> the flat 1024-element ``QliMetaData`` buffer.

        The native kernel leaves every element it does not write at whatever ``at::empty``
        returned; this port fills those with 0 (see the module docstring).
        """
        table = [0] * _QLI_META_SIZE

        # LI Metadata (L867-883)
        for i in range(self.aic_core_num):
            base = _LI_METADATA_SIZE * i
            if i >= res.usedCoreNum:
                table[base + _LI_CORE_ENABLE] = 0               # AIC disabled
                continue
            table[base + _LI_CORE_ENABLE] = 1                   # AIC enabled
            # FA start
            table[base + _LI_BN2_START] = 0 if i == 0 else res.bN2End[i - 1]
            table[base + _LI_M_START] = 0 if i == 0 else res.gS1End[i - 1]
            table[base + _LI_S2_START] = 0 if i == 0 else res.s2End[i - 1]
            # FA end
            table[base + _LI_BN2_END] = res.bN2End[i]
            table[base + _LI_M_END] = res.gS1End[i]
            table[base + _LI_S2_END] = res.s2End[i]
            table[base + _LI_FIRST_LD_DATA_WORKSPACE_IDX] = res.firstFdDataWorkspaceIdx[i]

        # LD Metadata (L886-899)
        for i in range(self.aiv_core_num):
            base = _LI_BLOCK + _LD_METADATA_SIZE * i
            if i >= res.fdRes.fdUsedVecNum:
                table[base + _LD_CORE_ENABLE] = 0               # AIV disabled
                continue
            table[base + _LD_CORE_ENABLE] = 1                   # AIV enabled
            cur_fd_idx = res.fdRes.fdIdx[i]
            table[base + _LD_BN2_IDX] = res.fdRes.fdBN2Idx[cur_fd_idx]
            table[base + _LD_M_IDX] = res.fdRes.fdMIdx[cur_fd_idx]
            table[base + _LD_WORKSPACE_IDX] = res.fdRes.fdWorkspaceIdx[cur_fd_idx]
            table[base + _LD_WORKSPACE_NUM] = res.fdRes.fdS2SplitNum[cur_fd_idx]
            table[base + _LD_M_START] = res.fdRes.fdMStart[i]
            table[base + _LD_M_NUM] = res.fdRes.fdMNum[i]
        return table


def _resolve_core_nums(device: torch.device, aic_core_num, aiv_core_num):
    """``PlatformInfo::GetCubeCoreNum()`` / ``GetVectorCoreNum()`` (aclnn op_api L96-97)."""
    if aic_core_num is not None and aiv_core_num is not None:
        return int(aic_core_num), int(aiv_core_num)
    detected_aic, detected_aiv = _DEFAULT_AIC_CORE_NUM, _DEFAULT_AIV_CORE_NUM
    if device.type == "npu":
        try:
            import torch_npu  # noqa: F401

            props = torch.npu.get_device_properties(device)
            cube = int(getattr(props, "cube_core_num", 0) or 0)
            vector = int(getattr(props, "vector_core_num", 0) or 0)
            if cube > 0 and vector > 0 and vector % cube == 0:
                detected_aic, detected_aiv = cube, vector
        except Exception:  # pragma: no cover - torch_npu not installed
            pass
    if aic_core_num is None:
        aic_core_num = detected_aic
    if aiv_core_num is None:
        aiv_core_num = detected_aiv
    return int(aic_core_num), int(aiv_core_num)


def _resolve_soc_version(device: torch.device, soc_version) -> str:
    """``PlatformInfo::GetSocLongVersion()`` (aclnn op_api L98).

    ``ProcessSocVersion`` (aicpu.cpp L230-240) only tests whether the string contains
    ``Ascend950``; every other SoC takes the 910B branch.
    """
    if soc_version is not None:
        return str(soc_version)
    if device.type == "npu":
        try:
            import torch_npu  # noqa: F401

            try:
                return str(torch.npu.get_device_name(device))
            except Exception:
                return str(torch.npu.get_device_name(device.index or 0))
        except Exception:  # pragma: no cover - torch_npu not installed
            pass
    return ""


def npu_vllm_quant_lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    batch_size: int = 0,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = _INT64_MAX,
    next_tokens: int = _INT64_MAX,
    cmp_ratio: int = 1,
    device: str = "npu",
    aic_core_num: int | None = None,
    aiv_core_num: int | None = None,
    soc_version: str | None = None,
) -> torch.Tensor:
    """Schedule table for the quantized lightning indexer (AscendC AICPU -> Python).

    Drop-in replacement for ``torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata``:
    the parameter names, order and defaults match the registered schema
    (torch_binding.cpp L1245), so the operator can be taken over through the usual adapter.

    Args:
        num_heads_q / num_heads_k: query / key head counts; the native operator hard-requires
            64 and 1 (CheckSingleParam L100-108).
        head_dim / query_quant_mode / key_quant_mode: accepted for signature compatibility;
            the metadata kernel never reads them.
        actual_seq_lengths_query: ``int32 [B]``; with ``layout_query="TND"`` it is the
            *exclusive* prefix sum of the query lengths (``query_start_loc[1:]``), which
            ``GetS1SeqSize`` differentiates.  If omitted, ``max_seqlen_q`` is used for every
            batch.
        actual_seq_lengths_key: ``int32 [B]`` per-batch KV length (``layout_key`` =
            ``PA_BSND``/``BSND``) or its inclusive prefix sum (``TND``).
        batch_size: fallback batch size used when a length tensor is absent; the lengths
            actually used are ``min(queryBatch, kvBatch)`` (ParamsInit L257).
        max_seqlen_q / max_seqlen_k: fallbacks for absent length tensors; ``max_seqlen_k`` is
            scaled by ``cmp_ratio`` (L289).
        layout_query: ``"TND"`` or ``"BSND"``; layout_key: ``"PA_BSND"``, ``"TND"`` or
            ``"BSND"`` (CheckSingleParam L110-126).
        sparse_count: indexer top-k; unused by the metadata kernel.
        sparse_mode: 0 (no mask) or 3 (right-down causal, what every call site passes);
            anything else is rejected, like the native op (L128-132).
        pre_tokens / next_tokens: must be ``INT64_MAX`` (L134-142).
        cmp_ratio: 1..128 (L210); a power of two on the 910B branch, 1/4/128 on Ascend950
            (L215-226).
        device: device string used when no length tensor is given; otherwise the output
            follows the input tensors.
        aic_core_num / aiv_core_num: AIC/AIV core count overrides; ``None`` reads them from the
            device properties, i.e. from ``PlatformInfo`` exactly like the native operator.
        soc_version: SoC long version override used to select the cost-model branch; ``None``
            reads the device name.

    Returns:
        ``torch.Tensor`` of shape ``(1024,)``, dtype ``torch.int32`` -- ``QliMetaData``:
        ``LI[36][8]`` in ``[0, 288)`` and ``LD[72][8]`` in ``[288, 864)`` (QLI metadata header
        L70-73).  ``LI[core][0]`` is the enable flag, the next six values are the
        ``[start, end)`` cut points of ``bN2 / m / s2`` for that core and ``LI[core][7]`` is
        the first flash-decode workspace index.  Every AIV entry is 0 because the native
        operator never records flash-decode heads (``supportFd_`` is a fixed ``false``).
        Elements the native kernel leaves uninitialised read back as 0 here.
    """
    num_heads_q = int(num_heads_q)
    num_heads_k = int(num_heads_k)
    batch_size = int(batch_size)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    sparse_mode = int(sparse_mode)
    pre_tokens = int(pre_tokens)
    next_tokens = int(next_tokens)
    cmp_ratio = int(cmp_ratio)
    layout_query = str(layout_query)
    layout_key = str(layout_key)
    # int(head_dim) / int(query_quant_mode) / int(key_quant_mode) / int(sparse_count) are not
    # needed: the metadata kernel ignores them (see the module docstring).

    # ------------------------------------------------------------------ ParamsCheck
    # CheckSingleParam L72-144 / CheckExistence L146-170 / CheckFeature L207-228.  The native
    # kernel returns KERNEL_STATUS_PARAM_INVALID, which makes the aclnn call raise, hence
    # ValueError here.
    if num_heads_q != 64:
        raise ValueError(f"num_heads_q should only be 64, but got {num_heads_q}")
    if num_heads_k != 1:
        raise ValueError(f"num_heads_k should only be 1, but got {num_heads_k}")
    if batch_size < 0:
        raise ValueError(f"batch_size should not be negative, but got {batch_size}")
    if max_seqlen_q < 0:
        raise ValueError(f"max_seqlen_q should not be negative, but got {max_seqlen_q}")
    if max_seqlen_k < 0:
        raise ValueError(f"max_seqlen_k should not be negative, but got {max_seqlen_k}")
    if layout_query not in ("TND", "BSND"):
        raise ValueError(f"layout_query must be TND or BSND, but got {layout_query!r}")
    if layout_key not in ("PA_BSND", "TND", "BSND"):
        raise ValueError(f"layout_key must be PA_BSND/TND/BSND, but got {layout_key!r}")
    if layout_query == "TND" and layout_key == "BSND":
        raise ValueError("for layout_query TND, layout_key should be PA_BSND/TND")
    if layout_query == "BSND" and layout_key == "TND":
        raise ValueError("for layout_query BSND, layout_key should be PA_BSND/BSND")
    if sparse_mode not in (_SPARSE_MODE_DEFAULT_MASK, _SPARSE_MODE_RIGHT_DOWN_CAUSAL):
        raise ValueError(f"sparse_mode should be 0 or 3, but got {sparse_mode}")
    if pre_tokens != _INT64_MAX:
        raise ValueError(f"pre_tokens should only be 2^63-1, but got {pre_tokens}")
    if next_tokens != _INT64_MAX:
        raise ValueError(f"next_tokens should only be 2^63-1, but got {next_tokens}")
    # CheckExistence L146-170
    if layout_query == "TND" and actual_seq_lengths_query is None:
        raise ValueError("for layout_query TND, actual_seq_lengths_query must be provided")
    if layout_key in ("PA_BSND", "TND") and actual_seq_lengths_key is None:
        raise ValueError(
            f"for layout_key {layout_key}, actual_seq_lengths_key must be provided")
    if cmp_ratio < 1 or cmp_ratio > 128:
        raise ValueError(f"cmp_ratio should be in [1, 128], but got {cmp_ratio}")

    tensor_device = _resolve_device(actual_seq_lengths_query, actual_seq_lengths_key, device)
    soc = _resolve_soc_version(tensor_device, soc_version)
    if "Ascend950" not in soc:                      # ValidSocVersion::ASCEND910B
        if cmp_ratio & (cmp_ratio - 1):
            raise ValueError("for Atlas A3 cmp_ratio should be 1/2/4/8/16/32/64/128, "
                             f"but got {cmp_ratio}")
    elif cmp_ratio not in (1, 4, 128):              # ValidSocVersion::ASCEND950
        raise ValueError(f"for Ascend950 cmp_ratio should be 1/4/128, but got {cmp_ratio}")

    aic_num, aiv_num = _resolve_core_nums(tensor_device, aic_core_num, aiv_core_num)
    if aic_num == 0 or aiv_num == 0 or aiv_num % aic_num != 0:
        raise ValueError(f"core num invalid: aic={aic_num}, aiv={aiv_num}")
    if aic_num > _AIC_CORE_NUM or aiv_num > _AIV_CORE_NUM:
        raise ValueError(
            f"core nums exceed the QliMetaData tables: aic={aic_num} (max {_AIC_CORE_NUM}), "
            f"aiv={aiv_num} (max {_AIV_CORE_NUM})")

    scheduler = _Scheduler(
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        cmp_ratio=cmp_ratio,
        layout_query=layout_query,
        layout_key=layout_key,
        soc_version=soc,
        aic_core_num=aic_num,
        aiv_core_num=aiv_num,
    )

    # CheckConsistency L192-205
    if layout_query == "BSND" or (layout_query == "TND" and layout_key == "TND"):
        if scheduler.q_batch_size != scheduler.kv_batch_size:
            raise ValueError("with this layout pair actual_seq_lengths_query and "
                             "actual_seq_lengths_key must have the same length")
    if abs(scheduler.q_batch_size - scheduler.kv_batch_size) > 1:
        raise ValueError("the length difference between actual_seq_lengths_query and "
                         "actual_seq_lengths_key must not be greater than 1")

    split_res = _SplitResult(aic_num, aiv_num)
    scheduler.balance_schedule(split_res)
    table = scheduler.gen_metadata(split_res)
    return torch.tensor(table, dtype=torch.int32, device=tensor_device)
