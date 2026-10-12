"""lightning_indexer —— `npu_lightning_indexer` 的 Triton 等价实现

## 来源说明（重要，勿删）
`npu_lightning_indexer`（非量化版）在 **vllm-ascend 源码里没有 AscendC 实现**：
  - vllm-ascend 的 `csrc/attention/` 下没有任何 `OpDef` 注册名为 `LightningIndexer`
    （只有 `FusedLightningIndexerManage` / `FusedQuantLightningIndexerManage` /
      `VllmAscendQuantLightningIndexerV2` / `LightningIndexerQuant`）
  - `csrc/attention/lightning_indexer_v2/` 下**只有一个** 235 行的
    `op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h`，没有 op_host/def/tiling
  - 生产代码 `vllm_ascend/device/device_op.py` L591-592 明确写着：
        # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
        # So two branches are maintained temporarily.
        # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
  ⇒ 它由**外部 CANN 包**（或 torch_npu 内置）提供，不在本仓库源码范围内。

## 本实现的口径
按 Lead 决策，把 `lightning_indexer` 定义为 **量化版（`LightningIndexerQuant`，本仓库有源码）
在恒等 scale 下的特例**，并**复用已验证的量化内核**：

    query_dequant_scale ≡ 1.0 , key_dequant_scale ≡ 1.0
    ⇒ score[s1,n2,s2] = Σ_g relu( query[s1,n2*gs+g] · key[s2,n2] ) * weights[s1,n2*gs+g]

这与 `lightning_indexer_quant` 的公式（出处：`lightning_indexer_quant_service_cube.h`
L542-543 `reluPre=1` 随路做在 Cube 的 QK 上；`..._service_vector.h` L383 之后才乘 kScale）
在 scale=1 时**逐位相同**（fp16 里 `x * 1.0 == x` 精确）。

## 调用点风格（`vllm_ascend/device/device_op.py` L632-640）
    torch.ops._C_ascend.npu_lightning_indexer(
        query=q_li, key=kv_cache[idx], weights=weights,
        actual_seq_lengths_query=..., actual_seq_lengths_key=...,
        block_table=attn_metadata.block_table,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=2048, sparse_mode=3)
即「非量化版 = 量化版去掉 `query_dequant_scale`/`key_dequant_scale`/
`query_quant_mode`/`key_quant_mode`，其余完全相同」。

**索引类零容差**：sparse_indices 逐元素完全相等（不足 `sparse_count` 补 -1）。
**并列决胜**：AscendC 对相等 score 的先后未定义；本实现取**较小下标**（与量化版一致）。
"""

from __future__ import annotations

import torch

from .lightning_indexer_quant import lightning_indexer_quant

__all__ = ["lightning_indexer", "npu_lightning_indexer"]


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
) -> torch.Tensor:
    """等价于 `npu_lightning_indexer`（见模块 docstring 的来源说明与口径）。

    Args:
        query: int8  TND [T,N1,D] / BSND [B,S1,N1,D]
        key:   int8  PA_BSND [numBlocks, blockSize, N2, D] / BSND [B,S2,N2,D]
        weights: 逐 q head，[T,N1]（TND）或 [B,S1,N1]
        actual_seq_lengths_query/key, block_table, layout_*, sparse_count, sparse_mode:
            与量化版同名同义（TND 需 cu_seqlens；PA_BSND 需 block_table）
        query_quant_mode/key_quant_mode: 仅为签名兼容，非量化版无含义（只接受 0）

    Returns:
        sparse_indices int32，TND → [T,N2,K]；BSND → [B,S1,N2,K]
    """
    if int(query_quant_mode) != 0 or int(key_quant_mode) != 0:
        raise NotImplementedError(
            "非量化 lightning_indexer 无 quant_mode 语义（只接受 0）；"
            "需要量化请用 lightning_indexer_quant")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key 必须是 int8（与量化版一致）")

    tnd = layout_query == "TND"
    if tnd:
        T, N1, _ = query.shape
        rows_w = T
        B = None
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND 必须提供 actual_seq_lengths_query（cu_seqlens）")
        cu = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu.numel() >= 2:
            B = cu.numel() - 1
    else:
        B, S1, N1, _ = query.shape
        rows_w = B * S1
        T = rows_w

    # 恒等 scale：fp16 的 1.0（x * 1.0 在 fp16 里精确等于 x）
    qs = torch.ones(rows_w, N1, dtype=torch.float16, device=query.device)
    if layout_key in ("PA_BSND", "PA_BBND"):
        if block_table is None:
            raise ValueError("layout_key=PA_BSND 必须提供 block_table")
        num_blocks, block_size = key.shape[0], key.shape[1]
        ks = torch.ones(num_blocks, block_size, dtype=torch.float16, device=key.device)
    else:
        # BSND：量化版接受 [B,S2] 或 [S2]
        if B is None:
            raise ValueError("BSND key 需要能确定 batch 数")
        ks = torch.ones(B * key.shape[1], dtype=torch.float16, device=key.device).view(B, key.shape[1])

    return lightning_indexer_quant(
        query=query, key=key, weights=weights,
        query_dequant_scale=qs, key_dequant_scale=ks,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        query_quant_mode=0, key_quant_mode=0,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=sparse_count, sparse_mode=sparse_mode)


# 调用点名字别名（审计脚本按 `npu_<op>` → `<op>.py` 解析，两个名字都提供）
npu_lightning_indexer = lightning_indexer
