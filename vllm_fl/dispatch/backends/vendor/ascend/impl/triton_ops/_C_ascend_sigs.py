# -*- coding: utf-8 -*-
"""`torch.ops._C_ascend.*` 的 schema 签名表（24 个算子）。

用途
----
FlagGems 的接管机制（`_repos/FlagGems/src/flag_gems/patches/patch_util.py`）是::

    _VLLM_OPS_SIGNATURES = {lib_name: {op_name: signature}}     # L57-L102
    torch.library.define(f"{lib_name}::{op_name}", signature)   # L108
    lib = torch.library.Library(lib_name, "IMPL"); lib.impl(op_name, fn, key)  # L129-L158

本文件就是 `_C_ascend` 那一格的 `{op_name: signature}`；`signature` 是**不含算子名、
不含 `lib::` 前缀**的 schema 字符串（与 patch_util 里 `"rms_norm": "(Tensor(a!) result, ...) -> ()"` 同形）。

用法::

    from _C_ascend_sigs import _C_ASCEND_SIGS, to_patch_util_signatures
    patch_util._VLLM_OPS_SIGNATURES.update(to_patch_util_signatures())

出处（**只读参考，本文件不 import 任何 C++/仓库代码**）
-------------------------------------------------------
两个仓库都会往**同一个** `torch.ops._C_ascend` 命名空间注册（
`TORCH_LIBRARY(_C_ascend, ops)` / `TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)`），
所以 schema 的权威出处**因算子而异**：

* 仓库 VF = `_repos/vllm-plugin-FL-xmd/vllm-plugin-FL-feature-v024-ascend-platform-runtime/`
  - `csrc/ascend/torch_binding.cpp` 的 `TORCH_LIBRARY(_C_ascend, ops)` 块：**L1145-L1249**
* 仓库 VA = `_repos/vllm-ascend/`
  - `csrc/torch_binding.cpp` 的 `TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)` 块：**L2870-L3710**

⚠️ 三处「同名不同 schema」的冲突已逐条标注在 `PROVENANCE[...]["conflicts"]` 里
（`mla_preprocess` / `npu_sparse_attention_score` / `inplace_partial_rotary_mul`，另有
`dispatch_ffn_combine` 多一个 `world_size`）。本文件**以 VF（vllm-fl，本次接管的目标仓库）为准**，
因为生产调用点（`vllm_fl/**`）是按 VF 的 schema 写的。

⚠️ 两个算子**没有**权威出处（见 `NOT_FOUND` / `NOT_REGISTERED`），如实标注、未编造：
  * `npu_lightning_indexer` —— 由 **torch_npu(PTA)** 提供，两个仓库的 csrc 里都没有 `ops.def`；
    本文件给出的签名是**按调用点 kwargs 反推的**（低置信度，见 `_C_ASCEND_SIGS_INFERRED`）。
  * `dispatch_ffn_combine_bf16` —— **不是独立的 torch op**；bf16 变体由 `dispatch_ffn_combine`
    按 dtype 在内部分派（调用点只调 `dispatch_ffn_combine`，见 `moe_comm_method.py:339`）。

⚠️ **算子名**：task-12 / task-13 用的短名与实际注册名不完全一致，`torch.library.define`
必须用**实际注册名**。映射见 `TASK_NAME_TO_OP`（例如 `sparse_attention_score` →
`npu_sparse_attention_score`）。短名若直接拿去 define，FlagGems 的接管**永远不会命中**。

⚠️ **空白归一化**（实测，必读）：C++ 的 `ops.def` 容忍 `->(Tensor ...`（`->` 后无空格），
但 Python 的 `torch.library.define` **不容忍**，会直接抛
``expected schema to look like e.g. "(Tensor x) -> Tensor"``。
仓库里有两处这种写法（`grouped_matmul_swiglu_quant_v2`、`..._weight_nz_tensor_list`），
本文件已补上那一个空格（`_WHITESPACE_NORMALIZED`）。逗号后缺空格**没有影响**（实测 OK）。

⚠️ **接入后必须跑运行时自检**（裁决 C）：FlagGems 的 `_define_op_if_not_exists`
（`patch_util.py` L105-L110）**只在 op 未注册时才 define** —— 若目标环境已加载真实 `.so`
（VF 或 VA 的 `_C_ascend`），生效的就是 `.so` 的 schema。用
`verify_runtime_schemas(strict=True)`（`_C_ascend_ops_a.register()` 默认已挂钩）
把「加载的 .so 与我们表里 VF 版不一致」**显式报出来**，不要静默错配；
冲突清单见 `SCHEMA_CONFLICTS`。
"""

from __future__ import annotations

LIB_NAME = "_C_ascend"

# ---------------------------------------------------------------------------
# 分组（op 名 = **实际注册名**）
# ---------------------------------------------------------------------------

#: task-12（本任务）：attention 组 12 个
GROUP_A_OPS = [
    "compressor",
    "mla_preprocess",
    "indexer_compress_epilog_v2",
    "kv_compress_epilog",
    "store_kv_block",
    "npu_lightning_indexer",
    "npu_lightning_indexer_quant",
    "npu_vllm_quant_lightning_indexer",
    "npu_sparse_attention_score",
    "npu_sparse_flash_attention",
    "inplace_partial_rotary_mul",
    "npu_hc_pre_v2",
]

#: task-13：moe / 量化组 12 个
GROUP_B_OPS = [
    "dispatch_ffn_combine",
    "dispatch_ffn_combine_bf16",
    "moe_gating_top_k",
    "moe_gating_top_k_hash",
    "npu_moe_init_routing_custom",
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    "compressor_metadata",
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
    "npu_scatter_nd_update_v2",
]

#: task-12 描述里的字面写法（短名）→ 实际注册名
TASK_NAME_TO_OP = {
    "compressor": "compressor",
    "mla_preprocess": "mla_preprocess",
    "indexer_compress_epilog_v2": "indexer_compress_epilog_v2",
    "kv_compress_epilog": "kv_compress_epilog",
    "store_kv_block": "store_kv_block",
    "lightning_indexer": "npu_lightning_indexer",
    "lightning_indexer_quant": "npu_lightning_indexer_quant",
    "vllm_quant_lightning_indexer": "npu_vllm_quant_lightning_indexer",
    "sparse_attention_score": "npu_sparse_attention_score",
    "sparse_flash_attention": "npu_sparse_flash_attention",
    "inplace_partial_rotary_mul": "inplace_partial_rotary_mul",
    "hc_pre_v2": "npu_hc_pre_v2",
    "dispatch_ffn_combine": "dispatch_ffn_combine",
    "dispatch_ffn_combine_bf16": "dispatch_ffn_combine_bf16",
    "moe_gating_top_k": "moe_gating_top_k",
    "moe_gating_top_k_hash": "moe_gating_top_k_hash",
    "moe_init_routing_custom": "npu_moe_init_routing_custom",
    "grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_weight_nz": "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
    ),
    "compressor_metadata": "compressor_metadata",
    "sparse_attn_sharedkv": "npu_sparse_attn_sharedkv",
    "sparse_attn_sharedkv_metadata": "npu_sparse_attn_sharedkv_metadata",
    "scatter_nd_update_v2": "npu_scatter_nd_update_v2",
}

# ---------------------------------------------------------------------------
# 权威 schema（23 条：22 条有 csrc 出处 + 1 条按调用点反推）
# 每条上方注释 = 出处文件 + 行号；字符串按 C++ 源码原样拼接（含原有空白）
# ---------------------------------------------------------------------------

_C_ASCEND_SIGS = {
    # ======================= GROUP A：attention（12） =======================
    # 出处：VF csrc/ascend/torch_binding.cpp L1229（`TORCH_LIBRARY(_C_ascend, ops)` 块 L1145-L1249）
    #       VA csrc/torch_binding.cpp L3313-L3323 同名同参（**无冲突**）
    #       GE 侧：VF csrc/ascend/attention/compressor/op_host/compressor_def.cpp L190 `OP_ADD(Compressor, ...)`
    "compressor": (
        "(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, Tensor ape, "
        "Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, Tensor? state_block_table, "
        "Tensor? cu_seqlens, Tensor? seqused, Tensor? start_pos, int rope_head_dim, "
        "int cmp_ratio, int coff, float norm_eps, int rotary_mode, int cache_mode) -> Tensor"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1147-L1158
    #   ⚠️ 该 ops.def 在 VF 里被 `#ifdef VLLM_FL_ENABLE_MLAPO_DIRECT_KERNEL` 包着（L1146/L1159），
    #      未开该宏的构建**不注册此 op** → patch_util 会走 `_define_op_if_not_exists` 自己补。
    #   VA 变体：csrc/torch_binding.cpp L2971-L2981，差别仅 `Tensor? cos, Tensor? sin`（VH 里 cos/sin 必选）
    #   函数签名对照：VF csrc/ascend/mla_preprocess/mla_preprocess_torch_adpt.h L33
    "mla_preprocess": (
        "(Tensor hiddenState, Tensor wdqkv, Tensor? descale0, "
        "Tensor gamma1, Tensor? beta1, Tensor wuq, Tensor? descale1, Tensor "
        "gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache, Tensor "
        "kv_cache_rope, Tensor slotmapping, Tensor? quant_scale0, Tensor? "
        "quant_offset0, Tensor? bias0, Tensor? quant_scale1, Tensor? "
        "quant_offset1, Tensor? bias1, Tensor? ctkv_scale, Tensor? "
        "q_nope_scale, str? cache_mode, str? quant_mode, bool? "
        "enable_inner_out, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! "
        "q_out1, Tensor! kv_cache_out1, Tensor! inner_out) -> (Tensor q_out0, "
        "Tensor kv_cache_out0, Tensor q_out1, Tensor kv_cache_out1, Tensor "
        "inner_out)"
    ),
    # 出处：VA csrc/torch_binding.cpp L3560-L3567（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_def.cpp L51 `OP_ADD(IndexerCompressEpilogV2)`
    "indexer_compress_epilog_v2": (
        "(Tensor(a!) indexer_compress_cache, Tensor x, Tensor slot_mapping, int layout=2) -> ()"
    ),
    # 出处：VA csrc/torch_binding.cpp L3471-L3481（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp L56 `OP_ADD(KvCompressEpilog)`
    "kv_compress_epilog": (
        "(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
        "int quant_group_size, int quant_mode, bool round_scale_flag, int layout) -> ()"
    ),
    # 出处：VA csrc/torch_binding.cpp L3636-L3638（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/attention/store_kv_block/op_host/store_kv_block_def.cpp L64 `OP_ADD(StoreKVBlock)`
    #       注：原串为 `...group_key_idx,Tensor group_key_cache_idx...`（逗号后无空格），此处保持原样
    "store_kv_block": (
        "(Tensor key_in, Tensor key_cache_in, Tensor group_len, Tensor group_key_idx,"
        "Tensor group_key_cache_idx, int block_size=0) -> ()"
    ),
    # 出处：VA csrc/torch_binding.cpp L3599-L3605
    #       ⚠️ 注册名是 `npu_lightning_indexer_quant`（task 短名 `lightning_indexer_quant`）
    #       GE 侧：VA csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_def.cpp L84 `OP_ADD(LightningIndexerQuant)`
    "npu_lightning_indexer_quant": (
        "(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, "
        "int query_quant_mode=0, int key_quant_mode=0, "
        "str layout_query='BSND', str layout_key='BSND',"
        "int sparse_count=2048, int sparse_mode=3) -> Tensor"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1231
    #       ⚠️ 注册名是 `npu_vllm_quant_lightning_indexer`（task 短名 `vllm_quant_lightning_indexer`）
    #       GE 侧：VF csrc/ascend/attention/vllm_quant_lightning_indexer/op_host/vllm_quant_lightning_indexer_def.cpp L152 `OP_ADD(VllmQuantLightningIndexer)`
    "npu_vllm_quant_lightning_indexer": (
        '(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, '
        'Tensor key_dequant_scale, int query_quant_mode=0, int key_quant_mode=0, '
        'Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, '
        'Tensor? block_table=None, Tensor? metadata=None, str layout_query="BSND", '
        'str layout_key="BSND", int sparse_count=2048, int sparse_mode=3, '
        'int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, '
        'int cmp_ratio=1, bool return_value=False) -> '
        '(Tensor sparse_indices, Tensor sparse_values)'
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1191-L1199（**以 VF 为准**：生产调用点
    #       vllm_fl/attention/ascend/sparse_attn_m3.py L91-L105 用的是这里的大驼峰名字）
    #   ⚠️ 冲突：VA csrc/torch_binding.cpp L3641-L3652 是**另一套全 snake_case** schema：
    #       npu_sparse_attention_score(Tensor query, Tensor key, Tensor value, Tensor select_idx,
    #       Tensor block_table, *, Tensor? select_num_idx=None, ..., str kv_input_layout="BNSD",
    #       int num_key_value_heads=1, float scale_value=1.0, int block_size=128, int top_k=16,
    #       int inner_precise=0, ScalarType? attention_out_dtype=None) -> Tensor
    #       两套**参数名不可互换** → 若目标环境加载的是 VA 的 .so，adapter 必须另接一套。
    #   GE 侧：VF csrc/ascend/attention/sparse_attention_score/op_host/sparse_attention_score_def.cpp L85
    #           `OP_ADD(SparseAttentionScore)`；函数签名 VF .../sparse_attention_score_torch_adpt.h L145/L182
    "npu_sparse_attention_score": (
        "(Tensor query, Tensor key, Tensor value, "
        "Tensor selectIdx, Tensor blockTable, Tensor? selectNumIdx=None, "
        "Tensor? qDequantScale=None, Tensor? kDequantScale=None, "
        "Tensor? vDequantScale=None, Tensor? actualSeqLengths=None, "
        "Tensor? actualSeqLengthsKv=None, str qInputLayout=\"TND\", "
        "str kvInputLayout=\"TND\", int numKeyValueHeads=0, "
        "float scaleValue=0.0, int blockSize=128, int topK=0, "
        "int innerPrecise=4) -> Tensor"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1232-L1242
    #       VA csrc/torch_binding.cpp L3136-L3146 同名同参（**无冲突**）
    #       函数签名：VF csrc/ascend/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h L93
    #       GE 侧：VF .../sparse_flash_attention/op_host/sparse_flash_attention_def.cpp L101 `OP_ADD(SparseFlashAttention)`
    "npu_sparse_flash_attention": (
        "(Tensor query, Tensor key, Tensor value, "
        "Tensor sparse_indices, float scale_value, *, "
        "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, "
        "Tensor? key_rope=None, int sparse_block_size=1, "
        "str layout_query='BSND', str layout_kv='BSND', "
        "int sparse_mode=3, int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int attention_mode=2, "
        "bool return_softmax_lse=False) -> "
        "(Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1246
    #       GE 侧：VF csrc/ascend/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp L83
    #   ⚠️ 冲突：VA csrc/torch_binding.cpp L3453-L3458 多一个末位 `, bool negate_sin=False`
    #       （VA: "(Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice, bool negate_sin=False) -> ()"）
    "inplace_partial_rotary_mul": (
        "(Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice) -> ()"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1226
    #       VA csrc/torch_binding.cpp L3435-L3441 同名同参，仅返回名不同（(Tensor y, Tensor post, Tensor comb_frag)）
    #       GE 侧：VF csrc/ascend/moe/hc_pre/op_host/hc_pre_def.cpp L76 `OP_ADD(HcPre)`（无独立 hc_pre_v2 def.cpp）
    "npu_hc_pre_v2": (
        "(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, "
        "int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> "
        "(Tensor out0, Tensor out1, Tensor out2)"
    ),
    # ==================== GROUP B：moe / 量化（12） ====================
    # 出处：VF csrc/ascend/torch_binding.cpp L1160-L1164（含 `Tensor! out, Tensor! expert_token_nums`）
    #       GE 侧：VF csrc/ascend/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_def.cpp L98 `OP_ADD(DispatchFFNCombine)`
    #   ⚠️ 冲突：VA csrc/torch_binding.cpp L3225-L3229 末位多 `, int world_size=0`
    "dispatch_ffn_combine": (
        "(Tensor x, Tensor[] weight1, Tensor[] weight2, "
        "Tensor expert_idx, Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, "
        "Tensor[] bias2, Tensor probs, str group, int max_output_size, "
        "Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, "
        "float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1200-L1203
    #       GE 侧：VF csrc/ascend/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp L70 `OP_ADD(MoeGatingTopK)`
    #       函数签名：VF csrc/ascend/moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h L19
    "moe_gating_top_k": (
        "(Tensor x, int k, int k_group, int group_count, "
        "int group_select_mode, int renorm, int norm_type, bool out_flag, "
        "float routed_scaling_factor, float eps, Tensor? bias_opt=None) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1204-L1209
    #       GE 侧：VF csrc/ascend/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp L145 `OP_ADD(MoeGatingTopKHash)`
    "moe_gating_top_k_hash": (
        "(Tensor x, int k, Tensor? bias=None, "
        "Tensor? input_ids=None, Tensor? tid2eid=None, int k_group=1, "
        "int group_count=1, float routed_scaling_factor=1.0, "
        "float eps=1e-20, int group_select_mode=0, int renorm=0, "
        "int norm_type=0, bool out_flag=False) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1217-L1223
    #       GE 侧：VF csrc/ascend/moe/moe_init_routing_custom/op_host/moe_init_routing_custom_def.cpp L104 `OP_ADD(MoeInitRoutingCustom)`
    #       函数签名：VF csrc/ascend/moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h L19
    "npu_moe_init_routing_custom": (
        "(Tensor x, Tensor expert_idx, *, "
        "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
        "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
        "int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    ),
    # 出处：VA csrc/torch_binding.cpp L3090-L3095（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp L255
    "grouped_matmul_swiglu_quant_v2": (
        "(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,  Tensor group_list,  "
        "Tensor? smooth_scale=None,"
        "Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, "
        "int? dequant_dtype=0, int? quant_mode=0,"
        "int? quant_dtype=0, bool transpose_weight=False, int group_list_type=0, "
        "int[2] tuning_config=[],float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale)"
        # ↑ 原串为 `...->"(Tensor output...`（`->` 后**无空格**，见 VA L3093-L3094）。
        #   C++ 的 `ops.def` 接受这种写法，但 Python 的 `torch.library.define` 要求
        #   `->` 之后必须有空白，否则报
        #   `expected schema to look like e.g. "(Tensor x) -> Tensor"`。
        #   此处**只补了一个空格**，schema 语义不变（已实测修复 `define` 失败）。
    ),
    # 出处：VA csrc/torch_binding.cpp L3075-L3079（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant/op_host/（`OP_ADD` 在 tiling 之外，见 gmm/ 目录）
    "grouped_matmul_swiglu_quant_weight_nz": (
        "(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "Tensor group_list, *, Tensor? bias=None,"
        "Tensor? offset=None, float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale, Tensor output_offset)"
    ),
    # 出处：VA csrc/torch_binding.cpp L3082-L3088（VF csrc/ascend/ 无此 op → 未找到）
    #       GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        "(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,"
        "Tensor group_list, *,"
        "Tensor? bias=None, Tensor? offset=None, float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale, Tensor output_offset)"
        # ↑ 原串为 `...->"(Tensor output...`（`->` 后无空格，VA L3085-L3086）→
        #   原因同 `grouped_matmul_swiglu_quant_v2`，只补空格。
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1230
    #       VA csrc/torch_binding.cpp L3326-L3333 同名同参（**无冲突**）
    #       GE 侧：VF csrc/ascend/attention/compressor_metadata/op_host/compressor_metadata_def.cpp L69 `OP_ADD(CompressorMetadata)`
    "compressor_metadata": (
        "(Tensor rope_cos, Tensor rope_sin, Tensor cu_seqlens, Tensor start_pos, "
        "Tensor kv_block_table, int kv_block_size, int slot_mapping_format, "
        "int compress_ratio, int num_compressed_tokens, int num_reqs_actual) "
        "-> (Tensor, Tensor, Tensor)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1243（**以 VF 为准**：VF 里 cu_seqlens_* 是 Tensor?）
    #   ⚠️ 注册名是 `npu_sparse_attn_sharedkv`（task-13 短名 `sparse_attn_sharedkv`）
    #   VA 变体 csrc/torch_binding.cpp L3367-L3393 同名同参（逐字一致）
    #   GE 侧：VF csrc/ascend/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp L126 `OP_ADD(SparseAttnSharedkv)`
    "npu_sparse_attn_sharedkv": (
        "(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, "
        "Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, Tensor? metadata=None, "
        "float softmax_scale=0, int cmp_ratio=0, int ori_mask_mode=4, int cmp_mask_mode=3, "
        "int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", "
        "bool return_softmax_lse=False) -> (Tensor out, Tensor softmax_lse)"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1244（返回 `-> Tensor`；VA L3421 写作 `-> (Tensor metadata)`，语义相同）
    #       GE 侧：VF csrc/ascend/attention/sparse_attn_sharedkv/op_host/（同目录 def.cpp）
    "npu_sparse_attn_sharedkv_metadata": (
        "(int num_heads_q, int num_heads_kv, int head_dim, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, "
        "Tensor? seqused_kv=None, int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, "
        "int ori_topk=0, int cmp_topk=0, int cmp_ratio=4, int ori_mask_mode=4, int cmp_mask_mode=3, "
        "int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", "
        "bool has_ori_kv=True, bool has_cmp_kv=True, str device=\"npu\") -> Tensor"
    ),
    # 出处：VF csrc/ascend/torch_binding.cpp L1248
    #       GE 侧：VF csrc/ascend/moe/scatter_nd_update_v2/op_host/scatter_nd_update_v2_def.cpp L83 `OP_ADD(ScatterNdUpdateV2)`
    #   ⚠️ VA 里对应的是另一个名字 `npu_scatter_nd_update_sk`（VA csrc/torch_binding.cpp L3591-L3593），参数名相同
    "npu_scatter_nd_update_v2": "(Tensor(a!) var, Tensor indices, Tensor update) -> ()",
}

# ---------------------------------------------------------------------------
# 无权威出处：按调用点反推（低置信度）
# ---------------------------------------------------------------------------

#: `npu_lightning_indexer`（task 短名 `lightning_indexer`）
#:
#: **未找到权威 schema**：两个仓库的 csrc 里都没有 `ops.def("npu_lightning_indexer...`,
#: 也没有 `*_def.cpp` 的 `OP_ADD`（`_repos/vllm-ascend/csrc/attention/lightning_indexer_v2/`
#: 下**只有 op_kernel**，一个 .h，见 `op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h`）。
#: 该 op 由 **torch_npu(PTA)** 提供（调用点注释
#: `csrc/torch_binding.cpp` L3597-L3598 也说明 `npu_lightning_indexer_quant` 「is planned to be
#: integrated into PTA」，非量化版已在 PTA 里）。
#:
#: 下面这条是**按调用点 kwargs 反推**的，`*` 之后的可选参数与默认值**未经源码证实**：
#:   VF vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py L632-L643
#:   VF vllm_fl/attention/ascend/sfa_v1.py（同形调用）
#: ⚠️ 置信度：低。接管前必须用 `torch.ops._C_ascend.npu_lightning_indexer._schemas` 或
#:    `torch._C._jit_get_operation("_C_ascend::npu_lightning_indexer")` 在目标环境实测校对。
_C_ASCEND_SIGS_INFERRED = {
    "npu_lightning_indexer": (
        "(Tensor query, Tensor key, Tensor weights, *, "
        "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "Tensor? block_table=None, str layout_query='TND', str layout_key='PA_BSND', "
        "int sparse_count=2048, int sparse_mode=3) -> Tensor"
    ),
}

# ---------------------------------------------------------------------------
# 没有对应 torch op 的条目
# ---------------------------------------------------------------------------

#: `dispatch_ffn_combine_bf16` **不是独立的 torch op**，没有任何 `ops.def`。
#:
#: 证据：
#:   * VF csrc/ascend/torch_binding.cpp 只有 `ops.def("dispatch_ffn_combine(...)`（L1160）；
#:     VA csrc/torch_binding.cpp 同样只有 `dispatch_ffn_combine`（L3226）。
#:   * bf16 变体是 **GE/AscendC 层的另一个算子**：
#:     `csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_def.cpp` L92
#:     `OP_ADD(DispatchFFNCombineBF16);`，由 `dispatch_ffn_combine` 按 dtype 在内部分派。
#:   * 生产调用点只有一个：
#:     VF vllm_fl/dispatch/backends/vendor/ascend/impl/moe/moe_comm_method.py L339-L355
#:     调 `torch.ops._C_ascend.dispatch_ffn_combine(...)`（无 `_bf16` 后缀）。
#:
#: ⇒ 接管入口是 `dispatch_ffn_combine`，不是 `dispatch_ffn_combine_bf16`。
#:   若要让 bf16 走 Triton，应在 `dispatch_ffn_combine` 的 adapter 里按 `x.dtype` 分派到
#:   `_analysis/triton_ops/dispatch_ffn_combine_bf16.py`。
NOT_REGISTERED: dict[str, str] = {
    "dispatch_ffn_combine_bf16": (
        "无独立 torch op / 无 ops.def：bf16 变体由 GE 算子 DispatchFFNCombineBF16 "
        "(csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_def.cpp L92) "
        "实现，torch 层只有 dispatch_ffn_combine（VF torch_binding.cpp L1160 / VA L3226）。"
        "接管入口 = dispatch_ffn_combine，按 dtype 分派。"
    ),
}

#: 与 C++ 原串相比**做过空白归一化**的条目（只补 `->` 后的一个空格，语义不变）
WHITESPACE_NORMALIZED: dict[str, str] = {
    "grouped_matmul_swiglu_quant_v2": (
        "VA L3093-L3094 原串 `...->\"(Tensor output...`：`->` 后无空格；"
        "已改成 `-> (`，否则 torch.library.define 失败"
    ),
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        "VA L3085-L3086 原串 `...->\"(Tensor output...`：同上"
    ),
}

#: 24 条里所有拿不到「权威 torch schema」的条目 → 状态说明
NOT_FOUND: dict[str, str] = {    "npu_lightning_indexer": (
        "csrc 里无 ops.def / 无 *_def.cpp OP_ADD（VA csrc/attention/lightning_indexer_v2/ 只有 "
        "op_kernel，无 op_host）；op 由 torch_npu(PTA) 提供。已按调用点 kwargs 反推，见 "
        "_C_ASCEND_SIGS_INFERRED（置信度低）。"
    ),
    "dispatch_ffn_combine_bf16": NOT_REGISTERED["dispatch_ffn_combine_bf16"],
}

# ---------------------------------------------------------------------------
# 出处明细（文件 + 行号），供审计/报告直接引用
# ---------------------------------------------------------------------------

VF = (
    "_repos/vllm-plugin-FL-xmd/vllm-plugin-FL-feature-v024-ascend-platform-runtime/"
)
VA = "_repos/vllm-ascend/"

PROVENANCE: dict[str, dict] = {
    "compressor": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1229",
        "also": [VA + "csrc/torch_binding.cpp L3313-L3323"],
        "ge": VF + "csrc/ascend/attention/compressor/op_host/compressor_def.cpp L190 (OP_ADD)",
        "conflicts": [],
    },
    "mla_preprocess": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1147-L1158",
        "also": [VA + "csrc/torch_binding.cpp L2971-L2981"],
        "ge": VF + "csrc/ascend/mla_preprocess/mla_preprocess_torch_adpt.h L33 (no OP_ADD in csrc)",
        "conflicts": [
            "VF: `Tensor gamma2, Tensor cos, Tensor sin` (cos/sin 必选)；"
            "VA L2974: `Tensor gamma2, Tensor? cos, Tensor? sin`（可选）",
            "VF 的 ops.def 被 `#ifdef VLLM_FL_ENABLE_MLAPO_DIRECT_KERNEL` 包裹（L1146/L1159）",
        ],
    },
    "indexer_compress_epilog_v2": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3560-L3567",
        "also": [],
        "ge": VA + "csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_def.cpp L51 (OP_ADD)",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "kv_compress_epilog": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3471-L3481",
        "also": [],
        "ge": VA + "csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp L56 (OP_ADD)",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "store_kv_block": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3636-L3638",
        "also": [],
        "ge": VA + "csrc/attention/store_kv_block/op_host/store_kv_block_def.cpp L64 (OP_ADD)",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "npu_lightning_indexer": {
        "kind": "call_site_inferred",
        "source": VF + "vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py",
        "line": "L632-L643",
        "also": [],
        "ge": "",
        "conflicts": [],
        "warning": "无 ops.def / 无 OP_ADD；op 由 torch_npu(PTA) 提供。默认值未经证实。",
    },
    "npu_lightning_indexer_quant": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3599-L3605",
        "also": [],
        "ge": VA + "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_def.cpp L84 (OP_ADD)",
        "conflicts": [],
    },
    "npu_vllm_quant_lightning_indexer": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1231",
        "also": [],
        "ge": VF + "csrc/ascend/attention/vllm_quant_lightning_indexer/op_host/vllm_quant_lightning_indexer_def.cpp L152 (OP_ADD)",
        "conflicts": [],
    },
    "npu_sparse_attention_score": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1191-L1199",
        "also": [VA + "csrc/torch_binding.cpp L3641-L3652"],
        "ge": VF + "csrc/ascend/attention/sparse_attention_score/op_host/sparse_attention_score_def.cpp L85 (OP_ADD)",
        "conflicts": [
            "VA L3641-L3652 是另一套全 snake_case schema（select_idx/block_table/.../"
            "str kv_input_layout=\"BNSD\"/int num_key_value_heads=1/float scale_value=1.0/"
            "int block_size=128/int top_k=16/int inner_precise=0/ScalarType? attention_out_dtype=None）。"
            "参数名与 VF 版**不可互换**。",
        ],
    },
    "npu_sparse_flash_attention": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1232-L1242",
        "also": [VA + "csrc/torch_binding.cpp L3136-L3146"],
        "ge": VF + "csrc/ascend/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp L101 (OP_ADD)",
        "conflicts": [],
    },
    "inplace_partial_rotary_mul": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1246",
        "also": [VA + "csrc/torch_binding.cpp L3453-L3458"],
        "ge": VF + "csrc/ascend/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp L83 (OP_ADD)",
        "conflicts": [
            "VA L3453-L3458 末位多 `bool negate_sin=False`；VF 版没有该参数。",
        ],
    },
    "npu_hc_pre_v2": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1226",
        "also": [VA + "csrc/torch_binding.cpp L3435-L3441"],
        "ge": VF + "csrc/ascend/moe/hc_pre/op_host/hc_pre_def.cpp L76 (OP_ADD(HcPre))",
        "conflicts": [
            "VA L3440 返回名不同：`(Tensor y, Tensor post, Tensor comb_frag)` vs VF `(Tensor out0, Tensor out1, Tensor out2)`",
        ],
    },
    "dispatch_ffn_combine": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1160-L1164",
        "also": [VA + "csrc/torch_binding.cpp L3225-L3229"],
        "ge": VF + "csrc/ascend/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_def.cpp L98 (OP_ADD)",
        "conflicts": ["VA L3228 末位多 `int world_size=0`"],
    },
    "dispatch_ffn_combine_bf16": {
        "kind": "not_registered",
        "source": "",
        "line": "",
        "also": [],
        "ge": VF + "csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_def.cpp L92 (OP_ADD(DispatchFFNCombineBF16))",
        "conflicts": [],
        "warning": NOT_REGISTERED["dispatch_ffn_combine_bf16"],
    },
    "moe_gating_top_k": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1200-L1203",
        "also": [],
        "ge": VF + "csrc/ascend/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp L70 (OP_ADD)",
        "conflicts": [],
    },
    "moe_gating_top_k_hash": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1204-L1209",
        "also": [],
        "ge": VF + "csrc/ascend/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp L145 (OP_ADD)",
        "conflicts": [],
    },
    "npu_moe_init_routing_custom": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1217-L1223",
        "also": [],
        "ge": VF + "csrc/ascend/moe/moe_init_routing_custom/op_host/moe_init_routing_custom_def.cpp L104 (OP_ADD)",
        "conflicts": [],
    },
    "grouped_matmul_swiglu_quant_v2": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3090-L3095",
        "also": [],
        "ge": VA + "csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp L255",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "grouped_matmul_swiglu_quant_weight_nz": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3075-L3079",
        "also": [],
        "ge": VA + "csrc/gmm/grouped_matmul_swiglu_quant/op_host/",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": {
        "kind": "binding",
        "source": VA + "csrc/torch_binding.cpp",
        "line": "L3082-L3088",
        "also": [],
        "ge": VA + "csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/",
        "conflicts": ["VF csrc/ascend/ 下无此 op（未找到）"],
    },
    "compressor_metadata": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1230",
        "also": [VA + "csrc/torch_binding.cpp L3326-L3333"],
        "ge": VF + "csrc/ascend/attention/compressor_metadata/op_host/compressor_metadata_def.cpp L69 (OP_ADD)",
        "conflicts": [],
    },
    "npu_sparse_attn_sharedkv": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1243",
        "also": [VA + "csrc/torch_binding.cpp L3367-L3393"],
        "ge": VF + "csrc/ascend/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp L126 (OP_ADD)",
        "conflicts": [],
    },
    "npu_sparse_attn_sharedkv_metadata": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1244",
        "also": [VA + "csrc/torch_binding.cpp L3396-L3423"],
        "ge": VF + "csrc/ascend/attention/sparse_attn_sharedkv/op_host/",
        "conflicts": [
            "VA L3421 返回写作 `-> (Tensor metadata)`，VF 写作 `-> Tensor`（语义相同）",
        ],
    },
    "npu_scatter_nd_update_v2": {
        "kind": "binding",
        "source": VF + "csrc/ascend/torch_binding.cpp",
        "line": "L1248",
        "also": [],
        "ge": VF + "csrc/ascend/moe/scatter_nd_update_v2/op_host/scatter_nd_update_v2_def.cpp L83 (OP_ADD)",
        "conflicts": [
            "VA 对应名字为 `npu_scatter_nd_update_sk`（VA csrc/torch_binding.cpp L3591-L3593），参数名相同",
        ],
    },
}

# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

#: 权威 + 反推（23 条）= 24 条减去 `dispatch_ffn_combine_bf16`
_C_ASCEND_ALL_SIGS: dict[str, str] = {**_C_ASCEND_SIGS, **_C_ASCEND_SIGS_INFERRED}

# ---------------------------------------------------------------------------
# 外形参数解析（纯标准库；本模块不依赖 torch）
# ---------------------------------------------------------------------------


def _split_top_level(s: str) -> list[str]:
    """按**顶层**逗号切分（忽略 `()` / `[]` 内部的逗号，如 `Tensor(a!)` / `int[2]`）。"""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_schema_params(schema: str) -> list[dict]:
    """把 `"(Tensor(a!) x, int k=1, *, Tensor? b=None) -> Tensor"` 解析成形参列表。

    返回 `[{"name","ann","default","kwonly"}, ...]`；`*` 之后的标 `kwonly=True`。
    兼容 torch 运行时 `_schemas` 的归一化写法（`Tensor($0! -> ) out`、
    `float eps=9.9999999999999995e-21` 等）—— 名字取每段的**最后一个 token**。
    """
    s = schema.strip()
    assert s.startswith("("), f"schema 不以 ( 开头: {s[:60]!r}"
    depth, end = 0, None
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, f"schema 括号不闭合: {s[:80]!r}"
    args_str = s[1:end]
    rest = s[end + 1:].strip()
    assert rest.startswith("->"), f"schema 缺少 `->`: {rest[:60]!r}"

    params, kwonly = [], False
    for raw in _split_top_level(args_str):
        p = raw.strip()
        if not p:
            continue
        if p == "*":
            kwonly = True
            continue
        head, sep, dflt = p.partition("=")
        decl = head.strip()
        toks = decl.split()
        assert len(toks) >= 2, f"无法解析形参声明: {decl!r}"
        name, ann = toks[-1], " ".join(toks[:-1])
        params.append({
            "name": name,
            "ann": ann,
            "default": dflt.strip() if sep else None,
            "kwonly": kwonly,
        })
    return params


def parse_runtime_schema(raw, lib_name: str = LIB_NAME) -> list[dict]:
    """解析 `torch.ops.<lib>.<op>._schemas` 里的 `"<lib>::<op>(...) -> ..."`。

    ⚠️ 实测（torch 2.13）：`_schemas` 的 value **不是 `str`，而是
    `torch._C.FunctionSchema`**（`repr` 看着像字符串，`.strip()` 直接 AttributeError）。
    本函数先用 `str()` 归一化，字符串路径只作兜底 ——
    **首选** `runtime_schema_params()`（走 `FunctionSchema.arguments`，不解析字符串）。
    """
    s = str(raw).strip()
    if "::" in s:
        s = s.split("::", 1)[1]          # 去掉 `_C_ascend::`
    if "(" in s and not s.startswith("("):
        s = s[s.index("("):]             # 去掉算子名
    return parse_schema_params(s)


def runtime_schema_params(schema_obj) -> list[tuple]:
    """从运行时 schema 对象取 `[(形参名, 是否 keyword-only), ...]`。

    **首选** `FunctionSchema.arguments`（结构化，不解析字符串；`Argument.kwarg_only`
    就是 schema 里 `*` 之后的标记）；拿不到时退回字符串解析。
    """
    args = getattr(schema_obj, "arguments", None)
    if args is not None:
        try:
            return [(a.name, bool(getattr(a, "kwarg_only", False))) for a in args]
        except Exception:
            pass
    return [(p["name"], bool(p["kwonly"]))
            for p in parse_runtime_schema(schema_obj)]


#: VF / VA 两仓库「同名不同 schema」的算子（裁决 C：**以 VF 为准**，不一致要显式报错）
SCHEMA_CONFLICTS: dict[str, dict] = {
    "mla_preprocess": {
        "vf": "Tensor gamma2, Tensor cos, Tensor sin（cos/sin 必选）",
        "va": "Tensor gamma2, Tensor? cos, Tensor? sin（可选）",
        "impact": "形参名集合相同 → 名字级自检不会报错；但 VF 版还被 "
                  "`#ifdef VLLM_FL_ENABLE_MLAPO_DIRECT_KERNEL`（L1146/L1159）包裹，"
                  "未开宏的构建根本不注册该 op",
    },
    "npu_sparse_attention_score": {
        "vf": "大驼峰 selectIdx/blockTable/actualSeqLengths/.../blockSize/topK/innerPrecise",
        "va": "全 snake_case select_idx/block_table/... + `ScalarType? attention_out_dtype`",
        "impact": "**参数名不可互换** → 若目标环境是 VA 的 .so，adapter 必然 TypeError；"
                  "`verify_runtime_schemas()` 会抓到",
    },
    "inplace_partial_rotary_mul": {
        "vf": "5 参（无 negate_sin）",
        "va": "6 参（末尾多 `bool negate_sin=False`）",
        "impact": "多出的是可选参数 → 自检在**个数**上都会报",
    },
    "dispatch_ffn_combine": {
        "vf": "14 参（无 world_size）",
        "va": "15 参（末尾多 `int world_size=0`）",
        "impact": "同上",
    },
}


def verify_runtime_schemas(strict: bool = False, lib_name: str = LIB_NAME,
                           out_stats: dict | None = None) -> list[str]:
    """比对**运行时实际注册**的 `<lib_name>::<op>` schema 与本地表的形参。

    为什么必须做（裁决 C）：FlagGems 的 `_define_op_if_not_exists`
    （`patch_util.py` L105-L110）**只在 op 未注册时才 define**。若目标环境已加载真实
    `.so`（VF 或 VA 的 `_C_ascend`），**生效的就是 `.so` 的 schema**；对
    `SCHEMA_CONFLICTS` 里的算子，adapter 的关键字参数就会对不上。
    本函数把这种错配**显式暴露**，而不是静默错配。

    比较口径：形参**名字 / 顺序 / kwonly / 个数**（决定关键字调用能否成立）。
    **不比默认值** —— torch 会把 `0.0` 归一化成 `0.`、`Tensor! out` 归一化成
    `Tensor($0! -> ) out`，属纯外表差异。未注册的 op 跳过（交给 patch_util 的 define 补）。

    ⚠️ **自检不允许静默失效**：op 已注册但 schema 取不到/解析失败时，会把
    「自检失效」本身作为一条 problem 报出来（而不是 `continue` 掉）—— 一个静默通过的
    自检比没有自检更危险（本函数初版就踩过：`_schemas` 的 value 是 `FunctionSchema`
    而非 `str`，字符串解析抛错被吞掉，反证测试立刻抓出来了）。

    Args:
        strict: True 时发现问题直接抛 `RuntimeError`。
        lib_name: 命名空间名（默认 `_C_ascend`；测试里用探针命名空间做反证）。
        out_stats: 可选；会被填入 `{"lib_name","total","registered","checked"}`，
            用于证明自检**非空跑**（`checked == 0` 时「没有问题」毫无意义）。

    Returns:
        问题描述列表（空 = 一致，或没有可比对的已注册 op）。
    """
    try:
        import torch
    except Exception:  # 无 torch：本检查不适用
        if out_stats is not None:
            out_stats.update({"lib_name": lib_name, "total": len(_C_ASCEND_ALL_SIGS),
                              "registered": 0, "checked": 0, "no_torch": True})
        return []

    problems: list[str] = []
    registered = checked = 0
    for op, sig in _C_ASCEND_ALL_SIGS.items():
        try:
            ns = getattr(torch.ops, lib_name)
        except Exception:
            if out_stats is not None:
                out_stats.update({"lib_name": lib_name, "total": len(_C_ASCEND_ALL_SIGS),
                                  "registered": 0, "checked": 0, "no_namespace": True})
            return []          # 命名空间不存在 → 无可比对
        if not hasattr(ns, op):
            continue           # 未注册 → 合法跳过（patch_util 会 define）
        registered += 1
        try:
            schemas = getattr(getattr(ns, op), "_schemas", None) or {}
            schema_obj = schemas.get("") or next(iter(schemas.values()), None)
            if schema_obj is None:
                problems.append(f"{op}: 已注册但取不到 _schemas —— 自检失效，必须修复")
                continue
            gnames = runtime_schema_params(schema_obj)
            if not gnames:
                problems.append(f"{op}: 运行时 schema 解析出 0 个形参 —— 自检失效")
                continue
        except Exception as e:
            problems.append(
                f"{op}: 运行时 schema 解析失败（**自检失效，必须修复**）："
                f"{type(e).__name__}: {e}")
            continue
        checked += 1
        want = parse_schema_params(sig)
        wnames = [(p["name"], bool(p["kwonly"])) for p in want]
        if wnames != gnames:
            first = next((i for i, (a, b) in enumerate(zip(wnames, gnames)) if a != b),
                         min(len(wnames), len(gnames)))
            problems.append(
                f"{op}: 运行时 schema 与本表不一致（首个差异在第 {first} 个形参）\n"
                f"      运行时: {gnames[:first + 1]}\n"
                f"      本表  : {wnames[:first + 1]}\n"
                f"      很可能是加载了另一仓库（VA）的 .so 或 schema 已更新；"
                f"冲突清单见 SCHEMA_CONFLICTS"
            )
    if out_stats is not None:
        out_stats.update({"lib_name": lib_name, "total": len(_C_ASCEND_ALL_SIGS),
                          "registered": registered, "checked": checked})
    if strict and problems:
        raise RuntimeError(
            f"运行时 {lib_name} schema 与本表不一致（拒绝静默错配）：\n  - "
            + "\n  - ".join(problems)
        )
    return problems


def to_patch_util_signatures() -> dict[str, dict[str, str]]:
    """返回可直接 `update()` 进 `patch_util._VLLM_OPS_SIGNATURES` 的字典。"""
    return {LIB_NAME: dict(_C_ASCEND_ALL_SIGS)}


def schema_of(task_name_or_op: str) -> str:
    """按 task 短名或实际注册名取 schema；`dispatch_ffn_combine_bf16` 会抛 KeyError（它没有 schema）。"""
    op = TASK_NAME_TO_OP.get(task_name_or_op, task_name_or_op)
    return _C_ASCEND_ALL_SIGS[op]


def coverage_report() -> list[tuple[str, str, str]]:
    """24 条逐条状态：`(task 短名, 实际注册名, 状态)`；状态 ∈ {binding, binding(无OP_ADD), inferred, not_registered}。"""
    rows = []
    for task_name, op in TASK_NAME_TO_OP.items():
        prov = PROVENANCE.get(op, {})
        kind = prov.get("kind", "?")
        if kind == "binding":
            extra = "" if prov.get("ge") else "(no OP_ADD)"
            status = "binding" + extra
        elif kind == "call_site_inferred":
            status = "inferred(低置信度)"
        elif kind == "not_registered":
            status = "not_registered"
        else:
            status = "?"
        rows.append((task_name, op, status))
    return rows


def check_coverage() -> None:
    """自检：24 个 task 短名都有归属，且权威集合与 group 列表自洽。"""
    expected = set(GROUP_A_OPS) | set(GROUP_B_OPS)
    assert len(GROUP_A_OPS) == 12, len(GROUP_A_OPS)
    assert len(GROUP_B_OPS) == 12, len(GROUP_B_OPS)
    assert len(set(GROUP_A_OPS)) == 12 and len(set(GROUP_B_OPS)) == 12
    assert not (set(GROUP_A_OPS) & set(GROUP_B_OPS)), "A/B 组 op 名重叠"
    assert set(TASK_NAME_TO_OP.values()) == expected, (
        set(TASK_NAME_TO_OP.values()) ^ expected
    )
    assert set(PROVENANCE) == expected, set(PROVENANCE) ^ expected
    missing = expected - set(_C_ASCEND_ALL_SIGS)
    assert missing == {"dispatch_ffn_combine_bf16"}, missing


if __name__ == "__main__":  # pragma: no cover - 人工核对用
    check_coverage()
    print(f"{LIB_NAME}: {len(_C_ASCEND_ALL_SIGS)} 条 schema "
          f"(权威 {len(_C_ASCEND_SIGS)} + 反推 {len(_C_ASCEND_SIGS_INFERRED)})；"
          f"not_registered {len(NOT_REGISTERED)}；共 {len(TASK_NAME_TO_OP)} 个 task 算子")
    for task_name, op, status in coverage_report():
        print(f"  {task_name:52s} -> {op:52s} {status}")
