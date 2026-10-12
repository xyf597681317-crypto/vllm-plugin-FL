# -*- coding: utf-8 -*-
"""FlagGems 接管 ``torch.ops._C_ascend.*`` —— **B 组（moe / 量化 / 状态类）12 个 adapter**。

本文件是 **task-13** 的交付物，与 task-12 的 ``_C_ascend_sigs.py`` 配套：
``_C_ascend_sigs.py`` 提供 24 条 schema（只读依赖），本文件提供 B 组 12 个
adapter（可直接被 FlagGems 的 ``patch_util.patch_vllm_lib`` 使用）。

接管机制（FlagGems 侧，已读源码）
----------------------------------
``_repos/FlagGems/src/flag_gems/patches/patch_util.py``::

    _VLLM_OPS_SIGNATURES = {lib_name: {op_name: signature}}       # L57-L102
    torch.library.define(f"{lib_name}::{op_name}", signature)     # L108
    lib = torch.library.Library(lib_name, "IMPL")
    lib.impl(op_name, fn, key)                                    # L129-L158

本文件对外提供三样东西：

* :data:`SIGS_B` —— ``{实际注册名: patch_util 风格 signature}``（11 条；
  ``dispatch_ffn_combine_bf16`` **不是独立 torch op**，见下）
* :data:`ADAPTERS_B` —— ``{实际注册名: adapter 函数}``
* :func:`patch` —— 自己完成 ``define`` + ``Library("IMPL").impl``（不依赖 FlagGems 内部）
* :func:`to_patch_util_signatures` —— 返回 ``{"_C_ascend": SIGS_B}``，可直接 update 进
  ``patch_util._VLLM_OPS_SIGNATURES``

约定（与 task-12 一致，务必遵守）
--------------------------------
1. **短名 ≠ 注册名**：task 给的 12 个短名里有 4 个实际带 ``npu_`` 前缀
   （见 :data:`TASK_NAME_TO_OP_B`）。``torch.library.define`` 必须用**注册名**，
   否则 FlagGems 的接管永远不会命中。
2. **adapter 必须能接受关键字调用**：生产调用点（``vllm_fl/**``）基本全 kwargs。
   adapter 的参数名 / 顺序 / kwarg-only 边界与 schema **逐字对齐**，
   由 :func:`check_adapter_signatures` 静态校验（不需要 torch）。
3. **``NotImplementedError`` 一律透传**：adapter 内**没有任何 try/except**，
   不吞错、不改写异常类型；只有对「已确认超出 Triton 实现覆盖范围」的取值
   才提前抛 ``NotImplementedError``，并在注释里给出出处行号。
4. **``dispatch_ffn_combine_bf16`` 没有 torch schema**：VF/VA 两个仓库都只注册
   ``dispatch_ffn_combine``（VF ``csrc/ascend/torch_binding.cpp`` L1160 /
   VA ``csrc/torch_binding.cpp`` L3226），bf16 变体由 C++ 按
   ``weight1[0].scalar_type()`` 内部分派到 ``aclnnDispatchFFNCombineBF16``
   （VF ``csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h``
   L50 ``at::kChar`` → W8A8；L66 ``at::kBFloat16 || at::kHalf`` → BF16 OPL）。
   因此本文件：
     * ``dispatch_ffn_combine`` 的 adapter **按 weight dtype 分派**到 int8 / bf16
       两个 Triton 实现（复刻原 C++ 行为，也是 task-12 的建议）；
     * 同时保留一个**独立可调用**的 ``dispatch_ffn_combine_bf16`` adapter
       （签名与 int8 变体同形，来自调用点签名 + ``dispatch_ffn_combine_bf16_def.cpp``
       L18-L89，**不是注册 schema，不进 SIGS_B**）。
5. ``dispatch_ffn_combine_bf16`` 的精度判据经过一次重要修订（原来用 half-ULP，
   原理不可达；现改用**按幅度分段**：``|ref|max<=1`` 用 README 绝对地板（bf16 4e-3），
   ``|ref|max>1`` 用输出 dtype 完整 1 ULP；实测 7/7 PASS），演进史见
   ``_analysis/mc2_后端问题记录.md`` **第七节**。adapter 不关心判据，只做参数透传。

⚠️ 无 NPU 时的行为
Lead 裁决（2026-10-10，已落实到代码）
-------------------------------------
1. **``dispatch_ffn_combine`` 的 bias ⇒ adapter 显式丢弃**（``status=resolved_drop_bias``）：
   原算子**两个分支都丢弃** ``bias1/bias2``（int8：``torch_adpt.h`` L50-L65 未传入 aclnn；
   bf16：L69-L70 ``(void)bias1; (void)bias2;``），而生产**总是**传非空 bias
   （``moe_comm_method.py`` L334-L355）⇒ adapter 给实现传 ``None`` 复刻原语义，
   **不改** ``_analysis/triton_ops/`` 基线（实现里的拒绝护栏保留）。
   配套可证伪测试见 ``test_ops_b.py`` ④c。
   ⚠️ 风险：若生产未来真的用 bias，本「丢弃」即为错，须改为透传 + 补实现。
2. **GMM ×3（``..._weight_nz`` / ``..._weight_nz_tensor_list`` / ``v2``）⇒ 保留显式拒绝，
   如实记为缺口**（``status=gap_impl_missing``）：这三条 bias/assist matrix **真的参与计算**
   （``_v2_torch_adpt.h`` L28/L49/L59；``_torch_adpt.h`` L41-L42/L86-L87）⇒ 是**实现缺口**，
   本轮不补，登记在 :data:`GAPS_B`（GAP-B1/B2/B3），Lead 后续排期。
3. **``npu_sparse_attn_sharedkv_metadata`` ⇒ 如实留缺口**：AiCPU 算子、无 AI Core 内核，
   不属于「24 个 AscendC 算子」范围；adapter 保留完整 schema 签名 + 显式拒绝（GAP-B4）。
4. 认可：``moe_init_routing_custom`` 的 ``quant_mode=0`` 显式拒绝（GAP-B6）、
   ``npu_sparse_attn_sharedkv`` 依 tiling 契约（L1185-L1193）镜像拒绝非法 ``cmp_ratio``。
5. **GAP-B5 已关闭**（Lead 2026-10-10 核实）：生产 FUSED_MC2 未量化路径传的是
   **空张量占位**（``fused_moe.py`` L245-L249 TODO + **L259-L264**：``w1`` 是单元素 list ⇒ E=1，
   ``w1_scale = [torch.tensor([], dtype=torch.int64)]``），bf16 实现的
   ``_check_scale_unused`` 对空张量 ⇒ ``allclose((1,0),(1,0)) == True`` ⇒ **接受**，不阻塞。
   登记迁到 :data:`PRODUCTION_CONFLICTS` 的 ``resolved_scale_empty_placeholder`` 条目；
   测试 ④c 覆盖 int64/float32 空张量占位与负例。

📌 **缺口清单以 :data:`GAPS_B` 为准（5 条：GAP-B1/B2/B3/B4/B6），不得声称「已覆盖」
（GAP-B5 已关闭、不再计入）。**

⚠️ 无 NPU 时的行为
------------------

本模块**不 import torch / triton**（懒加载），因此在没有 torch 的宿主机上也能
import 并做签名静态校验；只有在真正调用 adapter 时才需要 torch + triton，
并会把 ``_analysis/triton_ops/`` 的实现按需加载进来。
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

# ===========================================================================
# 0. 出处常量（只写文档用，不 import 任何仓库代码）
# ===========================================================================

VF = "_repos/vllm-plugin-FL-xmd/vllm-plugin-FL-feature-v024-ascend-platform-runtime/"
VA = "_repos/vllm-ascend/"
VF_BINDING = VF + "csrc/ascend/torch_binding.cpp"
VA_BINDING = VA + "csrc/torch_binding.cpp"
TRITON_OPS_REF = "_analysis/triton_ops/"

LIB_NAME = "_C_ascend"

#: task-13 的 12 个**短名**（题面给的写法）
GROUP_B_TASK_OPS = [
    "dispatch_ffn_combine",
    "dispatch_ffn_combine_bf16",
    "moe_gating_top_k",
    "moe_gating_top_k_hash",
    "moe_init_routing_custom",
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    "compressor_metadata",
    "sparse_attn_sharedkv",
    "sparse_attn_sharedkv_metadata",
    "scatter_nd_update_v2",
]

#: 短名 → **实际注册名**（与 ``_C_ascend_sigs.TASK_NAME_TO_OP`` 保持一致，只取 B 组）
TASK_NAME_TO_OP_B = {
    "dispatch_ffn_combine": "dispatch_ffn_combine",
    "dispatch_ffn_combine_bf16": "dispatch_ffn_combine_bf16",  # 无 torch schema（见模块 docstring）
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

#: 实际注册名 → task 短名（反查）
OP_TO_TASK_NAME_B = {v: k for k, v in TASK_NAME_TO_OP_B.items()}


# ===========================================================================
# 1. schema 表（第一步产物；signature 与 _C_ascend_sigs.py 逐字一致）
# ===========================================================================
# 格式：patch_util 风格 —— **不含算子名、不含 `lib::` 前缀**
# （对照 patch_util.py L59 `"rms_norm": "(Tensor(a!) result, ...) -> ()"`）。

SIGS_B: dict[str, str] = {
    # ---------------- dispatch_ffn_combine ----------------
    # 权威：VF_BINDING L1160-L1164（TORCH_LIBRARY(_C_ascend, ops) 块 L1145-L1249）
    #   ⚠️ 冲突：VA_BINDING L3225-L3229 末位多一个 `, int world_size=0`；
    #      本文件以 VF 为准（生产调用点 vllm_fl/** 按 VF 写），
    #      VA 的 world_size 作为 keyword-only 扩展参数接受（非 0 → NotImplementedError）。
    # GE 侧：VF dispatch_ffn_combine_def.cpp L18-L98（OP_ADD L98），输入 a/w1/w2/expertIdx/
    #        scale1/scale2/probs/xActiveMaskOptional，输出 out / expert_token_nums
    # 生产调用点：vllm_fl/.../moe/moe_comm_method.py L339-L355（全 kwargs）
    "dispatch_ffn_combine": (
        "(Tensor x, Tensor[] weight1, Tensor[] weight2, "
        "Tensor expert_idx, Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, "
        "Tensor[] bias2, Tensor probs, str group, int max_output_size, "
        "Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, "
        "float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
    ),
    # ---------------- moe_gating_top_k ----------------
    # 权威：VF_BINDING L1200-L1203；VA_BINDING L3233-L3247 同名同参（无冲突）
    # GE 侧：VF moe_gating_top_k_def.cpp L18-L70（输入 x/bias，输出 y/expert_idx/out，
    #        attr k/k_group/group_count/group_select_mode/renorm/norm_type/out_flag/
    #        routed_scaling_factor/eps）
    # torch 层形参名：VF moe_gating_top_k_torch_adpt.h L19（`bias_opt`）
    # 生产调用点：vllm_fl/.../device/device_op.py L147-L159（全 kwargs，含 bias_opt）
    "moe_gating_top_k": (
        "(Tensor x, int k, int k_group, int group_count, "
        "int group_select_mode, int renorm, int norm_type, bool out_flag, "
        "float routed_scaling_factor, float eps, Tensor? bias_opt=None) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)"
    ),
    # ---------------- moe_gating_top_k_hash ----------------
    # 权威：VF_BINDING L1204-L1209（13 个形参）
    #   ⚠️ 冲突：VA_BINDING L3291-L3310 末尾多 bias_vl / image_sentinel_lo /
    #      image_sentinel_count（本文件按 VF 为准，VA 那 3 个作为 keyword-only 扩展接受，
    #      非默认值 → NotImplementedError）
    # GE 侧：VF moe_gating_top_k_hash_def.cpp L18-L145（输入 x/bias/input_ids/tid2eid）
    # 生产调用点：vllm_fl/.../impl/moe/experts_selector.py L266
    "moe_gating_top_k_hash": (
        "(Tensor x, int k, Tensor? bias=None, "
        "Tensor? input_ids=None, Tensor? tid2eid=None, int k_group=1, "
        "int group_count=1, float routed_scaling_factor=1.0, "
        "float eps=1e-20, int group_select_mode=0, int renorm=0, "
        "int norm_type=0, bool out_flag=False) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)"
    ),
    # ---------------- npu_moe_init_routing_custom ----------------
    # 权威：VF_BINDING L1217-L1223（注意 `*` 在 scale/offset 之前 → 二者是 kwarg-only）
    # GE 侧：VF moe_init_routing_custom_def.cpp L18-L104（attr 默认值：quant_mode=-1）
    # torch 层实现：VF moe_init_routing_custom_torch_adpt.h L19-L117
    # 生产调用点：vllm_fl/.../device/device_op.py L116-L126、impl/device_operator.py L169-L179
    "npu_moe_init_routing_custom": (
        "(Tensor x, Tensor expert_idx, *, "
        "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
        "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
        "int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    ),
    # ---------------- grouped_matmul_swiglu_quant_v2 ----------------
    # 权威：VA_BINDING L3090-L3095（VF csrc/ascend/ 无此 op → VF 未找到）
    # GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/..._def.cpp L255
    # 生产调用点：vllm_fl/.../impl/moe/moe_mlp.py L343-L353（W4A8 路径，传 weight_assist_matrix）
    "grouped_matmul_swiglu_quant_v2": (
        "(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,  Tensor group_list,  "
        "Tensor? smooth_scale=None,"
        "Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, "
        "int? dequant_dtype=0, int? quant_mode=0,"
        "int? quant_dtype=0, bool transpose_weight=False, int group_list_type=0, "
        "int[2] tuning_config=[],float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale)"
        # ↑ 原串 `->` 后无空格（VA L3093-L3094），C++ 接受但 Python `torch.library.define`
        #   要求 `->` 后有空白；此处只补一个空格，schema 语义不变（已实测）。
    ),
    # ---------------- grouped_matmul_swiglu_quant_weight_nz ----------------
    # 权威：VA_BINDING L3075-L3079（VF 无此 op）
    # GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant/（同源单张量版；本仓库无独立 AscendC 内核）
    # aclnn 适配：VA csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/
    #            grouped_matmul_swiglu_quant_torch_adpt.h L20-L60
    # 生产调用点：vllm_fl/.../device/device_op.py L225-L233（kwargs，含 bias）
    "grouped_matmul_swiglu_quant_weight_nz": (
        "(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "Tensor group_list, *, Tensor? bias=None,"
        "Tensor? offset=None, float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale, Tensor output_offset)"
    ),
    # ---------------- grouped_matmul_swiglu_quant_weight_nz_tensor_list ----------------
    # 权威：VA_BINDING L3082-L3088（VF 无此 op）
    # GE 侧：VA csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/
    # aclnn 适配：同目录 grouped_matmul_swiglu_quant_torch_adpt.h L62-L99（bias/offset 会传给 aclnn）
    # 生产调用点：vllm_fl/.../impl/moe/moe_mlp.py L175-L183、L360-L368（kwargs，含 bias）
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        "(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,"
        "Tensor group_list, *,"
        "Tensor? bias=None, Tensor? offset=None, float swiglu_limit=0.0) -> "
        "(Tensor output, Tensor output_scale, Tensor output_offset)"
        # ↑ 原串 `->` 后无空格（VA L3086-L3087），`torch.library.define` 会报
        #   "expected schema to look like ..."；此处只补一个空格（与 task-12 修复一致）。
    ),
    # ---------------- compressor_metadata ----------------
    # 权威：VF_BINDING L1230；VA_BINDING L3326-L3333 同名同参（无冲突）
    # GE 侧：VF compressor_metadata_def.cpp L8-L69（输入 ropeCos/ropeSin/cuSeqlens/startPos/
    #        kvBlockTable，输出 compressCos/compressSin/slotMapping，
    #        attr kvBlockSize/slotMappingFormat/cmpRatio/actualNumReqs）
    # 生产调用点：vllm_fl/attention/ascend/dsa_v1.py L1555-L1566（**位置参数**，10 个，顺序同 schema）
    "compressor_metadata": (
        "(Tensor rope_cos, Tensor rope_sin, Tensor cu_seqlens, Tensor start_pos, "
        "Tensor kv_block_table, int kv_block_size, int slot_mapping_format, "
        "int compress_ratio, int num_compressed_tokens, int num_reqs_actual) "
        "-> (Tensor, Tensor, Tensor)"
    ),
    # ---------------- npu_sparse_attn_sharedkv ----------------
    # 权威：VF_BINDING L1243；VA_BINDING L3367-L3393 同名同参（逐字一致）
    # GE 侧：VF sparse_attn_sharedkv_def.cpp L19-L126（attr 默认值：softmax_scale=1.0,
    #        cmp_ratio=1, ori_win_left=127, layout_kv="PA_ND"）
    #   ⚠️ GE 侧 tiling 只接受 cmp_ratio ∈ {4,128}（且仅在 CFA/SCFA 模式，即给了 cmp_kv 时）：
    #      VF sparse_attn_sharedkv_tiling.cpp L1185-L1193
    # 生产调用点：vllm_fl/attention/ascend/dsa_v1.py L1988-L2004（q 位置 + 其余全 kwargs）
    "npu_sparse_attn_sharedkv": (
        "(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, "
        "Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, Tensor? metadata=None, "
        "float softmax_scale=0, int cmp_ratio=0, int ori_mask_mode=4, int cmp_mask_mode=3, "
        "int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", "
        "bool return_softmax_lse=False) -> (Tensor out, Tensor softmax_lse)"
    ),
    # ---------------- npu_sparse_attn_sharedkv_metadata ----------------
    # 权威：VF_BINDING L1244（返回 `-> Tensor`；VA L3421 写作 `-> (Tensor metadata)`，语义相同）
    # GE 侧：**AiCPU 算子**，没有 *_def.cpp：
    #        VF csrc/ascend/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/
    #        + op_graph/sparse_attn_sharedkv_metadata_proto.h L23-L50（REG_OP）
    # 生产调用点：vllm_fl/.../device/device_op.py L767-L775（返回可调用 op，kwargs 含 device）
    #        + vllm_fl/attention/ascend/dsa_v1.py L787-L802（全 kwargs）
    "npu_sparse_attn_sharedkv_metadata": (
        "(int num_heads_q, int num_heads_kv, int head_dim, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, "
        "Tensor? seqused_kv=None, int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, "
        "int ori_topk=0, int cmp_topk=0, int cmp_ratio=4, int ori_mask_mode=4, int cmp_mask_mode=3, "
        "int ori_win_left=128, int ori_win_right=0, str layout_q=\"BSND\", str layout_kv=\"PA_ND\", "
        "bool has_ori_kv=True, bool has_cmp_kv=True, str device=\"npu\") -> Tensor"
    ),
    # ---------------- npu_scatter_nd_update_v2 ----------------
    # 权威：VF_BINDING L1248；C++ 实现 VF_BINDING L300-L303：
    #        `strides = var.strides()` 后 `EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, update, strides)`
    #        → **strides 是 torch 层内部推导的，不在 schema 里**
    # GE 侧：VF scatter_nd_update_v2_def.cpp L24-L83（输入 var/indices/updates，
    #        输出 var，attr strides(REQUIRED,ListInt) / use_locking）
    # 生产调用点：vllm_fl/.../device/device_op.py L797（位置参数）、
    #        impl/linearnorm/minimax_m3_qknorm_rope.py L164-L185
    #   ⚠️ VA 里对应的是**另一个名字** `npu_scatter_nd_update_sk`（VA_BINDING L3591-L3593），
    #      参数名相同（var/indices/update）——两者不要混用。
    "npu_scatter_nd_update_v2": "(Tensor(a!) var, Tensor indices, Tensor update) -> ()",
}

#: 每条 schema 的**权威出处**（供交付报告 / 自测打印）。格式：`文件:行号`（+ 补充出处）
PROVENANCE_HINT: dict[str, str] = {
    "dispatch_ffn_combine": (
        f"权威: {VF_BINDING} L1160-L1164（TORCH_LIBRARY(_C_ascend) L1145-L1249）；"
        "冲突: VA_BINDING L3225-L3229 多 int world_size=0（本文件以 VF 为准）；"
        f"GE: {VF}csrc/ascend/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_def.cpp L18-L98；"
        f"aclnn 适配: {VF}csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h L21-L93"
    ),
    "moe_gating_top_k": (
        f"权威: {VF_BINDING} L1200-L1203（VA_BINDING L3233-L3247 同名同参）；"
        f"GE: {VF}csrc/ascend/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp L18-L70；"
        f"torch 层实现: {VF}csrc/ascend/moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h L19"
    ),
    "moe_gating_top_k_hash": (
        f"权威: {VF_BINDING} L1204-L1209（13 形参）；"
        "冲突: VA_BINDING L3291-L3310 多 bias_vl/image_sentinel_lo/image_sentinel_count（本文件以 VF 为准）；"
        f"GE: {VF}csrc/ascend/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp L18-L145"
    ),
    "npu_moe_init_routing_custom": (
        f"权威: {VF_BINDING} L1217-L1223（`*` 在 scale/offset 之前 → kwarg-only）；"
        f"GE: {VF}csrc/ascend/moe/moe_init_routing_custom/op_host/moe_init_routing_custom_def.cpp L18-L104"
        "（quant_mode attr 默认 -1）"
    ),
    "grouped_matmul_swiglu_quant_v2": (
        f"权威: {VA_BINDING} L3090-L3095（VF csrc/ascend/ 无此 op → 未找到）；"
        f"GE: {VA}csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp L255；"
        f"aclnn 适配: {VA}csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h L20-L36"
    ),
    "grouped_matmul_swiglu_quant_weight_nz": (
        f"权威: {VA_BINDING} L3075-L3079（VF 无此 op → 未找到）；"
        f"aclnn 适配: {VA}csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/"
        "grouped_matmul_swiglu_quant_torch_adpt.h L20-L60（单张量版，纯别名转发）"
    ),
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        f"权威: {VA_BINDING} L3082-L3088（VF 无此 op → 未找到）；"
        f"GE: {VA}csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_host/"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp L23-L51；"
        f"aclnn 适配: {VA}csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/"
        "grouped_matmul_swiglu_quant_torch_adpt.h L62-L99（bias/offset 会传给 aclnn）"
    ),
    "compressor_metadata": (
        f"权威: {VF_BINDING} L1230（VA_BINDING L3326-L3333 同名同参）；"
        f"GE: {VF}csrc/ascend/attention/compressor_metadata/op_host/compressor_metadata_def.cpp L8-L69"
    ),
    "npu_sparse_attn_sharedkv": (
        f"权威: {VF_BINDING} L1243（VA_BINDING L3367-L3393 同名同参逐字一致）；"
        f"GE: {VF}csrc/ascend/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp L19-L126；"
        f"cmp_ratio 约束: {VF}csrc/ascend/attention/sparse_attn_sharedkv/op_host/"
        "sparse_attn_sharedkv_tiling.cpp L1185-L1193（CFA/SCFA 模式下只接受 {4,128}）"
    ),
    "npu_sparse_attn_sharedkv_metadata": (
        f"权威: {VF_BINDING} L1244（VA_BINDING L3421 写作 `-> (Tensor metadata)`，语义相同）；"
        "⚠️ **无 *_def.cpp**：该算子是 AiCPU 算子 → "
        f"{VF}csrc/ascend/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/ + "
        f"op_graph/sparse_attn_sharedkv_metadata_proto.h L23-L50（REG_OP）"
    ),
    "npu_scatter_nd_update_v2": (
        f"权威: {VF_BINDING} L1248；strides 内部推导见同文件 L300-L303（`var.strides()`）；"
        f"GE: {VF}csrc/ascend/moe/scatter_nd_update_v2/op_host/scatter_nd_update_v2_def.cpp L24-L83"
        "（attr strides REQUIRED）；"
        f"⚠️ VA 里对应另一个名字 npu_scatter_nd_update_sk（{VA_BINDING} L3591-L3593）"
    ),
}

#: 本组有 torch schema 的 11 个注册名（= 12 个短名 − dispatch_ffn_combine_bf16）
REGISTERED_OPS_B = list(SIGS_B)

#: `dispatch_ffn_combine_bf16` **没有权威 torch schema**（VF/VA 都没注册）。
#: 下面这条是**合成的**（仅用于文档 / 单测里 define 到临时命名空间验证参数名），
#: 来源：`dispatch_ffn_combine` 的 schema（同形）+ bf16 变体的调用点签名
#: （`_analysis/triton_ops/dispatch_ffn_combine_bf16.py` L293-L311）+ GE 侧
#: `dispatch_ffn_combine_bf16_def.cpp` L18-L89。**不要注册到 `_C_ascend`。**
SYNTHETIC_SIG_DISPATCH_FFN_COMBINE_BF16 = (
    "(Tensor x, Tensor[] weight1, Tensor[] weight2, "
    "Tensor expert_idx, Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, "
    "Tensor[] bias2, Tensor probs, str group, int max_output_size, "
    "Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, "
    "float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
)


# ===========================================================================
# 2. 实现映射（注册名 → _analysis/triton_ops/ 里的纯 Triton 实现）
# ===========================================================================

IMPL_B: dict[str, tuple[str, str]] = {
    "dispatch_ffn_combine": ("dispatch_ffn_combine", "dispatch_ffn_combine"),
    "moe_gating_top_k": ("moe_gating_top_k", "moe_gating_top_k"),
    "moe_gating_top_k_hash": ("moe_gating_top_k_hash", "moe_gating_top_k_hash"),
    "npu_moe_init_routing_custom": ("moe_init_routing_custom", "moe_init_routing_custom"),
    "grouped_matmul_swiglu_quant_v2": (
        "grouped_matmul_swiglu_quant_v2",
        "grouped_matmul_swiglu_quant_v2",
    ),
    "grouped_matmul_swiglu_quant_weight_nz": (
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
        "grouped_matmul_swiglu_quant_weight_nz",
    ),
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    ),
    "compressor_metadata": ("compressor_metadata", "compressor_metadata"),
    "npu_sparse_attn_sharedkv": ("sparse_attn_sharedkv", "sparse_attn_sharedkv"),
    "npu_sparse_attn_sharedkv_metadata": None,  # 无 Triton 实现 —— 见 IMPL_MISSING
    "npu_scatter_nd_update_v2": ("scatter_nd_update_v2", "scatter_nd_update_v2"),
}

#: `dispatch_ffn_combine` 的 bf16 分支实现（按 weight dtype 分派时使用）
IMPL_DFC_BF16 = ("dispatch_ffn_combine_bf16", "dispatch_ffn_combine_bf16")

#: 没有 Triton 实现、adapter 只能显式拒绝的算子
IMPL_MISSING: dict[str, str] = {
    "npu_sparse_attn_sharedkv_metadata": (
        "本组 39 个 Triton 交付里**没有** sparse_attn_sharedkv_metadata 的实现"
        "（_analysis/triton_ops/ 下无同名文件，sparse_attn_sharedkv.py 的 __all__ 只有 "
        "sparse_attn_sharedkv）。VF 侧该算子是 **AiCPU** 算子"
        "（csrc/ascend/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/ + "
        "op_graph/sparse_attn_sharedkv_metadata_proto.h L23-L50），"
        "没有可复用的 AscendC AI Core 内核，也就没有被重写成 Triton。"
        "adapter 保留完整 schema 签名并显式抛 NotImplementedError（不静默返回假张量）。"
    ),
}

#: 实现文件里已知的 NotImplementedError 分支（出处 = 文件:行）
#:
#: * 绝大多数由 adapter **原样透传**；
#: * ``dispatch_ffn_combine`` 的 ``bias1/bias2`` 两条例外：adapter 按 Lead 裁决
#:   **显式丢弃 bias**（复刻原算子语义，见 :func:`dispatch_ffn_combine` docstring），
#:   因此这两条在 adapter 路径上**不可达**，护栏仍保留在实现里。
NOT_IMPLEMENTED_BRANCHES_B: dict[str, list[str]] = {
    "dispatch_ffn_combine": [
        "dispatch_ffn_combine.py L274 type 与 W8A8 int8 变体不匹配（adapter 透传）",
        "dispatch_ffn_combine.py L276/L281 bias1/bias2 非空 —— ⚠️ **adapter 已按下述裁决丢弃 bias，"
        "在 adapter 路径上不可达**（护栏保留在实现里；直调 triton_ops 仍会拒绝）",
        "dispatch_ffn_combine.py L296/L298 scale1/scale2 缺失（adapter 透传）",
        "dispatch_ffn_combine.py L307 probs 为 None（adapter 透传）",
        "dispatch_ffn_combine.py L322 本地派发行数 > max_output_size（容量裁剪在 hccl dispatch 阶段；adapter 透传）",
        "dispatch_ffn_combine_bf16.py L319/L362 type 不匹配 / 超 max_output_size（adapter 透传）",
        "dispatch_ffn_combine_bf16.py L321/L326 bias 非空 —— ⚠️ 同 int8 分支，adapter 已丢弃 bias",
        "dispatch_ffn_combine_bf16.py L287 scale 非空且非全 1 —— ⚠️ **生产不会触发**"
        "（Lead 已核实：FUSED_MC2 未量化路径传的是空张量占位，fused_moe.py L261-L264；"
        "空张量 allclose 恒 True ⇒ 接受。见 PRODUCTION_CONFLICTS 的 "
        "resolved_scale_empty_placeholder 条目；仍是实现侧护栏，adapter 原样透传 scale）",
        "dispatch_ffn_combine_bf16.py L346 probs 为 None（adapter 透传）",
    ],
    "moe_gating_top_k": [],
    "moe_gating_top_k_hash": [],
    "npu_moe_init_routing_custom": [
        "moe_init_routing_custom.py L287 drop_pad_mode != 0（assert，非 NotImplementedError）",
        "moe_init_routing_custom.py L288 quant_mode ∉ {-1, 1}（assert；docstring L55 明确 quant_mode=0 静态量化不在范围内）",
    ],
    "grouped_matmul_swiglu_quant_v2": [
        "grouped_matmul_swiglu_quant_v2.py L171 bias 非空",
        "grouped_matmul_swiglu_quant_v2.py L176 weight_assist_matrix 非空（W4A8/A4W4 路径）",
        "grouped_matmul_swiglu_quant_v2.py L180 transpose_weight=True",
        "grouped_matmul_swiglu_quant_v2.py L213 smooth_scale 维度 ∉ {1,2}",
    ],
    "grouped_matmul_swiglu_quant_weight_nz": [
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list.py L175 bias 非空",
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list.py L180 offset 非空",
    ],
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": [
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list.py L175 bias 非空",
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list.py L180 offset 非空",
    ],
    "compressor_metadata": [],
    "npu_sparse_attn_sharedkv": [
        "sparse_attn_sharedkv.py L310 layout_q ∉ {BSND,TND}",
        "sparse_attn_sharedkv.py L312 layout_kv ∉ {BSND,PA_ND,PA_BSND,PA_BBND}",
        "sparse_attn_sharedkv.py L314 ori_sparse_indices 非空",
        "sparse_attn_sharedkv.py L316 ori/cmp_kv_stride != 0",
        "sparse_attn_sharedkv.py L318 ori_mask_mode != 4",
        "sparse_attn_sharedkv.py L320 cmp_mask_mode != 3",
        "（另：L321-L324 非法 metadata dtype / ori_kv 与 cmp_kv 同时缺失 → ValueError）",
    ],
    "npu_sparse_attn_sharedkv_metadata": ["本文件 adapter 直接抛 NotImplementedError（无 Triton 实现）"],
    "npu_scatter_nd_update_v2": [
        "scatter_nd_update_v2.py 无 NotImplementedError；adapter 对 var 非连续显式拒绝（见实现注释）",
    ],
}

#: ⚠️ **生产调用点 vs 已交付 Triton 实现**的分歧登记表。
#:
#: ``status`` 取值（**Lead 2026-10-10 裁决后的状态**）：
#:
#: * ``resolved_drop_bias`` —— 原算子就丢弃 ⇒ adapter **显式丢弃**（已实现，附可证伪测试）
#: * ``gap_impl_missing``  —— 生产**会用**、实现**未覆盖** ⇒ adapter 保留**显式拒绝**，
#:   记入缺口清单（:data:`GAPS_B`），**需补实现**（本轮不补，不阻塞主线）
#: * ``not_applicable``    —— 不属于「24 个 AscendC AI Core 算子」范围（如 AiCPU 算子）
PRODUCTION_CONFLICTS = [
    {
        "op": "dispatch_ffn_combine",
        "status": "resolved_drop_bias",
        "production": "moe_comm_method.py L334-L355（enable_fused_mc2=1 时**总是**传非空 "
        "bias1/bias2=w1_scale_bias/w2_scale_bias，并在 L334-L336 assert 它们不为 None）",
        "ascendc_original": "dispatch_ffn_combine_torch_adpt.h：int8 分支 L50-L65 **不把 bias 传给 "
        "aclnn**；bf16/fp16 分支 L69-L70 显式 `(void)bias1; (void)bias2;` → 原算子**丢弃** bias",
        "triton_impl": "dispatch_ffn_combine.py L276/L281、dispatch_ffn_combine_bf16.py L321/L326 "
        "在 bias 非空时抛 NotImplementedError（实现侧的诚实护栏，保留不动）",
        "resolution": "**Lead 裁决（2026-10-10）：adapter 显式丢弃 bias**（复刻原算子语义）——"
        "adapter 给实现传 bias1=None/bias2=None；护栏仍留在 triton_ops 基线里。"
        "配套测试：传非空 bias 与传 None 走**同一条**实现调用（test_ops_b ④c 用 stub 断言收参为 None）。",
        "risk": "⚠️ 若生产未来版本真的开始让 bias 参与计算，这个「丢弃」就是错的 —— "
        "届时应改为透传 bias 并让实现补上 bias 广播。",
    },
    {
        "op": "dispatch_ffn_combine(bf16 分支 scale)",
        "status": "resolved_scale_empty_placeholder",
        "production": "FUSED_MC2 未量化路径：`fused_moe.py` **L259-L260** 把权重包成**单元素 list**"
        "（`w1 = [layer.w13_weight]` ⇒ E=1），**L261-L264** 给 scale/bias 传**空张量占位**："
        "`w1_scale = [torch.tensor([], dtype=torch.int64)]`、"
        "`w1_scale_bias = [torch.tensor([], dtype=torch.float32)]`；L245-L249 的 TODO 注释明确写着"
        "这是「非量化场景」下因 C++ 签名不接受 None 而被迫使用的 placeholder。",
        "ascendc_original": "bf16 分支把 scale1/scale2 传给 aclnn（torch_adpt.h L71-L77），"
        "空张量占位不参与计算",
        "triton_impl": "`dispatch_ffn_combine_bf16.py` L281-L290 `_check_scale_unused`：scale 为 "
        "None / 空 list 直接返回；否则 `_as_stacked` → `.to(torch.float32)` → "
        "`torch.allclose(st, ones_like(st))`。空张量堆叠后形状 (1,0)，与 ones_like 的 allclose "
        "**恒为 True** ⇒ **接受空张量占位**（int64 空张量也走同一条路径）。",
        "resolution": "**Lead 裁决（2026-10-10）：GAP-B5 关闭**（原缺口条目已从 GAPS_B 移除，"
        "仅保留此处登记以追溯）。adapter 对 scale **原样透传**（不转换、不丢弃）；"
        "测试 ④c 覆盖 int64/float32 空张量占位 + 全 1 scale + 负例（非全 1 必须拒绝）。",
        "risk": "⚠️ 仅当传入**非空且非全 1** 的 scale 时才拒绝 —— 当前证据表明不会发生："
        "W8A8 量化走 `weight_dtype == at::kChar` 的 **int8 分支**（torch_adpt.h L50），"
        "到不了 bf16 分支。**残留边界（E 一致性）**：占位 scale 是单元素 list，"
        "而 `_as_stacked` 有 `len(scale) == E` 断言；FUSED_MC2 未 split 时 `w1` 也是单元素 list"
        "（L259-L260 ⇒ E=1）→ 成立。若未来出现「split tensor list（E>1）＋单元素占位 scale」"
        "的组合会命中该断言（实现与调用点的共同约束，非 adapter 引入）。",
    },
    {
        "op": "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
        "status": "gap_impl_missing",
        "production": "moe_mlp.py L360-L368（`bias=bias1`）",
        "ascendc_original": "torch_adpt.h L62-L99：bias/offset **会**传给 "
        "aclnnGroupedMatmulSwigluQuantWeightNzTensorList（**L86-L87**）⇒ 真的参与计算",
        "triton_impl": "L175/L180 bias/offset 非空 → NotImplementedError",
        "resolution": "**Lead 裁决：保留显式拒绝 + 记入缺口清单**（真缺口，需补实现；"
        "本轮目标是集成跑通，不阻塞主线）。",
        "risk": "生产 `_custom_gmm_swiglu_enabled` 路径在 bias1 非 None 时会抛错。",
    },
    {
        "op": "grouped_matmul_swiglu_quant_weight_nz",
        "status": "gap_impl_missing",
        "production": "device_op.py L225-L233（kwargs 含 bias）",
        "ascendc_original": "单张量版走同一 aclnn 系列，bias **真的参与计算**"
        "（torch_adpt.h L20-L51 单张量版把 bias 传给 aclnn）",
        "triton_impl": "复用 tensor_list 实现（同一函数），bias 非空 → NotImplementedError",
        "resolution": "同上：保留显式拒绝 + 记入缺口清单（需补实现）。",
        "risk": "bias 非 None 的调用会抛错。",
    },
    {
        "op": "grouped_matmul_swiglu_quant_v2",
        "status": "gap_impl_missing",
        "production": "moe_mlp.py L343-L353（W4A8 路径：`weight_assist_matrix=bias1`、`dequant_mode=0`）",
        "ascendc_original": "v2 支持 W4A8/A4W4 的 assist matrix（int4 打包），"
        "且 `bias_real = bias.value_or(at::Tensor())` **真的传给 aclnn**"
        "（grouped_matmul_swiglu_quant_v2_torch_adpt.h L28/L49/L59）",
        "triton_impl": "L176 weight_assist_matrix 非空 → NotImplementedError（该实现只覆盖 int8/A8W8）",
        "resolution": "同上：保留显式拒绝 + 记入缺口清单（需补实现）。",
        "risk": "`use_w4a8_per_channel_gmm_swiglu` 为真时抛错；W8A8 路径（不传 assist matrix）不受影响。",
    },
    {
        "op": "npu_sparse_attn_sharedkv_metadata",
        "status": "not_applicable",
        "production": "device_op.py L767-L775 + dsa_v1.py L787-L802（kwargs，含 device）",
        "ascendc_original": "**AiCPU 算子**（op_kernel_aicpu/ + op_graph/..._proto.h L23-L50），"
        "没有 AI Core 内核 ⇒ 本来就不属于「24 个 AscendC 算子」范围",
        "triton_impl": "无 Triton 实现（_analysis/triton_ops/ 下无对应文件）",
        "resolution": "**Lead 裁决：如实留缺口** —— adapter 保留完整 schema 签名 + 调用时显式抛 "
        "NotImplementedError（不返回假张量）。",
        "risk": "DSA 路径若真的调用该 metadata op，FlagGems 接管后会抛错。",
    },
]

#: 📋 **缺口清单**（Lead 要求「必须如实写进缺口清单，不能让它看起来已覆盖」）。
#: 每条：`(id, 算子, 类型, 生产会用?, 是否已实现, 建议)`
GAPS_B: list[dict[str, str]] = [
    {
        "id": "GAP-B1",
        "op": "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
        "kind": "实现未覆盖（生产会用）",
        "production_site": "moe_mlp.py L360-L368（bias=bias1）",
        "implemented": "否 —— adapter 显式拒绝（bias/offset 非空）",
        "action": "需补实现：读 op_kernel 的 bias/offset 广播规则并重新验证（Lead 排期到 M1-M4 之后）",
    },
    {
        "id": "GAP-B2",
        "op": "grouped_matmul_swiglu_quant_weight_nz",
        "kind": "实现未覆盖（生产会用）",
        "production_site": "device_op.py L225-L233（bias）",
        "implemented": "否 —— 与 GAP-B1 共用实现，同样显式拒绝",
        "action": "同 GAP-B1（补 bias/offset 后两者一起解决）",
    },
    {
        "id": "GAP-B3",
        "op": "grouped_matmul_swiglu_quant_v2",
        "kind": "实现未覆盖（生产会用，W4A8/A4W4 路径）",
        "production_site": "moe_mlp.py L343-L353（weight_assist_matrix=bias1, dequant_mode=0）",
        "implemented": "否 —— 只覆盖 int8/A8W8；assist matrix 非空即拒绝",
        "action": "需补实现：int4 打包的 assist matrix + bias（新工作，需读 arch35 op_kernel）",
    },
    {
        "id": "GAP-B4",
        "op": "npu_sparse_attn_sharedkv_metadata",
        "kind": "AiCPU 算子，无 Triton 实现（不适用）",
        "production_site": "device_op.py L767-L775 + dsa_v1.py L787-L802",
        "implemented": "否 —— adapter 保留完整 schema 签名后显式抛 NotImplementedError",
        "action": "如确需接管，需按 AiCPU 逻辑另写 host 侧实现（不属于本组 Triton 交付范围）",
    },
    {
        "id": "GAP-B6",
        "op": "npu_moe_init_routing_custom",
        "kind": "实现未覆盖（schema 默认值路径）",
        "production_site": "schema 默认 quant_mode=0（静态量化）；生产 device_op.py L116-L126 总显式传参（默认 -1）",
        "implemented": "否 —— adapter 对 quant_mode=0 显式抛 NotImplementedError",
        "action": "若未来有调用点使用静态量化，需补实现",
    },
    # 注：原 GAP-B5（dispatch_ffn_combine bf16 分支 scale 分歧）已由 Lead 核实并**关闭**，
    #     编号保留空缺以维持追溯；登记已迁到 PRODUCTION_CONFLICTS 的
    #     `status="resolved_scale_empty_placeholder"` 条目（生产传空张量占位 ⇒ 实现接受）。
]


# ===========================================================================
# 3. 实现加载器（懒加载，模块 import 期不需要 torch / triton）
# ===========================================================================

#: Triton 实现目录（默认 `<repo>/_analysis/triton_ops`，可用环境变量覆盖）
TRITON_OPS_DIR = Path(
    os.environ.get(
        "FLAGGEMS_TRITON_OPS_DIR",
        str(Path(__file__).resolve().parents[1] / "triton_ops"),
    )
)

_IMPL_CACHE: dict[tuple[str, str], object] = {}


def _load_impl(module: str, func: str):
    """按需加载 ``_analysis/triton_ops/<module>.py`` 的 ``<func>``。

    这些实现文件用**相对 import**（``from .common import ...``），所以必须作为包
    ``triton_ops.<module>`` 导入 —— 因此这里把 ``triton_ops`` 的**父目录**加入
    ``sys.path``（只在缺失时插入一次）。
    """
    key = (module, func)
    if key in _IMPL_CACHE:
        return _IMPL_CACHE[key]

    root = TRITON_OPS_DIR
    if not (root / f"{module}.py").is_file():
        raise NotImplementedError(
            f"找不到 Triton 实现 {root / (module + '.py')}；"
            f"可用环境变量 FLAGGEMS_TRITON_OPS_DIR 指向 _analysis/triton_ops 所在目录。"
        )
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod = importlib.import_module(f"{root.name}.{module}")
    fn = getattr(mod, func, None)
    if fn is None:
        raise NotImplementedError(f"{root.name}.{module} 里没有 {func}()")
    _IMPL_CACHE[key] = fn
    return fn


def _impl_for(op_name: str):
    """取某注册名的 Triton 实现（无实现 → NotImplementedError）。"""
    spec = IMPL_B.get(op_name)
    if spec is None:
        raise NotImplementedError(IMPL_MISSING.get(op_name, f"{op_name} 没有 Triton 实现"))
    return _load_impl(*spec)


def _torch():
    import torch  # 局部 import：保证本模块在无 torch 的宿主机上仍可 import

    return torch


# ===========================================================================
# 4. adapter（12 个；参数名 / 顺序 / kwarg-only 边界与 schema 对齐）
# ===========================================================================
# 规则：
#   * 函数体**只做参数映射 + 必要的「超范围取值」显式拒绝**，没有任何 try/except；
#   * 传参一律用**关键字**，保证参数名与实现签名同名，改名会在第一次调用时立刻暴露；
#   * 每个不支持的取值分支都在注释里指出实现文件:行号。


def dispatch_ffn_combine(
    x,
    weight1,
    weight2,
    expert_idx,
    scale1,
    scale2,
    bias1,
    bias2,
    probs,
    group,
    max_output_size,
    out,
    expert_token_nums,
    x_active_mask=None,
    swiglu_limit=1000000.0,
    *,
    world_size=0,
    type=None,
    local_expert_base=0,
):
    """接管 ``_C_ascend::dispatch_ffn_combine``（schema: SIGS_B，VF_BINDING L1160-L1164）。

    **按 weight dtype 分派**（复刻 VF ``dispatch_ffn_combine_torch_adpt.h`` L41/L50/L66 的
    C++ 行为，也是 task-12 的建议，见 ``_C_ascend_sigs.py`` NOT_REGISTERED 注释）：

    * ``weight1[0].dtype == torch.int8`` → ``triton_ops/dispatch_ffn_combine.py``（W8A8）
    * ``weight1[0].dtype ∈ {bfloat16, float16}`` → ``triton_ops/dispatch_ffn_combine_bf16.py``

    已实测精度（bf16 变体）：**7/7 PASS**，判据为「按幅度分段」——
    ``|ref|max<=1`` 用 README 绝对地板（bf16 4e-3），``|ref|max>1`` 用输出 dtype 完整 1 ULP；
    判据演进史见 ``_analysis/mc2_后端问题记录.md`` **第七节**。

    **bias 语义对齐（Lead 裁决 2026-10-10）**：原 AscendC 算子的**两个分支都丢弃
    ``bias1``/``bias2``** —— int8 分支压根没把 bias 放进 aclnn 参数列表
    （``dispatch_ffn_combine_torch_adpt.h`` **L50-L65**），bf16/fp16 分支显式
    ``(void)bias1; (void)bias2;``（同文件 **L69-L70**）；而生产调用点
    （``vllm_fl/.../moe/moe_comm_method.py`` **L339-L355**）**总是**传非空
    ``bias1/bias2 = w1_scale_bias/w2_scale_bias``（L334-L336 还 assert 它们不为 None）。
    所以本 adapter **显式给实现传 None**，复刻「丢弃」语义；实现里
    「bias 非空即拒绝」的安全护栏仍然保留（诚实），只是不再挡在生产路径上。

    ⚠️ **风险（必须随实现一起演进）**：若生产未来版本真的开始让 bias 参与计算，
    这个「丢弃」就是错的 —— 届时应改为把 bias 透传给实现、并让实现补上 bias 广播。
    记录见 :data:`GAPS_B`。

    透传的 NotImplementedError：type / scale1 / scale2 / probs / max_output_size
    （见 :data:`NOT_IMPLEMENTED_BRANCHES_B`）。⚠️ bf16 分支的 scale 分歧**已由 Lead 核实关闭**
    （生产 FUSED_MC2 未量化路径传**空张量占位**：``fused_moe.py`` L245-L249 TODO + L259-L264，
    实现的 ``_check_scale_unused``（``dispatch_ffn_combine_bf16.py`` L281-L290）对空张量
    ``allclose`` 恒为 True ⇒ 接受）——adapter 对 scale **原样透传**，见
    :data:`PRODUCTION_CONFLICTS` 的 ``resolved_scale_empty_placeholder`` 条目与测试 ④c。
    """
    torch = _torch()
    if int(world_size) != 0:
        # VA_BINDING L3228 的 extension 参数（VF schema 没有）；跨卡 hccl 不在实现范围内
        raise NotImplementedError(
            f"world_size={world_size}：仅 VA schema 有此参数，且对应跨卡 hccl 通信，"
            "不在纯 Triton 本地计算实现范围内（VF_BINDING L1160-L1164 无此参数）。"
        )
    w0 = weight1[0] if isinstance(weight1, (list, tuple)) else weight1
    dtype = w0.dtype
    if dtype == torch.int8:
        impl = _load_impl("dispatch_ffn_combine", "dispatch_ffn_combine")
    elif dtype in (torch.bfloat16, torch.float16):
        impl = _load_impl(*IMPL_DFC_BF16)
    else:
        # 原 C++ 只有 at::kChar / at::kBFloat16 / at::kHalf 三条路
        raise NotImplementedError(
            f"weight1[0].dtype={dtype}：原实现只分派 int8(W8A8, torch_adpt.h L50) 与 "
            f"bf16/fp16(torch_adpt.h L66)，本 adapter 同样只覆盖这两种。"
        )
    return impl(
        x=x,
        weight1=weight1,
        weight2=weight2,
        expert_idx=expert_idx,
        scale1=scale1,
        scale2=scale2,
        # ⚠️ 原算子两分支都丢弃 bias1/bias2（torch_adpt.h L50-L65 int8 未传入 aclnn；
        #    L69-L70 bf16 分支 `(void)bias1; (void)bias2;`）→ 这里显式传 None 复刻该语义
        #    （生产 moe_comm_method.py L339-L355 总是传非空 bias）。详见 docstring 与 GAPS_B。
        bias1=None,
        bias2=None,
        probs=probs,
        group=group,
        max_output_size=max_output_size,
        out=out,
        expert_token_nums=expert_token_nums,
        x_active_mask=x_active_mask,
        swiglu_limit=swiglu_limit,
        type=type,
        local_expert_base=local_expert_base,
    )


def dispatch_ffn_combine_bf16(
    x,
    weight1,
    weight2,
    expert_idx,
    scale1,
    scale2,
    bias1,
    bias2,
    probs,
    group,
    max_output_size,
    out,
    expert_token_nums,
    x_active_mask=None,
    swiglu_limit=1000000.0,
    *,
    type=None,
    local_expert_base=0,
):
    """**没有权威 torch schema**（VF/VA 都只注册 ``dispatch_ffn_combine``）的 bf16 变体 adapter。

    参数名与 ``dispatch_ffn_combine`` 同形（来自调用点签名 + bf16 def.cpp L18-L89），
    可直接调用 / 单测；**不要**把它 define 成 ``_C_ascend::dispatch_ffn_combine_bf16``
    （那会造出一个生产环境不存在的 op）。接管入口仍是 ``dispatch_ffn_combine`` + dtype 分派。

    精度判据说明：见 ``_analysis/mc2_后端问题记录.md`` 第七节（按幅度分段，实测 7/7 PASS）。

    **bias 语义与 ``dispatch_ffn_combine`` 完全一致**：原 bf16 分支显式
    ``(void)bias1; (void)bias2;``（``dispatch_ffn_combine_torch_adpt.h`` **L69-L70**）
    → 本 adapter 同样**丢弃** bias（传 None），理由与风险见
    :func:`dispatch_ffn_combine` 的 docstring 与 :data:`GAPS_B`。
    """
    impl = _load_impl(*IMPL_DFC_BF16)
    return impl(
        x=x,
        weight1=weight1,
        weight2=weight2,
        expert_idx=expert_idx,
        scale1=scale1,
        scale2=scale2,
        # ⚠️ 复刻原 bf16 分支的 `(void)bias1; (void)bias2;`（torch_adpt.h L69-L70）
        bias1=None,
        bias2=None,
        probs=probs,
        group=group,
        max_output_size=max_output_size,
        out=out,
        expert_token_nums=expert_token_nums,
        x_active_mask=x_active_mask,
        swiglu_limit=swiglu_limit,
        type=type,
        local_expert_base=local_expert_base,
    )


def moe_gating_top_k(
    x,
    k,
    k_group,
    group_count,
    group_select_mode,
    renorm,
    norm_type,
    out_flag,
    routed_scaling_factor,
    eps,
    bias_opt=None,
):
    """接管 ``_C_ascend::moe_gating_top_k``（schema: SIGS_B，VF_BINDING L1200-L1203）。

    实现：``triton_ops/moe_gating_top_k.py`` L168（签名里有 ``bias``/``bias_opt`` 两个别名，
    torch schema 用的是 ``bias_opt``，这里只传 ``bias_opt``）。
    生产调用点：``device_op.py`` L147-L159（全 kwargs，含 ``bias_opt``）。
    无 NotImplementedError 分支（见 NOT_IMPLEMENTED_BRANCHES_B）。
    """
    impl = _impl_for("moe_gating_top_k")
    return impl(
        x=x,
        k=k,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
        bias_opt=bias_opt,
    )


def moe_gating_top_k_hash(
    x,
    k,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    routed_scaling_factor=1.0,
    eps=1e-20,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    *,
    bias_vl=None,
    image_sentinel_lo=129257,
    image_sentinel_count=5,
):
    """接管 ``_C_ascend::moe_gating_top_k_hash``（schema: SIGS_B，VF_BINDING L1204-L1209）。

    ⚠️ 参数顺序按 **VF**（``k`` 在第 2 位）；实现 ``triton_ops/moe_gating_top_k_hash.py``
    L280 的参数顺序不同（``k`` 在第 5 位）→ 因此这里**全部按关键字传参**。

    末尾 3 个 keyword-only 参数是 **VA_BINDING L3306-L3308 的扩展**（VL 场景），
    VF schema 里没有；Triton 实现也不支持 → 非默认值时显式抛 NotImplementedError。
    生产调用点：``experts_selector.py`` L266。
    """
    if bias_vl is not None or int(image_sentinel_lo) != 129257 or int(image_sentinel_count) != 5:
        raise NotImplementedError(
            "bias_vl / image_sentinel_lo / image_sentinel_count 是 VA_BINDING L3306-L3308 "
            "的 VL 扩展参数，VF schema（L1204-L1209，本次接管依据）没有它们，"
            "已交付的 Triton 实现（moe_gating_top_k_hash.py L280）也不支持。"
        )
    impl = _impl_for("moe_gating_top_k_hash")
    return impl(
        x=x,
        k=k,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=k_group,
        group_count=group_count,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
    )


def moe_init_routing_custom(
    x,
    expert_idx,
    *,
    scale=None,
    offset=None,
    active_num=-1,
    expert_capacity=-1,
    expert_num=-1,
    drop_pad_mode=0,
    expert_tokens_num_type=0,
    expert_tokens_num_flag=False,
    quant_mode=0,
    active_expert_range=(),
    row_idx_type=0,
):
    """接管 ``_C_ascend::npu_moe_init_routing_custom``（schema: SIGS_B，VF_BINDING L1217-L1223）。

    ⚠️ schema 里 ``*`` 在 ``scale`` 之前 → ``scale``/``offset`` 都是 **kwarg-only**，
    所以本 adapter 也把 ``*`` 放在同一位置（位置参数只有 ``x``/``expert_idx``）。
    生产调用点：``device_op.py`` L116-L126、``device_operator.py`` L169-L179（全 kwargs）。

    ⚠️ ``quant_mode`` 的 schema 默认值是 **0（静态量化）**，而 GE 侧 def.cpp L95 的
    attr 默认是 ``-1``（非量化）。已交付的 Triton 实现只覆盖 ``quant_mode ∈ {-1, 1}``
    （``moe_init_routing_custom.py`` docstring L55、assert L288）→ 默认值 0 会显式抛错。
    生产调用点总是显式传 ``quant_mode``（默认 -1），所以不受影响。
    """
    if int(quant_mode) == 0:
        raise NotImplementedError(
            "quant_mode=0（静态量化）不在已交付 Triton 实现范围内："
            "moe_init_routing_custom.py docstring L55 / assert L288 只覆盖 quant_mode ∈ {-1, 1}。"
            "注意 torch schema（VF_BINDING L1222）的默认值是 0，而 GE def.cpp L95 的默认是 -1；"
            "调用方需显式传 quant_mode=-1（非量化）或 1（动态量化）。"
        )
    impl = _impl_for("npu_moe_init_routing_custom")
    return impl(
        x=x,
        expert_idx=expert_idx,
        scale=scale,
        offset=offset,
        active_num=active_num,
        expert_capacity=expert_capacity,
        expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag,
        quant_mode=quant_mode,
        active_expert_range=list(active_expert_range),
        row_idx_type=row_idx_type,
    )


def grouped_matmul_swiglu_quant_v2(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=0,
    quant_mode=0,
    quant_dtype=0,
    transpose_weight=False,
    group_list_type=0,
    tuning_config=(),
    swiglu_limit=0.0,
):
    """接管 ``_C_ascend::grouped_matmul_swiglu_quant_v2``（schema: SIGS_B，VA_BINDING L3090-L3095）。

    实现：``triton_ops/grouped_matmul_swiglu_quant_v2.py`` L147（只覆盖 int8/A8W8）。
    schema 的 ``tuning_config`` 默认是 ``[]``，实现不用它，这里用等价的 ``()`` 作默认，并原样透传。

    透传的 NotImplementedError（L171/L176/L180/L213）：bias 非空 / weight_assist_matrix 非空
    （W4A8/A4W4）/ transpose_weight=True / smooth_scale 维度不支持。
    ⚠️ 生产 ``moe_mlp.py`` L343-L353 的 W4A8 路径传 ``weight_assist_matrix=bias1`` → 会抛错，
    见 PRODUCTION_CONFLICTS[3]。
    """
    impl = _impl_for("grouped_matmul_swiglu_quant_v2")
    return impl(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        smooth_scale=smooth_scale,
        weight_assist_matrix=weight_assist_matrix,
        bias=bias,
        dequant_mode=dequant_mode,
        dequant_dtype=dequant_dtype,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        transpose_weight=transpose_weight,
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        swiglu_limit=swiglu_limit,
    )


def grouped_matmul_swiglu_quant_weight_nz(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    bias=None,
    offset=None,
    swiglu_limit=0.0,
):
    """接管 ``_C_ascend::grouped_matmul_swiglu_quant_weight_nz``（schema: SIGS_B，VA_BINDING L3075-L3079）。

    实现：``triton_ops/grouped_matmul_swiglu_quant_weight_nz_tensor_list.py`` L241
    （本仓库无独立 AscendC 内核 → 与 tensor_list 版共用数学，见该文件 docstring L1-L60）。
    schema 里 ``bias``/``offset``/``swiglu_limit`` 在 ``*`` 之后 → adapter 同样 kwarg-only。
    生产调用点：``device_op.py`` L225-L233。
    透传的 NotImplementedError：bias 非空 / offset 非空（L175/L180）。
    """
    impl = _impl_for("grouped_matmul_swiglu_quant_weight_nz")
    return impl(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        bias=bias,
        offset=offset,
        swiglu_limit=swiglu_limit,
    )


def grouped_matmul_swiglu_quant_weight_nz_tensor_list(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    bias=None,
    offset=None,
    swiglu_limit=0.0,
):
    """接管 ``_C_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list``
    （schema: SIGS_B，VA_BINDING L3082-L3088）。

    实现：``triton_ops/grouped_matmul_swiglu_quant_weight_nz_tensor_list.py`` L227。
    生产调用点：``moe_mlp.py`` L175-L183 / L360-L368（kwargs，含 ``bias``）。
    透传的 NotImplementedError：bias 非空 / offset 非空（L175/L180）——
    注意 AscendC 侧 aclnn **确实接收** bias/offset（torch_adpt.h L82-L94），
    见 PRODUCTION_CONFLICTS[1]。
    """
    impl = _impl_for("grouped_matmul_swiglu_quant_weight_nz_tensor_list")
    return impl(
        x=x,
        weight=weight,
        weight_scale=weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        bias=bias,
        offset=offset,
        swiglu_limit=swiglu_limit,
    )


def compressor_metadata(
    rope_cos,
    rope_sin,
    cu_seqlens,
    start_pos,
    kv_block_table,
    kv_block_size,
    slot_mapping_format,
    compress_ratio,
    num_compressed_tokens,
    num_reqs_actual,
):
    """接管 ``_C_ascend::compressor_metadata``（schema: SIGS_B，VF_BINDING L1230）。

    **名字映射**（schema 名 → 实现名，``triton_ops/compressor_metadata.py`` L178）：

    ==================== ==================
    schema（torch 层）    实现（AscendC 层）
    ==================== ==================
    ``compress_ratio``    ``cmp_ratio``
    ``num_compressed_tokens`` ``num_rows``（= 输出行数，实现 docstring L198）
    ``num_reqs_actual``   ``actual_num_reqs``
    ==================== ==================

    GE 侧 def.cpp L58-L61 的 attr 正是 ``kvBlockSize``/``slotMappingFormat``/``cmpRatio``/
    ``actualNumReqs``，且 **没有** num_compressed_tokens（它由输出行数推得）。
    生产调用点 ``dsa_v1.py`` L1555-L1566 是**位置参数**（10 个，顺序同 schema）。
    无 NotImplementedError 分支；实现里是 assert（形状/ dtype）。
    """
    impl = _impl_for("compressor_metadata")
    return impl(
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cu_seqlens=cu_seqlens,
        start_pos=start_pos,
        kv_block_table=kv_block_table,
        num_rows=num_compressed_tokens,
        kv_block_size=kv_block_size,
        slot_mapping_format=slot_mapping_format,
        cmp_ratio=compress_ratio,
        actual_num_reqs=num_reqs_actual,
    )


def sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=0,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=128,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
):
    """接管 ``_C_ascend::npu_sparse_attn_sharedkv``（schema: SIGS_B，VF_BINDING L1243）。

    实现：``triton_ops/sparse_attn_sharedkv.py`` L265。生产调用点 ``dsa_v1.py``
    L1988-L2004（``q`` 位置 + 其余全 kwargs）。

    两处 adapter 级检查（都**不是**吞错，而是把原算子的契约显式化）：

    1. **非法 ``cmp_ratio``**：GE 侧 tiling 只在 CFA/SCFA 模式（即给了 ``cmp_kv``）时
       校验 ``cmp_ratio ∈ {4, 128}``（``sparse_attn_sharedkv_tiling.cpp`` L1185-L1193）。
       纯 Triton 实现本身不校验 → adapter 在这里镜像原契约；未给 ``cmp_kv``（SWA 模式）
       时 ``cmp_ratio`` 不参与计算，原样透传（生产 ``dsa_v1.py`` L1986-L2004 就是这种用法）。
    2. **``return_softmax_lse=True``**：AscendC/torch 契约返回 ``softmax_lse``
       （fp32，形状 = ``q.shape[:-1] + [1]``，见 VF_BINDING L146-L153），
       而 Triton 实现返回**两个**张量 ``(out, lse_max, lse_sum)``，且
       ``lse_max`` 形状是 ``[B,N2,S1,gs]``/``[N2,T,gs]``、语义是**未取对数的分母和**
       （``sparse_attn_sharedkv.py`` L431-L439 与其测试 L76-L84）。两者形状/语义都不同，
       没有经过验证的映射 → 显式拒绝，避免静默给出错误结果。生产调用点只用返回值的
       ``[0]`` 且从不传 ``return_softmax_lse=True``，不受影响。

    其余 NotImplementedError 全部来自实现（L310/L312/L314/L316/L318/L320）并**原样透传**。
    """
    if cmp_kv is not None and int(cmp_ratio) not in (4, 128):
        raise NotImplementedError(
            f"cmp_ratio={cmp_ratio}：原算子 tiling 在 CFA/SCFA 模式（有 cmp_kv）下只接受 "
            "cmp_ratio ∈ {4, 128}（VF sparse_attn_sharedkv_tiling.cpp L1185-L1193）。"
            "已交付的 Triton 实现不校验该值 → adapter 镜像原契约显式拒绝。"
        )
    if return_softmax_lse:
        raise NotImplementedError(
            "return_softmax_lse=True：AscendC/torch 契约的 softmax_lse 是 fp32 "
            "q.shape[:-1]+[1]（VF_BINDING L146-L153）；Triton 实现返回 "
            "(out, lse_max, lse_sum)，lse_max 形状 [B,N2,S1,gs]/[N2,T,gs]、语义为未取对数的"
            "分母和（sparse_attn_sharedkv.py L431-L439）—— 无已验证映射，显式拒绝以免静默算错。"
        )
    impl = _impl_for("npu_sparse_attn_sharedkv")
    out, lse_max, _lse_sum = impl(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=False,  # 上面已拒绝 True；实现据此返回空 lse（与契约一致）
    )
    return out, lse_max


def sparse_attn_sharedkv_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=4,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=128,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    has_ori_kv=True,
    has_cmp_kv=True,
    device="npu",
):
    """接管 ``_C_ascend::npu_sparse_attn_sharedkv_metadata``（schema: SIGS_B，VF_BINDING L1244）。

    ⚠️ **没有 Triton 实现**（``_analysis/triton_ops/`` 下无对应文件）。VF 侧该算子是
    **AiCPU** 算子（``op_kernel_aicpu/`` + ``op_graph/sparse_attn_sharedkv_metadata_proto.h``
    L23-L50），没有 AscendC AI Core 内核可移植。adapter 保留**完整 schema 签名**
    （这样 ``torch.library.define`` 能成功、kwargs 调用点不会因签名不符而静默落到别处），
    调用时显式抛 NotImplementedError，不返回假张量、不静默降级。

    生产调用点：``device_op.py`` L767-L775（把该 op 当可调用对象返回）+ ``dsa_v1.py``
    L787-L802（kwargs，含 ``device``）。
    """
    raise NotImplementedError(IMPL_MISSING["npu_sparse_attn_sharedkv_metadata"])


def scatter_nd_update_v2(var, indices, update):
    """接管 ``_C_ascend::npu_scatter_nd_update_v2``（schema: SIGS_B，VF_BINDING L1248）。

    schema 只有 3 个形参（``var``/``indices``/``update``），而 GE 侧 def.cpp L71 要求
    ``strides``(REQUIRED, ListInt) —— 原 C++ 实现是**内部推导**的：
    VF_BINDING L300-L303 ``at::IntArrayRef var_stride = var.strides();`` 然后传给 aclnn。
    adapter 复刻该行为，并只取前 ``K = indices.shape[-1]`` 个 stride（实现断言
    ``len(strides) == K``；前 K 个 stride 已包含尾部维度乘积，线性索引语义不变）。

    **原地语义**：schema 是 ``Tensor(a!) var ... -> ()``。实现返回的是新张量，
    因此这里传 ``out=var`` 让它直接写回；若实现因 ``var`` 非连续而返回了副本，
    再 ``var.copy_(y)`` 保证 schema 的原地语义成立。

    ``var`` 非连续时（生产调用点确实出现过）：在 **连续副本** 上计算，再把结果
    ``copy_`` 回原来的步幅视图。⚠️ 此时必须传**副本自己的 stride**——原实现是按
    ``strides`` 做线性寻址的，传原张量的 stride 会与副本的内存布局不符。
    2026-10-12 实测：DeepSeek-V4 生成阶段即传非连续 ``var``，旧版直接
    ``NotImplementedError`` 让整个引擎挂掉。

    无 NotImplementedError 分支（``scatter_nd_update_v2.py`` 内没有）。
    """
    k = int(indices.shape[-1])
    if var.is_contiguous():
        impl = _impl_for("npu_scatter_nd_update_v2")
        y = impl(var, indices, update, [int(s) for s in var.stride()[:k]], out=var)
        if y is not var:
            var.copy_(y)
        return None
    base = var.detach().contiguous()
    impl = _impl_for("npu_scatter_nd_update_v2")
    y = impl(base, indices, update, [int(s) for s in base.stride()[:k]], out=base)
    var.copy_(y if y is not None else base)
    return None


# ===========================================================================
# 5. 注册表
# ===========================================================================

#: 注册名 → adapter（11 个注册 op；`dispatch_ffn_combine_bf16` 见 ADAPTERS_B_BY_TASK_NAME）
ADAPTERS_B: dict[str, object] = {
    "dispatch_ffn_combine": dispatch_ffn_combine,
    "moe_gating_top_k": moe_gating_top_k,
    "moe_gating_top_k_hash": moe_gating_top_k_hash,
    "npu_moe_init_routing_custom": moe_init_routing_custom,
    "grouped_matmul_swiglu_quant_v2": grouped_matmul_swiglu_quant_v2,
    "grouped_matmul_swiglu_quant_weight_nz": grouped_matmul_swiglu_quant_weight_nz,
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        grouped_matmul_swiglu_quant_weight_nz_tensor_list
    ),
    "compressor_metadata": compressor_metadata,
    "npu_sparse_attn_sharedkv": sparse_attn_sharedkv,
    "npu_sparse_attn_sharedkv_metadata": sparse_attn_sharedkv_metadata,
    "npu_scatter_nd_update_v2": scatter_nd_update_v2,
}

#: 短名 → adapter（12 个，含没有 schema 的 `dispatch_ffn_combine_bf16`）
ADAPTERS_B_BY_TASK_NAME: dict[str, object] = {
    "dispatch_ffn_combine": dispatch_ffn_combine,
    "dispatch_ffn_combine_bf16": dispatch_ffn_combine_bf16,
    "moe_gating_top_k": moe_gating_top_k,
    "moe_gating_top_k_hash": moe_gating_top_k_hash,
    "moe_init_routing_custom": moe_init_routing_custom,
    "grouped_matmul_swiglu_quant_v2": grouped_matmul_swiglu_quant_v2,
    "grouped_matmul_swiglu_quant_weight_nz": grouped_matmul_swiglu_quant_weight_nz,
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": (
        grouped_matmul_swiglu_quant_weight_nz_tensor_list
    ),
    "compressor_metadata": compressor_metadata,
    "sparse_attn_sharedkv": sparse_attn_sharedkv,
    "sparse_attn_sharedkv_metadata": sparse_attn_sharedkv_metadata,
    "scatter_nd_update_v2": scatter_nd_update_v2,
}


def to_patch_util_signatures() -> dict[str, dict[str, str]]:
    """返回可直接 ``update()`` 进 ``patch_util._VLLM_OPS_SIGNATURES`` 的字典。"""
    return {LIB_NAME: dict(SIGS_B)}


def to_patch_util_adapters() -> dict[str, dict[str, object]]:
    """返回 ``{lib_name: {op_name: adapter}}``（FlagGems/Lead 侧自行 impl）。"""
    return {LIB_NAME: dict(ADAPTERS_B)}


# ===========================================================================
# 6. schema 解析 / 签名自检（**不需要 torch**）
# ===========================================================================


def _split_top_level(text: str) -> list[str]:
    """按顶层逗号切分（忽略 ``[]``/``()``/``{}`` 内的逗号）。"""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse_schema_params(signature: str) -> dict[str, list[str]]:
    """从 patch_util 风格 signature 里解析形参名（不依赖 torch）。

    返回 ``{"positional": [...], "kwarg_only": [...], "all": [...]}``。
    ``Tensor(a!) var`` / ``Tensor[]? weight`` / ``int[2] tuning_config=[]`` 这类都能正确取到名字。
    """
    body = signature.split("->")[0].strip()
    if "(" not in body or ")" not in body:
        raise ValueError(f"schema 不合法: {signature!r}")
    inner = body[body.index("(") + 1 : body.rindex(")")]
    positional: list[str] = []
    kwarg_only: list[str] = []
    seen_star = False
    for chunk in _split_top_level(inner):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "*":
            seen_star = True
            continue
        decl = chunk.split("=")[0].strip()
        name = decl.split()[-1]
        if seen_star:
            kwarg_only.append(name)
        else:
            positional.append(name)
    return {"positional": positional, "kwarg_only": kwarg_only, "all": positional + kwarg_only}


def _schema_required_count(signature: str) -> int:
    """schema 里**必填位置参数**的个数（没有 ``=默认值`` 的）。"""
    body = signature.split("->")[0].strip()
    inner = body[body.index("(") + 1 : body.rindex(")")]
    n = 0
    for chunk in _split_top_level(inner):
        chunk = chunk.strip()
        if not chunk or chunk == "*":
            continue
        if "=" not in chunk:
            n += 1
    return n


def check_adapter_signatures(verbose: bool = False) -> list[str]:
    """静态校验 11 条 schema 与 adapter 的参数名/顺序/kwarg-only 边界一致。

    检查项：
      1. adapter 不得有 ``*args`` / ``**kwargs``（否则 ``inspect.signature`` 无法比对，
         也说明没有真的对齐 schema）；
      2. schema 的**位置参数**必须是 adapter 位置参数序列的前缀（顺序一致）；
      3. schema 的 **kwarg-only 参数**必须是 adapter kwarg-only 序列的前缀（顺序一致）；
      4. 必填位置参数个数一致（默认值不能被搬走）；
      5. adapter 多出来的参数**必须都有默认值**（本组的 extras：VA 扩展 / 实现专用）。

    返回问题列表（空 = 全部通过）。
    """
    problems: list[str] = []
    for op_name, signature in SIGS_B.items():
        fn = ADAPTERS_B[op_name]
        params = list(inspect.signature(fn).parameters.values())
        if any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params
        ):
            problems.append(f"{op_name}: adapter 出现 *args/**kwargs，无法与 schema 对齐")
            continue
        pos = [
            p.name
            for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        kw = [p.name for p in params if p.kind == p.KEYWORD_ONLY]
        want = parse_schema_params(signature)

        if pos[: len(want["positional"])] != want["positional"]:
            problems.append(
                f"{op_name}: 位置参数不匹配 schema\n"
                f"    schema  : {want['positional']}\n"
                f"    adapter : {pos}"
            )
        if kw[: len(want["kwarg_only"])] != want["kwarg_only"]:
            problems.append(
                f"{op_name}: kwarg-only 参数不匹配 schema\n"
                f"    schema  : {want['kwarg_only']}\n"
                f"    adapter : {kw}"
            )
        want_required = _schema_required_count(signature)
        got_required = sum(
            1
            for p in params
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        if want_required != got_required:
            problems.append(
                f"{op_name}: 必填位置参数个数 {got_required} != schema 的 {want_required}"
            )
        extras = pos[len(want["positional"]) :] + kw[len(want["kwarg_only"]) :]
        for name in extras:
            p = inspect.signature(fn).parameters[name]
            if p.default is p.empty:
                problems.append(f"{op_name}: 多出的参数 {name} 没有默认值")
        if verbose:
            print(
                f"  {op_name:52s} pos={len(pos)} kw={len(kw)} extras={extras or '-'}"
            )
    return problems


def check_against_task12_sigs() -> list[str]:
    """与 task-12 的 ``_C_ascend_sigs.py`` 交叉核对（可选依赖；文件不在就跳过）。

    比对内容：本组有 schema 的 11 个注册名，signature 必须**逐字相同**；
    以及 12 条短名 → 注册名映射一致。

    健壮性：对方文件被重命名/重构（属性名变了）时**不抛异常**，而是给出可读的 problem，
    这样 Lead 侧重跑本自测不会因为对方内部改名而崩。
    """
    problems: list[str] = []
    try:
        import _C_ascend_sigs as other  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [f"[skip] 无法 import _C_ascend_sigs（{exc}）"]
    all_sigs = getattr(other, "_C_ASCEND_ALL_SIGS", None)
    task_map = getattr(other, "TASK_NAME_TO_OP", None)
    if not isinstance(all_sigs, dict) or not isinstance(task_map, dict):
        return [
            "[skip] _C_ascend_sigs 里找不到 _C_ASCEND_ALL_SIGS / TASK_NAME_TO_OP（对方可能重构过），"
            "跳过交叉核对"
        ]
    for op_name, signature in SIGS_B.items():
        theirs = all_sigs.get(op_name)
        if theirs is None:
            problems.append(f"{op_name}: task-12 的 _C_ascend_sigs 里没有这条")
        elif theirs != signature:
            problems.append(
                f"{op_name}: signature 与 task-12 不一致\n    本文件: {signature}\n    task-12: {theirs}"
            )
    for short, op in TASK_NAME_TO_OP_B.items():
        theirs = task_map.get(short)
        if theirs != op:
            problems.append(f"短名映射不一致: {short} → 本文件 {op} / task-12 {theirs}")
    return problems


# ===========================================================================
# 7. 注册（真的接管 `_C_ascend`）
# ===========================================================================


def patch(lib_name: str = LIB_NAME, key: str = "PrivateUse1", verbose: bool = True):
    """``define`` + ``Library(IMPL).impl`` —— 把本组 11 个 op 接管到 Triton 实现。

    用法（Lead 侧集成）::

        from _C_ascend_ops_b import patch
        patch()                     # 默认 _C_ascend + PrivateUse1

    与 FlagGems 的关系：等价于把 ``SIGS_B`` update 进
    ``patch_util._VLLM_OPS_SIGNATURES``，再 ``patch_vllm_lib("_C_ascend", op, ADAPTERS_B[op], key)``。
    返回 ``(lib, [(op_name, "defined"|"already_defined"), ...])``。
    """
    torch = _torch()
    status: list[tuple[str, str]] = []
    for op_name, signature in SIGS_B.items():
        qualname = f"{lib_name}::{op_name}"
        try:
            torch.library.define(qualname, signature)
            status.append((op_name, "defined"))
        except Exception as exc:  # noqa: BLE001 —— 已存在（厂家 .so 先注册）是正常情况
            msg = str(exc).lower()
            if any(
                k in msg
                for k in ("already", "same name", "already registered", "overriding")
            ):
                status.append((op_name, "already_defined"))
            else:
                raise
        if verbose:
            print(f"[_C_ascend_ops_b] {status[-1][1]:16s} {qualname}")

    lib = torch.library.Library(lib_name, "IMPL")
    for op_name, fn in ADAPTERS_B.items():
        lib.impl(op_name, fn, key)
        if verbose:
            print(f"[_C_ascend_ops_b] impl             {lib_name}::{op_name} [{key}]")
    return lib, status


def describe() -> str:
    """人类可读清单（供交付报告 / 自测打印）。"""
    lines = [
        f"B 组（task-13）：{len(GROUP_B_TASK_OPS)} 个短名 / {len(SIGS_B)} 条注册 schema / "
        f"{len(ADAPTERS_B)} 个注册 adapter",
        "-" * 100,
    ]
    for short in GROUP_B_TASK_OPS:
        op = TASK_NAME_TO_OP_B[short]
        spec = IMPL_B.get(op)
        impl = "** 无 Triton 实现（显式 NotImplementedError）**" if spec is None else f"{spec[0]}.py::{spec[1]}"
        lines.append(f"{short:52s} → {op}")
        lines.append(f"{'':52s}   impl: {impl}")
    return "\n".join(lines)


__all__ = [
    "LIB_NAME",
    "GROUP_B_TASK_OPS",
    "TASK_NAME_TO_OP_B",
    "OP_TO_TASK_NAME_B",
    "SIGS_B",
    "PROVENANCE_HINT",
    "REGISTERED_OPS_B",
    "SYNTHETIC_SIG_DISPATCH_FFN_COMBINE_BF16",
    "IMPL_B",
    "IMPL_DFC_BF16",
    "IMPL_MISSING",
    "NOT_IMPLEMENTED_BRANCHES_B",
    "PRODUCTION_CONFLICTS",
    "GAPS_B",
    "TRITON_OPS_DIR",
    "ADAPTERS_B",
    "ADAPTERS_B_BY_TASK_NAME",
    "to_patch_util_signatures",
    "to_patch_util_adapters",
    "parse_schema_params",
    "check_adapter_signatures",
    "check_against_task12_sigs",
    "patch",
    "describe",
    # adapters
    "dispatch_ffn_combine",
    "dispatch_ffn_combine_bf16",
    "moe_gating_top_k",
    "moe_gating_top_k_hash",
    "moe_init_routing_custom",
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    "compressor_metadata",
    "sparse_attn_sharedkv",
    "sparse_attn_sharedkv_metadata",
    "scatter_nd_update_v2",
]

if __name__ == "__main__":  # pragma: no cover - 人工核对用
    print(describe())
