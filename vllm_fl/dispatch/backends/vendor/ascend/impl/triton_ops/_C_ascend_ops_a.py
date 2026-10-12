# -*- coding: utf-8 -*-
"""FlagGems 集成 A：`torch.ops._C_ascend.*` 的 **attention 组 12 个 adapter**。

机制（照抄 `_repos/FlagGems/src/flag_gems/patches/patch_util.py`）
---------------------------------------------------------------
::

    # patch_util.py L105-L110
    torch.library.define(f"{lib_name}::{op_name}", signature)
    # L129 / L153-L158
    lib = torch.library.Library(lib_name, "IMPL"); lib.impl(op_name, fn, dispatch_key)

本模块提供 `_C_ASCEND_IMPLS_A = {op_name: adapter}` 与 `register()`；
schema 由同目录 `_C_ascend_sigs.py` 提供（`_C_ASCEND_SIGS`），二者**必须成对使用**。

设计约束（都有实测依据）
------------------------
1. **参数名/顺序/默认值 100% 对齐 schema**。
   torch 的自定义算子分发会按 schema 把参数**归一化成位置参数**（`*` 之后的按关键字传），
   所以 adapter 的形参名与顺序必须和 schema 一致，否则 `*` 之后的关键字参数会
   `TypeError: unexpected keyword argument`。自测里用 `inspect.signature` 逐条比对
   （见 `test_ops_a.py`）。
2. **adapter 一律惰性 import Triton 实现**（函数体内 import）。
   宿主机没有 `triton`，若模块级 import，`import _C_ascend_ops_a` 立刻失败，
   也就无法在 CPU 上做 `torch.library.define` 验证。
3. **`NotImplementedError` 一律透传**：所有 adapter **没有** `try/except`。
   Triton 实现里显式不支持的取值必须原样冒到调用方，绝不静默吞掉。
   各算子未实现分支的清单见文件末尾 `NOT_IMPLEMENTED_BRANCHES`。
4. **`Tensor(a!)` / `-> ()` 的原地语义要还原**：
   Triton 实现多数返回**新张量**（或返回被写坏的 cache），而 schema 声明的是原地别名。
   adapter 负责把结果 `copy_()` 回 schema 声明的可变张量，并按 schema 的返回个数返回。
5. **只写本 scope 的 3 个文件**：不碰 `_analysis/triton_ops/`（已验证基线）、
   不碰 `_repos/FlagGems/`（集成 patch 由 Lead 统一接入）。

Triton 实现来源
---------------
全部来自 `_analysis/triton_ops/`（已在容器 `vllm-ascend-fanyc-0923` 实测通过）。
`_OP_TO_IMPL` 给出算子 → 模块/函数/返回形状的完整映射。
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

from _C_ascend_sigs import LIB_NAME

# ---------------------------------------------------------------------------
# 惰性加载 `_analysis/triton_ops/<module>.py`
# ---------------------------------------------------------------------------

#: `_analysis/`（`triton_ops` 包的父目录）——必须进 sys.path 才能 `import triton_ops.*`
_TRITON_OPS_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPL_CACHE: dict = {}


def _impl(module_name: str, attr: str):
    """惰性导入 `_analysis/triton_ops/<module_name>.py` 的 `attr` 并缓存。

    只在 adapter 被真正调用时触发；宿主机（无 triton）import 本模块不会触发。
    """
    key = (module_name, attr)
    fn = _IMPL_CACHE.get(key)
    if fn is None:
        if _TRITON_OPS_PARENT not in sys.path:
            sys.path.insert(0, _TRITON_OPS_PARENT)
        mod = importlib.import_module(f"triton_ops.{module_name}")
        fn = getattr(mod, attr)
        _IMPL_CACHE[key] = fn
    return fn


def _empty_like_batch(t: torch.Tensor) -> torch.Tensor:
    """构造「空张量」占位（某些 schema 的返回值在关闭开关时是空的）。"""
    return torch.empty(0, dtype=t.dtype if t is not None else torch.float32,
                       device=t.device if t is not None else None)


# ===========================================================================
# 1. compressor
# schema 出处：VF csrc/ascend/torch_binding.cpp L1229
#             （VA csrc/torch_binding.cpp L3313-L3323 同名同参）
# Triton 实现：_analysis/triton_ops/compressor.py :: compressor
#              返回 (cmp_kv [?, D], state_cache_out)；按 coff/cmp_ratio 内部分派到
#              compressor_coff2.py（coff=2）/ compressor_bigr.py（cmp_ratio>16）
# ===========================================================================
def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    cache_mode: int,
):
    """`compressor` → compressor.py。

    生产调用点：`vllm_fl/attention/ascend/dsa_v1.py:2051`（`coff=2` 可能）、
    `dsa_v1.py:2344 / 2553 / 2779`、`vllm_fl/attention/ascend/context_parallel/dsa_cp.py:1421 / 1561`。

    ⚠️ schema 返回**只有一个** `Tensor`，但 Triton 实现返回 `(out, state_cache_out)`，
    且 `state_cache_out` 是 `state_cache.clone()` 后的**新张量**（`compressor.py:418`）。
    schema 把 `state_cache` 声明为 `Tensor(a!)`（原地可变），所以 adapter
    **把 `sc_out` copy 回 `state_cache`**，只返回 `out` —— 这样 alias 语义与真实 op 一致。

    ⚠️ `rotary_mode`：生产传 `2`（INTERLEAVE）。实现内部在
    `compressor.py:445-447` 对 `rotary_mode==2` 且 `(D-RD)%RD!=0 或 BLK_D%RD!=0`
    抛 `NotImplementedError`；`coff` 非 {1,2} 抛 `ValueError`（`compressor.py:332-335`）；
    `coff==2` 走 `compressor_coff2`（其 L389/L475 另有 `NotImplementedError`）。
    这些错误**全部透传**。
    """
    out, sc_out = _impl("compressor", "compressor")(
        x=x,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        state_cache=state_cache,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )
    if state_cache is not None and sc_out is not None:
        # 还原 schema 的 Tensor(a!) 原地语义（真实 op 就是原地写 state_cache）
        state_cache.copy_(sc_out)
    return out


# ===========================================================================
# 2. mla_preprocess
# schema 出处：VF csrc/ascend/torch_binding.cpp L1147-L1158
#             ⚠️ 被 `#ifdef VLLM_FL_ENABLE_MLAPO_DIRECT_KERNEL`(L1146/L1159) 包裹
#             VA 变体：csrc/torch_binding.cpp L2971-L2981（cos/sin 变可选）
# 函数签名：VF csrc/ascend/mla_preprocess/mla_preprocess_torch_adpt.h L33-L42
# Triton 实现：_analysis/triton_ops/mla_preprocess.py :: mla_preprocess
#              返回 (q, kv_cache_out0, q2, kv_cache_out1)，量化分支转到 mla_preprocess_int8.py
# ===========================================================================

#: 裁决 A（Lead 2026-10-10）：`cache_mode != "int8_nzcache"` 时**原算子接收但不消费**、
#: adapter 因此不转发给 Triton 实现的参数名 —— **只允许且仅允许这两个**。
#: 出处：`mla_preprocess.h` L47-L49（CACHE_MODE_KROPE_CTKV=1 / INT8_NZCACHE=2）+
#:       `mla_preprocess_mix_bf16.hpp` L2439/L2617/L2670（消费处全在
#:       `if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE)` 内）+
#:       `mla_preprocess_torch_adpt.h` L80-L86（无值时传空占位张量，不报错）。
#: `test_ops_a.py` 第 10 步会对这个**字面量**做 AST 校验（防止将来被悄悄扩大）。
_MLA_IGNORED_UNLESS_INT8_NZ = ("ctkv_scale", "q_nope_scale")


def mla_preprocess(
    hiddenState: torch.Tensor,
    wdqkv: torch.Tensor,
    descale0: torch.Tensor | None,
    gamma1: torch.Tensor,
    beta1: torch.Tensor | None,
    wuq: torch.Tensor,
    descale1: torch.Tensor | None,
    gamma2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wuk: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    slotmapping: torch.Tensor,
    quant_scale0: torch.Tensor | None,
    quant_offset0: torch.Tensor | None,
    bias0: torch.Tensor | None,
    quant_scale1: torch.Tensor | None,
    quant_offset1: torch.Tensor | None,
    bias1: torch.Tensor | None,
    ctkv_scale: torch.Tensor | None,
    q_nope_scale: torch.Tensor | None,
    cache_mode: str | None,
    quant_mode: str | None,
    enable_inner_out: bool | None,
    q_out0: torch.Tensor,
    kv_cache_out0: torch.Tensor,
    q_out1: torch.Tensor,
    kv_cache_out1: torch.Tensor,
    inner_out: torch.Tensor,
):
    """`mla_preprocess` → mla_preprocess.py（量化路径 → mla_preprocess_int8.py）。

    生产调用点：`vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:349-380`
    （14 个位置参数 + 13 个关键字参数；注意它同时把 `decode_k_nope`/`decode_k_pe`
    传给位置 12/13 **和** `kv_cache_out0`/`kv_cache_out1`）。

    参数名映射（schema → Triton 实现，**位置一一对应**）：
        hiddenState → hidden_state    descale0 → deq_scale_qkv
        wuq → wuq                     descale1 → qb_deq_scl
        kv_cache → *（见下）*          kv_cache_rope → *（见下）*
        slotmapping → slot_mapping
        q_out0 → q                    q_out1 → q2
        kv_cache_out0/1 → kv_cache_out0/1     inner_out → inner_out

    ⚠️ **`kv_cache` / `kv_cache_rope` 不被消费**：Triton 实现没有「读 kv_cache」的入参
    （它的 `decode_k_nope`/`decode_k_pe` 形参只用于**拒绝**非 None 值，见下）。
    真实 op 的 kernel 也是把结果写进 `keycache_out`(= `kv_cache_out0`)。
    生产调用点上 `kv_cache`/`kv_cache_rope` 与 `kv_cache_out0`/`kv_cache_out1`
    **是同一张张量**（device_op.py:361-377），所以行为一致；若某调用点传了不同的张量，
    按 schema（`kv_cache` 是普通输入、`kv_cache_out0` 才是 `Tensor!`）应以 `*_out*` 为准。

    ⚠️ **`decode_k_nope` / `decode_k_pe` 槽位显式传 None**：实现 `mla_preprocess.py:354`
    对这两个槽位（实现里借用来表达「decode-only 输入」）非 None 时会抛
    `NotImplementedError`。真实调用点把 kv_cache 缓冲区放在这两个位置上，
    所以 adapter **必须传 None**，否则生产路径必挂。

    ⚠️ `enable_inner_out=True` 时返回第 5 个元素 `inner_out`；关闭/未传时返回空张量。

    ⚠️ 未实现分支（**全部透传，不吞**；出处 `_analysis/triton_ops/mla_preprocess.py`）：
      * `quant_mode ∈ {"per_token_quant_symm"(1), "per_token_quant_asymm"(2)}` → L363-365
        `NotImplementedError`
      * `beta1` 非 None → L354 `NotImplementedError`
      * `no_quant` 下 `descale0`/`descale1`/`quant_scale0/1`/`quant_offset0/1`/`bias0/1`
        任一非 None → L369-372 `NotImplementedError`
      * `cache_mode` 非法（不在 kvcache/krope_ctkv/nzcache/int8_nzcache）→ L376-377
        `NotImplementedError`
      * `cache_mode == "int8_nzcache"` 但 `ctkv_scale`/`q_nope_scale` 缺失 → L378-379 `ValueError`
        （该模式下这两个 scale 是真被消费的）
      * ⚠️ 实现 L355-356 还会在 `cache_mode != "int8_nzcache"` 时**拒绝**非空的
        `ctkv_scale`/`q_nope_scale`。**adapter 在裁决 A 下已在调用前把它们置 None**
        （复刻原算子「接收但不消费」语义），因此这条护栏在生产路径上不再被触发 —— 见下。

    裁决 A（Lead 2026-10-10 核实并裁决）：**非 `int8_nzcache` 时丢弃
    `ctkv_scale` / `q_nope_scale`**，复刻原算子语义「接收但不消费」：:

        出处（原算子侧）：
          csrc/ascend/mla_preprocess/op_host/mla_preprocess.h L47-L49
              CACHE_MODE_KROPE_CTKV = 1  "double in and double out"
              CACHE_MODE_INT8_NZCACHE = 2 "high performance KV NZ format/quant"
          csrc/ascend/mla_preprocess/op_kernel/mla_preprocess_mix_bf16.hpp
              L2439 / L2617 / L2670：ctkvScale / qNopeScale 的消费全部在
              `if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) { ... }` 里
          csrc/ascend/mla_preprocess/mla_preprocess_torch_adpt.h L80-L86
              `CtkvScale = ctkv_scale.has_value() ? ctkv_scale.value() : at::empty({1}, ...kHalf...);`
              → 无值时用**空占位张量**照样往下走，**从不解引用拒绝**

    ⇒ 在 `krope_ctkv`（生产值）下这两个 scale 传不传都不影响结果；
    ⇒ 只有 `int8_nzcache` 下才**必须原样透传**。

    ⚠️ **只丢这两个**：`quant_scale0`/`quant_offset0`/`bias0`/`quant_scale1`/`quant_offset1`/
    `bias1` 是原算子真正喂给 kernel 的，**必须照传**（`test_ops_a.py` 第 10 步有可证伪测试，
    含 AST 校验丢弃集合字面量与逐键透传断言）。

    ⚠️ **风险声明**：若生产将来在 `krope_ctkv` 下**真的开始消费**这两个 scale，
    此处丢弃即为错误 —— 届时必须改为透传并让实现消费它们。
    """
    impl = _impl("mla_preprocess", "mla_preprocess")
    # kv_cache / kv_cache_rope：如果调用点只给了它们而没给 *_out*，用它们当输出缓冲
    kv_out0 = kv_cache_out0 if kv_cache_out0 is not None else kv_cache
    kv_out1 = kv_cache_out1 if kv_cache_out1 is not None else kv_cache_rope

    # ---- 裁决 A：非 int8_nzcache 时丢弃 ctkv_scale / q_nope_scale（只这两个）----
    # 这些名字是原算子**接收但不消费**的（出处见 docstring 的「裁决 A」），
    # 若不丢弃会被 Triton 实现 L355-356 的过严护栏拒绝，导致生产 MLA 路径失败。
    if cache_mode != "int8_nzcache":
        _ignored = {_name: None for _name in _MLA_IGNORED_UNLESS_INT8_NZ}
    else:
        _ignored = {}

    q, kv_ret0, q2, kv_ret1 = impl(
        hidden_state=hiddenState,
        wdqkv=wdqkv,
        deq_scale_qkv=descale0,
        gamma1=gamma1,
        beta1=beta1,
        wuq=wuq,
        qb_deq_scl=descale1,
        gamma2=gamma2,
        cos=cos,
        sin=sin,
        wuk=wuk,
        # ↓ 这两个槽位在实现里是「拒绝非 None」的哨兵，必须显式 None（见 docstring）
        decode_k_nope=None,
        decode_k_pe=None,
        slot_mapping=slotmapping,
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        # ↓ 裁决 A：非 int8_nzcache 时这里恒为 None（只丢这两个，见 docstring「裁决 A」）
        ctkv_scale=_ignored.get("ctkv_scale", ctkv_scale),
        q_nope_scale=_ignored.get("q_nope_scale", q_nope_scale),
        cache_mode=cache_mode,
        quant_mode=quant_mode,
        enable_inner_out=bool(enable_inner_out) if enable_inner_out is not None else False,
        inner_out=inner_out,
        q_out0=q_out0,
        kv_cache_out0=kv_out0,
        q_out1=q_out1,
        kv_cache_out1=kv_out1,
    )
    inner_ret = inner_out if inner_out is not None else _empty_like_batch(hiddenState)
    return q, kv_ret0, q2, kv_ret1, inner_ret


# ===========================================================================
# 3. indexer_compress_epilog_v2
# schema 出处：VA csrc/torch_binding.cpp L3560-L3567
#             （VF csrc/ascend/ 无此 op）
# Triton 实现：_analysis/triton_ops/indexer_compress_epilog_v2.py
#              ⚠️ 形参顺序 (x, slot_mapping, indexer_compress_cache, ...) 与 schema 不同！
# ===========================================================================
def indexer_compress_epilog_v2(
    indexer_compress_cache: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: int = 2,
):
    """`indexer_compress_epilog_v2` → indexer_compress_epilog_v2.py。schema `-> ()`。

    生产调用点（**全部用关键字**）：
    `vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:1571 / 1587 / 1607`
    （`indexer_compress_cache=indexer_full_cache.view(torch.uint8), x=kv,
    slot_mapping=..., layout=2`）。

    ⚠️ 参数名与 schema 一致（不是实现的名字），实现按关键字调用 →
    `indexer_compress_cache` 落到实现的第 3 个形参。

    ⚠️ 实现的 `block_stride=0` 缺省即「取 `cache.stride(0)`，否则 `cacheBs*cacheD`」
    （`indexer_compress_epilog_v2.py:173-174`），与调用点传 `cache.stride(0)` 等价，
    故 adapter 不暴露 `block_stride`（schema 里也没有）。

    ⚠️ 未实现分支（透传）：`layout != 2` → L149-150 `NotImplementedError`；
    dtype/形状/容量不符 → `ValueError`。

    ⚠️ 真实 op 是**原地**写 cache（`Tensor(a!)`），实现也确实 `tl.store` 进传入的 cache，
    返回同一个 cache；schema 返回 `()`，所以 adapter 返回 `None`。
    """
    _impl("indexer_compress_epilog_v2", "indexer_compress_epilog_v2")(
        x=x,
        slot_mapping=slot_mapping,
        indexer_compress_cache=indexer_compress_cache,
        layout=layout,
    )
    return None


# ===========================================================================
# 4. kv_compress_epilog
# schema 出处：VA csrc/torch_binding.cpp L3471-L3481（VF csrc/ascend/ 无此 op）
# Triton 实现：_analysis/triton_ops/kv_compress_epilog.py
#              ⚠️ 形参顺序 (x, slot_mapping, kv_compress_cache, ...) 与 schema 不同！
# ===========================================================================
def kv_compress_epilog(
    kv_compress_cache: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_group_size: int,
    quant_mode: int,
    round_scale_flag: bool,
    layout: int,
):
    """`kv_compress_epilog` → kv_compress_epilog.py。schema `-> ()`。

    生产调用点（**全部用关键字**）：
    `vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:1542-1550`
    （`quant_group_size=64, quant_mode=2, round_scale_flag=True, layout=1`）
    ⇒ **生产走 quant_mode=2（MXFP8 / E8M0 单字节 scale）**。

    参数名映射：`round_scale_flag` → 实现的 `round_scale`
    （实现里 `bool(int(round_scale))`，传 bool `True` 也正确）。

    ⚠️ 未实现分支（**透传**，出处 `kv_compress_epilog.py`）：
      * `quant_mode ∉ {1,2}` → L210-211 `NotImplementedError`
      * `layout ∉ {1,2}` → L212-214 `NotImplementedError`
      * `quant_mode==1` 且 `quant_group_size != 128` → L217-219 `NotImplementedError`
      * `quant_mode==2` 且 `quant_group_size ∉ {64,128}` → L220-222 `NotImplementedError`
      * cache dtype 非 `fp8_e4m3fn`/`uint8`（即 **`fp8_e5m2` 未实现**）→ L229-232
        `NotImplementedError`
    （其余形状/容量检查是 `ValueError`。）

    ⚠️ `kv_compress_cache` 是 `Tensor(a!)`：真实 op 原地写，实现也 `tl.store` 进传入的
    cache 的 uint8 视图；schema 返回 `()`，adapter 返回 `None`。
    """
    _impl("kv_compress_epilog", "kv_compress_epilog")(
        x=x,
        slot_mapping=slot_mapping,
        kv_compress_cache=kv_compress_cache,
        quant_mode=quant_mode,
        layout=layout,
        quant_group_size=quant_group_size,
        round_scale=round_scale_flag,
    )
    return None


# ===========================================================================
# 5. store_kv_block
# schema 出处：VA csrc/torch_binding.cpp L3636-L3638（VF csrc/ascend/ 无此 op）
# Triton 实现：_analysis/triton_ops/store_kv_block.py :: store_kv_block
# ===========================================================================
def store_kv_block(
    key_in: torch.Tensor,
    key_cache_in: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int = 0,
):
    """`store_kv_block` → store_kv_block.py。schema `-> ()`。

    生产调用点（**全位置参数**）：
    `vllm_fl/attention/ascend/sfa_v1.py:1839-1846 / 1857-1864`
    （`(k_li, kv_cache[i], group_len, group_key_idx, group_key_cache_idx, block_size)`）。

    ⚠️ 真实 op 原地写 `key_cache_in`，实现也 `tl.store(KEYCACHE + ...)` 原地写；
    `key_cache_in` **必须是 contiguous**（实现里 `key_cache_in.reshape(-1)`，
    非连续张量 reshape 会返回副本 ⇒ 原地写入丢失）。调用点传的是 `kv_cache[...]` 视图，
    见 docstring 末尾 `CAVEATS` 第 2 条。

    ⚠️ 未实现/拒绝分支：三个 idx 张量非 int32、长度不一致、dtype 非 int8/fp16/bf16、
    `keyCacheIn` dtype 与 `keyIn` 不一致 → `ValueError`（无 `NotImplementedError`）。
    """
    _impl("store_kv_block", "store_kv_block")(
        key_in=key_in,
        key_cache_in=key_cache_in,
        group_len=group_len,
        group_key_idx=group_key_idx,
        group_key_cache_idx=group_key_cache_idx,
        block_size=block_size,
    )
    return None


# ===========================================================================
# 6. npu_lightning_indexer
# schema 出处：**未找到权威出处**（两个仓库 csrc 里都没有 ops.def / OP_ADD；
#             op 由 torch_npu(PTA) 提供）→ 见 _C_ascend_sigs.py 的
#             `_C_ASCEND_SIGS_INFERRED`（按调用点 kwargs 反推，置信度低）
# Triton 实现：_analysis/triton_ops/lightning_indexer.py :: lightning_indexer
# ===========================================================================
def npu_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
):
    """`npu_lightning_indexer` → lightning_indexer.py。返回 sparse_indices。

    ⚠️ **本条的 schema 没有权威出处**（见 `_C_ascend_sigs.py`）：
    参数名取自生产调用点 `vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:632-643`
    与 `vllm_fl/attention/ascend/sfa_v1.py`，`*` 之后的可选参数与默认值**未证实**。
    接入前请在目标环境用 `torch._C._jit_get_operation("_C_ascend::npu_lightning_indexer")`
    或 `torch.ops._C_ascend.npu_lightning_indexer._schemas` 校对。

    ⚠️ 实现里 `query_quant_mode`/`key_quant_mode` 只接受 0（非量化版无量化语义），
    非 0 抛 `NotImplementedError`（`lightning_indexer.py:77-80`）；但**这两个参数不在
    反推 schema 里**（调用点也没传）。
    query/key 非 int8 → `ValueError`；TND 缺 `actual_seq_lengths_query` → `ValueError`。
    """
    return _impl("lightning_indexer", "lightning_indexer")(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )


# ===========================================================================
# 7. npu_lightning_indexer_quant
# schema 出处：VA csrc/torch_binding.cpp L3599-L3605
# Triton 实现：_analysis/triton_ops/lightning_indexer_quant.py :: lightning_indexer_quant
# ===========================================================================
def npu_lightning_indexer_quant(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
):
    """`npu_lightning_indexer_quant` → lightning_indexer_quant.py。返回 sparse_indices(int32)。

    生产调用点（**全部关键字**）：
    `vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:602-617`
    （`layout_query="TND", layout_key="PA_BSND", query_quant_mode=0, key_quant_mode=0,
    sparse_count=2048, sparse_mode=3`）。

    ⚠️ schema 的 `layout_key` 默认是 `'BSND'`，实现的默认是 `"PA_BSND"`；
    adapter **总是显式传值**，所以不受两边默认值差异影响。

    ⚠️ 未实现分支（**透传**，出处 `lightning_indexer_quant.py`）：
      * `layout_query ∉ {BSND,TND}` → L158-159 `NotImplementedError`
      * `layout_key ∉ {BSND,PA_BSND,PA_BBND}` → L160-161 `NotImplementedError`
      * `query_quant_mode != 0` 或 `key_quant_mode != 0` → L162-164 `NotImplementedError`
      * 其余（sparse_mode、dtype、形状、长度）为 `ValueError`
    """
    return _impl("lightning_indexer_quant", "lightning_indexer_quant")(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )


# ===========================================================================
# 8. npu_vllm_quant_lightning_indexer
# schema 出处：VF csrc/ascend/torch_binding.cpp L1231
# Triton 实现：_analysis/triton_ops/vllm_quant_lightning_indexer.py
# ===========================================================================
def npu_vllm_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    cmp_ratio: int = 1,
    return_value: bool = False,
):
    """`npu_vllm_quant_lightning_indexer` → vllm_quant_lightning_indexer.py。

    返回 `(sparse_indices, sparse_values)`；`return_value=False` 时第二个是空张量
    （与 schema `-> (Tensor sparse_indices, Tensor sparse_values)` 一致）。

    生产调用点（**全部关键字**）：`vllm_fl/attention/ascend/dsa_v1.py:2100 / 2393 / 2658`、
    `vllm_fl/attention/ascend/context_parallel/dsa_cp.py:1648`
    （`layout_query="TND", layout_key="PA_BSND", cmp_ratio=4, sparse_mode=3`）。

    ⚠️ 未实现分支（**透传**，出处 `vllm_quant_lightning_indexer.py`）：
      * `layout_query ∉ {BSND,TND}` → L209-210
      * `layout_key ∉ {BSND,PA_BSND,PA_BBND}` → L211-212
      * `query_quant_mode != 0` / `key_quant_mode != 0` → L215-217
      * `pre_tokens`/`next_tokens ∉ {None, INT64_MAX}` → L218-220
      以上均为 `NotImplementedError`；`metadata` 非 int32 → `ValueError`。
    """
    return _impl("vllm_quant_lightning_indexer", "vllm_quant_lightning_indexer")(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        metadata=metadata,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
    )


# ===========================================================================
# 9. npu_sparse_attention_score
# schema 出处：VF csrc/ascend/torch_binding.cpp L1191-L1199（大驼峰，**以 VF 为准**）
#             ⚠️ VA csrc/torch_binding.cpp L3641-L3652 是另一套全 snake_case schema
# Triton 实现：_analysis/triton_ops/sparse_attention_score.py :: sparse_attention_score
# ===========================================================================

#: 本实现实际固定的输入布局（schema 的两个 layout 属性**不在实现的参数表里**，
#: 原样传下去会被 `**kwargs` 静默吞掉 ⇒ adapter 必须显式拒绝其它取值）
_SCORE_Q_LAYOUT_OK = ("TND",)
_SCORE_KV_LAYOUT_OK = ("TND",)


def npu_sparse_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selectIdx: torch.Tensor,
    blockTable: torch.Tensor,
    selectNumIdx: torch.Tensor | None = None,
    qDequantScale: torch.Tensor | None = None,
    kDequantScale: torch.Tensor | None = None,
    vDequantScale: torch.Tensor | None = None,
    actualSeqLengths: torch.Tensor | None = None,
    actualSeqLengthsKv: torch.Tensor | None = None,
    qInputLayout: str = "TND",
    kvInputLayout: str = "TND",
    numKeyValueHeads: int = 0,
    scaleValue: float = 0.0,
    blockSize: int = 128,
    topK: int = 0,
    innerPrecise: int = 4,
):
    """`npu_sparse_attention_score` → sparse_attention_score.py。**只返回 attentionOut**。

    生产调用点（**位置参数 + 大驼峰关键字**）：
    `vllm_fl/attention/ascend/sparse_attn_m3.py:91-105`
    （`q, key, value, topk_idx, block_table, selectNumIdx=..., actualSeqLengths=...,
    actualSeqLengthsKv=..., numKeyValueHeads=..., scaleValue=..., blockSize=...,
    topK=..., innerPrecise=...`），调用方写 `result = ...`（**不拆包**）。

    ⚠️ schema 返回**只有一个 `Tensor`**，但 Triton 实现返回 `(attentionOut, softmaxLse)`。
    所以 adapter **只返回 attentionOut**（与生产调用点 `out.copy_(result)` 一致）。
    若需要 lse，实现里有 `lse=` / `write_lse=` 关键字（**本 schema 无法表达**）。

    ⚠️ 实现同时接受大驼峰（AscendC 原名）与 snake_case 两套参数名；
    adapter 按 schema 传大驼峰，实现内部别名回填（`sparse_attention_score.py:285-315`）。

    ⚠️ **`qInputLayout` / `kvInputLayout` 显式拒绝非默认值**：
    实现的参数表里**没有**这两个名字（会被 `**kwargs`(L269) 静默吞掉），
    它固定按 `query=[T,N1,D]`（TND）+ `key/value=[numBlocks, blockSize, KVH, D]`（分块 KV）
    寻址。为了不「静默忽略布局」，adapter 对非 `"TND"` 取值抛 `NotImplementedError`。
    生产调用点 `sparse_attn_m3.py:91-105` **没传**这两个参数 → 走 schema 默认 `"TND"`，不受影响。

    ⚠️ `innerPrecise` **被实现接受但不被消费**（`inner_precise` 在 L247 声明、L311 别名回填后
    再无使用）。生产值 `_SPARSE_ATTN_INNER_PRECISE = 4`（sparse_attn_m3.py:45）恰等于 schema
    默认值 4，故当前无论传什么都不改变结果；这是**已知缺口**（见 CAVEATS），不是静默分支。

    ⚠️ 未实现/拒绝分支：`selectIdx`/`blockTable` 缺失 → `ValueError`（L316-318）；
    `topK != selectIdx.shape[-1]` → `ValueError`（L333-334）；
    layout 非 TND → 本 adapter 的 `NotImplementedError`。
    实现本身**没有** `NotImplementedError` 分支（fp8 FullQuant 已实现）。
    """
    if (qInputLayout not in _SCORE_Q_LAYOUT_OK) or (kvInputLayout not in _SCORE_KV_LAYOUT_OK):
        # 这两个属性实现收不到（会进 **kwargs），因此必须显式拒绝而不是静默丢弃
        raise NotImplementedError(
            "npu_sparse_attention_score: 本 Triton 实现固定按 query=[T,N1,D](TND) + "
            "key/value=[numBlocks, blockSize, KVH, D](分块 KV) 寻址，"
            f"不支持 qInputLayout={qInputLayout!r} / kvInputLayout={kvInputLayout!r}"
            "（实现参数表里没有这两个名字，原样透传会被 **kwargs 静默吞掉）"
        )
    out, _lse = _impl("sparse_attention_score", "sparse_attention_score")(
        query=query,
        key=key,
        value=value,
        selectIdx=selectIdx,
        blockTable=blockTable,
        selectNumIdx=selectNumIdx,
        qDequantScale=qDequantScale,
        kDequantScale=kDequantScale,
        vDequantScale=vDequantScale,
        actualSeqLengths=actualSeqLengths,
        actualSeqLengthsKv=actualSeqLengthsKv,
        qInputLayout=qInputLayout,
        kvInputLayout=kvInputLayout,
        numKeyValueHeads=numKeyValueHeads,
        scaleValue=scaleValue,
        blockSize=blockSize,
        topK=topK,
        innerPrecise=innerPrecise,
    )
    return out


# ===========================================================================
# 10. npu_sparse_flash_attention
# schema 出处：VF csrc/ascend/torch_binding.cpp L1232-L1242
#             （VA csrc/torch_binding.cpp L3136-L3146 逐字一致）
# Triton 实现：_analysis/triton_ops/sparse_flash_attention.py :: sparse_flash_attention
# ===========================================================================
def npu_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    *,
    block_table: torch.Tensor | None = None,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
):
    """`npu_sparse_flash_attention` → sparse_flash_attention.py。

    返回 `(attention_out, softmax_max, softmax_sum)`（与 schema 三返回一致；
    `return_softmax_lse=False` 时后两个是空张量）。

    生产调用点：`vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:682-698`
    （`q, k, v, sparse_indices` 位置；`scale_value=...` 位置；其余关键字；
    真实组合 = `layout_query="TND", layout_kv="PA_BSND", attention_mode=2`）。

    ⚠️ 未实现分支（**透传**，出处 `sparse_flash_attention.py`）：
      * `layout_query ∉ {BSND,TND}` → L228-229 `NotImplementedError`
      * `layout_kv ∉ {BSND,PA_BSND,PA_BBND}` → L230-231 `NotImplementedError`
      * `attention_mode != 2` → L232-233 `NotImplementedError`
      * `pre_tokens`/`next_tokens ∉ {None, INT64_MAX}` → L234-236 `NotImplementedError`
      其余（sparse_mode、query_rope/key_rope 缺失、形状）为 `ValueError`。
    """
    return _impl("sparse_flash_attention", "sparse_flash_attention")(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )


# ===========================================================================
# 11. inplace_partial_rotary_mul
# schema 出处：VF csrc/ascend/torch_binding.cpp L1246
#             ⚠️ VA csrc/torch_binding.cpp L3453-L3458 多一个 `bool negate_sin=False`
# Triton 实现：_analysis/triton_ops/inplace_partial_rotary_mul.py
# ===========================================================================
def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    rotary_mode: str,
    partial_slice,
):
    """`inplace_partial_rotary_mul` → inplace_partial_rotary_mul.py。schema `-> ()`。

    生产调用点（**13 处**，全位置参数 + 后两个关键字）：
    `vllm_fl/attention/ascend/dsa_v1.py:1762-1768 / 1837 / 1864 / 1948 / 1971 / 2267 /
    2291 / 2525 / 2836`、`context_parallel/dsa_cp.py:1371 / 1385 / 1520 / 1633`
    （典型：`(o_proj_input.unsqueeze(1), cos, -sin, rotary_mode="interleave",
    partial_slice=[nope_head_dim, head_dim])`）。

    ⚠️ `Tensor(a!) x`：Triton 实现**返回新张量** `y = x.clone()`（
    `inplace_partial_rotary_mul.py:256`），所以 adapter **把 y copy 回 x**，
    按 schema 返回 `None`。

    ⚠️ 未实现/拒绝分支：`rotary_mode` 非 `{half,interleave,quarter,interleave-half}`
    （或 int 0..3）、`r1/r2` 缺失、`x` 非 4D、`partial_slice` 越界、dtype 不符
    → 全部 `ValueError`（本实现**没有** `NotImplementedError`）。
    """
    y = _impl("inplace_partial_rotary_mul", "inplace_partial_rotary_mul")(
        x=x,
        r1=r1,
        r2=r2,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
    )
    x.copy_(y)
    return None


# ===========================================================================
# 12. npu_hc_pre_v2
# schema 出处：VF csrc/ascend/torch_binding.cpp L1226
#             （VA csrc/torch_binding.cpp L3435-L3441 同名同参，仅返回名不同）
# Triton 实现：_analysis/triton_ops/hc_pre.py :: hc_pre
#              ⚠️ 实现的后两个形参是 (hc_eps, norm_eps)，与 schema 的
#                 (norm_eps, hc_eps) **顺序相反** ⇒ adapter 必须按关键字传！
# ===========================================================================
def npu_hc_pre_v2(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """`npu_hc_pre_v2` → hc_pre.py。返回 `(y, post, comb_frag)`。

    生产调用点：`vllm_fl/models/deepseek_v4_ascend.py:969-971`
    （全位置参数：`x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters,
    norm_eps, hc_eps`）。

    ⚠️ **必须按关键字调用实现**：`hc_pre.py:156-164` 的形参顺序是
    `(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps)`
    —— `hc_eps` 与 `norm_eps` 与 schema **相反**。按位置传会把两个 eps 弄反。

    ⚠️ 未实现/拒绝分支：`x` 非 3D/4D、dtype 非 bf16、`hc_mult` 非 {1,2,4,8,16}、
    `x.shape[-2] != hc_mult`、`hc_fn.shape != (2*hm+hm*hm, hm*d)`、`hc_scale.numel()!=3`
    → 全部 `assert`/`ValueError`（本实现**没有** `NotImplementedError`）。
    """
    return _impl("hc_pre", "hc_pre")(
        x=x,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )


# ===========================================================================
# 注册表 / 辅助
# ===========================================================================

#: op 名（**实际注册名**）→ adapter。键必须与 `_C_ascend_sigs._C_ASCEND_SIGS` 的键一致。
_C_ASCEND_IMPLS_A = {
    "compressor": compressor,
    "mla_preprocess": mla_preprocess,
    "indexer_compress_epilog_v2": indexer_compress_epilog_v2,
    "kv_compress_epilog": kv_compress_epilog,
    "store_kv_block": store_kv_block,
    "npu_lightning_indexer": npu_lightning_indexer,
    "npu_lightning_indexer_quant": npu_lightning_indexer_quant,
    "npu_vllm_quant_lightning_indexer": npu_vllm_quant_lightning_indexer,
    "npu_sparse_attention_score": npu_sparse_attention_score,
    "npu_sparse_flash_attention": npu_sparse_flash_attention,
    "inplace_partial_rotary_mul": inplace_partial_rotary_mul,
    "npu_hc_pre_v2": npu_hc_pre_v2,
}

#: op → Triton 实现文件 / 函数 / 返回形状（供报告与审计）
_OP_TO_IMPL = {
    "compressor": ("_analysis/triton_ops/compressor.py", "compressor",
                   "(cmp_kv, state_cache_out) → adapter 只返回 cmp_kv 并把 sc_out 写回 state_cache"),
    "mla_preprocess": ("_analysis/triton_ops/mla_preprocess.py", "mla_preprocess",
                       "(q, kv_cache_out0, q2, kv_cache_out1) → adapter 补第 5 个 inner_out"),
    "indexer_compress_epilog_v2": ("_analysis/triton_ops/indexer_compress_epilog_v2.py",
                                   "indexer_compress_epilog_v2", "cache（原地）→ adapter 返回 None"),
    "kv_compress_epilog": ("_analysis/triton_ops/kv_compress_epilog.py",
                           "kv_compress_epilog", "cache（原地）→ adapter 返回 None"),
    "store_kv_block": ("_analysis/triton_ops/store_kv_block.py",
                       "store_kv_block", "key_cache_in（原地）→ adapter 返回 None"),
    "npu_lightning_indexer": ("_analysis/triton_ops/lightning_indexer.py",
                              "lightning_indexer", "sparse_indices"),
    "npu_lightning_indexer_quant": ("_analysis/triton_ops/lightning_indexer_quant.py",
                                    "lightning_indexer_quant", "sparse_indices"),
    "npu_vllm_quant_lightning_indexer": (
        "_analysis/triton_ops/vllm_quant_lightning_indexer.py",
        "vllm_quant_lightning_indexer", "(sparse_indices, sparse_values)"),
    "npu_sparse_attention_score": ("_analysis/triton_ops/sparse_attention_score.py",
                                   "sparse_attention_score",
                                   "(attentionOut, softmaxLse) → adapter 只返回 attentionOut"),
    "npu_sparse_flash_attention": ("_analysis/triton_ops/sparse_flash_attention.py",
                                   "sparse_flash_attention",
                                   "(attention_out, softmax_max, softmax_sum)"),
    "inplace_partial_rotary_mul": ("_analysis/triton_ops/inplace_partial_rotary_mul.py",
                                   "inplace_partial_rotary_mul",
                                   "y（新张量）→ adapter copy_ 回 x 并返回 None"),
    "npu_hc_pre_v2": ("_analysis/triton_ops/hc_pre.py", "hc_pre",
                      "(y, post, comb_frag)"),
}

#: 未实现分支清单（**如实记录**；adapter 不吞任何异常）
NOT_IMPLEMENTED_BRANCHES = {
    "compressor": [
        "rotary_mode=2(INTERLEAVE) 且 (D-RD)%RD!=0 或 BLK_D%RD!=0 → NotImplementedError "
        "(compressor.py:445-447)",
        "coff=2 → 走 compressor_coff2.py，其 L389 / L475 另有 NotImplementedError",
    ],
    "mla_preprocess": [
        "quant_mode='per_token_quant_symm'(1) / 'per_token_quant_asymm'(2) → "
        "NotImplementedError (mla_preprocess.py:363-365)",
        "beta1 非 None → NotImplementedError (L354)",
        "cache_mode != 'int8_nzcache' 时 ctkv_scale / q_nope_scale 非 None → "
        "NotImplementedError (L355-356)；**adapter 已在裁决 A 下于调用前置 None，"
        "故生产路径不再命中这条**（原始调用点 device_op.py:370-372 正是非空组合）",
        "no_quant 下 descale0/descale1/quant_scale0/1/quant_offset0/1/bias0/1 任一非 None → "
        "NotImplementedError (L369-372)",
        "cache_mode 非法（不在 kvcache/krope_ctkv/nzcache/int8_nzcache）→ "
        "NotImplementedError (L376-377)",
    ],
    "indexer_compress_epilog_v2": ["layout != 2 → NotImplementedError "
                                   "(indexer_compress_epilog_v2.py:149-150)"],
    "kv_compress_epilog": [
        "quant_mode ∉ {1,2} → NotImplementedError (L210-211)",
        "layout ∉ {1,2} → NotImplementedError (L212-214)",
        "quant_mode=1 且 quant_group_size != 128 → NotImplementedError (L217-219)",
        "quant_mode=2 且 quant_group_size ∉ {64,128} → NotImplementedError (L220-222)",
        "cache dtype 非 fp8_e4m3fn/uint8（含 **fp8_e5m2**）→ NotImplementedError (L229-232)",
    ],
    "store_kv_block": [],
    "npu_lightning_indexer": [
        "query_quant_mode != 0 或 key_quant_mode != 0 → NotImplementedError "
        "(lightning_indexer.py:77-80)（这两个参数不在反推 schema 里）",
    ],
    "npu_lightning_indexer_quant": [
        "layout_query ∉ {BSND,TND} → NotImplementedError (L158-159)",
        "layout_key ∉ {BSND,PA_BSND,PA_BBND} → NotImplementedError (L160-161)",
        "query_quant_mode != 0 / key_quant_mode != 0 → NotImplementedError (L162-164)",
    ],
    "npu_vllm_quant_lightning_indexer": [
        "layout_query ∉ {BSND,TND} → NotImplementedError (L209-210)",
        "layout_key ∉ {BSND,PA_BSND,PA_BBND} → NotImplementedError (L211-212)",
        "query_quant_mode != 0 / key_quant_mode != 0 → NotImplementedError (L215-217)",
        "pre_tokens/next_tokens ∉ {None, INT64_MAX} → NotImplementedError (L218-220)",
    ],
    "npu_sparse_attention_score": [
        "本 adapter 显式拒绝 qInputLayout/kvInputLayout != \"TND\"（实现收不到这两个属性，"
        "原样透传会被 **kwargs 静默吞掉）",
    ],
    "npu_sparse_flash_attention": [
        "layout_query ∉ {BSND,TND} → NotImplementedError (L228-229)",
        "layout_kv ∉ {BSND,PA_BSND,PA_BBND} → NotImplementedError (L230-231)",
        "attention_mode != 2 → NotImplementedError (L232-233)",
        "pre_tokens/next_tokens ∉ {None, INT64_MAX} → NotImplementedError (L234-236)",
    ],
    "inplace_partial_rotary_mul": [],
    "npu_hc_pre_v2": [],
}

#: 已知语义 caveat（**需要 Lead/用户确认**，不得当成已解决）
CAVEATS = {
    "__schema_conflicts__": [
        "裁决 C（Lead 2026-10-10）：VF / VA 两仓库对 4 个算子注册了**同名不同 schema**"
        "（mla_preprocess / npu_sparse_attention_score / inplace_partial_rotary_mul / "
        "dispatch_ffn_combine），本表**以 VF 为准**（生产调用点在 vllm_fl）。",
        "**运行时自检**：`register(verify_schemas=True)`（默认）会调 "
        "`_C_ascend_sigs.verify_runtime_schemas(strict=True)`，比对**实际生效的** "
        "`torch.ops._C_ascend.<op>._schemas` 与本地表的形参名/kwonly/个数；"
        "不一致直接 `RuntimeError`（典型：目标环境是 VA 的 .so）——**不静默错配**。",
        "为什么必要：FlagGems 的 `_define_op_if_not_exists`（patch_util.py L105-L110）"
        "**只在 op 未注册时才 define**；已有 .so 注册时生效的是 .so 的 schema，"
        "我们的 schema 表根本不会被使用。",
        "注意自检**不比默认值**：torch 会把 `0.0` 归一化成 `0.`、`Tensor! out` 归一化成 "
        "`Tensor($0! -> ) out`，属纯外表差异；决定关键字调用能否成立的是名字/kwonly/个数。",
    ],
    "mla_preprocess": [
        "schema 的 `kv_cache` / `kv_cache_rope` 是普通输入（非 Tensor!），Triton 实现里没有"
        "对应的读入参（它的 decode_k_nope/decode_k_pe 槽位只用来拒绝非 None 值）⇒ "
        "adapter 把 kv_cache/kv_cache_rope 仅当作 `kv_cache_out0/1` 缺失时的兜底输出缓冲。"
        "生产调用点 device_op.py:361-377 里二者是同一张张量，行为一致；"
        "**若出现 `kv_cache` 与 `kv_cache_out0` 不同张量的调用点，需要重新确认语义**。",
        "adapter 显式传 `decode_k_nope=None, decode_k_pe=None`。真实调用点把 kv_cache "
        "缓冲区放在这两个位置，若原样透传会必然触发实现的 NotImplementedError。",
        "裁决 A（Lead 2026-10-10）：`cache_mode != \"int8_nzcache\"` 时 adapter **丢弃** "
        "`ctkv_scale`/`q_nope_scale`（**只这两个**），复刻原算子「接收但不消费」语义。"
        "出处：mla_preprocess.h L47-L49 + mix_bf16.hpp L2439/L2617/L2670"
        "（消费处全在 `if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE)` 内）+ "
        "mla_preprocess_torch_adpt.h L80-L86（无值传空占位、从不解引用拒绝）。"
        "此前「非 int8_nzcache 即拒绝」是实现的**过严护栏**，会让生产 MLA 路径直接失败；"
        "现由 adapter 覆盖。",
        "**风险声明**：若生产将来在 `krope_ctkv` 下**真的开始消费**这两个 scale，"
        "此丢弃即为错误 —— 届时必须改为透传并让实现消费它们。",
        "`quant_scale0/quant_offset0/bias0/quant_scale1/quant_offset1/bias1` **必须照传**"
        "（原算子真的喂给 kernel）；`test_ops_a.py` 第 10 步逐键断言「只丢这两个」。",
    ],
    "store_kv_block": [
        "实现内部 `key_cache_in.reshape(-1)`：若传入非 contiguous 张量，reshape 返回副本 ⇒ "
        "原地写入会丢失。调用点 sfa_v1.py:1841 传 `kv_cache[i]`（切片视图），"
        "请确认其在 `c8_enable_reshape_optim` 路径下是 contiguous。",
        "`block_size` 被实现接受但**未被使用**（store_kv_block.py 注释：tiling 里用于 UB 容量校验）"
        "——属性级 no-op，不影响数值。",
    ],
    "npu_lightning_indexer": [
        "schema 无权威出处（torch_npu/PTA 提供），本条的参数名/默认值是按调用点反推的，"
        "接入前必须实测校对（见 _C_ascend_sigs.py 的 _C_ASCEND_SIGS_INFERRED）。",
    ],
    "npu_sparse_attention_score": [
        "schema（VF 版）只声明 1 个返回值 → adapter 丢弃 softmaxLse。"
        "若目标环境加载的是 VA 的 .so（另一套全 snake_case schema），"
        "adapter 的参数名对不上，需要另接一套（见 _C_ascend_sigs.py 的 conflicts）。",
        "`innerPrecise` 被实现接受但**未被消费**（sparse_attention_score.py L247 声明、"
        "L311 别名回填后无使用）。生产值 4 == schema 默认值 4，当前不影响结果；"
        "若将来调用点传非 4 的值，精度语义会与 AscendC 原实现不一致——属**已知缺口**。",
        "`qInputLayout`/`kvInputLayout` 不在实现参数表里，adapter 已改为对非 \"TND\" 取值"
        "抛 NotImplementedError（避免静默丢弃）。",
    ],
}


def to_patch_util_impls() -> dict:
    """返回 `{op_name: adapter}`；供 Lead 直接 `lib.impl(op, fn, key)` 或并入 `_VLLM_OPS_IMPLS`。"""
    return dict(_C_ASCEND_IMPLS_A)


def register(lib_name: str = LIB_NAME, dispatch_key: str = "PrivateUse1",
             verify_schemas: bool = True):
    """把本组 12 个 adapter 注册进 `torch.library.Library(lib_name, "IMPL")`。

    ⚠️ 调用前必须先用 `patch_util.init_vllm_libraries()`（或等价逻辑）确保
    `_C_ascend` 命名空间与各 op 的 schema 已存在，否则 `lib.impl` 会报
    "operator does not exist"。

    Args:
        verify_schemas: **默认 True**（裁决 C）：注册前先跑
            `_C_ascend_sigs.verify_runtime_schemas()`，把「运行时实际生效的 schema
            与本表的形参不一致」（典型原因：目标环境加载的是 **VA** 的 .so，见
            `SCHEMA_CONFLICTS`）**显式报错**，而不是静默错配 —— 那时 adapter 的
            关键字参数会对不上。确认无事后可显式关掉。

    返回创建出来的 `Library` 对象（**必须保持引用**，否则析构会注销注册）。
    """
    if verify_schemas:
        from _C_ascend_sigs import verify_runtime_schemas
        problems = verify_runtime_schemas(lib_name=lib_name, strict=True)
        if problems:  # strict=True 已抛，这里只是双保险
            raise RuntimeError(
                "运行时 schema 与本表不一致，拒绝静默错配：\n  - " + "\n  - ".join(problems))
    lib = torch.library.Library(lib_name, "IMPL")
    for op_name, fn in _C_ASCEND_IMPLS_A.items():
        lib.impl(op_name, fn, dispatch_key)
    return lib


def check_against_sigs() -> list:
    """与 `_C_ascend_sigs` 对账：返回不一致项列表（空 = 一致）。"""
    from _C_ascend_sigs import _C_ASCEND_ALL_SIGS

    problems = []
    for op in _C_ASCEND_IMPLS_A:
        if op not in _C_ASCEND_ALL_SIGS:
            problems.append(f"{op}: 有 adapter 但 _C_ascend_sigs 里没有 schema")
    for op in _C_ASCEND_ALL_SIGS:
        if op not in _C_ASCEND_IMPLS_A:
            # Group A 只覆盖 12 个；Group B（task-13）的 op 不在这里报错
            problems.append(f"{op}: 有 schema 但本组（A）没有 adapter（应属 B 组）")
    return problems


if __name__ == "__main__":  # pragma: no cover - 人工核对用
    print(f"Group A adapters: {len(_C_ASCEND_IMPLS_A)}")
    for op, fn in _C_ASCEND_IMPLS_A.items():
        mod, attr, ret = _OP_TO_IMPL[op]
        n_nie = len(NOT_IMPLEMENTED_BRANCHES.get(op, []))
        print(f"  {op:34s} -> {mod} :: {attr}")
        print(f"      返回: {ret}")
        print(f"      NotImplementedError 分支: {n_nie}")
