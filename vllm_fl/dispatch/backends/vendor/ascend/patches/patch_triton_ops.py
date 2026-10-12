# Copyright (c) 2026 BAAI. All rights reserved.
"""Register the migrated Triton operators as ``_C_ascend::*`` implementations.

This module is the vendor-owned replacement for the former FlagGems monkey
patch: the operator implementations, their torch schemas and the adapters now
live in this repository under ``impl/triton_ops`` (``_C_ascend_sigs.py``,
``_C_ascend_ops_a.py``, ``_C_ascend_ops_b.py`` and ``_C_ascend_bridge.py``).

:func:`apply_triton_ops_patch` defines any schema the compiled ``_C_ascend``
extension does not provide and registers every adapter for the NPU dispatch
key, so ``torch.ops._C_ascend.<op>`` runs the Triton kernel instead of the
AscendC kernel.

The takeover is **on by default** -- these operators live in this repository
next to the vendor's own Triton kernels (``ops/triton/**``) and are used the
same way.  ``USE_FLAGGEMS`` is the FlagGems backend switch and has nothing to
do with this package; ``VLLM_FL_TRITON_OPS=0`` is the only explicit opt-out
(for AscendC <-> Triton A/B runs) and ``VLLM_FL_TRITON_OPS_SKIP`` keeps
individual operators on their AscendC kernel.
"""

from __future__ import annotations

import functools
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_LIB_NAME = "_C_ascend"
_DISPATCH_KEY = "PrivateUse1"

#: ``<...>/vendor/ascend`` -- the operator package lives next to ``patches/``.
_ASCEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRITON_OPS_DIR = os.path.join(_ASCEND_DIR, "impl", "triton_ops")

_installed = False
_first_calls: set[str] = set()
_lib = None

#: Operators that must keep their AscendC/CANN kernel because the migrated
#: Triton implementation cannot run them yet. Every entry is backed by a
#: reproduction on 910_93 (2026-10-12); the remaining 19 adapters are enabled
#: and `DeepSeek-V4-Flash-w8a8-mtp` (TP=8) finishes with RC=0 and produces
#: tokens with six of them running on Triton (compressor, compressor_metadata,
#: inplace_partial_rotary_mul, npu_hc_pre_v2, npu_scatter_nd_update_v2,
#: moe_gating_top_k_hash).
#:
#: * ``npu_sparse_attn_sharedkv_metadata`` -- AiCPU operator, the migration
#:   contains no implementation at all; the adapter raises ``NotImplementedError``.
#:   It is called on every decode step (``attention/ascend/dsa_v1.py``).
#: * ``npu_sparse_attn_sharedkv`` -- the BiShengIR ``ub overflow`` (8691712 bits
#:   vs 1572864) is fixed by chunking the D axis of ``_saskv_out_kernel``
#:   (``SASKV_BLK_D_OUT=128``) with ``BLK_K=8``, and the wild addresses that made
#:   the device log report ``the current working operator name is
#:   _saskv_out_kernel`` / ``vector core execution is abnormal`` are clamped.
#:   The kernel now compiles and runs, but the TND+PA cases -- which is the
#:   production layout -- are numerically wrong (17 pass / 9 fail, errors
#:   ~1.5e-1 against a 1e-3 budget), so it stays on AscendC until that is fixed.
#: * ``npu_moe_init_routing_custom`` -- the dynamic-quant (``quant_mode=1``,
#:   ``scale=None``) path is implemented and matches the AscendC semantics in the
#:   standalone tests, but at the production row width the kernel still trips the
#:   BiShengIR pipeline; keep it on AscendC until that tile is fixed.
#: * ``npu_vllm_quant_lightning_indexer`` -- the Triton implementation is the
#:   "core math" version (BSND, cmp_ratio=1). The TND/decode call site passes
#:   ``query_start_loc[1:]`` (length B, no leading 0) while the implementation
#:   originally expected ``[0, a1, ...]``; that is now accepted, but the op has
#:   not been re-validated end to end in a model run yet.
#:
#: ``VLLM_FL_TRITON_OPS_SKIP`` replaces this list for experiments (exact names,
#: comma separated).
_DEFAULT_SKIPPED = (
    "npu_sparse_attn_sharedkv_metadata",
    "npu_sparse_attn_sharedkv",
    "npu_moe_init_routing_custom",
    "npu_vllm_quant_lightning_indexer",
)


def triton_ops_enabled() -> bool:
    """**默认启用**的 Triton 接管。

    这些算子的 Triton 实现和厂商自己的 Triton 算子（``ops/triton/**``）住在同一个
    包树里（``impl/triton_ops/``），因此和它们一样：**import 即生效，不需要任何
    开关**。``USE_FLAGGEMS`` 属于 FlagGems 后端的开关，与本包无关，这里不再读它。

    ``VLLM_FL_TRITON_OPS=0`` 是唯一的显式关闭手段，用于做
    AscendC ↔ Triton 的对照实验；不设置时一律启用。
    """
    explicit = os.environ.get("VLLM_FL_TRITON_OPS", "").strip()
    if explicit:
        return explicit not in ("0", "false", "False", "no")
    return True


def triton_ops_skipped() -> set[str]:
    """Operators that must keep their AscendC kernel.

    ``VLLM_FL_TRITON_OPS_SKIP`` is an escape hatch: a comma separated list of
    operator names whose adapters are not registered.  It defaults to
    :data:`_DEFAULT_SKIPPED`; set it explicitly (e.g. to a single name) to
    override that list for an experiment.
    """
    raw = os.environ.get("VLLM_FL_TRITON_OPS_SKIP")
    if raw is None:
        return set(_DEFAULT_SKIPPED)
    return {name.strip() for name in raw.split(",") if name.strip()}


def _traced(op_name: str, fn):
    """Wrap an adapter so the first dispatched call is visible in the log.

    The marker is emitted at most once per operator and per process; it exists
    to prove from a model log that ``_C_ascend`` dispatches to Triton and not to
    the AscendC kernel.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if op_name not in _first_calls:
            _first_calls.add(op_name)
            logger.info(
                "[triton_ops] %s dispatched to the Triton implementation", op_name
            )
            print(
                f"[triton_ops] {op_name} dispatched to the Triton implementation",
                flush=True,
            )
        return fn(*args, **kwargs)

    return wrapper


def _load_native_library() -> None:
    """Best-effort import of the compiled ``_C_ascend`` extension.

    Its schemas must be visible before we decide whether an operator still needs
    ``torch.library.define``, otherwise a later import of the extension would
    try to register the very same names.
    """
    try:
        import vllm_fl._C_ascend  # noqa: F401
    except Exception as exc:  # pragma: no cover - absent extension is handled
        logger.debug("triton_ops: vllm_fl._C_ascend not importable: %s", exc)


def _is_op_defined(op_name: str) -> bool:
    lib = getattr(torch.ops, _LIB_NAME, None)
    return lib is not None and hasattr(lib, op_name)


def apply_triton_ops_patch() -> bool:
    """Install the ``_C_ascend`` Triton implementations. Idempotent per process.

    Returns ``True`` when the adapters are registered, ``False`` when the
    takeover is disabled or the package is unavailable.
    """
    global _installed, _lib

    if _installed:
        return True

    if not triton_ops_enabled():
        logger.info(
            "triton_ops: VLLM_FL_TRITON_OPS=0; keeping the AscendC operators"
        )
        return False

    if not os.path.isdir(_TRITON_OPS_DIR):
        logger.warning("triton_ops: package directory missing: %s", _TRITON_OPS_DIR)
        return False
    if _TRITON_OPS_DIR not in sys.path:
        sys.path.insert(0, _TRITON_OPS_DIR)

    _load_native_library()

    from _C_ascend_bridge import to_patch_util_adapters, to_patch_util_signatures

    schemas = dict(to_patch_util_signatures().get(_LIB_NAME, {}))
    adapters = dict(to_patch_util_adapters().get(_LIB_NAME, {}))

    # 补齐原 23 条清单之外、但 csrc/ascend 里同样注册了的算子
    # （含厂商 Triton FLA 链依赖的 chunk_fwd_o / chunk_gated_delta_rule_fwd_h）。
    # 任何一步失败都不影响主注册链路。
    try:
        from _C_ascend_extra import build as _build_extra_ops

        extra_schemas, extra_adapters = _build_extra_ops()
        for name, text in extra_schemas.items():
            schemas.setdefault(name, text)
        for name, fn in extra_adapters.items():
            adapters.setdefault(name, fn)
        logger.info(
            "triton_ops: %d extra operators merged in (total schemas=%d, adapters=%d)",
            len(extra_adapters), len(schemas), len(adapters),
        )
    except Exception as exc:  # pragma: no cover - extras are best effort
        logger.warning("triton_ops: extra operator registry unavailable: %s", exc)

    if not schemas or not adapters:
        logger.warning(
            "triton_ops: empty registry (schemas=%d, adapters=%d)",
            len(schemas),
            len(adapters),
        )
        return False

    defined = 0
    for op_name, schema in schemas.items():
        if not _is_op_defined(op_name):
            torch.library.define(f"{_LIB_NAME}::{op_name}", schema)
            defined += 1

    # ⚠️ 注册方式**必须**用 ``torch.library.impl(op, key)``，不能用
    # ``torch.library.Library(name, "IMPL").impl(op, fn, key)``。
    # 后者在这套 torch_npu 上会被 `VariableFallbackKernel.cpp:250 npu_cpu_fallback`
    # 截获——**对本包"新建 schema"的算子**（`_C_ascend_extra.py` 那 11 个、
    # 以及所有未编译进原生 `_C_ascend.so` 的 vllm-ascend 算子），adapter 收到的是
    # **搬到 CPU 的张量**，于是静默退回 CPU 执行（只打一条
    # "not currently supported on the NPU backend and will fall back to run on the
    # CPU" 警告，不报错）。表现为：日志里明明有 "[triton_ops] X dispatched"，
    # 实际却在 CPU 上跑——模型里就是每 token 来回搬数据 → 极慢 → 看起来像卡死。
    # 独立复现：`_probe_ops_5.py`(旧写法 → adapter 内 device=cpu + SUSPECT REMOTE
    # ERROR) / `_probe_ops_6.py`(新写法 → device=npu:0)。
    skipped = triton_ops_skipped()
    registered = 0
    for op_name, fn in adapters.items():
        if op_name in skipped:
            logger.warning(
                "triton_ops: %s skipped by VLLM_FL_TRITON_OPS_SKIP; "
                "its AscendC kernel stays active",
                op_name,
            )
            continue
        torch.library.impl(f"{_LIB_NAME}::{op_name}", _DISPATCH_KEY)(
            _traced(op_name, fn)
        )
        registered += 1

    _installed = True
    logger.info(
        "triton_ops: registered %d/%d _C_ascend adapters (%d schemas newly "
        "defined, %d skipped) for dispatch key %s",
        registered,
        len(adapters),
        defined,
        len(skipped),
        _DISPATCH_KEY,
    )
    print(
        f"[triton_ops] _C_ascend 已注册（Triton）: {defined} 条新建 schema, "
        f"{registered}/{len(adapters)} 个 adapter（skip={sorted(skipped)}）, "
        f"dispatch_key={_DISPATCH_KEY}",
        flush=True,
    )
    return True


__all__ = ["apply_triton_ops_patch", "triton_ops_enabled", "triton_ops_skipped"]
