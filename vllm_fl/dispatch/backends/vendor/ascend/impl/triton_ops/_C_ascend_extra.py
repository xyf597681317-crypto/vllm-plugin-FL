# -*- coding: utf-8 -*-
"""额外接管的 `_C_ascend` 算子（把它们从 AscendC 切到本仓库的 Triton 实现）。

原迁移清单（`_C_ascend_sigs.py` / `_C_ascend_ops_*.py`，合计 130 KB）只收录了 23 条 schema，
而 `csrc/ascend/torch_binding.cpp` 的 ``TORCH_LIBRARY(_C_ascend, ops)`` 块实际注册了 27 个算子。
本文件把差额补齐——**全部走 Triton，不留 AscendC 回退**。

其中 `chunk_gated_delta_rule_fwd_h` 与 `chunk_fwd_o` 尤其关键：厂商自己的 Triton FLA 链
（`ops/triton/fla/chunk.py` L115 / L197）就在调它们。

适配层按算子定制，处理三类差异（都是实测踩出来的）：
  * 入口函数名 ≠ 模块名（`causal_conv1d.py` 的入口是 `causal_conv1d_fn`）；
  * schema 首参是**原地写**的输出张量（`npu_causal_conv1d_custom` 的 `output` → 实现侧 `out=`）；
  * 形参名大小写/命名不同（`A_log`→`a_log`、`smooth_scale`→`smooth_scale1`、
    `npu_hc_pre` 的 `norm_eps`/`hc_eps` 顺序与实现相反）。

schema 逐字抄自 `csrc/ascend/torch_binding.cpp`。
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import sys

import torch

logger = logging.getLogger(__name__)

#: op 名 -> (schema, impl 模块名, 入口函数名)
EXTRA_OPS: dict[str, tuple[str, str, str]] = {
    "chunk_gated_delta_rule_fwd_h": (
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=Non"
        "e, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_st"
        "ate=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_se"
        "qlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? trans"
        "pose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor fina"
        "l_state_out)",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_gated_delta_rule_fwd_h",
    ),
    "chunk_fwd_o": (
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tens"
        "or? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_i"
        "ndices=None, int? chunk_size=None, bool? transpose_state_layout=False) -"
        "> Tensor",
        "chunk_fwd_o",
        "chunk_fwd_o",
    ),
    "npu_recurrent_gated_delta_rule": (
        "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, T"
        "ensor(a!) state, *, Tensor? beta=None, float? scale=None, Tensor? actual"
        "_seq_lengths=None, Tensor? ssm_state_indices=None, Tensor? num_accepted_"
        "tokens=None, Tensor? g=None, Tensor? gk=None) -> Tensor",
        "recurrent_gated_delta_rule",
        "recurrent_gated_delta_rule",
    ),
    "npu_fused_gdn_gating": (
        "npu_fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, f"
        "loat beta=1.0, float threshold=20.0) -> (Tensor g, Tensor beta_output)",
        "fused_gdn_gating",
        "fused_gdn_gating",
    ),
    "npu_causal_conv1d_custom": (
        "npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor "
        "conv_state, Tensor? bias_opt, Tensor? query_start_loc_opt, Tensor? cache"
        "_indices_opt, Tensor? initial_state_mode_opt, Tensor? num_accepted_token"
        "s_opt, int activation_mode, int pad_slot_id, int run_mode) -> Tensor",
        "causal_conv1d",
        "causal_conv1d_fn",
    ),
    "npu_dequant_swiglu_quant": (
        "npu_dequant_swiglu_quant(Tensor x, *, Tensor? weight_scale=None, Tensor?"
        " activation_scale=None, Tensor? bias=None, Tensor? quant_scale=None, Ten"
        "sor? quant_offset=None, Tensor? group_index=None, bool activate_left=Tru"
        "e, int quant_mode=0, int swiglu_mode=0, float clamp_limit=0.0, float glu"
        "_alpha=1.0, float glu_bias=0.0) -> (Tensor y, Tensor scale)",
        "dequant_swiglu_quant",
        "dequant_swiglu_quant",
    ),
    "npu_hc_post": (
        "npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tens"
        "or",
        "hc_post",
        "hc_post",
    ),
    "npu_hc_pre": (
        "npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int "
        "hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) -> (Tensor"
        " out0, Tensor out1, Tensor out2)",
        "hc_pre",
        "hc_pre",
    ),
    "npu_hc_pre_inv_rms": (
        "npu_hc_pre_inv_rms(Tensor x, float epsilon=1e-20) -> Tensor",
        "hc_pre_inv_rms",
        "hc_pre_inv_rms",
    ),
    "npu_hc_pre_sinkhorn": (
        "npu_hc_pre_sinkhorn(Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor "
        "hc_base, Tensor x, int hc_mult, int hc_sinkhorn_iters, float hc_eps) -> "
        "(Tensor out0, Tensor out1, Tensor out2)",
        "hc_pre_sinkhorn",
        "hc_pre_sinkhorn",
    ),
    "npu_rms_norm_dynamic_quant": (
        "npu_rms_norm_dynamic_quant(Tensor x, Tensor gamma, Tensor? smooth_scale="
        "None, Tensor? beta=None, float epsilon=1e-6) -> (Tensor y_out, Tensor sc"
        "ale_out)",
        "rms_norm_dynamic_quant",
        "rms_norm_dynamic_quant",
    ),
}


#: 本文件所在目录 = triton_ops 包目录；实现文件内部用的是相对导入
#: （``from .common import ...``），所以**必须按包导入**（``triton_ops.<module>``），
#: 不能当顶层模块 import —— 否则报 "attempted relative import with no known parent package"。
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
_PKG_NAME = os.path.basename(_PKG_DIR)


def _import_impl(module: str):
    """按包导入实现模块（与 ``_C_ascend_ops_b.py::_impl`` 同一套做法）。"""
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    return importlib.import_module(f"{_PKG_NAME}.{module}")


def _split_top_level(s: str) -> list[str]:
    """按顶层逗号切分（不切括号/方括号内部的逗号）。"""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _list_params(schema: str) -> tuple[str, ...]:
    """从 schema 里取出所有 ``int[]`` 参数名（这些需要 list -> tensor）。"""
    head = schema.split(") ->", 1)[0]
    inner = head[head.index("(") + 1:]
    names = []
    for chunk in _split_top_level(inner):
        c = chunk.strip()
        if "int[]" not in c:
            continue
        tail = c.split("[]", 1)[1]
        name = re.split(r"[=,]", tail.lstrip("? "))[0].strip()
        if name:
            names.append(name)
    return tuple(names)


def _as_tensor(value, device):
    if value is None or torch.is_tensor(value):
        return value
    return torch.tensor(list(value), dtype=torch.int64, device=device)


def _first_device(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if torch.is_tensor(a):
            return a.device
    return torch.device("npu")


def _make_adapter(op_name: str, schema: str, module: str, func: str):
    list_params = _list_params(schema)

    def adapter(*args, **kwargs):
        impl = getattr(_import_impl(module), func)
        dev = _first_device(args, kwargs)
        for name in list_params:
            if name in kwargs:
                kwargs[name] = _as_tensor(kwargs[name], dev)
        return impl(*args, **kwargs)

    adapter.__name__ = f"adapter_{op_name}"
    adapter.__qualname__ = adapter.__name__
    return adapter


# --------------------------------------------------------------------------
# 定制适配（通用适配覆盖不了的）
# --------------------------------------------------------------------------
def _adapter_npu_recurrent_gated_delta_rule(query, key, value, state, *, beta=None, scale=None,
                                            actual_seq_lengths=None, ssm_state_indices=None,
                                            num_accepted_tokens=None, g=None, gk=None):
    """schema 的参数顺序与实现不同：

    schema : ``(query, key, value, Tensor(a!) state, *, beta, scale, actual_seq_lengths,
               ssm_state_indices, num_accepted_tokens, g, gk)``
    实现   : ``(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
               g, gk, num_accepted_tokens, scale_value, *, scale)``

    按关键字整包转发会撞位（实测 `got multiple values for argument 'beta'`），
    故这里显式按实现的位置顺序转发，并把 schema 的 ``scale`` 绑到 ``scale_value``。

    另外两处语义差异（**都是实测踩出来的静默错值风险**）：
      * 实现返回 ``(out, final_state)`` 二元组，而 schema 声明 ``-> Tensor`` ⇒ 必须只回传 ``out``；
        否则调用方拿到 tuple，会在下游（`vllm_fl/.../fla/chunk.py:341` 的 beta 形状断言）炸掉。
      * schema 把 ``state`` 标为 ``Tensor(a!)``（**原地写**），实现却是返回新的 ``final_state``
        ⇒ 这里显式 ``state.copy_(final)``，否则调用方的 state 永远不更新（**静默错值**）。
    """
    fn = getattr(_import_impl("recurrent_gated_delta_rule"), "recurrent_gated_delta_rule")
    res = fn(
        query, key, value, beta, state,
        actual_seq_lengths, ssm_state_indices,
        g=g, gk=gk, num_accepted_tokens=num_accepted_tokens,
        scale_value=1.0 if scale is None else scale,
    )
    out, final = res if isinstance(res, tuple) else (res, None)
    if torch.is_tensor(final) and torch.is_tensor(state) and tuple(final.shape) == tuple(state.shape):
        state.copy_(final)
    return out


def _adapter_npu_causal_conv1d_custom(output, x, weight, conv_state=None, bias_opt=None,
                                      query_start_loc_opt=None, cache_indices_opt=None,
                                      initial_state_mode_opt=None, num_accepted_tokens_opt=None,
                                      activation_mode=0, pad_slot_id=-1, run_mode=0):
    """`npu_causal_conv1d_custom`：schema 首参是**原地写**的 `output`，而实现入口
    `causal_conv1d_fn(x, weight, ...)` 首参是 `x`、输出走关键字 `out=`。
    直接按位置转发会把 `output` 当成 `x`（实测报
    `module 'triton_ops.causal_conv1d' has no attribute 'causal_conv1d'`），故显式映射。

    调用点契约（`impl/gdn.py:199` 等四处）用的是 `output`（原地写），返回值也回传同一张量。
    """
    fn = getattr(_import_impl("causal_conv1d"), "causal_conv1d_fn")
    return fn(
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias_opt,
        query_start_loc_opt=query_start_loc_opt,
        cache_indices_opt=cache_indices_opt,
        initial_state_mode_opt=initial_state_mode_opt,
        num_accepted_tokens_opt=num_accepted_tokens_opt,
        activation_mode=activation_mode,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        out=output,
    )


def _adapter_npu_rms_norm_dynamic_quant(x, gamma, smooth_scale=None, beta=None, epsilon=1e-6):
    """schema 只有一个 `smooth_scale`、返回 ``(y_out, scale_out)`` 两元组；
    实现拆成 `smooth_scale1/smooth_scale2`、返回 **四**元组 ``(y1, y2, s1, s2)``。
    AscendC 的单 smooth 路径 = 第一组输出 ⇒ 取 ``(y1, s1)``（否则调用方拿到 4 元组会炸）。
    """
    fn = getattr(_import_impl("rms_norm_dynamic_quant"), "rms_norm_dynamic_quant")
    res = fn(x, gamma, smooth_scale1=smooth_scale, beta=beta, epsilon=epsilon)
    if isinstance(res, tuple):
        if len(res) >= 4:
            return res[0], res[2]
        if len(res) == 2:
            return res
    return res


def _adapter_npu_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """schema 首参名是 `A_log`，实现形参是 `a_log`（关键字调用会炸）。"""
    fn = getattr(_import_impl("fused_gdn_gating"), "fused_gdn_gating")
    return fn(A_log, a, b, dt_bias, beta=beta, threshold=threshold)


def _adapter_npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult=4,
                        hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6):
    """schema 的 (norm_eps, hc_eps) 与实现 `hc_pre(..., hc_eps=..., norm_eps=...)` **顺序相反**。"""
    fn = getattr(_import_impl("hc_pre"), "hc_pre")
    return fn(x, hc_fn, hc_scale, hc_base, hc_mult=hc_mult,
              hc_sinkhorn_iters=hc_sinkhorn_iters, hc_eps=hc_eps, norm_eps=norm_eps)


#: 需要特殊适配的算子
CUSTOM_ADAPTERS = {
    "npu_recurrent_gated_delta_rule": _adapter_npu_recurrent_gated_delta_rule,
    "npu_causal_conv1d_custom": _adapter_npu_causal_conv1d_custom,
    "npu_rms_norm_dynamic_quant": _adapter_npu_rms_norm_dynamic_quant,
    "npu_fused_gdn_gating": _adapter_npu_fused_gdn_gating,
    "npu_hc_pre": _adapter_npu_hc_pre,
}


def build() -> tuple[dict[str, str], dict[str, object]]:
    """返回 (schemas, adapters)。任何单个算子导入失败都不影响其余算子。"""
    schemas: dict[str, str] = {}
    adapters: dict[str, object] = {}
    for op_name, (schema, module, func) in EXTRA_OPS.items():
        try:
            _import_impl(module)
        except Exception as exc:  # pragma: no cover - 缺实现时跳过
            logger.warning("extra triton op %s unavailable: %s", op_name, exc)
            continue
        schemas[op_name] = schema
        adapters[op_name] = CUSTOM_ADAPTERS.get(
            op_name, _make_adapter(op_name, schema, module, func))
    return schemas, adapters
