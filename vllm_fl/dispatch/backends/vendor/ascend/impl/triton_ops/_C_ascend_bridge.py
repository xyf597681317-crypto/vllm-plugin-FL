# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FlagGems ↔ `_C_ascend` 集成的**桥接层**（Lead 维护）。

## 这个文件解决什么问题

vllm-fl 的 AscendC 算子走 `torch.ops._C_ascend.<op>` 调用。要在 FlagGems 里接管它们，
需要三样东西（见 `flag_gems/patches/patch_util.py` 的既有机制）：

1. `_VLLM_OPS_SIGNATURES[lib][op]` —— schema 字符串（`torch.library.define` 用）
2. `init_vllm_libraries()` 的 `libs` dict 里要有该 `lib` 的 `Library` 对象
3. `_VLLM_OPS_IMPLS[lib][op]` —— adapter 实现函数

`patch_util.py` 原本只有 `_C` / `_moe_C` / `_vllm_fa3_C` / `_C_cache_ops`。
**本模块提供 `_C_ascend` 的第 1、3 项**，第 2 项由 `apply_patch_util_changes()` 打印出
需要手工施加的补丁（我们**不改 FlagGems 上游源码**，避免与上游冲突）。

## 数据来源（两组并行交付，均已自测）

* `_C_ascend_sigs.py`    —— 23 条 schema（权威出处：两仓库 `torch_binding.cpp` 的 `TORCH_LIBRARY` 块）
* `_C_ascend_ops_a.py`   —— attention 组 12 个 adapter
* `_C_ascend_ops_b.py`   —— moe/量化组 12 个 adapter

## 已知分歧与风险（如实记录，不隐藏）

* **9 个算子的实际注册名带 `npu_` 前缀**（task 短名直接用会永不命中）→ 见 `TASK_NAME_TO_OP`
* **`npu_lightning_indexer` schema 是调用点反推的**（两仓库都无 `ops.def`，op 由 torch_npu 提供）
  → 接入后**必须**用 `torch.ops._C_ascend.npu_lightning_indexer._schemas` 实测校对
* **VF/VA 有 4 处同名不同 schema**（`mla_preprocess` / `npu_sparse_attention_score` /
  `inplace_partial_rotary_mul` / `dispatch_ffn_combine`）→ **以 VF 为准**（生产在 vllm_fl）
* **`dispatch_ffn_combine_bf16` 不是独立 torch op**（由 `dispatch_ffn_combine` 按 dtype 内部分派）
"""

from __future__ import annotations

import sys
from pathlib import Path

LIB_NAME = "_C_ascend"

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _import_groups():
    """惰性导入两组（模块内不 import torch/triton，宿主机也能 import 本桥接层）。"""
    import _C_ascend_sigs as sigsA
    import _C_ascend_ops_a as groupA

    groupB = None
    try:
        import _C_ascend_ops_b as groupB  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - B 组未就绪时仍可用 A 组
        groupB = None
    return sigsA, groupA, groupB


def to_patch_util_signatures() -> dict:
    """返回 `{LIB_NAME: {op_name: schema}}`，可直接 `_VLLM_OPS_SIGNATURES.update(...)`。

    ⚠️ `_C_ascend_sigs.to_patch_util_signatures()` 已经返回**修正过 `->` 空格**的版本
    （C++ `ops.def` 容忍 `->(Tensor`、Python `torch.library.define` 不容忍）。
    """
    sigsA, _, _ = _import_groups()
    sigs = sigsA.to_patch_util_signatures()
    # 兼容：若返回的不是 {LIB_NAME: {...}} 形态，则包一层
    if LIB_NAME not in sigs:
        inner = sigs.get("signatures", sigs)
        return {LIB_NAME: dict(inner)}
    return {LIB_NAME: dict(sigs[LIB_NAME])}


def to_patch_util_adapters() -> dict:
    """返回 `{LIB_NAME: {op_name: adapter}}`，可直接 `_VLLM_OPS_IMPLS.update(...)`。"""
    _, groupA, groupB = _import_groups()
    adapters: dict = dict(getattr(groupA, "_C_ASCEND_IMPLS_A", {}))
    if groupB is not None:
        adapters.update(groupB.to_patch_util_adapters().get(LIB_NAME, {}))
    return {LIB_NAME: adapters}


def task_name_to_op() -> dict:
    """task 短名 → 实际注册名（9 个带 `npu_` 前缀）。"""
    sigsA, _, _ = _import_groups()
    return dict(sigsA.TASK_NAME_TO_OP)


def coverage() -> dict:
    """覆盖情况自检：schema 数 / adapter 数 / 差集。"""
    sigs, adapters = to_patch_util_signatures()[LIB_NAME], to_patch_util_adapters()[LIB_NAME]
    return {
        "lib": LIB_NAME,
        "n_schemas": len(sigs),
        "n_adapters": len(adapters),
        "schema_without_adapter": sorted(set(sigs) - set(adapters)),
        "adapter_without_schema": sorted(set(adapters) - set(sigs)),
    }


# --------------------------------------------------------------------------
# FlagGems `patch_util.py` 需要施加的两处改动（我们不改上游源码，只提供补丁文本）
# --------------------------------------------------------------------------
PATCH_UTIL_LIB_ENTRY = '''\
    "_C_ascend": ("torch_npu", "vllm_fl._C_ascend"),
'''

PATCH_UTIL_INSTRUCTIONS = f"""\
要在 FlagGems 里启用 `{LIB_NAME}` 接管，`src/flag_gems/patches/patch_util.py` 需加两处：

(1) `_ensure_vllm_library_exists()` 的 `module_map` 里增加一行：
{PATCH_UTIL_LIB_ENTRY}
    ⚠️ 若不需自动 import C++ 扩展（我们让 `torch.library.define` 自建命名空间），
       直接给 `module_map` 加 `"{LIB_NAME}": ()` 亦可 —— `_ensure_vllm_library_exists`
       会返回 False，随后 `_define_op_if_not_exists()` 用 `torch.library.define` 建 op。

(2) `init_vllm_libraries()` 里增加：
        {LIB_NAME}_lib = torch.library.Library("{LIB_NAME}", "IMPL")
    并把它加进 `libs` dict（key 为 "{LIB_NAME}"）。

(3) 注册实现（`apply_gems_patches_to_vllm()` 或 `patch_empty_vllm()` 里）：
        from flag_gems_integration._C_ascend_bridge import (
            to_patch_util_signatures, to_patch_util_adapters, LIB_NAME)
        _VLLM_OPS_SIGNATURES.update(to_patch_util_signatures())
        _VLLM_OPS_IMPLS.update(to_patch_util_adapters())
    ⚠️ `trigger` 时机：`_VLLM_OPS_SIGNATURES` 要在 `init_vllm_libraries()` **之前**更新，
       否则 `_define_op_if_not_exists` 拿不到 schema。
"""


def apply_patch_util_changes(patch_util_path: str | None = None, dry_run: bool = True) -> str:
    """就地修改 FlagGems 的 `patch_util.py`，把 `_C_ascend` 接进去。

    默认 `dry_run=True` **只返回将要施加的改动**，不写盘。
    传 `dry_run=False` + `patch_util_path` 才实际改写（会先备份 `.bak`）。
    """
    if dry_run or not patch_util_path:
        return PATCH_UTIL_INSTRUCTIONS

    p = Path(patch_util_path)
    src = p.read_text(encoding="utf-8")
    orig = src

    # (1) module_map
    anchor = '    module_map = {\n'
    if LIB_NAME not in src.split("module_map = {", 1)[-1].split("}", 1)[0]:
        src = src.replace(anchor, anchor + f'        "{LIB_NAME}": (),\n', 1)

    # (2) Library + libs dict
    if f'{LIB_NAME}_lib = torch.library.Library' not in src:
        src = src.replace(
            "    vllm_C_cache_ops_lib = torch.library.Library(",
            f'    {LIB_NAME}_lib = torch.library.Library("{LIB_NAME}", "IMPL")\n'
            "    vllm_C_cache_ops_lib = torch.library.Library(",
            1,
        )
    if f'"{LIB_NAME}": {LIB_NAME}_lib,' not in src:
        src = src.replace(
            '        "_C_cache_ops": vllm_C_cache_ops_lib,\n',
            '        "_C_cache_ops": vllm_C_cache_ops_lib,\n'
            f'        "{LIB_NAME}": {LIB_NAME}_lib,\n',
            1,
        )

    if src == orig:
        return "no change (already patched?)"
    p.with_suffix(p.suffix + ".bak").write_text(orig, encoding="utf-8")
    p.write_text(src, encoding="utf-8")
    return f"patched {p} (backup: {p.name}.bak)"


if __name__ == "__main__":  # pragma: no cover - 人工核对用
    cov = coverage()
    print(f"[{LIB_NAME}] schemas={cov['n_schemas']} adapters={cov['n_adapters']}")
    if cov["schema_without_adapter"]:
        print(f"  有 schema 无 adapter: {cov['schema_without_adapter']}")
    if cov["adapter_without_schema"]:
        print(f"  有 adapter 无 schema: {cov['adapter_without_schema']}")
    print()
    print(PATCH_UTIL_INSTRUCTIONS)
