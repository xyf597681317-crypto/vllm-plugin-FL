# -*- coding: utf-8 -*-
"""验证 `_C_ascend_extra.py`（Lead 的文件，只读）：注册项 + 哪些算子能用 AscendC 当裁判。

用法（容器内）：
    DSH_NPU=npu:14 python3 -W ignore _probe_extra_reg.py
"""
import io
import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)          # ← 教训：换卡必须 set_device
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def op_exists(name):
    try:
        return hasattr(torch.ops._C_ascend, name), None
    except Exception as ex:  # noqa: BLE001
        return False, f"{type(ex).__name__}: {str(ex)[:80]}"


def main():
    print(f"[reg] dev={DEV} triton={torch.__version__} VLLM_FL_TRITON_OPS="
          f"{os.environ.get('VLLM_FL_TRITON_OPS', '<unset>')}", flush=True)
    import triton_ops._C_ascend_extra as extra

    print("== A) build() ==", flush=True)
    warns = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            schemas, adapters = extra.build()
        except Exception as ex:  # noqa: BLE001
            print(f"  build() 抛异常: {type(ex).__name__}: {ex}", flush=True)
            return
        warns = list(w)
    print(f"  schemas={len(schemas)}  adapters={len(adapters)}  warnings={len(warns)}", flush=True)
    for w_ in warns[:10]:
        print(f"    WARN: {w_.category.__name__}: {str(w_.message)[:120]}", flush=True)
    for name in extra.EXTRA_OPS:
        custom = name in extra.CUSTOM_ADAPTERS
        print(f"    {name:36s} schema={'Y' if name in schemas else 'N'} "
              f"adapter={'Y' if name in adapters else 'N'} "
              f"{'CUSTOM' if custom else 'generic'}", flush=True)
    # schema 与实现是否真能 import
    print("== A2) 每条 schema 的 impl 模块 import 检查 ==", flush=True)
    for name, (_schema, module, func) in extra.EXTRA_OPS.items():
        try:
            mod = extra._import_impl(module)
            has = hasattr(mod, func)
            print(f"    {name:36s} {module}.{func} -> {'OK' if has else 'MISSING FUNC'}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"    {name:36s} IMPORT FAIL {type(ex).__name__}: {str(ex)[:80]}", flush=True)

    print("== B) torch.ops._C_ascend 存在性（= 能否拿 AscendC 当裁判） ==", flush=True)
    print("  -- B1) 裸 torch_npu（未加载 patch）--", flush=True)
    for name in extra.EXTRA_OPS:
        ok, err = op_exists(name)
        print(f"    {name:36s} {ok}" + (f"  ({err})" if err else ""), flush=True)

    print("  -- B2) import patch 模块之后 --", flush=True)
    try:
        import vllm_fl.dispatch.backends.vendor.ascend.patch  # noqa: F401
        print("    patch 模块 import OK", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"    patch import 失败: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    for name in extra.EXTRA_OPS:
        ok, err = op_exists(name)
        print(f"    {name:36s} {ok}" + (f"  ({err})" if err else ""), flush=True)

    print("  -- B3) 显式 apply_triton_ops_patch() 之后 --", flush=True)
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_triton_ops import (
            apply_triton_ops_patch)
        res = apply_triton_ops_patch()
        print(f"    apply_triton_ops_patch() -> {res}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"    调用失败: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    for name in extra.EXTRA_OPS:
        ok, err = op_exists(name)
        print(f"    {name:36s} {ok}" + (f"  ({err})" if err else ""), flush=True)
    print("[reg] done", flush=True)


if __name__ == "__main__":
    main()
