"""swiglu_group_quant —— 不可用证据复核 + 显式拒绝验证（**不是**数值精度测试）。

本测试刻意**不做数值对拍**：本算子在目标环境 910_93 上没有可运行的真算子，
也没有能判定布局的硬件手段，任何"对拍出来的 PASS"都可能是假 PASS。
它只验证两件事，符合 Lead 要求的"诚实标注"：
  1. 容器里 `torch.ops._C_ascend.npu_swiglu_group_quant` 确实不存在（保存现场证据）；
  2. 本包入口 `swiglu_group_quant()` 显式抛 `NotImplementedError`（而不是静默算错）。
"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
_mod = importlib.import_module(_PKG + ".swiglu_group_quant")

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


emit("=" * 96)
emit("swiglu_group_quant: 可用性现场复核 + 显式拒绝验证（无数值对拍）")
emit("=" * 96)

# 1) 现场复核：目标环境是否存在该算子
has = hasattr(torch.ops._C_ascend, "npu_swiglu_group_quant")
related = []
try:
    related = [n for n in dir(torch.ops._C_ascend) if "swiglu" in n.lower()]
except Exception as e:  # noqa: BLE001
    related = [f"<enum failed: {e}>"]
emit(f"  [1] torch.ops._C_ascend.npu_swiglu_group_quant 存在? {has}")
emit(f"      相关 swiglu 算子: {related[:20]}")
emit(f"      设备: {torch.npu.get_device_name(0)}")

# 2) 显式拒绝
raised = None
try:
    _mod.swiglu_group_quant(torch.zeros(1, 8, device=DEV), None, None)
except NotImplementedError as e:
    raised = str(e)
except Exception as e:  # noqa: BLE001
    raised = f"WRONG-EXC {type(e).__name__}: {e}"

ok = (raised is not None) and raised.startswith("swiglu_group_quant 仅注册 ascend950")
emit(f"  [2] 入口抛 NotImplementedError 且理由正确? {ok}")
if raised:
    emit(f"      理由: {raised[:200]}...")

emit()
verdict = "全部通过 ✅（不可用已如实标注，且入口不会静默算错）" if ok else "存在问题 ❌"
emit(f"总体: {verdict}")
emit("说明：本算子无数值 PASS 结论；在 ascend950 环境按模块 docstring §1-§5 实现后再补精度测试。")

txt = "\n".join(L)
open("/root/dsh_work/swiglu_group_quant_test.txt", "w", encoding="utf-8").write(txt + "\n")
