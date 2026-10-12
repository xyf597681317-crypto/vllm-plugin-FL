# -*- coding: utf-8 -*-
"""定案第二步：在**生产量级**形状上比 文档/Triton vs 原生（小形状可能落在退化路径上）。

判据用"全 1"这种手算最干净的图案：x 全 1 ⇒ Σx² = R ⇒ 文档语义 y = 1/sqrt(1+eps) ≈ 1.0；
以及"单个 1" ⇒ y = 1/sqrt(1/R + eps) = sqrt(R)。

用法：DSH_NPU=npu:15 python3 -W ignore _probe_native_r2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:15")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_fl.ascend_custom_ops import enable_custom_op  # noqa: E402
import triton_ops.hc_pre_inv_rms as ir  # noqa: E402

print(f"[r2] dev={DEV} enable_custom_op() -> {enable_custom_op()}", flush=True)
EPS = 1e-6


def case(shape, pattern):
    xd = torch.zeros(*shape, dtype=torch.float64)
    flat = xd.reshape(-1, shape[-2] * shape[-1])          # [rows, R]
    if pattern == "ones":
        flat[:] = 1.0
    elif pattern == "one_first":
        flat[:, 0] = 1.0
    elif pattern == "one_last":
        flat[:, -1] = 1.0
    R = shape[-2] * shape[-1]
    s = float((flat[0] ** 2).sum())
    doc = 1.0 / (s / R + EPS) ** 0.5
    x = xd.to(torch.bfloat16).to(DEV)
    try:
        n = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, EPS)
        torch.npu.synchronize()
        nv = [round(float(v), 6) for v in n.detach().cpu().flatten()[:3]]
    except Exception as ex:  # noqa: BLE001
        nv = f"EXC {type(ex).__name__}: {str(ex)[:80]}"
    t = ir.hc_pre_inv_rms(x, EPS)
    torch.npu.synchronize()
    tv = [round(float(v), 6) for v in t.detach().cpu().flatten()[:3]]
    print(f"  shape={str(tuple(shape)):22s} R={R:6d} {pattern:10s} "
          f"手算(文档)={doc:.6f} | Triton={tv} | 原生={nv}", flush=True)


def main():
    print("== 小形状（上一轮用的退化档） ==", flush=True)
    case((2, 4, 32), "ones")
    print("== 生产量级 ==", flush=True)
    case((64, 4, 512), "ones")          # R=2048
    case((64, 4, 512), "one_first")
    case((64, 4, 512), "one_last")
    case((16, 4, 4096), "ones")         # R=16384（hc=4,d=4096 ⇒ DSV4 的 hc_dim 量级）
    case((8, 4, 7168), "ones")
    case((1, 64, 4, 512), "ones")       # 4-D：模型里真正的调用形态
    print("[r2] done", flush=True)


if __name__ == "__main__":
    main()
