# -*- coding: utf-8 -*-
"""定案：原生 `hc_pre_inv_rms` 的**实际归约窗口/归一化分母**是什么。

已确证的权威事实（读容器里的原生 AscendC 源码：
`_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/.../ascendc/hc_pre_inv_rms/`）：
```
Cast(castLocal, xLocal, R);  Mul(castLocal, castLocal, castLocal, R);   // x²
ReduceSumHalfInterval(y[idx], castLocal[idx*rAlignB32], R);            // Σ_{R} x²
float meanCof = 1.0f / R;   Muls(y, y, meanCof);   Adds(y, y, epsilon);
Duplicate(one, 1.0f);  Sqrt(y, y);  Div(y, one, y);                     // y = 1/sqrt(Σ/R + eps)
```
⇒ **原生公式与"文档/Triton"完全一致**（`1/sqrt(Σx²/R + eps)`），分歧只可能出在 **R 的取值**。
本探针用"前 k 个元素为 1、其余为 0"的图案扫 k，直接测出有效归一化：

    文档口径 R=128 → y = 1/sqrt(k/128 + eps)
    若有效分母是 64（= R/2） → y = 1/sqrt(k/64 + eps)
    若把和算了两遍（2Σ/R）   → y = 1/sqrt(2k/128 + eps)  ← 与上一行数值相同
再补一档"1 只放在后半段（位置 64..127）"，用于区分"只归约前半个窗口"。

用法：DSH_NPU=npu:15 python3 -W ignore _probe_native_r.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:15")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagFems/src".replace("FlagFems", "FlagGems"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_fl.ascend_custom_ops import enable_custom_op  # noqa: E402
import triton_ops.hc_pre_inv_rms as ir  # noqa: E402

print(f"[R] dev={DEV} enable_custom_op() -> {enable_custom_op()}", flush=True)

HC, D, R = 4, 32, 128
EPS = 1e-6


def build(xd):
    return xd.to(torch.bfloat16).to(DEV)


def report(tag, xd_np):
    x = build(xd_np)
    try:
        n = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, EPS)
        torch.npu.synchronize()
        nv = float(n.detach().cpu().flatten()[0])
    except Exception as ex:  # noqa: BLE001
        nv = float("nan")
        print(f"    原生 EXC {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    t = ir.hc_pre_inv_rms(x, EPS)
    torch.npu.synchronize()
    tv = float(t.detach().cpu().flatten()[0])
    s = float((xd_np.double() ** 2).sum())
    h_doc = 1.0 / (s / R + EPS) ** 0.5
    h_half = 1.0 / (s / (R // 2) + EPS) ** 0.5
    print(f"  {tag:34s} Σx²={s:8.1f} | 文档/Triton={tv:.6f} | 原生={nv:.6f} | "
          f"若R=128→{h_doc:.6f} 若R=64→{h_half:.6f} | 原生/文档={nv / h_doc:.4f}", flush=True)


def main():
    xd = torch.zeros(2, HC, D, dtype=torch.float64)
    print("== A) 前 k 个元素为 1（k 扫） ==", flush=True)
    for k in (4, 8, 16, 32, 64, 96, 128):
        xd.zero_()
        xd[0].reshape(-1)[:k] = 1.0
        report(f"前 {k} 个 1", xd)
    print("== B) 1 只放在后半段（位置 64..127） ==", flush=True)
    xd.zero_()
    xd[0].reshape(-1)[64:128] = 1.0
    report("后半段 64 个 1", xd)
    print("== C) 1 只放在最后一个元素（位置 127） ==", flush=True)
    xd.zero_()
    xd[0].reshape(-1)[127] = 1.0
    report("仅位置 127 = 1", xd)
    print("== D) 全 1（Σ=128） ==", flush=True)
    xd.zero_()
    xd[0] = 1.0
    report("全 1", xd)
    print("[R] done", flush=True)


if __name__ == "__main__":
    main()
