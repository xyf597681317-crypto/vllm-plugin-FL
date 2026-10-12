# -*- coding: utf-8 -*-
"""task-3 探针：生产宽度下 gmm_v2 / gmm_v1 的 UB 可行性（只读，调用 host 函数原样跑）。

- `grouped_matmul_swiglu_quant_v2`                     （本任务写范围内）
- `grouped_matmul_swiglu_quant_weight_nz_tensor_list`  （**同族、同样的 host tiling**，
  `blk_m=16 / blk_k=next_pow2(K) / blk_n=next_pow2(I)`；不在本任务写范围，只测不改）

生产 I = moe_intermediate_size ∈ {512, 704, 768, 1024, 1408, 1536, 2048, 2304}；
K = hidden ∈ {512, 7168, ...}。打印 BiShengIR 原始 ub overflow 数字。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_ub2.py
"""
import os
import re
import sys

import torch
import torch_npu  # noqa: F401

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
from triton_ops.grouped_matmul_swiglu_quant_v2 import (  # noqa: E402
    grouped_matmul_swiglu_quant_v2)
from triton_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list import (  # noqa: E402
    grouped_matmul_swiglu_quant_weight_nz_tensor_list)


def _report(tag, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  [OK   ] {tag}", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001
        msg = " ".join(str(ex).split())
        m = re.search(r"ub overflow, requires (\d+) bits while (\d+) bits available", msg)
        if m:
            print(f"  [UB!  ] {tag}  requires={m.group(1)} bits (available={m.group(2)})", flush=True)
        else:
            print(f"  [FAIL ] {tag}  {type(ex).__name__}: {msg[:200]}", flush=True)
        return False


def case_v2(M, K, I, G=1):
    N = 2 * I
    torch.manual_seed(71)
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-128, 128, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.001
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.001
    gl = torch.full((G,), M, dtype=torch.int64, device=DEV)
    tag = f"gmm_v2  M={M} K={K} I={I}(N={N}) G={G}"

    def fn():
        grouped_matmul_swiglu_quant_v2(x, list(w), list(ws), xs, gl)
    return _report(tag, fn)


def case_v1(M, K, I, G=1):
    N = 2 * I
    torch.manual_seed(72)
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-128, 128, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.001
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.001
    gl = torch.full((G,), M, dtype=torch.int64, device=DEV)
    tag = f"gmm_v1  M={M} K={K} I={I}(N={N}) G={G}"

    def fn():
        grouped_matmul_swiglu_quant_weight_nz_tensor_list(x, list(w), list(ws), xs, gl)
    return _report(tag, fn)


def main():
    print(f"[ub2] dev={DEV}", flush=True)
    print("== A) gmm_v2：生产 I × 小 K（隔离 BLK_N 影响） ==", flush=True)
    for I in (256, 512, 2048):
        case_v2(64, 128, I)
    print("== B) gmm_v2：生产 I × 生产 K（DeepSeek hidden=7168 / Qwen=2048/4096） ==", flush=True)
    for K in (2048, 4096, 7168):
        for I in (512, 2048):
            case_v2(64, K, I)
    print("== C) gmm_v1（同族，不在写范围） ==", flush=True)
    for K in (128, 2048, 7168):
        for I in (256, 512, 2048):
            case_v1(64, K, I)
    print("[ub2] done", flush=True)


if __name__ == "__main__":
    main()
