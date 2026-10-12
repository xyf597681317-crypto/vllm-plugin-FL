# -*- coding: utf-8 -*-
"""task-3 探针：BiShengIR UB 容量边界（grouped_matmul_swiglu_quant_v2 的 BLK_N 上限）。

生产 `moe_intermediate_size` ∈ {512, 704, 768, 1024, 1408, 1536, 2048, 2304}（见
/public-flash/models/*/config.json），而 gmm_v2 的 I = weight_scale 长度的一半
= moe_intermediate_size，`blk_n = next_pow2(I)`（`grouped_matmul_swiglu_quant_v2.py` host 侧）。
本探针在**不改生产文件**的前提下，用生产 `_gmm_v2_kernel` 原样跑不同 BLK_N/BLK_K/BLK_M，
打印 BiShengIR 的原始报错文本，确定「能编译 / ub overflow」的分界与可用余量。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_ub.py
"""
import os
import re
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
from triton_ops.grouped_matmul_swiglu_quant_v2 import _gmm_v2_kernel  # noqa: E402
from triton_ops.common import next_pow2  # noqa: E402


def case(M, K, I, blk_m=16, blk_k=None, blk_n=None):
    """原样跑生产 kernel，返回 (ok, 摘要)。"""
    E, N = 1, 2 * I
    blk_k = blk_k or max(next_pow2(K), 16)
    blk_n = blk_n or max(next_pow2(I), 16)
    torch.manual_seed(61)
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-128, 128, (E, K, N), dtype=torch.int8, device=DEV)
    ws = (torch.rand(E, N, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    xs = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.001)
    st = torch.zeros(E, dtype=torch.int32, device=DEV)
    en = torch.full((E,), M, dtype=torch.int32, device=DEV)
    y = torch.empty((M, I), dtype=torch.int8, device=DEV)
    ys = torch.empty((M,), dtype=torch.float32, device=DEV)
    tag = f"M={M} K={K} I={I} BLK_M={blk_m} BLK_K={blk_k} BLK_N={blk_n}"
    try:
        _gmm_v2_kernel[(E, triton.cdiv(M, blk_m))](
            x, w, ws, xs, x, st, en, y, ys, K, N, I,
            BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
            HAS_SMOOTH_COL=False, HAS_SMOOTH_ROW=False, LIMIT_ON=False, SWIGLU_LIMIT=0.0)
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


def main():
    print(f"[ub] dev={DEV} triton={triton.__version__}", flush=True)
    print("== A) BLK_N 分界（K=128, M=16, BLK_M=16, BLK_K=128）==", flush=True)
    for I in (100, 128, 129, 200, 256, 257, 300, 512, 1024):
        case(16, 128, I)
    print("== B) 生产 I 值（= moe_intermediate_size，取自 /public-flash/models/*/config.json）==", flush=True)
    for I in (512, 704, 768, 1024, 1408, 1536, 2048, 2304):
        case(16, 128, I)
    print("== C) 缩小 BLK_M / BLK_K 能否救回大 BLK_N ==", flush=True)
    for I, bm, bk in ((512, 1, 128), (512, 16, 16), (512, 16, 32), (2048, 1, 16), (2048, 16, 16)):
        case(16, 128, I, blk_m=bm, blk_k=bk)
    print("== D) 小 BLK_N 在大 K 下是否仍 OK（余量参考） ==", flush=True)
    for I, bk in ((256, 512), (256, 1024)):
        case(16, bk, I)
    print("[ub] done", flush=True)


if __name__ == "__main__":
    main()
