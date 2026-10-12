# -*- coding: utf-8 -*-
"""验证：新 1-D 写法在「非 2 幂列宽」下的逐元素正确性。

旧 2-D `[BLK_D, BLK_C]` + 2-D 掩码 store 写法在 cols=17/24/33/48 上会静默丢写
（见 _probe_moe_store.py / _probe_moe_bisect.py）；本探针覆盖
非量化 / 动态量化(scale=None, 生产形态) / 动态量化(带 smooth) 三条路径 × 各种列宽。
用法（容器内）：python3 -W ignore _probe_moe_fix.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_ops.moe_init_routing_custom import (  # noqa: E402
    CUMSUM, DYNAMIC_QUANT, GATHER, NON_QUANT, moe_init_routing_custom,
)

DEV = os.environ.get("MOE_DEV", "npu:1")


def ref_quant(x, ei, scale, k):
    e = ei.reshape(-1).long().cpu()
    order = torch.argsort(e, stable=True)
    xv = x.detach().cpu().float()[order // k]
    sv = xv if scale is None else xv * scale.detach().cpu().float()[e[order]]
    s = sv.abs().amax(dim=1) / 127.0
    q = sv / s[:, None]
    q16 = q.half().float()
    r = torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))
    return r.clamp(-128, 127).to(torch.int8), s


def run(cols, quant, smooth, dtype=torch.float16, N=16, k=2, E=4, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(N, cols, dtype=torch.float32, device=DEV) * 2.0).to(dtype)
    ei = torch.randint(0, E, (N, k), dtype=torch.int32, device=DEV)
    scale = None
    if quant and smooth:
        scale = torch.rand(E, cols, dtype=torch.float32, device=DEV) * 0.5 + 0.75
    ex, ridx, cnt, esc = moe_init_routing_custom(
        x, ei, scale, None, expert_num=E, expert_tokens_num_type=CUMSUM,
        expert_tokens_num_flag=True,
        quant_mode=(DYNAMIC_QUANT if quant else NON_QUANT), row_idx_type=GATHER)
    torch.npu.synchronize()
    tag = f"cols={cols:5d} {'quant' if quant else 'raw  '} " \
          f"{'smooth' if smooth else 'plain ':6s} {str(dtype)[6:]:9s}"
    if quant:
        rx, s_ref = ref_quant(x, ei, scale, k)
        d = ex.detach().cpu().long() - rx.long()
        mx = d.abs().max().item()
        nbad = (d.abs() > 1).sum().item()
        e_s = (esc.detach().cpu().double() - s_ref.double()).abs().max().item()
        ok = (mx <= 1) and e_s <= 2e-6 * max(1.0, s_ref.abs().max().item())
        print(f"{tag} x_err={mx:4d} n(|d|>1)={nbad:4d} scale_err={e_s:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok
    rx = x.detach().cpu()[torch.argsort(ei.reshape(-1).long().cpu(), stable=True) // k]
    d = (ex.detach().cpu().double() - rx.double()).abs().max().item()
    print(f"{tag} x_err={d} {'PASS' if d == 0 else 'FAIL'}", flush=True)
    return d == 0


if __name__ == "__main__":
    print(f"[probe-fix] dev={DEV} triton={triton.__version__}", flush=True)
    cols_list = (17, 24, 33, 48, 64, 100, 128, 256, 1024, 7168)
    allok = True
    for cols in cols_list:
        allok &= run(cols, quant=False, smooth=False)
        allok &= run(cols, quant=True, smooth=False)     # 生产形态 scale=None
        allok &= run(cols, quant=True, smooth=True)      # 带 smooth
    for cols in (33, 64, 128, 7168):
        allok &= run(cols, quant=False, smooth=False, dtype=torch.bfloat16)
        allok &= run(cols, quant=True, smooth=False, dtype=torch.bfloat16)
        allok &= run(cols, quant=True, smooth=True, dtype=torch.float32)
    print(f"\n[probe-fix] {'ALL PASS' if allok else 'FAIL'}", flush=True)
