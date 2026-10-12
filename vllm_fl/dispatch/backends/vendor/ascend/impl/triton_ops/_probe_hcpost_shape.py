# -*- coding: utf-8 -*-
"""hc_post 生产 shape 编译失败最小复现（报错原文：
    spec/triton/language/core.py in where -> ValueError: Cannot make_shape_compatible:
    incompatible dimensions at index 1: 16 and 128）。

生产事实（`/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp/config.json`）：
    hidden_size = 4096, hc_mult = 4, hc_eps = 1e-6
调用点（`vllm_fl/models/deepseek_v4_ascend.py:975`）：
    npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
    ⇒ x [1,n,D]、residual [1,n,H,D]、post [1,n,H]、comb [1,n,H,H]

本探针：按上面的调用点形态扫 (H, D) 组合，找出触发该 `where` 维度错的组合，
并对能编译的组合与 fp64 参考逐元素比对（max|Δ| / exact）。

用法：DSH_NPU=npu:12 python3 -W ignore _probe_hcpost_shape.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:12")
torch.npu.set_device(DEV)          # ← 必须
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_ops.hc_post import hc_post  # noqa: E402
from triton_ops.common import next_pow2  # noqa: E402


def make(n, H, D, t1=torch.bfloat16, t2=torch.bfloat16, seed=5):
    torch.manual_seed(seed)
    x = ((torch.rand(1, n, D, device=DEV) * 2 - 1) * 0.4).to(t1)
    r = ((torch.rand(1, n, H, D, device=DEV) * 2 - 1) * 0.4).to(t1)
    p = ((torch.rand(1, n, H, device=DEV) * 2 - 1) * 0.4).to(t2)
    c = ((torch.rand(1, n, H, H, device=DEV) * 2 - 1) * 0.4).to(t2)
    return x, r, p, c


def ref(x, r, p, c):
    xd = x.detach().cpu().double()
    rd = r.detach().cpu().double()
    pd = p.detach().cpu().double()
    cd = c.detach().cpu().double()
    # y[...,h,d] = p[...,h]*x[...,d] + sum_k comb[...,k,h]*r[...,k,d]
    y = pd.unsqueeze(-1) * xd.unsqueeze(-2) + torch.einsum("...kh,...kd->...hd", cd, rd)
    return y


def tile(H, D):
    BLK_H = max(2, next_pow2(H))
    BLK_D = min(next_pow2(D), max(16, 4096 // BLK_H))
    return BLK_H, BLK_D


def case(n, H, D, t1=torch.bfloat16, t2=torch.bfloat16):
    bh, bd = tile(H, D)
    tag = f"n={n} H={H:3d} D={D:5d} dtype={str(t1).split('.')[-1]} tile(BLK_H={bh},BLK_D={bd})"
    x, r, p, c = make(n, H, D, t1, t2)
    try:
        y = hc_post(x, r, p, c)
        torch.npu.synchronize()
    except Exception as ex:  # noqa: BLE001
        first = " ".join(str(ex).split())[:200]
        print(f"  [FAIL] {tag}\n         {type(ex).__name__}: {first}", flush=True)
        if os.environ.get("HC_TRACE"):
            traceback.print_exc()
        return False
    ry = ref(x, r, p, c)
    got = y.detach().cpu().double()
    dd = (got - ry).abs()
    md = float(dd.max())
    amax = float(ry.abs().max())
    ulp = (2.0 ** -8 if t1 == torch.bfloat16 else 2.0 ** -11) * max(amax, 1e-12)
    ex = float((dd == 0).double().mean()) * 100
    ok = md <= ulp and tuple(y.shape) == tuple(ry.shape)
    print(f"  [{'OK  ' if ok else 'BAD '}] {tag}  max|Δ|={md:.3e} (ulp_floor={ulp:.1e}) "
          f"exact={ex:.2f}% |y|max={amax:.3f} y.shape={tuple(y.shape)}", flush=True)
    return ok


def main():
    print(f"[hcpost-shape] dev={DEV}", flush=True)
    print("== A) 生产/提示组合 ==", flush=True)
    # 生产：hidden_size=4096, hc_mult=4 → H=4, D=4096（另试几种 hc 布局）
    for (n, H, D) in [(64, 4, 4096), (64, 4, 128), (64, 16, 128), (64, 16, 4096),
                      (64, 8, 128), (64, 16, 256), (64, 4, 512), (64, 16, 512)]:
        case(n, H, D)
    print("== B) dtype 组合（T1/T2 可不同） ==", flush=True)
    case(32, 16, 128, torch.float16, torch.float16)
    case(32, 16, 128, torch.bfloat16, torch.float32)
    case(32, 4, 4096, torch.bfloat16, torch.bfloat16)
    print("== C) 2-D 形态（L0/op_host 形态 x [B,D]） ==", flush=True)
    try:
        x = ((torch.rand(32, 128, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        r = ((torch.rand(32, 16, 128, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        p = ((torch.rand(32, 16, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        c = ((torch.rand(32, 16, 16, device=DEV) * 2 - 1) * 0.4).to(torch.bfloat16)
        y = hc_post(x, r, p, c)
        torch.npu.synchronize()
        ry = ref(x, r, p, c)
        md = float((y.detach().cpu().double() - ry).abs().max())
        print(f"  [OK  ] 2-D B=32 H=16 D=128 max|Δ|={md:.3e} y.shape={tuple(y.shape)}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  [FAIL] 2-D B=32 H=16 D=128 {type(ex).__name__}: "
              f"{' '.join(str(ex).split())[:200]}", flush=True)
    print("[hcpost-shape] done", flush=True)


if __name__ == "__main__":
    main()
