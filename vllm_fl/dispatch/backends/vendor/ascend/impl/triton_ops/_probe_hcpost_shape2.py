# -*- coding: utf-8 -*-
"""按生产 shape 逐个编译 hc 家族，定位那条
    `in where -> Cannot make_shape_compatible: incompatible dimensions at index 1: 16 and 128`
到底出自哪个算子。

生产（DeepSeek-V4-Flash-w8a8-mtp/config.json）：hidden_size=4096, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6
⇒ hc_mix = 2*hm + hm^2 = 24；hc_fn [24, hm*d=16384]；hc_pre 的 host 用 blk_m=16 / blk_d=128。

用法：DSH_NPU=npu:12 python3 -W ignore _probe_hcpost_shape2.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:12")
torch.npu.set_device(DEV)
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import triton_ops.hc_post as hcpost_mod          # noqa: E402
import triton_ops.hc_pre as hcpre_mod            # noqa: E402
import triton_ops.hc_pre_sinkhorn as hcsk_mod    # noqa: E402
import triton_ops.hc_pre_inv_rms as hcir_mod     # noqa: E402

HM, D, N = 4, 4096, 16
HMIX = 2 * HM + HM * HM


def run(tag, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  [OK  ] {tag}", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001
        msg = " ".join(str(ex).split())[:260]
        print(f"  [FAIL] {tag}\n         {type(ex).__name__}: {msg}", flush=True)
        if os.environ.get("HC_TRACE"):
            traceback.print_exc()
        return False


def main():
    print(f"[hc-family] dev={DEV} hm={HM} d={D} rows={N} hmix={HMIX}", flush=True)
    torch.manual_seed(9)
    x3 = ((torch.rand(N, HM, D, device=DEV) * 2 - 1) * 0.3).to(torch.bfloat16)

    # 1) hc_pre（npu_hc_pre_v2）
    hc_fn = (torch.rand(HMIX, HM * D, device=DEV) * 0.05).to(torch.float32)
    hc_scale = torch.rand(3, device=DEV)
    hc_base = torch.rand(HMIX, device=DEV)
    run(f"hc_pre  x[{N},{HM},{D}] hc_fn[{HMIX},{HM * D}] (blk_m=16, blk_d=128)",
        lambda: hcpre_mod.hc_pre(x3, hc_fn, hc_scale, hc_base, hc_mult=HM,
                                 hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6))

    # 2) hc_pre_sinkhorn
    mixes = torch.rand(N, HMIX, device=DEV) * 0.1
    rsqrt = torch.rand(N, device=DEV) + 0.5
    run(f"hc_pre_sinkhorn  mixes[{N},{HMIX}] x[{N},{HM},{D}]",
        lambda: hcsk_mod.hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x3,
                                         hc_mult=HM, hc_sinkhorn_iters=20, hc_eps=1e-6))

    # 3) hc_pre_inv_rms
    run(f"hc_pre_inv_rms  x[{N},{HM},{D}]",
        lambda: hcir_mod.hc_pre_inv_rms(x3, 1e-6))

    # 4) hc_post（生产调用点形态）
    xp = ((torch.rand(1, N, D, device=DEV) * 2 - 1) * 0.3).to(torch.bfloat16)
    rp = ((torch.rand(1, N, HM, D, device=DEV) * 2 - 1) * 0.3).to(torch.bfloat16)
    pp = ((torch.rand(1, N, HM, device=DEV) * 2 - 1) * 0.3).to(torch.bfloat16)
    cp = ((torch.rand(1, N, HM, HM, device=DEV) * 2 - 1) * 0.3).to(torch.bfloat16)
    run(f"hc_post  x[1,{N},{D}] residual[1,{N},{HM},{D}]",
        lambda: hcpost_mod.hc_post(xp, rp, pp, cp))

    # 5) 反向：生产里 hc_pre 的 x 也可能是 [1,n,hm,d]
    x4 = x3.view(1, N, HM, D)
    run(f"hc_pre 4D x[1,{N},{HM},{D}]",
        lambda: hcpre_mod.hc_pre(x4, hc_fn, hc_scale, hc_base, hc_mult=HM,
                                 hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6))
    print("[hc-family] done", flush=True)


if __name__ == "__main__":
    main()
