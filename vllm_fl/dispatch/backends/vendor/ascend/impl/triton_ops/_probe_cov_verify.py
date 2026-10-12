# -*- coding: utf-8 -*-
"""_probe_cov_verify.py —— 对 _probe_cov_sweep.py 报出的可疑项做**严格复核**。

复核三件事：
  A) moe_gating_top_k / moe_gating_top_k_hash 的 NaN/Inf 计数是否真实
     （改用 fp64 判定 + 逐元素 1-D 摘要，避免计数本身出问题）
  B) hc_pre_inv_rms 的正确返回 shape（是我期望写错，还是真问题）
  C) dequant_swiglu_quant 大 M 的异常是否稳定复现
"""
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:3").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")


def probe_nan(tag, t):
    """严格判定：用 fp64 比较 + 逐元素扫描（小张量）/ 分块扫描（大张量）。"""
    info = {"tag": tag, "shape": tuple(t.shape), "dtype": str(t.dtype)}
    # 方式1：.float() 再 isfinite
    try:
        tf = t.detach().float()
        c1 = int((~torch.isfinite(tf)).sum().item())
    except Exception as e:
        c1 = f"ERR {type(e).__name__}"
    # 方式2：先把张量搬到 CPU 再判（绕开设备端归约）
    tc = t.detach().cpu()
    if tc.is_floating_point():
        c2 = int((~torch.isfinite(tc.float())).sum().item())
        vals = tc.float().flatten()
        fin = vals[torch.isfinite(vals)]
        amax = float(fin.abs().max().item()) if fin.numel() else float("nan")
        # 逐元素数一遍（纯 CPU，慢但绝对可信）
        c3 = 0
        for i in range(0, tc.numel(), max(1, tc.numel() // 1000000 + 1)):
            pass
        sample = [float(x) for x in vals[:6].tolist()]
    else:
        c2, amax, sample = None, None, [int(x) for x in tc.flatten()[:6].tolist()]
    info.update(isfinite_dev=c1, isfinite_cpu=c2, absmax=amax, sample=sample,
                numel=tc.numel())
    print(f"  [{tag}] shape={info['shape']} dtype={info['dtype']} numel={tc.numel()}", flush=True)
    print(f"     设备端 ~isfinite 计数 = {c1}", flush=True)
    print(f"     CPU 端 ~isfinite 计数 = {c2}   |finite|max={amax}  sample={sample}", flush=True)
    return info


print(f"DEV={DEV} current={torch.npu.current_device()}", flush=True)

print("\n===== A) moe_gating_top_k 真实 NaN 复核 =====", flush=True)
from triton_ops.moe_gating_top_k import moe_gating_top_k
for N, E, k in [(8, 256, 8), (4096, 256, 8), (256, 1024, 8)]:
    try:
        torch.manual_seed(0)
        x = (torch.randn(N, E, device=DEV) * 0.5).to(torch.bfloat16)
        bias = (torch.randn(E, device=DEV) * 0.1).to(torch.bfloat16)
        res = moe_gating_top_k(x, k, k_group=8, group_count=8, group_select_mode=0,
                               renorm=0, norm_type=0, out_flag=False,
                               routed_scaling_factor=1.0, eps=1e-20, bias=bias)
        torch.npu.synchronize()
        print(f"  -- N={N} E={E} k={k}: 返回 {type(res)} len={len(res) if isinstance(res,(tuple,list)) else 1}",
              flush=True)
        outs = res if isinstance(res, (tuple, list)) else (res,)
        for i, o in enumerate(outs):
            if hasattr(o, "shape"):
                probe_nan(f"N={N} E={E} out[{i}]", o)
    except Exception as e:
        print(f"  -- N={N} E={E} k={k}: EXC {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n===== A2) moe_gating_top_k_hash 真实 NaN 复核 =====", flush=True)
from triton_ops.moe_gating_top_k_hash import moe_gating_top_k_hash
for N, E, k in [(8, 256, 8), (4096, 256, 8)]:
    try:
        torch.manual_seed(0)
        x = (torch.randn(N, E, device=DEV) * 0.5).to(torch.bfloat16)
        bias = (torch.randn(E, device=DEV) * 0.1).to(torch.bfloat16)
        res = moe_gating_top_k_hash(x, bias, None, None, k=k, k_group=8, group_count=8,
                                    renorm=0, norm_type=0, out_flag=False,
                                    routed_scaling_factor=1.0, eps=1e-20)
        torch.npu.synchronize()
        outs = res if isinstance(res, (tuple, list)) else (res,)
        print(f"  -- N={N} E={E} k={k}: 返回 {type(res)} len={len(outs)}", flush=True)
        for i, o in enumerate(outs):
            if hasattr(o, "shape"):
                probe_nan(f"hash N={N} E={E} out[{i}]", o)
    except Exception as e:
        print(f"  -- hash N={N} E={E} k={k}: EXC {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n===== B) hc_pre_inv_rms 返回 shape =====", flush=True)
from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms
for shape in [(4096, 4, 512), (33, 4, 100)]:
    try:
        x = (torch.randn(*shape, device=DEV) * 0.5).to(torch.bfloat16)
        y = hc_pre_inv_rms(x, 1e-6)
        torch.npu.synchronize()
        print(f"  shape={shape} -> 返回 shape={tuple(y.shape)} dtype={y.dtype}", flush=True)
        probe_nan(f"inv_rms {shape}", y)
    except Exception as e:
        print(f"  shape={shape}: EXC {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n===== C) dequant_swiglu_quant 大 M 复核 =====", flush=True)
from triton_ops.dequant_swiglu_quant import dequant_swiglu_quant
for M, H in [(16384, 512), (65536, 512), (4096, 2048)]:
    try:
        torch.manual_seed(9)
        x = torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8
        ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        a = torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25
        y, sc = dequant_swiglu_quant(x, ws, a, None, None, None, None, True, "dynamic",
                                     torch.int8, "rint", -1, 0, None, None, None)
        torch.npu.synchronize()
        probe_nan(f"dsq M={M} H={H}", y)
    except Exception as e:
        print(f"  M={M} H={H}: EXC {type(e).__name__}: {str(e)[:300]}", flush=True)
        print("   " + traceback.format_exc()[-400:].replace("\n", "\n   "), flush=True)

print("COVVERIFYDONE", flush=True)
