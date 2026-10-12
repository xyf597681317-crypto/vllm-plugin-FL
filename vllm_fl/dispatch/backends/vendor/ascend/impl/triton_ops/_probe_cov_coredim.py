# -*- coding: utf-8 -*-
"""_probe_cov_coredim.py —— 生产阻塞项复核：grid 维上限 65535

已知（_probe_cov_verify.py 实测）：
    dequant_swiglu_quant M=65536 → KernelLaunch failed:
    "value 65536 for parameter coreDim is invalid. Expected value: less than or equal to 65535"
    （kernel 名 _dsq_swiglu_kernel；grid = (M,) ⇒ 每行一个 program）

本探针回答：
  A) 精确阈值：M = 65534 / 65535 / 65536 / 65537 各是什么结果
  B) 是否**生产可达**：6656(=832*8) / 8192 / 16384 / 32768 / 49152 都通过 ⇒ 只有 ≥65536 挂
  C) 其它算子是否同病：moe_init_routing_custom / moe_gating_top_k / hc_post 等
     在 rows ≥ 65536 时的行为（grid 是否也是按行）
  D) 判据：这是"编译期 UB"还是"运行期 launch 失败"（分类不同，修法不同）
"""
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

print(f"DEV={DEV} current={torch.npu.current_device()}", flush=True)

from triton_ops.dequant_swiglu_quant import dequant_swiglu_quant

print("\n===== A) dequant_swiglu_quant 精确阈值扫描（H=512, swiglu_mode=0）=====", flush=True)
for M in [32768, 49152, 65534, 65535, 65536, 65537]:
    try:
        torch.manual_seed(9)
        x = torch.randint(-1000, 1000, (M, 1024), device=DEV, dtype=torch.int32) * 8
        ws = torch.rand(1024, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        a = torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25
        y, sc = dequant_swiglu_quant(x, ws, a, None, None, None, None, True, "dynamic",
                                     torch.int8, "rint", -1, 0, None, None, None)
        torch.npu.synchronize()
        print(f"  M={M:6d}  OK    shape={tuple(y.shape)}", flush=True)
        del x, ws, a, y, sc
        torch.npu.empty_cache()
    except Exception as e:
        msg = str(e).replace("\n", " ")[:220]
        print(f"  M={M:6d}  EXC   {type(e).__name__}: {msg}", flush=True)

print("\n===== B) 生产 shape 复核（M=832*8 / 8192 / 16384）=====", flush=True)
for M in [6656, 8192, 16384]:
    try:
        torch.manual_seed(9)
        x = torch.randint(-1000, 1000, (M, 1024), device=DEV, dtype=torch.int32) * 8
        ws = torch.rand(1024, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        a = torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25
        y, sc = dequant_swiglu_quant(x, ws, a, None, None, None, None, True, "dynamic",
                                     torch.int8, "rint", -1, 0, None, None, None)
        torch.npu.synchronize()
        import math
        tc = y.detach().cpu()
        nz = int((tc != 0).sum().item())
        print(f"  M={M:6d}  OK    shape={tuple(y.shape)} 非零元素={nz}/{tc.numel()} "
              f"({100.0*nz/tc.numel():.1f}%)", flush=True)
        del x, ws, a, y, sc
        torch.npu.empty_cache()
    except Exception as e:
        print(f"  M={M:6d}  EXC   {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n===== C) 其它算子在 rows>=65536 的行为 =====", flush=True)
cases = []


def try_case(tag, fn):
    try:
        r = fn()
        torch.npu.synchronize()
        t = r[0] if isinstance(r, (tuple, list)) else r
        print(f"  {tag:46s} OK    shape={tuple(t.shape)}", flush=True)
    except Exception as e:
        m = str(e).replace("\n", " ")
        flag = "coreDim" if "coreDim" in m else ""
        print(f"  {tag:46s} EXC {flag:9s} {type(e).__name__}: {m[:180]}", flush=True)


def mgating(N, E=256, k=8):
    from triton_ops.moe_gating_top_k import moe_gating_top_k
    torch.manual_seed(0)
    x = (torch.randn(N, E, device=DEV) * 0.5).to(torch.bfloat16)
    bias = (torch.randn(E, device=DEV) * 0.1).to(torch.bfloat16)
    return moe_gating_top_k(x, k, k_group=8, group_count=8, group_select_mode=0,
                            renorm=0, norm_type=0, out_flag=False,
                            routed_scaling_factor=1.0, eps=1e-20, bias=bias)


for N in [65535, 65536]:
    try_case(f"moe_gating_top_k N={N} E=256", lambda N=N: mgating(N))

print("\n===== D) 结论判据 =====", flush=True)
print("  coreDim 上限 = 65535 ⇒ grid 第 0 维 = M 的 kernel 在 M>=65536 时**运行期 launch 失败**",
      flush=True)
print("  （不是 UB 溢出、不是编译期断言；消息里出现 coreDim / rtKernelLaunch）", flush=True)
print("COREDIMDONE", flush=True)
