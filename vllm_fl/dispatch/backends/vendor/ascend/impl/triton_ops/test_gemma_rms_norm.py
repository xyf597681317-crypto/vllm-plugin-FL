"""验证 gemma_rms_norm 的 Triton 实现 vs 调用点语义（fp64 参考）。"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
gemma_rms_norm = importlib.import_module(_PKG + ".gemma_rms_norm").gemma_rms_norm

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def ref_impl(x, gamma, eps):
    xd = x.detach().cpu().double()
    N = xd.shape[-1]
    var = (xd * xd).sum(dim=-1, keepdim=True) / N
    rrms = 1.0 / torch.sqrt(var + eps)
    w = gamma.detach().cpu().double()
    y = xd * rrms * (1.0 + w)
    return y, rrms.squeeze(-1)


def run_case(name, shape, N, dtype, eps=1e-6, callsite=False):
    torch.manual_seed(17)
    x = (torch.randn(*shape, dtype=torch.float32, device=DEV) * 2.0).to(dtype)
    g = (torch.randn(N, dtype=torch.float32, device=DEV) * 0.3)
    if dtype != torch.float32:
        g = g.to(dtype)

    if callsite:
        # ← 调用点风格：npu_gemma_rms_norm(x, gamma, epsilon) -> (y, rstd)
        y, rstd = gemma_rms_norm(x=x, gamma=g, epsilon=eps)
    else:
        y, rstd = gemma_rms_norm(x, g, eps)
    torch.npu.synchronize()
    ry, rrms = ref_impl(x, g.float(), eps)

    # y 被量化到输出 dtype：与「参考值按该 dtype 正确舍入」比，容差取 1 ULP（|y| 最大处）
    ulp = (2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -11) * max(float(ry.abs().max()), 1e-12)
    ry2 = ry.to(dtype).double() if dtype != torch.float32 else ry
    ey = float((y.detach().cpu().double() - ry2).abs().max())
    # rstd/rrms 都是 fp32 标量：注意拉平后再比，否则 [N,1] vs [N] 会广播成伪误差
    er = float((rstd.detach().cpu().double().reshape(-1) - rrms.reshape(-1)).abs().max())
    ok = ey <= ulp and er <= 2e-6
    emit(f"  {name:40s} y_err={ey:.2e}(<= {ulp:.2e} 1ULP) rstd_err={er:.2e}(<=2e-06)  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 100)
emit("gemma_rms_norm: Triton vs 调用点语义（fp64 参考；y = x*rrms*(1+gamma)，rstd = rrms）")
emit("=" * 100)

allok = True
allok &= run_case("M=64  N=2048 bf16", (64, 2048), 2048, torch.bfloat16)
allok &= run_case("M=64  N=2048 fp16", (64, 2048), 2048, torch.float16)
allok &= run_case("M=17  N=1000 fp16 非2幂", (17, 1000), 1000, torch.float16)
allok &= run_case("M=8   N=4096 bf16", (8, 4096), 4096, torch.bfloat16)
allok &= run_case("M=32  N=512  fp32", (32, 512), 512, torch.float32)
allok &= run_case("M=4   N=128  bf16 eps=1e-5", (4, 128), 128, torch.bfloat16, eps=1e-5)
allok &= run_case("3D [4,8,256] fp16", (4, 8, 256), 256, torch.float16)
allok &= run_case("调用点风格 npu_gemma_rms_norm", (32, 1024), 1024, torch.float16, callsite=True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/gemma_rms_norm_test.txt", "w", encoding="utf-8").write(txt + "\n")
