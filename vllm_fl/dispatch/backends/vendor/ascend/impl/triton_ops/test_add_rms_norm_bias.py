"""验证 add_rms_norm_bias 的 Triton 实现 vs 调用点语义（fp64 参考）。"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
add_rms_norm_bias = importlib.import_module(_PKG + ".add_rms_norm_bias").add_rms_norm_bias

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def ref_impl(x1, x2, gamma, beta, eps):
    a = x1.detach().cpu().double()
    b = x2.detach().cpu().double()
    x = a + b
    N = x.shape[-1]
    rstd = 1.0 / torch.sqrt((x * x).sum(-1, keepdim=True) / N + eps)
    y = x * rstd * gamma.detach().cpu().double()
    if beta is not None:
        y = y + beta.detach().cpu().double()
    return y, rstd, x


def run_case(name, shape, N, dtype, beta_on=True, eps=1e-6, callsite=False):
    torch.manual_seed(23)
    x1 = (torch.randn(*shape, dtype=torch.float32, device=DEV)).to(dtype)
    x2 = (torch.randn(*shape, dtype=torch.float32, device=DEV)).to(dtype)
    g = (torch.randn(N, dtype=torch.float32, device=DEV) * 0.5 + 1.0).to(dtype)
    bt = (torch.randn(N, dtype=torch.float32, device=DEV) * 0.1).to(dtype) if beta_on else None

    if callsite:
        # ← 调用点风格：npu_add_rms_norm_bias(x1, x2, gamma, beta, epsilon) -> (y, rstd, x)
        y, rstd, xout = add_rms_norm_bias(x1, x2, g, bt, eps)
    else:
        y, rstd, xout = add_rms_norm_bias(x1=x1, x2=x2, gamma=g, beta=bt, epsilon=eps)
    torch.npu.synchronize()
    ry, rrstd, rx = ref_impl(x1, x2, g, bt, eps)

    # x/y 被量化到输出 dtype：与「参考值按该 dtype 正确舍入」比，容差取 1 ULP（|值| 最大处）
    ulp = (2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -11) * max(
        float(ry.abs().max()), float(rx.abs().max()), 1e-12)
    ry2 = ry.to(dtype).double() if dtype != torch.float32 else ry
    rx2 = rx.to(dtype).double() if dtype != torch.float32 else rx
    ey = float((y.detach().cpu().double() - ry2).abs().max())
    ex = float((xout.detach().cpu().double() - rx2).abs().max())
    er = float((rstd.detach().cpu().double().reshape(-1) - rrstd.reshape(-1)).abs().max())
    ok = ey <= ulp and ex <= ulp and er <= 2e-6
    emit(f"  {name:42s} y_err={ey:.2e} x_err={ex:.2e}(<= {ulp:.2e} 1ULP) "
         f"rstd_err={er:.2e}(<=2e-06)  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 108)
emit("add_rms_norm_bias: Triton vs 调用点语义（fp64 参考）")
emit("y = (x1+x2)*rstd*gamma + beta（bias 在 norm 之后）；rstd = 1/sqrt(mean((x1+x2)^2)+eps)")
emit("=" * 108)

allok = True
allok &= run_case("M=64 N=2048 bf16 +beta", (64, 2048), 2048, torch.bfloat16)
allok &= run_case("M=64 N=2048 fp16 +beta", (64, 2048), 2048, torch.float16)
allok &= run_case("M=17 N=1000 fp16 非2幂", (17, 1000), 1000, torch.float16)
allok &= run_case("M=32 N=512  fp32 +beta", (32, 512), 512, torch.float32)
allok &= run_case("M=8  N=4096 bf16 无 beta", (8, 4096), 4096, torch.bfloat16, beta_on=False)
allok &= run_case("3D [4,8,256] fp16 +beta", (4, 8, 256), 256, torch.float16)
allok &= run_case("M=4  N=128  bf16 eps=1e-5", (4, 128), 128, torch.bfloat16, eps=1e-5)
allok &= run_case("调用点风格 位置参数", (32, 1024), 1024, torch.float16, callsite=True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/add_rms_norm_bias_test.txt", "w", encoding="utf-8").write(txt + "\n")
