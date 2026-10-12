"""验证 fused_gdn_gating 的 Triton 实现 vs AscendC 语义。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.fused_gdn_gating import fused_gdn_gating  # noqa: E402

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))


def ref(a_log, a, b, dt_bias, beta, threshold):
    """严格按 AscendC 内核顺序计算，fp64 参考。"""
    x = a.detach().cpu().double() + dt_bias.detach().cpu().double().unsqueeze(0)
    bx = beta * x
    sp = torch.log1p(torch.exp(torch.clamp(bx, max=threshold))) / beta
    softplus = torch.where(bx <= threshold, sp, x)
    g = -torch.exp(a_log.detach().cpu().double()).unsqueeze(0) * softplus
    bo = torch.sigmoid(b.detach().cpu().double())
    return g, bo


def run_case(name, B, H, dtype, beta=1.0, threshold=20.0):
    torch.manual_seed(0)
    a_log = (torch.randn(H, dtype=torch.float32, device=DEV) * 0.5)
    dt_bias = (torch.randn(H, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
    a = (torch.randn(B, H, dtype=torch.float32, device=DEV) * 2.0).to(dtype)
    b = (torch.randn(B, H, dtype=torch.float32, device=DEV)).to(dtype)

    g, bo = fused_gdn_gating(a_log, a, b, dt_bias, beta, threshold)
    torch.npu.synchronize()
    rg, rbo = ref(a_log, a, b, dt_bias, beta, threshold)

    eg = (g.detach().cpu().double() - rg).abs().max().item()
    eb = (bo.detach().cpu().double() - rbo).abs().max().item()
    # g 输出 fp32：地板取 2e-6（数学等价写法 log(1+exp(x)) vs log1p(exp(x)) 的 fp32 舍入）
    # beta_output 转成 InDtype：按该 dtype 的量化地板
    floor_g = 2e-6
    floor_b = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[dtype]
    ok = eg <= floor_g and eb <= floor_b
    emit(f"  {name:34s} g_err={eg:.3e}(<= {floor_g:.0e}) "
         f"beta_err={eb:.3e}(<= {floor_b:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 72)
emit("fused_gdn_gating: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 72)

allok = True
allok &= run_case("B=64   H=32  bf16", 64, 32, torch.bfloat16)
allok &= run_case("B=256  H=128 bf16", 256, 128, torch.bfloat16)
allok &= run_case("B=64   H=32  fp16", 64, 32, torch.float16)
allok &= run_case("B=128  H=64  fp16", 128, 64, torch.float16)
allok &= run_case("B=33   H=17  bf16 (非整除)", 33, 17, torch.bfloat16)
allok &= run_case("B=64   H=32  bf16 beta=0.5 thr=10", 64, 32, torch.bfloat16, 0.5, 10.0)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/gdn_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
