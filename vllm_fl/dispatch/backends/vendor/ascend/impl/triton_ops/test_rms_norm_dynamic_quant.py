"""验证 rms_norm_dynamic_quant 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import importlib
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")   # 容器内可用独立包名隔离
rms_norm_dynamic_quant = importlib.import_module(
    _PKG + ".rms_norm_dynamic_quant").rms_norm_dynamic_quant

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
QMAX = 127.0
L = []


def emit(s=""):
    L.append(str(s))


def rint_even(t):
    """就近偶数取整（CAST_RINT / torch.round 语义）。"""
    return torch.round(t)


def ref_impl(x, gamma, s1, s2, beta, eps, do1, do2):
    xd = x.detach().cpu().double()
    g = gamma.detach().cpu().double()
    R, D = xd.shape
    ss = (xd * xd).sum(dim=1)
    rstd = 1.0 / torch.sqrt(ss * (1.0 / D) + eps)
    t = xd * rstd[:, None] * g[None, :]
    if beta is not None:
        t = t + beta.detach().cpu().double()[None, :]

    def quant(z):
        m = z.abs().amax(dim=1)
        safe = m > 0
        st = torch.where(safe, QMAX / m, torch.zeros_like(m))
        q = rint_even(z * st[:, None])
        sc = torch.where(safe, m / QMAX, torch.zeros_like(m))
        return q, sc

    out = []
    if do1:
        out.append(quant(t * (s1.detach().cpu().double()[None, :] if s1 is not None else 1.0)))
    else:
        out.append((None, None))
    if do2:
        out.append(quant(t * (s2.detach().cpu().double()[None, :] if s2 is not None else 1.0)))
    else:
        out.append((None, None))
    return out


def run_case(name, R, D, dtype, smooth1=True, smooth2=True, beta=False,
             eps=1e-6, output_mask=None, scale_mag=1.0):
    torch.manual_seed(7)
    x = (torch.randn(R, D, dtype=torch.float32, device=DEV) * 2.0).to(dtype)
    gamma = (torch.randn(D, dtype=torch.float32, device=DEV) * 0.5 + 1.0).to(dtype)
    s1 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5) * scale_mag).to(dtype) if smooth1 else None
    s2 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5) * scale_mag).to(dtype) if smooth2 else None
    bt = (torch.randn(D, dtype=torch.float32, device=DEV) * 0.1).to(dtype) if beta else None

    y1, y2, sc1, sc2 = rms_norm_dynamic_quant(x, gamma, s1, s2, bt, eps, output_mask)
    torch.npu.synchronize()

    do1 = (output_mask is None) or bool(output_mask[0])
    do2 = (output_mask is not None and bool(output_mask[1])) or (
        output_mask is None and s1 is not None and s2 is not None)
    ref = ref_impl(x, gamma, s1, s2, bt, eps, do1, do2)

    msgs = []
    ok = True
    for tag, got, (rq, rs) in (("y1", y1, ref[0]), ("y2", y2, ref[1])):
        if got is None:
            msgs.append(f"{tag}=None(未使能,{'OK' if rq is None else 'BAD'})")
            ok &= rq is None
            continue
        g = got.detach().cpu().double()
        diff = (g - rq).abs()
        exact = float((diff == 0).double().mean())
        md = float(diff.max())
        ok &= md <= 1.0 and exact >= 0.999
        msgs.append(f"{tag}: max|d|={md:.0f} exact={exact * 100:.3f}%")
    for tag, got, (rq, rs) in (("s1", sc1, ref[0]), ("s2", sc2, ref[1])):
        if got is None:
            continue
        e = float((got.detach().cpu().double() - rs).abs().max())
        ok &= e <= 2e-6
        msgs.append(f"{tag}_err={e:.2e}")
    # 反量化一致性：y * scale ≈ rms 输出
    for tag, got, sc, (rq, rs) in (("y1", y1, sc1, ref[0]), ("y2", y2, sc2, ref[1])):
        if got is None:
            continue
        deq = got.detach().cpu().double() * sc.detach().cpu().double()[:, None]
        refv = rq * rs[:, None]
        rel = float(((deq - refv).abs() / (refv.abs().amax(dim=1, keepdim=True) + 1e-12)).max())
        ok &= rel <= 0.02
        msgs.append(f"{tag}_deq_rel={rel:.2e}")
    emit(f"  {name:40s} {'PASS' if ok else 'FAIL'}  " + "  ".join(msgs))
    return ok


emit("=" * 110)
emit("rms_norm_dynamic_quant: Triton vs AscendC 语义（fp64 参考）")
emit("y 为 int8：判定 max|Δ|<=1 且逐元素相等率>=99.9%（fp32 与 fp64 在 .5 边界的取舍差异，最多差 1 LSB）")
emit("scale 为 fp32：判定 <=2e-6")
emit("=" * 110)

allok = True
allok &= run_case("R=64  D=1024 bf16 smooth1+2", 64, 1024, torch.bfloat16)
allok &= run_case("R=64  D=1024 fp16 smooth1+2", 64, 1024, torch.float16)
allok &= run_case("R=33  D=1000 bf16 (非整除)", 33, 1000, torch.bfloat16)
allok &= run_case("R=16  D=4096 fp16", 16, 4096, torch.float16)
allok &= run_case("R=48  D=512  bf16 +beta", 48, 512, torch.bfloat16, beta=True)
allok &= run_case("R=48  D=512  bf16 无 smooth", 48, 512, torch.bfloat16, smooth1=False, smooth2=False)
allok &= run_case("R=48  D=512  bf16 仅 smooth2", 48, 512, torch.bfloat16, smooth1=False, smooth2=True)
allok &= run_case("R=24  D=768  fp16 mask=[1,0]", 24, 768, torch.float16, output_mask=[True, False])
allok &= run_case("R=24  D=768  fp16 mask=[0,1]", 24, 768, torch.float16, output_mask=[False, True])
allok &= run_case("R=24  D=768  fp16 mask=[1,1]", 24, 768, torch.float16, output_mask=[True, True])
allok &= run_case("R=40  D=2048 bf16 eps=1e-5", 40, 2048, torch.bfloat16, eps=1e-5)
allok &= run_case("R=40  D=2048 bf16 小 smooth", 40, 2048, torch.bfloat16, scale_mag=0.05)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/rms_norm_dynamic_quant_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
