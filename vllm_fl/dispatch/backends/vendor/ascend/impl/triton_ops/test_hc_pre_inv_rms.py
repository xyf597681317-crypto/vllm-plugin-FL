"""验证 hc_pre_inv_rms 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = 2e-6  # y 输出 fp32


def emit(s=""):
    L.append(str(s))


def ref(x, eps):
    """fp64 参考：y[a] = 1 / sqrt(mean_r(x^2) + eps)。"""
    xd = x.detach().cpu().double()
    A = 1
    for d in xd.shape[:-2]:
        A *= int(d)
    R = int(xd.shape[-2]) * int(xd.shape[-1])
    x2 = xd.reshape(A, R)
    mean = (x2 * x2).sum(dim=1) / float(R)
    return 1.0 / torch.sqrt(mean + eps)


def run_case(name, shape, dtype, eps=1e-6, scale=1.0):
    torch.manual_seed(0)
    x = (torch.randn(*shape, dtype=torch.float32, device=DEV) * scale).to(dtype)

    y = hc_pre_inv_rms(x, eps)
    torch.npu.synchronize()
    r = ref(x, eps)

    got = y.detach().cpu().double().reshape(-1)
    err = (got - r).abs().max().item()
    ok = err <= FLOOR
    emit(f"  {name:42s} err={err:.3e} (<= {FLOOR:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 78)
emit("hc_pre_inv_rms: Triton vs AscendC 语义（fp64 参考，y 为 fp32）")
emit("=" * 78)

allok = True
# 4D: (b, s, hc, d) -> y (b, s, 1)
allok &= run_case("4D b=2 s=64 hc=4 d=512 bf16", (2, 64, 4, 512), torch.bfloat16)
allok &= run_case("4D b=4 s=32 hc=4 d=1024 fp16", (4, 32, 4, 1024), torch.float16)
allok &= run_case("4D b=3 s=17 hc=3 d=96 fp32 (非整除)", (3, 17, 3, 96), torch.float32)
# 3D: (b*s, hc, d) -> y (b*s, 1)
allok &= run_case("3D bs=128 hc=4 d=512 bf16", (128, 4, 512), torch.bfloat16)
allok &= run_case("3D bs=17 hc=2 d=64 fp16 (非整除)", (17, 2, 64), torch.float16)
allok &= run_case("3D bs=33 hc=4 d=100 fp32 (R 非 32 倍数)", (33, 4, 100), torch.float32)
allok &= run_case("3D bs=5 hc=4 d=4096 fp16 (大 R)", (5, 4, 4096), torch.float16)
allok &= run_case("3D bs=64 hc=4 d=512 bf16 eps=1e-5", (64, 4, 512), torch.bfloat16, 1e-5)
allok &= run_case("3D bs=8 hc=1 d=2048 bf16 (hc=1)", (8, 1, 2048), torch.bfloat16)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/hc_pre_inv_rms_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
