"""验证 hc_pre 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
hc_pre = importlib.import_module(_PKG + ".hc_pre").hc_pre

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)      # 逐行落盘：崩溃时也能看到已完成用例的结果


def ref_impl(x, fn, scale, base, hm, iters, hc_eps, norm_eps):
    """按 hc_pre_base.h / cube 的语义（fp64）。"""
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])
    bs, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(bs, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + norm_eps)
    mixes = xflat @ fn.detach().cpu().double().t()
    sc = scale.detach().cpu().double()
    bs_base = base.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * sc[0] + bs_base[:hm]) + hc_eps
    y = torch.einsum("bj,bjd->bd", pre, xf)
    post = 2.0 * torch.sigmoid(mixes[:, hm:2 * hm] * rq[:, None] * sc[1] + bs_base[hm:2 * hm])
    A = (mixes[:, 2 * hm:].reshape(bs, hm, hm) * rq[:, None, None] * sc[2]
         + bs_base[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + hc_eps
    A = A / (A.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + hc_eps)
        A = A / (A.sum(dim=1, keepdim=True) + hc_eps)
    return y.to(x.dtype), post, A


def run_case(name, bs, hm, d, iters=20, hc_eps=1e-6, norm_eps=1e-6, four_d=False):
    torch.manual_seed(13)
    hmix = 2 * hm + hm * hm
    K = hm * d
    xshape = (2, bs // 2, hm, d) if four_d else (bs, hm, d)
    x = (torch.randn(*xshape, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = (torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5)))
    scale = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    base = (torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1)

    y, post, comb = hc_pre(x, fn, scale, base, hm, iters, hc_eps, norm_eps)
    torch.npu.synchronize()
    ry, rpost, rcomb = ref_impl(x, fn, scale, base, hm, iters, hc_eps, norm_eps)

    ry2 = ry.reshape(y.shape).to(torch.bfloat16).double()
    rpost2 = rpost.reshape(post.shape)
    rcomb2 = rcomb.reshape(comb.shape)
    ulp = (2.0 ** -8) * max(float(ry2.abs().max()), 1e-12)
    ey = float((y.detach().cpu().double() - ry2).abs().max())
    ep = float((post.detach().cpu().double() - rpost2).abs().max())
    ec = float((comb.detach().cpu().double() - rcomb2).abs().max())
    ok = ey <= ulp and ep <= 2e-6 and ec <= 2e-6
    emit(f"  {name:40s} y_err={ey:.2e}(<= {ulp:.2e} 1ULP) post_err={ep:.2e} "
         f"comb_err={ec:.2e}(<=2e-06)  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 106)
emit("hc_pre: Triton vs AscendC 语义（fp64 参考）")
emit("y 为 bf16（与正确舍入的参考比，容差 1 ULP）；post / comb_frag 为 fp32（<=2e-6）")
emit("=" * 106)

allok = True
allok &= run_case("bs=64 hm=4 d=512 iters=20", 64, 4, 512)
allok &= run_case("bs=33 hm=4 d=100 非整除", 33, 4, 100)
allok &= run_case("bs=16 hm=2 d=256 (hmix=8)", 16, 2, 256)
allok &= run_case("bs=8  hm=8 d=128 (hmix=80)", 8, 8, 128)
allok &= run_case("bs=17 hm=4 d=64  iters=1", 17, 4, 64, iters=1)
allok &= run_case("bs=16 hm=4 d=256 iters=3", 16, 4, 256, iters=3)
allok &= run_case("bs=32 hm=4 d=300 eps=1e-5", 32, 4, 300, hc_eps=1e-5, norm_eps=1e-5)
allok &= run_case("4D [2,16,4,512]", 32, 4, 512, four_d=True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/hc_pre_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
