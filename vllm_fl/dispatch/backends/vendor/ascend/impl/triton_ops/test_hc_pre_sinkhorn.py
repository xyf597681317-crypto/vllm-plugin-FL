"""验证 hc_pre_sinkhorn 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def case(*a, **kw):
    global allok
    try:
        allok &= run_case(*a, **kw)
    except Exception as e:  # noqa: BLE001
        emit(f"  {a[0]:44s} EXCEPTION {type(e).__name__}: {str(e)[:140]}")
        allok = False


def ref(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps):
    """fp64 参考：严格按 hc_pre_sinkhorn_perf.h L95-202 的顺序。"""
    mx = mixes.detach().cpu().double().reshape(-1, mixes.shape[-1])
    rs = rsqrt.detach().cpu().double().view(-1, 1)
    sc = hc_scale.detach().cpu().double()
    bs = hc_base.detach().cpu().double()
    rows = mx.shape[0]
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(rows, hm, d)

    pre = torch.sigmoid(mx[:, :hm] * rs * sc[0] + bs[:hm]) + eps
    y = torch.einsum("ij,ijd->id", pre, xd.reshape(rows, hm, d))
    post = 2.0 * torch.sigmoid(mx[:, hm:2 * hm] * rs * sc[1] + bs[hm:2 * hm])

    A = (mx[:, 2 * hm:].reshape(rows, hm, hm) * rs[:, :, None] * sc[2]
         + bs[2 * hm:].reshape(hm, hm))
    m = A.max(dim=2, keepdim=True).values
    e = torch.exp(A - m)
    A = e / e.sum(dim=2, keepdim=True) + eps
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + eps)
        A = A / (A.sum(dim=1, keepdim=True) + eps)
    return y, post, A


def run_case(name, lead, hm, d, iters=20, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    rows = 1
    for v in lead:
        rows *= v

    def U(*shape, lo=1.0, off=0.0):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo + off

    mixes = U(*lead, 2 * hm + hm * hm, lo=1.0)
    rsqrt = U(rows, lo=0.5, off=1.0)
    hc_scale = U(3, lo=0.5, off=1.0)
    hc_base = U(2 * hm + hm * hm, lo=0.5)
    x = (U(*lead, hm, d, lo=0.3)).to(torch.bfloat16)

    y, post, comb = hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x,
                                    hm, iters, eps)
    torch.npu.synchronize()
    ry, rpost, rcomb = ref(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps)

    ey = (y.detach().cpu().double().reshape(rows, d) - ry).abs().max().item()
    ep = (post.detach().cpu().double().reshape(rows, hm) - rpost).abs().max().item()
    ec = (comb.detach().cpu().double().reshape(rows, hm, hm) - rcomb).abs().max().item()
    fy = FLOOR[x.dtype]
    shape_ok = (tuple(y.shape) == tuple(lead) + (d,)
                and tuple(post.shape) == tuple(lead) + (hm,)
                and tuple(comb.shape) == tuple(lead) + (hm, hm))
    ok = ey <= fy and ep <= 2e-6 and ec <= 2e-6 and shape_ok
    emit(f"  {name:44s} y_err={ey:.2e}(<= {fy:.0e}) post_err={ep:.2e} comb_err={ec:.2e} "
         f"(<= 2e-06)  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 96)
emit("hc_pre_sinkhorn: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 96)

allok = True
case("2D bs=64  hm=4 d=512 iters=20", (64,), 4, 512)
case("2D bs=33  hm=4 d=100 iters=20 (非整除)", (33,), 4, 100)
case("3D b=2 s=32 hm=4 d=512 iters=20", (2, 32), 4, 512)
case("2D bs=16  hm=3 d=256 iters=20", (16,), 3, 256)
case("2D bs=16  hm=4 d=256 iters=1 (仅一次归一)", (16,), 4, 256, iters=1)
case("2D bs=16  hm=4 d=256 iters=3", (16,), 4, 256, iters=3)
case("2D bs=8   hm=8 d=128 iters=20", (8,), 8, 128)
case("2D bs=17  hm=4 d=64  eps=1e-5", (17,), 4, 64, eps=1e-5)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/hc_pre_sinkhorn_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
