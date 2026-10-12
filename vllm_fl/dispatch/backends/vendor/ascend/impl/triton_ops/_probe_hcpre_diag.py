# -*- coding: utf-8 -*-
"""_probe_hcpre_diag.py —— 三条缺陷的诊断 + 修法候选 A/B（只调算子、不改文件）

① hc_pre @ d=7168 错（d=4096 对）
② hc_pre_sinkhorn @ R≥256 错
③ hc_pre_inv_rms @ 极小 R（x[2,2,4]，R=8）⇒ 我们自己的 kernel 触发 vector core exception

本探针同时跑**修法候选**（在探针内定义变体 kernel，不改生产文件），
用来在动生产文件前确定哪条改法有效。

跑法：DSH_NPU=npu:12 python3 -W ignore triton_ops/_probe_hcpre_diag.py
"""
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    enable_custom_op()
except Exception as e:
    print("native enable failed:", e, flush=True)


def md(a, b):
    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


# ============================================================ ③ inv_rms
print("=" * 100, flush=True)
print("③ hc_pre_inv_rms：现行实现 vs 变体A（静态循环+BLK_A=1）", flush=True)
print("=" * 100, flush=True)


@triton.jit
def _inv_v1(X, Y, A, R, EPS, BLK_A: tl.constexpr, BLK_R: tl.constexpr):
    """现行：while 循环（runtime bound）+ BLK_A 行块。"""
    pid = tl.program_id(0)
    rows = pid * BLK_A + tl.arange(0, BLK_A)
    rmask = rows < A
    cols = tl.arange(0, BLK_R)
    acc = tl.zeros((BLK_A, BLK_R), dtype=tl.float32)
    r = 0
    while r < R:
        col = r + cols
        m = rmask[:, None] & (col < R)[None, :]
        x = tl.load(X + rows[:, None] * R + col[None, :], mask=m, other=0.0).to(tl.float32)
        acc += x * x
        r += BLK_R
    s = tl.sum(acc, axis=1)
    y = 1.0 / tl.sqrt(s / R + EPS)
    tl.store(Y + rows, y, mask=rmask)


@triton.jit
def _inv_v2(X, Y, A, R, EPS, NCH: tl.constexpr, BLK_A: tl.constexpr, BLK_R: tl.constexpr):
    """变体A：`for` + constexpr 次数（NCH=ceil_div(R,BLK_R)），一行一 program 不做行块。"""
    pid = tl.program_id(0)
    row = pid
    cols = tl.arange(0, BLK_R)
    acc = tl.zeros((BLK_R,), dtype=tl.float32)
    for i in range(NCH):
        col = i * BLK_R + cols
        m = col < R
        x = tl.load(X + row * R + col, mask=m, other=0.0).to(tl.float32)
        acc += x * x
    s = tl.sum(acc, axis=0)
    y = 1.0 / tl.sqrt(s / R + EPS)
    tl.store(Y + row, y)


def ref_inv(x, eps):
    xd = x.detach().cpu().double()
    R = xd.shape[-2] * xd.shape[-1]
    return (1.0 / torch.sqrt((xd.reshape(xd.shape[0], -1).pow(2).sum(1) / R) + eps))


def ceil_div(a, b):
    return (a + b - 1) // b


for shape in [(2, 2, 4), (8, 1, 8), (4, 4, 4), (16, 4, 4), (17, 4, 16), (64, 4, 16)]:
    R = shape[-2] * shape[-1]
    A = 1
    for v in shape[:-2]:
        A *= v
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * 0.5).to(torch.bfloat16)
    r = ref_inv(x, 1e-6)
    line = f"  shape={str(shape):16s} A={A:4d} R={R:4d}"
    # 现行 v1
    try:
        from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms as cur
        y = cur(x, 1e-6)
        torch.npu.synchronize()
        line += f" | 现行 err={md(y.detach().cpu().float(), r.float()):.2e}"
    except Exception as e:
        line += f" | 现行 EXC {type(e).__name__}: {str(e)[:60]}"
    # 变体A v2
    try:
        blk_r = max(2, min(512, 1 << (R - 1).bit_length()))
        nch = ceil_div(R, blk_r)
        y2 = torch.empty((A,), dtype=torch.float32, device=DEV)
        _inv_v2[(A,)](x.contiguous().view(A, R), y2, A, R, 1e-6, NCH=nch,
                      BLK_A=1, BLK_R=blk_r)
        torch.npu.synchronize()
        line += f" | 变体A err={md(y2.detach().cpu().float(), r.float()):.2e}"
    except Exception as e:
        line += f" | 变体A EXC {type(e).__name__}: {str(e)[:60]}"
    print(line, flush=True)

# ============================================================ ① hc_pre d
print("\n" + "=" * 100, flush=True)
print("① hc_pre：d 扫描（找 d=7168 特有的东西）+ 与原生对照", flush=True)
print("=" * 100, flush=True)
from triton_ops.hc_pre import hc_pre  # noqa: E402


def make_hcpre(bs, hm, d, iters=20, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    sc = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    ba = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, sc, ba, hm, iters


def ref_hcpre_full(x, fn, sc, ba, hm, iters, rows):
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + 1e-6)
    fnq = fn.detach().cpu().double()
    mixes = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mixes += xflat[:, s:e] @ fnq[:, s:e].t()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * scd[0] + bbd[:hm]) + 1e-6
    A = (mixes[:, 2 * hm:].reshape(nr, hm, hm) * rq[:, None, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = A / e.sum(dim=2, keepdim=True) + 1e-6
    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + 1e-6)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    return torch.einsum("bj,bjd->bd", pre, xf), A


print(f"  {'d':>6} {'K':>8} {'triton_y':>11} {'native_y':>11} {'fp64_ref':>11}  {'triton_comb':>11} "
      f"{'native_comb':>11}", flush=True)
for d in [1024, 2048, 4096, 5120, 6144, 7168, 8192]:
    bs, hm, iters = 64, 4, 20
    rows = [0, 1, 2, 3, 5, 8, 13, 63]
    try:
        y, post, comb = hc_pre(*make_hcpre(bs, hm, d, iters), 1e-6, 1e-6)
        torch.npu.synchronize()
        ry, rA = ref_hcpre_full(*make_hcpre(bs, hm, d, iters), rows)
        ty = md(y.detach().cpu().double().reshape(bs, d)[rows], ry.to(torch.bfloat16).double())
        tc = md(comb.detach().cpu().double().reshape(bs, hm, hm)[rows], rA)
        nat_y = nat_c = float("nan")
        try:
            ny = torch.ops._C_ascend.npu_hc_pre(*make_hcpre(bs, hm, d, iters), 1e-6, 1e-6)
            torch.npu.synchronize()
            nat_y = md(ny[0].detach().cpu().double().reshape(bs, d)[rows], ry.to(torch.bfloat16).double())
            nat_c = md(ny[2].detach().cpu().double().reshape(bs, hm, hm)[rows], rA)
        except Exception as e:
            nat_y = f"REJ"
        print(f"  {d:>6} {hm * d:>8} {ty:>11.3e} {nat_y if isinstance(nat_y, str) else format(nat_y, '>11.3e')} "
              f"{'—':>11} {tc:>11.3e} {nat_c if isinstance(nat_c, str) else format(nat_c, '>11.3e')}",
              flush=True)
    except Exception as e:
        print(f"  {d:>6} EXC {type(e).__name__}: {str(e)[:80]}", flush=True)

print("DIAGDONE", flush=True)
