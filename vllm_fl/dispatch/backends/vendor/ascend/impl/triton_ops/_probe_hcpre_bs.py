# -*- coding: utf-8 -*-
"""_probe_hcpre_bs.py —— 修后：定位 hc_pre 残余缺陷的 bs 阈值 + 复核 sinkhorn 大 R

已确认修后状态（`_probe_cov_j3_hcpre3.py`）：
  hc_pre d=7168：bs=16/64 **OK**（修好了），bs=4096 **仍错**（y=4.148 post=1.255 comb=0.186）
  hc_pre d=4096：bs=4096/8192 OK
  hc_pre_sinkhorn：R=64/128/512 **全部 OK**（原来 R≥256 错）⇒ 该修有效

本探针：
  A) hc_pre d=7168，bs = 64→8192 逐档，找阈值（grid = bs_pad/16 ⇒ 4096 时 256 个 program）
  B) sinkhorn R = 256..4096 复核（Lead 要求的验收区间）
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    enable_custom_op()
except Exception:
    pass
from triton_ops.hc_pre import hc_pre  # noqa: E402
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402


def md(a, b):
    a, b = a.double(), b.double()
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


def make_hcpre(bs, hm, d, iters=20, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    sc = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    ba = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, sc, ba, hm, iters


def ref_hcpre(x, fn, sc, ba, hm, iters, rows):
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + 1e-6)
    fnq = fn.detach().cpu().double()
    mix = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mix += xflat[:, s:e] @ fnq[:, s:e].t()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mix[:, :hm] * rq[:, None] * scd[0] + bbd[:hm]) + 1e-6
    y = torch.einsum("bj,bjd->bd", pre, xf)
    post = 2.0 * torch.sigmoid(mix[:, hm:2 * hm] * rq[:, None] * scd[1] + bbd[hm:2 * hm])
    A = (mix[:, 2 * hm:].reshape(nr, hm, hm) * rq[:, None, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = A / e.sum(dim=2, keepdim=True) + 1e-6
    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + 1e-6)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    return y, post, A


print("===== A) hc_pre d=7168：bs 阈值扫描 =====", flush=True)
print(f"  {'bs':>6} {'bs_pad':>7} {'grid':>6} | {'y_err':>11} {'post_err':>11} {'comb_err':>11} verdict",
      flush=True)
for bs in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    rows = [0, 1, 2, 3, min(5, bs - 1), min(8, bs - 1), min(13, bs - 1), bs - 1]
    rows = sorted(set(r for r in rows if 0 <= r < bs))
    try:
        y, post, comb = hc_pre(*make_hcpre(bs, 4, 7168), 1e-6, 1e-6)
        torch.npu.synchronize()
        ry, rp, rA = ref_hcpre(*make_hcpre(bs, 4, 7168), rows)
        ey = md(y.detach().cpu().double().reshape(bs, 7168)[rows], ry.to(torch.bfloat16).double())
        ep = md(post.detach().cpu().double().reshape(bs, 4)[rows], rp)
        ec = md(comb.detach().cpu().double().reshape(bs, 4, 4)[rows], rA)
        ok = ey <= 2.5e-2 and ep <= 2e-6 and ec <= 2e-6
        print(f"  {bs:>6} {(bs + 15) // 16 * 16:>7} {(bs + 15) // 16:>6} | {ey:>11.3e} "
              f"{ep:>11.3e} {ec:>11.3e} {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        print(f"  {bs:>6} EXC {type(e).__name__}: {str(e)[:90]}", flush=True)

print("\n===== B) hc_pre_sinkhorn：R 扫描（修后复核 Lead 要求的区间）=====", flush=True)


def make_sk(R, hm, d, iters=20, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm

    def U(*s, lo=1.0, off=0.0):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo + off
    return (U(R, hmix, lo=1.0), U(R, lo=0.5, off=1.0), U(3, lo=0.5, off=1.0),
            U(hmix, lo=0.5), U(R, hm, d, lo=0.3).to(torch.bfloat16), hm, iters, eps)


def ref_sk(mixes, rsqrt, sc, bb, x, hm, iters, eps, rows):
    mx = mixes.detach().cpu().double().reshape(-1, mixes.shape[-1])[rows]
    rs = rsqrt.detach().cpu().double().view(-1, 1)[rows]
    scd, bbd = sc.detach().cpu().double(), bb.detach().cpu().double()
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    pre = torch.sigmoid(mx[:, :hm] * rs * scd[0] + bbd[:hm]) + eps
    y = torch.einsum("ij,ijd->id", pre, xd)
    post = 2.0 * torch.sigmoid(mx[:, hm:2 * hm] * rs * scd[1] + bbd[hm:2 * hm])
    A = (mx[:, 2 * hm:].reshape(len(rows), hm, hm) * rs[:, :, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + eps
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + eps)
        A = A / (A.sum(dim=1, keepdim=True) + eps)
    return y, post, A


print(f"  {'R':>6} {'d':>6} | {'y_err':>11} {'post_err':>11} {'comb_err':>11} verdict", flush=True)
for R, d in [(128, 512), (256, 512), (512, 512), (1024, 512), (2048, 512), (4096, 512),
             (4096, 7168), (256, 7168)]:
    rows = [0, 1, 2, 3, 5, 8, 13, R - 1] if R > 14 else list(range(min(R, 8)))
    rows = sorted(set(r for r in rows if 0 <= r < R))
    try:
        ins = make_sk(R, 4, d)
        y, post, comb = hc_pre_sinkhorn(*ins)
        torch.npu.synchronize()
        ry, rp, rc = ref_sk(*ins, rows=rows)
        ey = md(y.detach().cpu().double().reshape(R, d)[rows], ry.to(torch.bfloat16).double())
        ep = md(post.detach().cpu().double().reshape(R, 4)[rows], rp)
        ec = md(comb.detach().cpu().double().reshape(R, 4, 4)[rows], rc)
        ok = ey <= 4e-3 and ep <= 2e-6 and ec <= 2e-6
        print(f"  {R:>6} {d:>6} | {ey:>11.3e} {ep:>11.3e} {ec:>11.3e} {'OK' if ok else 'FAIL'}",
              flush=True)
    except Exception as e:
        print(f"  {R:>6} {d:>6} EXC {type(e).__name__}: {str(e)[:90]}", flush=True)
print("BSDONE", flush=True)
