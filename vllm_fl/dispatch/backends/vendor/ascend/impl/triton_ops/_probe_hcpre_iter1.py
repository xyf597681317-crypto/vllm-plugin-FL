# -*- coding: utf-8 -*-
"""_probe_hcpre_iter1.py —— 定位残余缺陷：hc_pre d=7168 + iters=1 + bs≥4096

已确认（`_probe_hcpre_verify.py`）：
  d=7168 bs=64  it=1  OK        d=7168 bs=64  it=20 OK
  d=7168 bs=4096 it=1 FAIL      d=7168 bs=4096 it=20 OK
  d=7168 bs=8192 it=1 FAIL      d=7168 bs=8192 it=20 OK
  d=4096 全部 OK（it=1/20 × bs=64/4096/8192）
⇒ 只在 **(d=7168, iters=1, bs≥4096)** 组合上错。

本探针把三个子阶段各自与 fp64 对照，定位错在哪个 kernel：
  A) rsqrt（`_hc_pre_rsqrt_kernel`）
  B) mixes（`_hc_pre_mixes_kernel`）
  C) 端到端 hc_pre（y = sinkhorn_kernel 算的；post/comb 同）
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.hc_pre import (  # noqa: E402
    hc_pre, _hc_pre_mixes_kernel, _hc_pre_rsqrt_kernel)


def md(a, b):
    a, b = a.double(), b.double()
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


def cdiv(a, b):
    return (a + b - 1) // b


def make(bs, hm, d, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    sc = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    ba = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, sc, ba


def ref_mix_rsqrt(x, fn, rows):
    hm = x.shape[1]
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(1) / K + 1e-6)
    fnq = fn.detach().cpu().double()
    mix = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mix += xflat[:, s:e] @ fnq[:, s:e].t()
    return mix, rq


print("===== 子阶段定位：d=7168, iters 扫描（固定 bs=4096）=====", flush=True)
print(f"  {'iters':>6} {'e2e_y':>12} {'e2e_post':>12} {'e2e_comb':>12}  说明", flush=True)
for IT in [1, 2, 3, 5, 20]:
    bs, d, hm = 4096, 7168, 4
    x, fn, sc, ba = make(bs, hm, d)
    hmix, K = 2 * hm + hm * hm, hm * d
    rows = sorted({0, 1, 2, bs // 2, bs - 2, bs - 1})
    rows = [r for r in rows if 0 <= r < bs]
    mix_ref, rq_ref = ref_mix_rsqrt(x, fn, rows)
    y, post, comb = hc_pre(x, fn, sc, ba, hm, IT, 1e-6, 1e-6)
    torch.npu.synchronize()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mix_ref[:, :hm] * rq_ref[:, None] * scd[0] + bbd[:hm]) + 1e-6
    xf = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    ry = torch.einsum("bj,bjd->bd", pre, xf)
    A = (mix_ref[:, 2 * hm:].reshape(len(rows), hm, hm) * rq_ref[:, None, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e_ = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e_ / e_.sum(dim=2, keepdim=True) + 1e-6
    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    for _ in range(IT - 1):                      # ← 参考按 IT 次迭代
        A = A / (A.sum(dim=2, keepdim=True) + 1e-6)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    rpost = 2.0 * torch.sigmoid(mix_ref[:, hm:2 * hm] * rq_ref[:, None] * scd[1] + bbd[hm:2 * hm])
    e_y = md(y.detach().cpu().double().reshape(bs, d)[rows], ry.to(torch.bfloat16).double())
    e_p = md(post.detach().cpu().double().reshape(bs, hm)[rows], rpost)
    e_c = md(comb.detach().cpu().double().reshape(bs, hm, hm)[rows], A)
    note = "y/post 只依赖 mixes/rsqrt ⇒ 与 iters 无关" if e_y < 2.5e-2 and e_p < 2e-6 else "**y/post 也错**"
    print(f"  {IT:>6} {e_y:>12.3e} {e_p:>12.3e} {e_c:>12.3e}  {note}", flush=True)

print("\n===== bs 扫描（固定 iters=1，用**同一份**参考）=====", flush=True)
print(f"  {'bs':>6} {'e2e_y':>12} {'e2e_post':>12} {'e2e_comb':>12}", flush=True)
for bs in [64, 1024, 4096, 8192]:
    d, hm = 7168, 4
    x, fn, sc, ba = make(bs, hm, d)
    rows = sorted({0, 1, 2, bs // 2, bs - 2, bs - 1})
    rows = [r for r in rows if 0 <= r < bs]
    mix_ref, rq_ref = ref_mix_rsqrt(x, fn, rows)
    y, post, comb = hc_pre(x, fn, sc, ba, hm, 1, 1e-6, 1e-6)
    torch.npu.synchronize()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mix_ref[:, :hm] * rq_ref[:, None] * scd[0] + bbd[:hm]) + 1e-6
    xf = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    ry = torch.einsum("bj,bjd->bd", pre, xf)
    A = (mix_ref[:, 2 * hm:].reshape(len(rows), hm, hm) * rq_ref[:, None, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e_ = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e_ / e_.sum(dim=2, keepdim=True) + 1e-6
    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    rpost = 2.0 * torch.sigmoid(mix_ref[:, hm:2 * hm] * rq_ref[:, None] * scd[1] + bbd[hm:2 * hm])
    print(f"  {bs:>6} {md(y.detach().cpu().double().reshape(bs, d)[rows], ry.to(torch.bfloat16).double()):>12.3e} "
          f"{md(post.detach().cpu().double().reshape(bs, hm)[rows], rpost):>12.3e} "
          f"{md(comb.detach().cpu().double().reshape(bs, hm, hm)[rows], A):>12.3e}", flush=True)
print("ITER1DONE", flush=True)
print("ITER1DONE", flush=True)
