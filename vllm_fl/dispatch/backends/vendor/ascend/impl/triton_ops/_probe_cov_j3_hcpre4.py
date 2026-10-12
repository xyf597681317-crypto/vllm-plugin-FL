# -*- coding: utf-8 -*-
"""_probe_cov_j3_hcpre4.py —— 定位阈值与错法（干净版三方对照的延续）

已确认（`_probe_cov_j3_hcpre3.py`，判据与取行代码均为单一实现）：
  hc_pre:            d=4096 全形状 OK（bs=4096/8192、iters=1/20）；**d=7168 起就错**（bs=64 也错、y=nan）
  hc_pre_sinkhorn:   R=64/128 我们 OK；**R≥512 起我们错**；原生在 R≤128 与 R=512(d=512) 的 comb 也错

本探针回答两件事：
  A) `hc_pre_sinkhorn` 的**阈值**：R = 1..1024 逐档，看从哪一档开始错（含"前 8 行是否本来就对"）
  B) **错法**：把我们与参考的 post/comb 按行并排，判断是"整行错"还是"少数行错"，
     并检查是否等于**按行平移**（post[i] ≈ ref[i+1]）——这能直接指认行索引问题
  C) `hc_pre` d=7168：确认是 d 相关而非 bs 相关（已在上一版看到 bs=64 也错），这里补打印哪一段 d 先错
"""
import importlib.util
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

_s = importlib.util.spec_from_file_location(
    "cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cov)
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402
from triton_ops.hc_pre import hc_pre  # noqa: E402

try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    enable_custom_op()
except Exception:
    pass


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


print("===== A) hc_pre_sinkhorn 阈值扫描（d=512 hm=4，看前 8 行）=====", flush=True)
print(f"{'R':>6} {'y_err':>11} {'post_err':>11} {'comb_err':>11}  verdict", flush=True)
for R in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    rows = cov.probe_rows(R, 8)
    ins = make_sk(R, 4, 512)
    y, post, comb = hc_pre_sinkhorn(*ins)
    torch.npu.synchronize()
    ry, rp, rc = ref_sk(*ins, rows=rows)
    d = 512
    ey = float((y.detach().cpu().double().reshape(R, d)[rows] - ry.to(torch.bfloat16).double()).abs().max())
    ep = float((post.detach().cpu().double().reshape(R, 4)[rows] - rp).abs().max())
    ec = float((comb.detach().cpu().double().reshape(R, 4, 4)[rows] - rc).abs().max())
    ok = ey <= 4e-3 and ep <= 2e-6 and ec <= 2e-6
    print(f"{R:>6} {ey:>11.3e} {ep:>11.3e} {ec:>11.3e}  {'OK' if ok else 'FAIL'}", flush=True)

print("\n===== B) 错法：R=512 的 post / comb 逐行对照（前 8 行 + 平移假设）=====", flush=True)
R = 512
rows = cov.probe_rows(R, 8)
ins = make_sk(R, 4, 512)
y, post, comb = hc_pre_sinkhorn(*ins)
torch.npu.synchronize()
ry, rp, rc = ref_sk(*ins, rows=rows)
got_p = post.detach().cpu().double().reshape(R, 4)[rows]
print(f"  rows={rows}", flush=True)
print(f"  got_post  = {[[round(float(v), 6) for v in r] for r in got_p[:4]]}", flush=True)
print(f"  ref_post  = {[[round(float(v), 6) for v in r] for r in rp[:4]]}", flush=True)
# 平移假设：got[i] 是否等于 ref 的**其它行**
full_rp = ref_sk(*ins, rows=list(range(R)))[1]
for i, r in enumerate(rows[:4]):
    hit = None
    for cand in range(max(0, r - 4), min(R, r + 5)):
        if float((got_p[i] - full_rp[cand]).abs().max()) < 1e-9:
            hit = cand
            break
    print(f"    row={r}: got == ref_row[{hit}]" if hit is not None else
          f"    row={r}: 不等于附近任何一行（非平移）", flush=True)
# 全量统计：有多少行对
all_p = post.detach().cpu().double().reshape(R, 4)
nbad = int((all_p - full_rp).abs().max(dim=1).values.gt(2e-6).sum().item())
print(f"  全量 R={R}: post 错行数 = {nbad}/{R}", flush=True)
all_y = y.detach().cpu().double().reshape(R, 512)
full_ry = ref_sk(*ins, rows=list(range(R)))[0].to(torch.bfloat16).double()
nbad_y = int((all_y - full_ry).abs().max(dim=1).values.gt(4e-3).sum().item())
print(f"  全量 R={R}: y 错行数 = {nbad_y}/{R}", flush=True)

print("\n===== C) hc_pre d=7168：错在哪里（bs=64）=====", flush=True)


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
    mixes = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mixes += xflat[:, s:e] @ fnq[:, s:e].t()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * scd[0] + bbd[:hm]) + 1e-6
    return torch.einsum("bj,bjd->bd", pre, xf)


for d in [4096, 7168]:
    bs = 64
    rows = cov.probe_rows(bs, 8)
    y, post, comb = hc_pre(*make_hcpre(bs, 4, d), 1e-6, 1e-6)
    torch.npu.synchronize()
    ry = ref_hcpre(*make_hcpre(bs, 4, d), rows)
    yc = y.detach().cpu().double().reshape(bs, d)[rows]
    nan = int(torch.isnan(yc).sum().item())
    full = y.detach().cpu().double().reshape(bs, d)
    allr = ref_hcpre(*make_hcpre(bs, 4, d), list(range(bs))).to(torch.bfloat16).double()
    nbad = int((full - allr).abs().max(dim=1).values.gt(2.4e-2).sum().item())
    print(f"  d={d}: 采样行 y_err={float((yc - ry.to(torch.bfloat16).double()).abs().max()):.3e} "
          f"nan={nan}/{yc.numel()} | 全量错行={nbad}/{bs}", flush=True)
print("J3HCPRE4DONE", flush=True)
