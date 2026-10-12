# -*- coding: utf-8 -*-
"""_probe_cov_j3_hcpre2.py —— hc_pre / hc_pre_sinkhorn：**三方对照**（原生 AscendC vs Triton vs fp64）

背景与动机：
1. 上一轮 `_probe_cov_j3_hcpre.py` 报 `hc_pre bs=4096 hm=4 d=512` 的 `y_err=6.148e+00`
   （阈值 ULP≈8.2e-3），而 `post`/`comb` 只有 ~1e-7 —— **必须定性**是 kernel 错还是参考错。
   已知该文件当时**参考侧有索引 bug**（把 `[rows]` 用了两次），所以先修参考再判。
2. Lead 广播：`vllm_fl.ascend_custom_ops.enable_custom_op()` 注册的**原生 payload 可作裁判**
   （aclnn 薄封装那条路不可用）。⇒ 本探针做三方对照，判据强度远高于"只能主机复算"。

## 判据（先确认 dtype/地板）
`hc_pre` / `hc_pre_sinkhorn` 都返回 `(y, post, comb)`：
  - `y` = 输入 dtype（bf16）⇒ 与原生比用**完全相同**（同 dtype 同实现语义），与 fp64 比用 1 ULP；
  - `post`/`comb` = fp32 ⇒ 2e-6。
三方对照的意义：
  Triton≈fp64 且 native≠fp64 ⇒ **原生有问题**（我们是对的）
  Triton≠fp64 且 native≈fp64 ⇒ **我们的 kernel 有问题**
  两者都≈fp64          ⇒ 都对（差异只是浮点路径）

跑法：DSH_NPU=npu:12 python3 -W ignore triton_ops/_probe_cov_j3_hcpre2.py
"""
import importlib.util
import os
import sys
import traceback

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

from triton_ops.hc_pre import hc_pre  # noqa: E402
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402

POST_FLOOR = 2e-6
NATIVE_OK = False
try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    NATIVE_OK = bool(enable_custom_op())
except Exception as e:
    print(f"[native] enable_custom_op 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
print(f"[native] enable_custom_op -> {NATIVE_OK}", flush=True)
for nm in ("npu_hc_pre", "npu_hc_pre_sinkhorn"):
    print(f"[native]   {nm}: hasattr={hasattr(torch.ops._C_ascend, nm)}", flush=True)


# ---------------------------------------------------------------- 输入构造
def make_hcpre(bs, hm, d, iters=20, seed=13):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm
    K = hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    scale = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    base = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, scale, base, hm, iters


def make_sk(R, hm, d, iters=20, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm

    def U(*shape, lo=1.0, off=0.0):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo + off

    mixes = U(R, hmix, lo=1.0)
    rsqrt = U(R, lo=0.5, off=1.0)
    hc_scale = U(3, lo=0.5, off=1.0)
    hc_base = U(hmix, lo=0.5)
    x = U(R, hm, d, lo=0.3).to(torch.bfloat16)
    return mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps


# ---------------------------------------------------------------- fp64 参考（只算采样行）
def ref_hcpre(x, fn, scale, base, hm, iters, hc_eps, norm_eps, rows):
    """返回 (y[rows,d], post[rows,hm], comb[rows,hm,hm]) —— **rows 只在这里用一次**。"""
    xf_all = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])
    xf = xf_all[rows]                       # [nr, hm, d]   ← 唯一一次按行取
    d = xf.shape[-1]
    K = hm * d
    xflat = xf.reshape(len(rows), K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + norm_eps)
    fnq = fn.detach().cpu().double()
    mixes = torch.zeros(len(rows), fnq.shape[0], dtype=torch.float64)
    CH = 8192
    for s in range(0, K, CH):
        e = min(K, s + CH)
        mixes += xflat[:, s:e] @ fnq[:, s:e].t()
    sc = scale.detach().cpu().double()
    bb = base.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * sc[0] + bb[:hm]) + hc_eps
    y = torch.einsum("bj,bjd->bd", pre, xf)          # [nr, d]
    post = 2.0 * torch.sigmoid(mixes[:, hm:2 * hm] * rq[:, None] * sc[1] + bb[hm:2 * hm])
    A = (mixes[:, 2 * hm:].reshape(len(rows), hm, hm) * rq[:, None, None] * sc[2]
         + bb[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + hc_eps
    A = A / (A.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + hc_eps)
        A = A / (A.sum(dim=1, keepdim=True) + hc_eps)
    return y, post, A


def ref_sk(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps, rows):
    mx = mixes.detach().cpu().double().reshape(-1, mixes.shape[-1])[rows]
    rs = rsqrt.detach().cpu().double().view(-1, 1)[rows]
    sc = hc_scale.detach().cpu().double()
    bb = hc_base.detach().cpu().double()
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    pre = torch.sigmoid(mx[:, :hm] * rs * sc[0] + bb[:hm]) + eps
    y = torch.einsum("ij,ijd->id", pre, xd)
    post = 2.0 * torch.sigmoid(mx[:, hm:2 * hm] * rs * sc[1] + bb[hm:2 * hm])
    A = (mx[:, 2 * hm:].reshape(len(rows), hm, hm) * rs[:, :, None] * sc[2]
         + bb[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + eps
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + eps)
        A = A / (A.sum(dim=1, keepdim=True) + eps)
    return y, post, A


def cmp3(tag, tri, nat, refv, rows, is_y, ulp=None, floor=None):
    """返回 (tri_vs_ref, nat_vs_ref, tri_vs_nat) 三个 max|Δ|。"""
    def err(a, b):
        a = a.detach().cpu().double()[rows] if a.dim() > 1 and a.shape[0] != len(rows) else a.detach().cpu().double()
        b = b.detach().cpu().double()
        return float((a - b).abs().max()) if a.numel() else 0.0
    e_tr = err(tri, refv)
    e_nr = err(nat, refv) if nat is not None else float("nan")
    e_tn = err(tri, nat) if nat is not None else float("nan")
    thr = ulp if is_y else floor
    print(f"    {tag:10s} triton_vs_fp64={e_tr:.3e}  native_vs_fp64={e_nr:.3e}  "
          f"triton_vs_native={e_tn:.3e}   (阈值 {thr:.2e})", flush=True)
    return e_tr, e_nr, e_tn, thr


def run_hcpre(name, bs, hm, d, iters=20):
    print(f"\n===== hc_pre {name} (bs={bs} hm={hm} d={d} iters={iters}) =====", flush=True)
    rows = cov.probe_rows(bs, 8)
    # 每阶段重建同 seed 输入（原生可能 in-place 改输入）
    x, fn, scale, base, hmv, itv = make_hcpre(bs, hm, d, iters)
    tri = hc_pre(x, fn, scale, base, hmv, itv, 1e-6, 1e-6)
    torch.npu.synchronize()
    nat = None
    if NATIVE_OK and hasattr(torch.ops._C_ascend, "npu_hc_pre"):
        try:
            x2, fn2, sc2, ba2, _, _ = make_hcpre(bs, hm, d, iters)
            nat = torch.ops._C_ascend.npu_hc_pre(x2, fn2, sc2, ba2, hmv, itv, 1e-6, 1e-6)
            torch.npu.synchronize()
        except Exception as e:
            print(f"    [native] 调用失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
            nat = None
    x3, fn3, sc3, ba3, _, _ = make_hcpre(bs, hm, d, iters)
    ry, rpost, rcomb = ref_hcpre(x3, fn3, sc3, ba3, hmv, itv, 1e-6, 1e-6, rows)
    ulp = (2.0 ** -8) * max(float(ry.to(torch.bfloat16).double().abs().max()), 1e-12)
    # y：三方都按采样行取
    yc = tri[0].detach().cpu().double().reshape(bs, d)[rows]
    yk = nat[0].detach().cpu().double().reshape(bs, d)[rows] if nat is not None else None
    yref = ry.to(torch.bfloat16).double()
    e_tr = float((yc - yref).abs().max())
    e_nr = float((yk - yref).abs().max()) if yk is not None else float("nan")
    e_tn = float((yc - yk).abs().max()) if yk is not None else float("nan")
    print(f"    y        triton_vs_fp64={e_tr:.3e}  native_vs_fp64={e_nr:.3e}  "
          f"triton_vs_native={e_tn:.3e}   (阈值 1ULP={ulp:.2e})", flush=True)
    for i, (nm, ridx) in enumerate([("post", 1), ("comb", 2)]):
        tc = tri[ridx].detach().cpu().double().reshape(bs, -1)[rows]
        nc = nat[ridx].detach().cpu().double().reshape(bs, -1)[rows] if nat is not None else None
        rc = (rpost if ridx == 1 else rcomb).reshape(len(rows), -1)
        a = float((tc - rc).abs().max())
        b = float((nc - rc).abs().max()) if nc is not None else float("nan")
        c = float((tc - nc).abs().max()) if nc is not None else float("nan")
        print(f"    {nm:8s} triton_vs_fp64={a:.3e}  native_vs_fp64={b:.3e}  "
              f"triton_vs_native={c:.3e}   (阈值 {POST_FLOOR:.0e})", flush=True)
    return e_tr, ulp


def run_sk(name, R, hm, d, iters=20, eps=1e-6):
    print(f"\n===== hc_pre_sinkhorn {name} (R={R} hm={hm} d={d} iters={iters}) =====", flush=True)
    rows = cov.probe_rows(R, 8)
    ins = make_sk(R, hm, d, iters, eps)
    tri = hc_pre_sinkhorn(*ins)
    torch.npu.synchronize()
    nat = None
    if NATIVE_OK and hasattr(torch.ops._C_ascend, "npu_hc_pre_sinkhorn"):
        try:
            ins2 = make_sk(R, hm, d, iters, eps)
            m2, r2, s2, b2, x2, hmv, itv, e2 = ins2
            nat = torch.ops._C_ascend.npu_hc_pre_sinkhorn(m2, r2, s2, b2, x2, hmv, itv, e2)
            torch.npu.synchronize()
        except Exception as e:
            print(f"    [native] 调用失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
            nat = None
    ins3 = make_sk(R, hm, d, iters, eps)
    ry, rpost, rcomb = ref_sk(*ins3, rows=rows)
    xd = ins3[4]
    fy = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[xd.dtype]
    tc = tri[0].detach().cpu().double().reshape(R, d)[rows]
    nc = nat[0].detach().cpu().double().reshape(R, d)[rows] if nat is not None else None
    yref = ry.to(xd.dtype).double()
    e_tr = float((tc - yref).abs().max())
    e_nr = float((nc - yref).abs().max()) if nc is not None else float("nan")
    e_tn = float((tc - nc).abs().max()) if nc is not None else float("nan")
    print(f"    y        triton_vs_fp64={e_tr:.3e}  native_vs_fp64={e_nr:.3e}  "
          f"triton_vs_native={e_tn:.3e}   (阈值 {fy:g})", flush=True)
    for ridx, nm in [(1, "post"), (2, "comb")]:
        a = float((tri[ridx].detach().cpu().double().reshape(R, -1)[rows]
                   - (rpost if ridx == 1 else rcomb).reshape(len(rows), -1)).abs().max())
        b = c = float("nan")
        if nat is not None:
            nv = nat[ridx].detach().cpu().double().reshape(R, -1)[rows]
            b = float((nv - (rpost if ridx == 1 else rcomb).reshape(len(rows), -1)).abs().max())
            c = float((tri[ridx].detach().cpu().double().reshape(R, -1)[rows] - nv).abs().max())
        print(f"    {nm:8s} triton_vs_fp64={a:.3e}  native_vs_fp64={b:.3e}  "
              f"triton_vs_native={c:.3e}   (阈值 {POST_FLOOR:.0e})", flush=True)


if __name__ == "__main__":
    print(f"J3-HCPRE2 | DEV={DEV} current={torch.npu.current_device()}", flush=True)
    which = os.environ.get("J3P2", "all")
    if which in ("all", "pre"):
        for bs, hm, d, it in [(64, 4, 512, 20), (64, 4, 100, 20), (4096, 4, 512, 20),
                              (4096, 4, 7168, 20), (4096, 4, 512, 1), (16, 8, 128, 20)]:
            try:
                run_hcpre(f"", bs, hm, d, it)
            except Exception:
                print(f"[EXC] hc_pre bs={bs} d={d} iters={it}\n{traceback.format_exc()[-400:]}",
                      flush=True)
    if which in ("all", "sk"):
        for R, hm, d in [(64, 4, 512), (4096, 4, 512), (4096, 4, 7168), (33, 4, 100)]:
            try:
                run_sk("", R, hm, d)
            except Exception:
                print(f"[EXC] sk R={R} d={d}\n{traceback.format_exc()[-400:]}", flush=True)
    print("J3HCPRE2DONE", flush=True)
