# -*- coding: utf-8 -*-
"""_probe_cov_j3_hcpre3.py —— hc_pre / hc_pre_sinkhorn 三方对照（**干净重写版**）

动机：上一版（`_probe_cov_j3_hcpre2.py`）的比较代码自己不一致
（R=64 时 `comb` 的 native_vs_fp64 == triton_vs_native，而同行 triton_vs_native==0），
**不能采信**。本文件把"取采样行 + 比较"写成**唯一一个**辅助函数，先断言 shape 再比较。

判据（先确认 dtype/地板）：
  `y` = 输入 dtype(bf16) ⇒ 与原生比要求「完全一致」（同 dtype 同语义）；与 fp64 比用 1 ULP。
  `post`/`comb` = fp32 ⇒ 2e-6。

三方对照的读法：
  native≈fp64 且 triton≠fp64 ⇒ **我们的 kernel 错**
  triton≈fp64 且 native≠fp64 ⇒ 原生错（我们对了）
  两者都≈fp64                ⇒ 都对

跑法：DSH_NPU=npu:12 python3 -W ignore triton_ops/_probe_cov_j3_hcpre3.py
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
    print(f"[native] enable 失败: {type(e).__name__}: {str(e)[:150]}", flush=True)
print(f"[native] enable_custom_op -> {NATIVE_OK} | "
      f"npu_hc_pre={hasattr(torch.ops._C_ascend, 'npu_hc_pre')} | "
      f"npu_hc_pre_sinkhorn={hasattr(torch.ops._C_ascend, 'npu_hc_pre_sinkhorn')}",
      flush=True)


def take(t, rows, shape):
    """把张量 reshape 成 shape 后按 rows 取。**唯一**的取行入口。"""
    d = t.detach().cpu().double().reshape(*shape)
    assert d.shape[0] > max(rows), f"take: shape={tuple(d.shape)} rows 越界 {max(rows)}"
    return d[rows]


def md(a, b):
    assert a.shape == b.shape, f"比较 shape 不一致 {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


# ---------------------------------------------------------------- 输入
def make_hcpre(bs, hm, d, iters=20, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
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

    return (U(R, hmix, lo=1.0), U(R, lo=0.5, off=1.0), U(3, lo=0.5, off=1.0),
            U(hmix, lo=0.5), U(R, hm, d, lo=0.3).to(torch.bfloat16), hm, iters, eps)


# ---------------------------------------------------------------- fp64 参考（只算采样行）
def ref_hcpre(x, fn, scale, base, hm, iters, hc_eps, norm_eps, rows):
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]   # 唯一一次取行
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + norm_eps)
    fnq = fn.detach().cpu().double()
    mixes = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mixes += xflat[:, s:e] @ fnq[:, s:e].t()
    sc, bb = scale.detach().cpu().double(), base.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * sc[0] + bb[:hm]) + hc_eps
    y = torch.einsum("bj,bjd->bd", pre, xf)
    post = 2.0 * torch.sigmoid(mixes[:, hm:2 * hm] * rq[:, None] * sc[1] + bb[hm:2 * hm])
    A = (mixes[:, 2 * hm:].reshape(nr, hm, hm) * rq[:, None, None] * sc[2]
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
    sc, bb = hc_scale.detach().cpu().double(), hc_base.detach().cpu().double()
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


def run_hcpre(bs, hm, d, iters=20):
    rows = cov.probe_rows(bs, 8)
    tag = f"hc_pre bs={bs} hm={hm} d={d} it={iters}"
    print(f"\n===== {tag} =====", flush=True)
    tri = hc_pre(*make_hcpre(bs, hm, d, iters), 1e-6, 1e-6)
    torch.npu.synchronize()
    nat = None
    if NATIVE_OK and hasattr(torch.ops._C_ascend, "npu_hc_pre"):
        try:
            x2, fn2, s2, b2, hmv, itv = make_hcpre(bs, hm, d, iters)
            nat = torch.ops._C_ascend.npu_hc_pre(x2, fn2, s2, b2, hmv, itv, 1e-6, 1e-6)
            torch.npu.synchronize()
        except Exception as e:
            print(f"    [native] 拒绝/失败: {type(e).__name__}: {str(e)[:160]}", flush=True)
            nat = None
    ry, rpost, rcomb = ref_hcpre(*make_hcpre(bs, hm, d, iters), 1e-6, 1e-6, rows)
    ulp = (2.0 ** -8) * max(float(ry.to(torch.bfloat16).double().abs().max()), 1e-12)
    specs = [("y", 0, (bs, d), ry.to(torch.bfloat16).double(), ulp),
             ("post", 1, (bs, hm), rpost, POST_FLOOR),
             ("comb", 2, (bs, hm, hm), rcomb, POST_FLOOR)]
    verdict = []
    for nm, idx, shp, rv, thr in specs:
        t_ = take(tri[idx], rows, shp)
        a = md(t_, rv)
        b = c = float("nan")
        if nat is not None:
            n_ = take(nat[idx], rows, shp)
            b, c = md(n_, rv), md(t_, n_)
        print(f"    {nm:5s} triton_vs_fp64={a:.3e}  native_vs_fp64={b:.3e}  "
              f"triton_vs_native={c:.3e}   (阈值 {thr:.2e})", flush=True)
        # 只在三方都拿得到时给判定
        ok_tri = a <= thr
        ok_nat = (b <= thr) if nat is not None else None
        verdict.append((nm, ok_tri, ok_nat))
    if nat is not None:
        bad = [v for v in verdict if not v[1]]
        nat_ok = all(v[2] for v in verdict)
        print(f"    ⇒ verdict: triton {'OK' if not bad else 'FAIL@' + ','.join(v[0] for v in bad)}"
              f" | native {'OK' if nat_ok else 'FAIL'}", flush=True)
    return verdict


def run_sk(R, hm, d, iters=20, eps=1e-6):
    rows = cov.probe_rows(R, 8)
    tag = f"hc_pre_sinkhorn R={R} hm={hm} d={d} it={iters}"
    print(f"\n===== {tag} =====", flush=True)
    tri = hc_pre_sinkhorn(*make_sk(R, hm, d, iters, eps))
    torch.npu.synchronize()
    nat = None
    if NATIVE_OK and hasattr(torch.ops._C_ascend, "npu_hc_pre_sinkhorn"):
        try:
            ins2 = make_sk(R, hm, d, iters, eps)
            nat = torch.ops._C_ascend.npu_hc_pre_sinkhorn(*ins2)
            torch.npu.synchronize()
        except Exception as e:
            print(f"    [native] 拒绝/失败: {type(e).__name__}: {str(e)[:160]}", flush=True)
            nat = None
    ins3 = make_sk(R, hm, d, iters, eps)
    ry, rpost, rcomb = ref_sk(*ins3, rows=rows)
    xd = ins3[4]
    fy = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[xd.dtype]
    verdict = []
    for nm, idx, shp, rv, thr in [("y", 0, (R, d), ry.to(xd.dtype).double(), fy),
                                  ("post", 1, (R, hm), rpost, POST_FLOOR),
                                  ("comb", 2, (R, hm, hm), rcomb, POST_FLOOR)]:
        t_ = take(tri[idx], rows, shp)
        a = md(t_, rv)
        b = c = float("nan")
        if nat is not None:
            n_ = take(nat[idx], rows, shp)
            b, c = md(n_, rv), md(t_, n_)
        print(f"    {nm:5s} triton_vs_fp64={a:.3e}  native_vs_fp64={b:.3e}  "
              f"triton_vs_native={c:.3e}   (阈值 {thr:.2e})", flush=True)
        verdict.append((nm, a <= thr, (b <= thr) if nat is not None else None))
    if nat is not None:
        bad = [v for v in verdict if not v[1]]
        nat_ok = all(v[2] for v in verdict)
        print(f"    ⇒ verdict: triton {'OK' if not bad else 'FAIL@' + ','.join(v[0] for v in bad)}"
              f" | native {'OK' if nat_ok else 'FAIL'}", flush=True)
    return verdict


if __name__ == "__main__":
    print(f"J3-HCPRE3 | DEV={DEV}", flush=True)
    which = os.environ.get("J3P3", "all")
    if which in ("all", "pre"):
        # d 只支持 4096/7168（原生自己报的约束）⇒ 生产形状走 4096/7168
        for bs, hm, d, it in [(4096, 4, 4096, 20), (4096, 4, 7168, 20), (64, 4, 7168, 20),
                              (8192, 4, 4096, 20), (4096, 4, 4096, 1), (16, 4, 7168, 20)]:
            try:
                run_hcpre(bs, hm, d, it)
            except Exception:
                print(f"[EXC] hc_pre bs={bs} d={d}\n{traceback.format_exc()[-350:]}", flush=True)
    if which in ("all", "sk"):
        for R, hm, d in [(64, 4, 512), (128, 4, 512), (512, 4, 512), (4096, 4, 512),
                         (4096, 4, 7168), (33, 4, 100)]:
            try:
                run_sk(R, hm, d)
            except Exception:
                print(f"[EXC] sk R={R} d={d}\n{traceback.format_exc()[-350:]}", flush=True)
    print("J3HCPRE3DONE", flush=True)
