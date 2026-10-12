# -*- coding: utf-8 -*-
"""_probe_cov_j3_hcpre.py —— J3：hc_pre / hc_pre_sinkhorn + **注册入口 eps 顺序**核对

## 判据（先确认 dtype/地板 —— 上一件就是栽在这里）
两个算子都返回 **(y, post, comb)**：
  - `y` = **输入 dtype**（bf16）⇒ 判据用 bf16 ULP（与 test_hc_pre.py 同口径：
    `ulp = 2**-8 * max|ref_y|`，即 bf16 的 1 ULP 量级）；
  - `post` / `comb` = **fp32** ⇒ 判据 2e-6（与两个测试文件一致）。
`_coverage_sweep.py::run_j3` 只看**一个**输出；所以：
  - 对 `y` 用 `run_j3(out_index=0, float_floor=ULP)`；
  - `post`/`comb` 在本文件里**单独比对**（固定 2e-6）。

## 额外：`npu_hc_pre` 注册入口的 eps 顺序（Lead 特别点名，此前无独立对拍）
schema：`npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)`
实现：`hc_pre(..., hc_eps=1e-6, norm_eps=1e-6)`  ← **位置相反**
`_C_ascend_extra._adapter_npu_hc_pre` 做了显式重映射。本探针用**两个不同的 eps 值**
（`norm_eps=1e-3, hc_eps=1e-7`）验证：
  A) 走 adapter（模拟注册入口调用）== 直接调用实现且 eps 正确传参  ⇒ 必须**逐位一致**
  B) 若按位置硬传（`hc_eps=1e-3, norm_eps=1e-7`，即假想的"没做重映射"）⇒ 应与 A **明显不同**
     （这一条是**灵敏度自检**：若 B 与 A 相同，说明我的测试根本验不出顺序问题，那结论无效）

跑法：DSH_NPU=npu:12 python3 -W ignore triton_ops/_probe_cov_j3_hcpre.py
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
J3 = []

from triton_ops.hc_pre import hc_pre  # noqa: E402
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402

POST_FLOOR = 2e-6


# ======================= hc_pre =======================
def build_hcpre(bs, hm, d, iters=20, four_d=False, seed=13):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm
    K = hm * d
    xshape = (2, bs // 2, hm, d) if four_d else (bs, hm, d)
    x = (torch.randn(*xshape, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = (torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5)))
    scale = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    base = (torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1)
    return (x, fn, scale, base, hm, iters)


def fn_hcpre(x, fn, scale, base, hm, iters):
    return hc_pre(x, fn, scale, base, hm, iters, 1e-6, 1e-6)


def _hcpre_core(x, fn, scale, base, hm, iters, hc_eps, norm_eps, rows):
    """fp64 参考（与 test_hc_pre.py::ref_impl 同语义），只算采样行。

    x: [bs,hm,d]（已 reshape 平铺）；rows: 采样行下标。
    """
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    d = xf.shape[-1]
    K = hm * d
    xflat = xf.reshape(len(rows), K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + norm_eps)
    fnq = fn.detach().cpu().double()
    # 分块做 x@fn^T（大 K 时避免一次性 [len(rows), K] 之外的内存峰值）
    mixes = torch.zeros(len(rows), fnq.shape[0], dtype=torch.float64)
    CH = 8192
    for s in range(0, K, CH):
        e = min(K, s + CH)
        mixes += xflat[:, s:e] @ fnq[:, s:e].t()
    sc = scale.detach().cpu().double()
    bb = base.detach().cpu().double()
    pre = torch.sigmoid(mixes[:, :hm] * rq[:, None] * sc[0] + bb[:hm]) + hc_eps
    y = torch.einsum("bj,bjd->bd", pre, xf)
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


def ref_rows_hcpre(x, fn, scale, base, hm, iters, rows=None, want_scale=False):
    y, _, _ = _hcpre_core(x, fn, scale, base, hm, iters, 1e-6, 1e-6, rows)
    return y


def run_hcpre(name, bs, hm, d, iters=20, four_d=False):
    ins = build_hcpre(bs, hm, d, iters, four_d)
    rows = cov.probe_rows(bs, 8)
    x, fn, scale, base, hmv, itv = ins
    y, post, comb = fn_hcpre(*ins)
    torch.npu.synchronize()
    ry, rpost, rcomb = _hcpre_core(x, fn, scale, base, hmv, itv, 1e-6, 1e-6, rows)
    # y：bf16 ULP 判据（与 test_hc_pre.py 一致）
    ry_bf = ry.reshape(y.shape[0], -1)[rows].to(torch.bfloat16).double()
    ulp = (2.0 ** -8) * max(float(ry_bf.abs().max()), 1e-12)
    ey = float((y.detach().cpu().double().reshape(y.shape[0], -1)[rows] - ry_bf).abs().max())
    ep = float((post.detach().cpu().double().reshape(bs, hm)[rows] - rpost).abs().max())
    ec = float((comb.detach().cpu().double().reshape(bs, hm, hm)[rows] - rcomb).abs().max())
    ok = ey <= ulp and ep <= POST_FLOOR and ec <= POST_FLOOR
    rec = {"name": name, "n_rows": bs, "rows": rows, "status": "PASS" if ok else "FAIL",
           "max_d": ey, "exact": float("nan"), "over1": 0,
           "boundary_bad": [], "extra": f" ulp={ulp:.2e} post={ep:.2e} comb={ec:.2e}",
           "crit": "y: bf16 1ULP | post/comb: 2e-6"}
    J3.append(rec)
    print(f"[J3 {'PASS' if ok else 'FAIL'}] {name:40s} n={bs:6d} 采样{len(rows)}行 "
          f"y_err={ey:.3e}(≤{ulp:.2e}) post={ep:.2e} comb={ec:.2e}", flush=True)
    return rec


# ======================= hc_pre_sinkhorn =======================
def build_sk(rows_lead, hm, d, iters=20, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    R = 1
    for v in rows_lead:
        R *= v
    hmix = 2 * hm + hm * hm

    def U(*shape, lo=1.0, off=0.0):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo + off

    mixes = U(*rows_lead, hmix, lo=1.0)
    rsqrt = U(R, lo=0.5, off=1.0)
    hc_scale = U(3, lo=0.5, off=1.0)
    hc_base = U(hmix, lo=0.5)
    x = (U(*rows_lead, hm, d, lo=0.3)).to(torch.bfloat16)
    return (mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps)


def fn_sk(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps):
    return hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps)


def ref_rows_sk(mixes, rsqrt, hc_scale, hc_base, x, hm, iters, eps, rows=None,
                want_scale=False):
    mx = mixes.detach().cpu().double().reshape(-1, mixes.shape[-1])[rows]
    rs = rsqrt.detach().cpu().double().view(-1, 1)[rows]
    sc = hc_scale.detach().cpu().double()
    bb = hc_base.detach().cpu().double()
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    pre = torch.sigmoid(mx[:, :hm] * rs * sc[0] + bb[:hm]) + eps
    return torch.einsum("ij,ijd->id", pre, xd)


def run_sk(name, rows_lead, hm, d, iters=20, eps=1e-6):
    ins = build_sk(rows_lead, hm, d, iters, eps)
    R = 1
    for v in rows_lead:
        R *= v
    rows = cov.probe_rows(R, 8)
    y, post, comb = fn_sk(*ins)
    torch.npu.synchronize()
    mixes, rsqrt, hc_scale, hc_base, x, hmv, itv, epsv = ins
    mx = mixes.detach().cpu().double().reshape(R, -1)[rows]
    rs = rsqrt.detach().cpu().double().view(-1, 1)[rows]
    sc = hc_scale.detach().cpu().double()
    bb = hc_base.detach().cpu().double()
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(R, hmv, d)[rows]
    ry = torch.einsum("ij,ijd->id", torch.sigmoid(
        mx[:, :hmv] * rs * sc[0] + bb[:hmv]) + epsv, xd)
    rpost = 2.0 * torch.sigmoid(mx[:, hmv:2 * hmv] * rs * sc[1] + bb[hmv:2 * hmv])
    A = (mx[:, 2 * hmv:].reshape(len(rows), hmv, hmv) * rs[:, :, None] * sc[2]
         + bb[2 * hmv:].reshape(hmv, hmv))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + epsv
    A = A / (A.sum(dim=1, keepdim=True) + epsv)
    for _ in range(itv - 1):
        A = A / (A.sum(dim=2, keepdim=True) + epsv)
        A = A / (A.sum(dim=1, keepdim=True) + epsv)
    fy = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[x.dtype]
    yc = y.detach().cpu().double().reshape(R, d)[rows]
    ey = float((yc - ry.to(x.dtype).double()).abs().max())
    ep = float((post.detach().cpu().double().reshape(R, hmv)[rows] - rpost).abs().max())
    ec = float((comb.detach().cpu().double().reshape(R, hmv, hmv)[rows] - A).abs().max())
    ok = ey <= fy and ep <= POST_FLOOR and ec <= POST_FLOOR
    rec = {"name": name, "n_rows": R, "rows": rows, "status": "PASS" if ok else "FAIL",
           "max_d": ey, "exact": float("nan"), "over1": 0, "boundary_bad": [],
           "extra": f" post={ep:.2e} comb={ec:.2e}", "crit": f"y: {fy:g} | post/comb: 2e-6"}
    J3.append(rec)
    print(f"[J3 {'PASS' if ok else 'FAIL'}] {name:40s} n={R:6d} 采样{len(rows)}行 "
          f"y_err={ey:.3e}(≤{fy:g}) post={ep:.2e} comb={ec:.2e}", flush=True)
    return rec


# ======================= 注册入口 eps 顺序核对 =======================
def check_adapter_eps():
    print("\n===== 注册入口 `npu_hc_pre` 的 (norm_eps, hc_eps) 顺序核对 =====", flush=True)
    try:
        import _C_ascend_extra as ex
        ad = ex._adapter_npu_hc_pre
        print("  已导入容器内 _C_ascend_extra._adapter_npu_hc_pre", flush=True)
    except Exception as e:
        print(f"  ⚠️ 无法导入 adapter（{type(e).__name__}: {str(e)[:120]}）；"
              f"改用本地等价实现做同判据", flush=True)

        def ad(x, hc_fn, hc_scale, hc_base, hc_mult=4, hc_sinkhorn_iters=20,
               norm_eps=1e-6, hc_eps=1e-6):
            return hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult=hc_mult,
                          hc_sinkhorn_iters=hc_sinkhorn_iters,
                          hc_eps=hc_eps, norm_eps=norm_eps)
    bs, hm, d, iters = 64, 4, 512, 20
    torch.manual_seed(13)
    hmix = 2 * hm + hm * hm
    K = hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    scale = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    base = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    # ⚠️ 必须用**两个不同**的 eps，否则顺序错了也看不出来
    NORM_EPS, HC_EPS = 1e-3, 1e-7
    a = ad(x, fn, scale, base, hm, iters, NORM_EPS, HC_EPS)          # 走注册入口（schema 顺序）
    torch.npu.synchronize()
    b = hc_pre(x, fn, scale, base, hm, iters, HC_EPS, NORM_EPS)      # 直接调实现，语义正确
    torch.npu.synchronize()
    c = hc_pre(x, fn, scale, base, hm, iters, NORM_EPS, HC_EPS)      # 假想"没做重映射"（按位置硬传）
    torch.npu.synchronize()
    same_ab = all(float((a[i].detach().cpu().double() - b[i].detach().cpu().double()).abs().max()) == 0.0
                  for i in range(3))
    diff_ac = max(float((a[i].detach().cpu().double() - c[i].detach().cpu().double()).abs().max())
                  for i in range(3))
    # 采样几行做"正确性"核对（adapter 路径 vs 正确 eps 的 fp64 参考）
    rows = cov.probe_rows(bs, 8)
    ry, rpost, rcomb = _hcpre_core(x, fn, scale, base, hm, iters, HC_EPS, NORM_EPS, rows)
    ay = a[0].detach().cpu().double().reshape(bs, d)[rows]
    ulp = (2.0 ** -8) * max(float(ry.to(torch.bfloat16).double().abs().max()), 1e-12)
    ey = float((ay - ry.to(torch.bfloat16).double()).abs().max())
    ep = float((a[1].detach().cpu().double().reshape(bs, hm)[rows] - rpost).abs().max())
    ec = float((a[2].detach().cpu().double().reshape(bs, hm, hm)[rows] - rcomb).abs().max())
    ok_bind = same_ab
    ok_sens = diff_ac > 1e-12
    ok_num = ey <= ulp and ep <= POST_FLOOR and ec <= POST_FLOOR
    print(f"  A) adapter(norm={NORM_EPS}, hc={HC_EPS}) 逐位 == 直接调实现(eps 正确) : {same_ab}", flush=True)
    print(f"  B) 灵敏度自检：与'按位置硬传(没做重映射)'的最大差 = {diff_ac:.3e} "
          f"⇒ {'能验出顺序问题' if ok_sens else '**验不出**，本结论无效'}", flush=True)
    print(f"  C) adapter 路径数值 vs fp64 参考：y_err={ey:.3e}(≤{ulp:.2e}) "
          f"post={ep:.2e} comb={ec:.2e}", flush=True)
    print(f"  ⇒ 结论: 绑定正确={ok_bind} 数值正确={ok_num} 测试有灵敏度={ok_sens}", flush=True)
    return ok_bind and ok_num and ok_sens


if __name__ == "__main__":
    print(f"J3-HCPRE | DEV={DEV} current={torch.npu.current_device()}", flush=True)
    which = os.environ.get("J3P", "all")
    if which in ("all", "pre"):
        print("\n===== J3: hc_pre（hm=4，含生产 d=7168 与非 2 幂）=====", flush=True)
        for bs, hm, d, it in [(64, 4, 512, 20), (64, 4, 100, 20), (33, 4, 300, 20),
                              (4096, 4, 512, 20), (4096, 4, 7168, 20), (4096, 4, 512, 1),
                              (32, 4, 512, 3), (16, 8, 128, 20), (16, 2, 256, 20)]:
            try:
                run_hcpre(f"hc_pre bs={bs} hm={hm} d={d} iters={it}", bs, hm, d, it)
            except Exception:
                print(f"[J3 EXC ] hc_pre bs={bs} hm={hm} d={d}\n{traceback.format_exc()[-400:]}",
                      flush=True)
    if which in ("all", "sk"):
        print("\n===== J3: hc_pre_sinkhorn =====", flush=True)
        for lead, hm, d in [((64,), 4, 512), ((64,), 4, 100), ((2, 32), 4, 512),
                            ((4096,), 4, 512), ((4096,), 4, 7168), ((33,), 4, 100),
                            ((8,), 8, 128), ((16,), 3, 256)]:
            try:
                run_sk(f"hc_pre_sinkhorn lead={lead} hm={hm} d={d}", lead, hm, d)
            except Exception:
                print(f"[J3 EXC ] hc_pre_sinkhorn {lead} hm={hm} d={d}\n"
                      f"{traceback.format_exc()[-400:]}", flush=True)
    if which in ("all", "adapter"):
        try:
            check_adapter_eps()
        except Exception:
            print("ADAPTER CHECK FAILED:\n" + traceback.format_exc()[-800:], flush=True)

    print("\n===== J3-HCPRE 汇总 =====", flush=True)
    nbad = sum(1 for r in J3 if r["status"] != "PASS")
    print(f"J3HCPRE TOTAL={len(J3)} NOT_PASS={nbad}", flush=True)
    print("J3HCPREDONE", flush=True)
