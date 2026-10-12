# -*- coding: utf-8 -*-
"""_probe_cov_j3_hc.py —— J3 数值比对：hc_post / hc_pre_inv_rms / rms_norm_dynamic_quant

背景：这三个算子 J2 只证明过"能编译+执行"，**数值从未在生产宽度验证**，
且 `D=7168`（生产 hidden）与 `D=1001`（非 2 幂）**从没被测过**。

## ⚠️ 调用约定（与 `_coverage_sweep.py::run_j3` 严格一致）
    ins = build_inputs()        # **参数元组**
    fn(*ins)                    # 展开
    ref_rows(*ins, rows=..., want_scale=...)   # 同样展开
⇒ `fn` 与 `ref_rows` 必须收**同样多的必需位置参数**。

参考实现全部复用各自 `test_*.py` 里的 fp64 语义（不新造第二份口径），
只加"按行采样"的包装，把成本从 O(全量) 降到 O(采样行)。

跑法：DSH_NPU=npu:13 python3 -W ignore triton_ops/_probe_cov_j3_hc.py
"""
import importlib.util
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:13").split(":")[-1])
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

from triton_ops.hc_post import hc_post  # noqa: E402
from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms  # noqa: E402
from triton_ops.rms_norm_dynamic_quant import rms_norm_dynamic_quant  # noqa: E402


# ============================ hc_post ============================
def build_hc_post(B, H, D, dt=torch.bfloat16, scale=0.4, seed=0):
    """返回 (x, residual, post, comb)。"""
    torch.manual_seed(seed)
    x = (torch.rand(B, D, device=DEV) * 2 - 1).mul(scale).to(dt)
    residual = (torch.rand(B, H, D, device=DEV) * 2 - 1).mul(scale).to(dt)
    post = ((torch.rand(B, H, device=DEV) * 2 - 1).mul(scale)).to(dt)
    comb = ((torch.rand(B, H, H, device=DEV) * 2 - 1).mul(scale)).to(dt)
    return (x, residual, post, comb)


def fn_hc_post(x, residual, post, comb):
    return hc_post(x, residual, post, comb)


def ref_rows_hc_post(x, residual, post, comb, rows=None, want_scale=False):
    """与 test_hc_post.py::ref 同语义：y[b,h,d] = post[b,h]*x[b,d] + Σ_k comb[b,k,h]*residual[b,k,d]。"""
    xd = x.detach().cpu().double()[rows]
    rd = residual.detach().cpu().double()[rows]
    pd = post.detach().cpu().double()[rows]
    cd = comb.detach().cpu().double()[rows]
    return pd[:, :, None] * xd[:, None, :] + torch.einsum("bkh,bkd->bhd", cd, rd)


# ============================ hc_pre_inv_rms ============================
def build_inv(shape, dt=torch.bfloat16, scale=0.5, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(*shape, device=DEV) * scale).to(dt)
    return (x, eps)


def fn_inv(x, eps):
    return hc_pre_inv_rms(x, eps)


def ref_rows_inv(x, eps, rows=None, want_scale=False):
    """参考语义（test_hc_pre_inv_rms.py）：把 (..., hc, D) 展平为 (-1, hc, D)，
    对每个行首维输出 1/sqrt(mean(x^2 over hc,D)+eps)，shape (-1,1)。"""
    xd = x.detach().cpu().double()
    lead = xd.shape[:-2]
    hc, D = xd.shape[-2], xd.shape[-1]
    flat = xd.reshape(-1, hc, D)
    m = flat.pow(2).mean(dim=(1, 2))
    return (1.0 / torch.sqrt(m + eps)).reshape(-1, 1)[rows]


# ============================ rms_norm_dynamic_quant ============================
def build_rmsdq(R, D, dt=torch.bfloat16, scale=0.5, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(R, D, device=DEV) * scale).to(dt)
    g = torch.rand(D, device=DEV, dtype=torch.float32) + 0.5
    return (x, g)


def fn_rmsdq(x, g):
    return rms_norm_dynamic_quant(x, g)


def ref_rows_rmsdq(x, g, rows=None, want_scale=False):
    """动态量化语义：y_i = RINT( (x_i/rms) * g_i / scale )，scale = max|row|/127。"""
    xd = x.detach().cpu().double()[rows]
    gd = g.detach().cpu().double()
    rms = torch.sqrt(xd.pow(2).mean(dim=1, keepdim=True) + 1e-6)
    t = (xd / rms) * gd[None, :]
    mx = t.abs().amax(dim=1)
    sc = mx / 127.0
    if want_scale:
        return sc
    return torch.round(t / sc[:, None])


def run(name, fn, build, ref, n_rows, **kw):
    return cov.run_j3(name, fn, build, ref, n_rows=n_rows, results=J3, **kw)


if __name__ == "__main__":
    print(f"J3-HC | DEV={DEV} current={torch.npu.current_device()}", flush=True)
    which = os.environ.get("J3H", "all")

    if which in ("all", "hcp"):
        print("\n===== J3: hc_post（B∈{64,4096} × D∈{512,1001,2048,7168}）=====", flush=True)
        # hc_post 输出 bf16 ⇒ 用 bf16 量化地板 4e-3（与 test_hc_post.py 的 FLOOR 一致），
        # **不能**要求逐位一致（输入 bf16 + fp64 参考 ⇒ exact 率天然只有 ~0.3%）。
        for B, H, D in [(64, 4, 512), (64, 4, 1001), (64, 4, 7168), (4096, 4, 512),
                        (4096, 4, 1001), (4096, 4, 2048), (4096, 4, 7168), (1, 4, 7168),
                        (33, 3, 100)]:
            run(f"hc_post B={B} H={H} D={D}", fn_hc_post,
                lambda B=B, H=H, D=D: build_hc_post(B, H, D), ref_rows_hc_post,
                n_rows=B, out_index=0, n_sample=8)

    if which in ("all", "inv"):
        print("\n===== J3: hc_pre_inv_rms（fp32 输出）=====", flush=True)
        # 输出 fp32 ⇒ 地板 2e-6（与 test_hc_pre_inv_rms.py 的 FLOOR 一致）
        for shape in [(4096, 4, 512), (4096, 4, 1001), (4096, 4, 7168), (33, 4, 100),
                      (1, 4, 7168), (2048, 4, 2048)]:
            run(f"hc_pre_inv_rms shape={shape}", fn_inv,
                lambda s=shape: build_inv(s), ref_rows_inv,
                n_rows=shape[0], out_index=0, n_sample=8)

    if which in ("all", "rmsdq"):
        print("\n===== J3: rms_norm_dynamic_quant（R≤4096 × D≤7168，含非 2 幂）=====", flush=True)
        for R, D in [(4096, 512), (4096, 1001), (4096, 7168), (33, 1000), (1, 7168), (2048, 2048)]:
            run(f"rms_norm_dynamic_quant R={R} D={D}", fn_rmsdq,
                lambda R=R, D=D: build_rmsdq(R, D), ref_rows_rmsdq,
                n_rows=R, out_index=0, tol=1, exact_min=0.999, n_sample=8)

    print("\n===== J3-HC 汇总 =====", flush=True)
    for r in J3:
        if r["status"] in ("PASS", "FAIL"):
            print(f"  {r['status']:4s} {r['name']:40s} max|Δ|={r.get('max_d')} "
                  f"exact={r.get('exact')} >1LSB={r.get('over1')} 边界异常={r.get('boundary_bad')}",
                  flush=True)
        else:
            print(f"  {r['status']:8s} {r['name']:40s} {r.get('detail', '')}", flush=True)
    nbad = sum(1 for r in J3 if r["status"] != "PASS")
    print(f"J3HC TOTAL={len(J3)} NOT_PASS={nbad}", flush=True)
    print("J3HCDONE", flush=True)
