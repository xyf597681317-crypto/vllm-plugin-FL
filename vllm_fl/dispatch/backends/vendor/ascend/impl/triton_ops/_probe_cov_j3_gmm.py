# -*- coding: utf-8 -*-
"""_probe_cov_j3_gmm.py —— J3 数值比对：grouped_matmul_swiglu_quant_v2 / _weight_nz_tensor_list

背景：这两个算子在**生产宽度**下 J2（能编译+执行）9/9、3/3 OK，但**数值从未验证过**。
本探针用 J3（`_coverage_sweep.py::probe_rows/run_j3`）：**只对采样行跑主机 fp64 独立复算**，
逐元素判 `max|Δ|` / `>1LSB` / **完全一致率** / **边界行**。

参考实现**复用**测试文件里已有的 `ref()`（fp64，同语义），只加"按行采样"的包装。

判据（与单测同口径）：`y_diff == 0`（int8 逐元素完全一致）且 `y_scale` 相对误差 ≤ 2e-6。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_cov_j3_gmm.py
"""
import importlib.util
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

_spec = importlib.util.spec_from_file_location(
    "cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)
J3 = []
run_j3 = cov.run_j3
probe_rows = cov.probe_rows

from triton_ops.grouped_matmul_swiglu_quant_v2 import grouped_matmul_swiglu_quant_v2  # noqa: E402
from triton_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list import (  # noqa: E402
    grouped_matmul_swiglu_quant_weight_nz_tensor_list as gmm_v1)

FLOOR_S = 2e-6


def _rint_fp16(x):
    q16 = x.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref_rows_v2(x, w, ws, xs, gl, smooth, glist_type, rows=None, want_scale=False):
    """按行采样版 fp64 参考（与 test_grouped_matmul_swiglu_quant_v2.py::ref 同语义）。

    签名与 `fn_v2` 一致（`*ins` 展开），`rows` 为采样行，`want_scale` 由框架传入。
    """
    ends_l = [int(e) for e in gl.detach().cpu().tolist()]
    G = len(w)
    I = w[0].shape[1] // 2
    xd = x.detach().cpu().double()
    XS = xs.detach().cpu().double()
    smt = smooth.detach().cpu().double() if smooth is not None else None
    y = torch.zeros(len(rows), I, dtype=torch.int64)
    ys = torch.zeros(len(rows), dtype=torch.float64)
    prev = 0
    for g in range(G):
        s, e = prev, ends_l[g]
        prev = e
        sel = [i for i, r in enumerate(rows) if s <= r < e]
        if not sel:
            continue
        rr = [rows[i] for i in sel]
        W = w[g].detach().cpu().double()
        WS = ws[g].detach().cpu().double()
        C = (xd[rr] * XS[rr, None]) @ (W * WS[None, :])
        gate, up = C[:, :I], C[:, I:]
        sw = (gate * torch.sigmoid(gate)) * up
        if smt is not None:
            sw = sw * (smt[g][None, :] if smt.dim() == 2 else smt[g])
        mx = sw.abs().amax(dim=1)
        y[sel] = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64)
        ys[sel] = mx / 127.0
    return ys if want_scale else y


def make_v2_inputs(M, K, I, G=2, smooth_kind="col", glist_type=0, seed=0):
    torch.manual_seed(seed)
    N = 2 * I
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = [torch.randint(-40, 41, (K, N), dtype=torch.int8, device=DEV) for _ in range(G)]
    ws = [torch.rand(N, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(G)]
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    per = max(1, M // G)
    cnt = [min(M, per * (i + 1)) - min(M, per * i) for i in range(G)]
    cnt[-1] = M - sum(cnt[:-1])
    cnts = torch.tensor(cnt, dtype=torch.int64, device=DEV)
    gl = cnts if glist_type == 1 else torch.cumsum(cnts, dim=0)
    smooth = None
    if smooth_kind == "col":
        smooth = torch.rand(G, I, dtype=torch.float32, device=DEV) * 0.8 + 0.6
    elif smooth_kind == "row":
        smooth = torch.rand(G, dtype=torch.float32, device=DEV) * 0.8 + 0.6
    return (x, w, ws, xs, gl, smooth, glist_type)


def fn_v2(x, w, ws, xs, gl, smooth, glist_type):
    """位置参数与 `ref_rows_v2` **逐个对应**（J3 对两者都用 `*ins` 展开）。"""
    return grouped_matmul_swiglu_quant_v2(
        x, weight=w, weight_scale=ws, x_scale=xs, group_list=gl, smooth_scale=smooth,
        weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0,
        quant_mode=0, quant_dtype=0, transpose_weight=False, group_list_type=glist_type,
        tuning_config=[0], swiglu_limit=0.0)


def run_v2(name, M, K, I, G=2, smooth_kind="col", glist_type=0, n_sample=8):
    """调用约定（与 `run_j3` 严格一致，改前请先读这三行）：
        build_inputs() -> **参数元组** ins
        run_j3 执行：`fn(*ins)` 且 `ref_rows(*ins, rows=..., want_scale=...)`
    ⇒ `fn` 与 `ref_rows` 必须收**同样多的位置参数**；任何一侧收"整包元组"都会 TypeError。
    """
    ins = make_v2_inputs(M, K, I, G, smooth_kind, glist_type)

    def call(*a):
        return fn_v2(*a)
    return run_j3(name, call, lambda: ins, ref_rows_v2,
                  n_rows=M, out_index=0, tol=0, exact_min=1.0, n_sample=n_sample,
                  scale_index=1, scale_tol=FLOOR_S, results=J3)


# ------------------------------------------------------------------ v1
def ref_rows_v1(x, w, ws, xs, gl, rows=None, want_scale=False):
    """签名与 `call_v1` 一致（`*ins` 展开）。"""
    ends_l = [int(e) for e in gl.detach().cpu().tolist()]
    G = len(w)
    I = w[0].shape[1] // 2
    xd = x.detach().cpu().double()
    XS = xs.detach().cpu().double()
    y = torch.zeros(len(rows), I, dtype=torch.int64)
    ys = torch.zeros(len(rows), dtype=torch.float64)
    prev = 0
    for g in range(G):
        s, e = prev, ends_l[g]
        prev = e
        sel = [i for i, r in enumerate(rows) if s <= r < e]
        if not sel:
            continue
        rr = [rows[i] for i in sel]
        W = w[g].detach().cpu().double()
        WS = ws[g].detach().cpu().double()
        C = (xd[rr] * XS[rr, None]) @ (W * WS[None, :])
        gate, up = C[:, :I], C[:, I:]
        sw = (gate * torch.sigmoid(gate)) * up
        mx = sw.abs().amax(dim=1)
        y[sel] = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64)
        ys[sel] = mx / 127.0
    return ys if want_scale else y


def make_v1_inputs(M, K, I, G=2, seed=0):
    torch.manual_seed(seed)
    N = 2 * I
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = [torch.randint(-40, 41, (K, N), dtype=torch.int8, device=DEV) for _ in range(G)]
    ws = [torch.rand(N, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(G)]
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    per = max(1, M // G)
    cnt = [min(M, per * (i + 1)) - min(M, per * i) for i in range(G)]
    cnt[-1] = M - sum(cnt[:-1])
    gl = torch.cumsum(torch.tensor(cnt, dtype=torch.int64, device=DEV), dim=0)
    return (x, w, ws, xs, gl)


def run_v1(name, M, K, I, G=2, n_sample=8):
    ins = make_v1_inputs(M, K, I, G)      # ins = (x, w, ws, xs, gl)

    def call(*a):
        x, w, ws, xs, gl = a
        y, ys, _off = gmm_v1(x, weight=w, weight_scale=ws, x_scale=xs, group_list=gl)
        return y, ys
    return run_j3(name, call, lambda: ins, ref_rows_v1,
                  n_rows=M, out_index=0, tol=0, exact_min=1.0, n_sample=n_sample,
                  scale_index=1, scale_tol=FLOOR_S, results=J3)


if __name__ == "__main__":
    print(f"J3-GMM | DEV={DEV} current={torch.npu.current_device()}", flush=True)
    which = os.environ.get("J3G", "all")
    if which in ("all", "v2"):
        print("\n===== J3: grouped_matmul_swiglu_quant_v2（生产宽度首次数值验证）=====", flush=True)
        for M, K, I, sk in [
            (64, 2048, 512, "col"), (64, 2048, 1024, "col"), (64, 2048, 1408, "col"),
            (64, 2048, 2048, "col"), (64, 2048, 2304, "col"), (64, 2048, 512, None),
            (64, 2048, 512, "row"), (1, 7168, 2048, "col"), (32, 7168, 2048, "col"),
            (4096, 2048, 512, "col"), (33, 2048, 1001, "col"),
        ]:
            tag = f"v2 M={M} K={K} I={I} G=2 smooth={sk}"
            try:
                run_v2(tag, M, K, I, 2, sk)
            except Exception:
                print(f"[J3 EXC ] {tag}\n{traceback.format_exc()[-500:]}", flush=True)
        # group_list_type=1（计数语义）
        run_v2("v2 M=64 K=2048 I=512 glist_type=1", 64, 2048, 512, 2, "col", 1)
    if which in ("all", "v1"):
        print("\n===== J3: grouped_matmul_swiglu_quant_weight_nz_tensor_list (v1) =====", flush=True)
        for M, K, I in [(64, 2048, 512), (64, 2048, 2048), (32, 7168, 2048), (4096, 2048, 512)]:
            tag = f"v1 M={M} K={K} I={I} G=2"
            try:
                run_v1(tag, M, K, I, 2)
            except Exception:
                print(f"[J3 EXC ] {tag}\n{traceback.format_exc()[-500:]}", flush=True)

    print("\n===== J3-GMM 汇总 =====", flush=True)
    for r in J3:
        if r["status"] in ("PASS", "FAIL"):
            print(f"  {r['status']:4s} {r['name']:44s} max|Δ|={r.get('max_d')} "
                  f"exact={r.get('exact')} >1LSB={r.get('over1')} 边界异常={r.get('boundary_bad')}"
                  f"{r.get('extra', '')}", flush=True)
        else:
            print(f"  {r['status']:8s} {r['name']:44s} {r.get('detail', '')}", flush=True)
    nbad = sum(1 for r in J3 if r["status"] != "PASS")
    print(f"J3GMM TOTAL={len(J3)} NOT_PASS={nbad}", flush=True)
    print("J3GMMDONE", flush=True)
