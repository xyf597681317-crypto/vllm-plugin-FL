# -*- coding: utf-8 -*-
"""task-6 探针：两个融合 GMM 在**生产宽度**下的编译 + 逐元素正确性 + UB 余量。

背景：生产 `I = moe_intermediate_size ∈ {512,1024,2048,2304}`、`K = hidden ∈ {2048,7168}`，
单块写法必然 `ub overflow`（限额 1,572,864 bits）。改法是 N 轴 + K 轴运行期分块。

本探针：
  A) v2 `grouped_matmul_swiglu_quant_v2`：K ∈ {2048,7168} × I ∈ {512,1024,2048,2304}，
     全部要求编译+执行通过，并与 **fp64 CPU 参考逐元素**比对（报 max|Δ|、>1LSB 个数、完全一致率）。
  B) v1 `grouped_matmul_swiglu_quant_weight_nz_tensor_list`：同形状集合同判据。
  C) UB 余量扫描：同一生产形状下强制不同 (BLK_M, BLK_K, BLK_N)，打印「编译通过 / ub overflow N bits」，
     用来证明当前选定的 tile 在限额内、以及超限时确实会被 BiShengIR 拦下。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_gmm_prod.py
"""
import os
import re
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
from triton_ops.grouped_matmul_swiglu_quant_v2 import (  # noqa: E402
    _gmm_v2_kernel, grouped_matmul_swiglu_quant_v2,
    ub_budget_bits, UB_LIMIT_BITS, BLK_K_CAP, BLK_N_CAP)
from triton_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list import (  # noqa: E402
    grouped_matmul_swiglu_quant_weight_nz_tensor_list)
from triton_ops.common import next_pow2  # noqa: E402


# ---------------------------------------------------------------- fp64 参考
def ref_forward(x, W_i8, ws, xs, ends, I, limit=0.0, chunk=256):
    """逐元素 fp64 参考（按 I 分块做 matmul，避免整张 W 转 fp64 占爆内存）。

    与 test_*.py 的参考同式：
        C = (x*x_scale) @ (W*ws) ; gate,up = C[:, :I], C[:, I:]
        sw = silu(gate)*up ; mx = max|sw| ; y = rint_fp16(sw*127/mx) ; y_scale = mx/127
    """
    xd = x.detach().cpu().double()
    Wc = W_i8.detach().cpu()
    wsd = ws.detach().cpu().double()
    xsd = xs.detach().cpu().double()
    M, K = xd.shape
    G, _, N = Wc.shape
    y = torch.zeros(M, I, dtype=torch.int64)
    ys = torch.zeros(M, dtype=torch.float64)
    prev = 0
    for g in range(G):
        s, e = prev, int(ends[g])
        prev = e
        if e <= s:
            continue
        xr = xd[s:e] * xsd[s:e, None]
        sw = torch.zeros(e - s, I, dtype=torch.float64)
        for c0 in range(0, I, chunk):
            c1 = min(I, c0 + chunk)
            wg = Wc[g][:, c0:c1].double() * wsd[g][None, c0:c1]
            wu = Wc[g][:, I + c0:I + c1].double() * wsd[g][None, I + c0:I + c1]
            cg = xr @ wg
            cu = xr @ wu
            if limit > 0:
                cg = cg.clamp(max=limit)
            sw[:, c0:c1] = (cg * torch.sigmoid(cg)) * cu
        mx = sw.abs().amax(dim=1)
        q16 = (sw * (127.0 / mx)[:, None]).float().half().float().double()
        y[s:e] = torch.where(q16 >= 0, torch.floor(q16 + 0.5),
                             torch.ceil(q16 - 0.5)).clamp(-128, 127).to(torch.int64)
        ys[s:e] = mx / 127.0
    return y, ys


def make_inputs(M, K, I, G, seed=101):
    N = 2 * I
    torch.manual_seed(seed)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    per = max(1, M // G)
    cnt = [min(M, per * (i + 1)) - min(M, per * i) for i in range(G)]
    cnt[-1] = M - sum(cnt[:-1])
    ends = torch.cumsum(torch.tensor(cnt, dtype=torch.int64, device=DEV), dim=0)
    return x, w, ws, xs, ends


def report(tag, y, ys, ry, rys, t_ms=None):
    d = (y.detach().cpu().long() - ry).abs()
    n_gt1 = int((d > 1).sum())
    exact = float((d == 0).double().mean())
    es = float((ys.detach().cpu().double() - rys).abs().max())
    fs = 2e-6 * max(1.0, float(rys.abs().max()))
    ok = (n_gt1 == 0) and (es <= fs)
    print(f"  {tag:44s} max|Δ|={int(d.max())} >1LSB={n_gt1} exact={exact * 100:.4f}% "
          f"scale_err={es:.2e}(<={fs:.1e})  {'PASS' if ok else 'FAIL'}"
          + (f"  [{t_ms:.0f} ms]" if t_ms else ""), flush=True)
    return ok


# ---------------------------------------------------------------- A/B 生产形状
def case_v2(M, K, I, G, smooth=None):
    x, w, ws, xs, ends = make_inputs(M, K, I, G)
    blk_k = min(max(next_pow2(K), 16), BLK_K_CAP)
    blk_n = min(max(next_pow2(I), 16), BLK_N_CAP)
    need = ub_budget_bits(16, blk_k, blk_n)
    tag = f"v2 K={K} I={I} G={G} tile=({blk_k},{blk_n}) ub≈{need}"
    try:
        t0 = torch.npu.Event(enable_timing=True)
        t1 = torch.npu.Event(enable_timing=True)
        t0.record()
        y, ys = grouped_matmul_swiglu_quant_v2(
            x, weight=[w[i] for i in range(G)], weight_scale=[ws[i] for i in range(G)],
            x_scale=xs, group_list=ends, smooth_scale=smooth, swiglu_limit=0.0)
        torch.npu.synchronize()
        t1.record()
        ms = t0.elapsed_time(t1)
    except Exception as ex:  # noqa: BLE001
        msg = " ".join(str(ex).split())
        m = re.search(r"ub overflow, requires (\d+) bits while (\d+) bits available", msg)
        print(f"  [{tag}] EXC {type(ex).__name__}: "
              + (f"ub overflow requires={m.group(1)}" if m else msg[:160]), flush=True)
        return False
    ry, rys = ref_forward(x, w, ws, xs, ends.cpu().tolist(), I)
    return report(tag, y, ys, ry, rys, ms)


def case_v1(M, K, I, G):
    x, w, ws, xs, ends = make_inputs(M, K, I, G, seed=202)
    blk_k = min(max(next_pow2(K), 16), BLK_K_CAP)
    blk_n = min(max(next_pow2(I), 16), BLK_N_CAP)
    need = ub_budget_bits(16, blk_k, blk_n)
    tag = f"v1 K={K} I={I} G={G} tile=({blk_k},{blk_n}) ub≈{need}"
    try:
        y, ys, _off = grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x, weight=[w[i] for i in range(G)], weight_scale=[ws[i] for i in range(G)],
            x_scale=xs, group_list=ends)
        torch.npu.synchronize()
    except Exception as ex:  # noqa: BLE001
        msg = " ".join(str(ex).split())
        m = re.search(r"ub overflow, requires (\d+) bits while (\d+) bits available", msg)
        print(f"  [{tag}] EXC {type(ex).__name__}: "
              + (f"ub overflow requires={m.group(1)}" if m else msg[:160]), flush=True)
        return False
    ry, rys = ref_forward(x, w, ws, xs, ends.cpu().tolist(), I)
    return report(tag, y, ys, ry, rys)


# ---------------------------------------------------------------- C UB 余量扫描
def ub_case(M, K, I, bm, bk, bn):
    """直接调 kernel，强制指定 tile，看编译是否通过、超限时报多少 bits。"""
    G = 1
    N = 2 * I
    torch.manual_seed(303)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    st = torch.zeros(G, dtype=torch.int32, device=DEV)
    en = torch.full((G,), M, dtype=torch.int32, device=DEV)
    y = torch.empty((M, I), dtype=torch.int8, device=DEV)
    ys = torch.empty((M,), dtype=torch.float32, device=DEV)
    pred = ub_budget_bits(bm, bk, bn)
    tag = f"K={K} I={I} tile(BM={bm},BK={bk},BN={bn}) 预测={pred}"
    try:
        _gmm_v2_kernel[(G, triton.cdiv(M, bm))](
            x, w, ws, xs, x, st, en, y, ys, K, N, I,
            BLK_M=bm, BLK_K=bk, BLK_N=bn,
            HAS_SMOOTH_COL=False, HAS_SMOOTH_ROW=False, LIMIT_ON=False, SWIGLU_LIMIT=0.0)
        torch.npu.synchronize()
        print(f"  [OK   ] {tag}  (预测 {pred} ≤ 限额 {UB_LIMIT_BITS} "
              f"{'✓' if pred <= UB_LIMIT_BITS else '✗ 预测与实际不符!'})", flush=True)
    except Exception as ex:  # noqa: BLE001
        msg = " ".join(str(ex).split())
        m = re.search(r"ub overflow, requires (\d+) bits while (\d+) bits available", msg)
        if m:
            print(f"  [UB!  ] {tag}  实测 requires={m.group(1)} "
                  f"(预测 {pred}, {'一致' if abs(int(m.group(1)) - pred) < 0.1 * pred else '偏差大'})",
                  flush=True)
        else:
            print(f"  [FAIL ] {tag}  {type(ex).__name__}: {msg[:140]}", flush=True)


def main():
    print(f"[gmm-prod] dev={DEV} triton={triton.__version__}", flush=True)
    print(f"[gmm-prod] UB 限额={UB_LIMIT_BITS} bits  BLK_N_CAP={BLK_N_CAP} BLK_K_CAP={BLK_K_CAP}",
          flush=True)
    print("== A) v2 生产宽度：K ∈ {2048,7168} × I ∈ {512,1024,2048,2304} ==", flush=True)
    okA = True
    for K in (2048, 7168):
        for I in (512, 1024, 2048, 2304):
            okA &= case_v2(32, K, I, 2)
    print("== B) v1 生产宽度（同形状集合） ==", flush=True)
    okB = True
    for K in (2048, 7168):
        for I in (512, 1024, 2048, 2304):
            okB &= case_v1(32, K, I, 2)
    print("== C) UB 余量扫描（K=2048, I=512） ==", flush=True)
    for bm, bk, bn in ((16, 128, 256), (16, 128, 128), (16, 64, 256),
                       (16, 128, 512), (16, 512, 256), (16, 2048, 512)):
        ub_case(32, 2048, 512, bm, bk, bn)
    print(f"[gmm-prod] A(PASS)={okA} B(PASS)={okB}", flush=True)
    print("[gmm-prod] done", flush=True)


if __name__ == "__main__":
    main()
