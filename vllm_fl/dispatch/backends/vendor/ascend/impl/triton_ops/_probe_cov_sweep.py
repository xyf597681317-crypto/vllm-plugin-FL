# -*- coding: utf-8 -*-
"""_probe_cov_sweep.py —— 生产形状覆盖扫描（J2：编译/执行/健全性判据）

判据（不依赖参考实现，因此对每个算子都适用）：
    OK              编译+执行通过、输出 shape 与期望一致、无 nan/inf
    UB_OVERFLOW     BiShengIR `ub overflow`（生产宽度顶爆 UB）
    COMPILER_ASSERT FLIR/Triton 断言崩溃（如 InterleaveOptimization.cpp:635）
    OTHER_EXC / COMPILE_ERROR / NOT_IMPLEMENTED / OOM / DEVICE_TRANSIENT
    NAN / INF       输出含 nan/inf ⇒ 强烈怀疑丢写/除零
    SHAPE_MISMATCH  输出 shape 与期望不符

⚠️ 本探针**只调用算子，不修改任何算子文件**。
设备：DSH_NPU=npu:2  跑法：
    COV_ONLY=all python3 -W ignore triton_ops/_probe_cov_sweep.py
"""
import os
import sys
import time
import traceback
from collections import Counter

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

RESULTS = []


def classify(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "ub overflow" in low:
        return "UB_OVERFLOW"
    if "out of memory" in low or "507899" in s:
        return "OOM"
    if "assertion" in low or "unreachable" in low or "interleave" in low:
        return "COMPILER_ASSERT"
    if "notimplemented" in low:
        return "NOT_IMPLEMENTED"
    if "507035" in s or "507057" in s:
        return "DEVICE_TRANSIENT"
    if "compile" in low or "bishengir" in low or "flir" in low:
        return "COMPILE_ERROR"
    return "OTHER_EXC"


def case(op, name, fn, expect_shape=None):
    t0 = time.time()
    try:
        out = fn()
        torch.npu.synchronize()
    except Exception as e:
        r = {"op": op, "case": name, "status": classify(e),
             "detail": f"{type(e).__name__}: {str(e)[:200]}", "ms": (time.time() - t0) * 1e3}
        RESULTS.append(r)
        print(f"[{r['status']:16s}] {op:26s} {name:34s} {r['detail'][:100]}", flush=True)
        return r
    ms = (time.time() - t0) * 1e3
    t = out[0] if isinstance(out, (tuple, list)) else out
    st, detail = "OK", ""
    if expect_shape is not None and tuple(t.shape) != tuple(expect_shape):
        st, detail = "SHAPE_MISMATCH", f"got {tuple(t.shape)} want {tuple(expect_shape)}"
    elif t.is_floating_point():
        tf = t.detach().float()
        nn = int(torch.isnan(tf).sum().item())
        ni = int(torch.isinf(tf).sum().item())
        if nn or ni:
            st, detail = ("NAN" if nn else "INF"), f"nan={nn} inf={ni}/{tf.numel()}"
    r = {"op": op, "case": name, "status": st, "detail": detail, "ms": ms,
         "shape": list(t.shape), "dtype": str(t.dtype)}
    RESULTS.append(r)
    print(f"[{st:16s}] {op:26s} {name:34s} {ms:8.1f}ms shape={tuple(t.shape)} {detail}", flush=True)
    return r


# ============================== GMM v2 / v1 ==============================
def sweep_gmm():
    from triton_ops.grouped_matmul_swiglu_quant_v2 import grouped_matmul_swiglu_quant_v2 as gv2
    from triton_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list import (
        grouped_matmul_swiglu_quant_weight_nz_tensor_list as gv1)

    def args_v2(M, K, G, N, smooth=True):
        torch.manual_seed(0)
        x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
        w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
        ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
        xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
        cnt = [M // G] * G
        cnt[-1] = M - sum(cnt[:-1])
        gl = torch.cumsum(torch.tensor(cnt, dtype=torch.int64, device=DEV), dim=0)
        smt = torch.rand(G, N // 2, dtype=torch.float32, device=DEV) * 0.8 + 0.6 if smooth else None
        return dict(weight=[w[i] for i in range(G)], weight_scale=[ws[i] for i in range(G)],
                    x_scale=xs, group_list=gl, smooth_scale=smt, weight_assist_matrix=None,
                    bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                    transpose_weight=False, group_list_type=0, tuning_config=[0],
                    swiglu_limit=0.0)

    def args_v1(M, K, G, N):
        torch.manual_seed(0)
        x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
        w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
        ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
        xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
        cnt = [M // G] * G
        cnt[-1] = M - sum(cnt[:-1])
        gl = torch.cumsum(torch.tensor(cnt, dtype=torch.int64, device=DEV), dim=0)
        return dict(weight=[w[i] for i in range(G)], weight_scale=[ws[i] for i in range(G)],
                    x_scale=xs, group_list=gl, swiglu_limit=0.0)

    # 生产轴：K ∈ {2048,7168}，I ∈ {512,1024,1408,1536,2048,2304} ⇒ N = 2I
    for M, K, G, I in [(64, 2048, 2, 512), (64, 2048, 2, 1024), (64, 2048, 2, 1408),
                       (64, 2048, 2, 1536), (64, 2048, 2, 2048), (64, 2048, 2, 2304),
                       (32, 7168, 2, 2048), (1, 7168, 2, 2048), (4096, 2048, 2, 512)]:
        N = 2 * I
        case("gmm_v2", f"M={M} K={K} G={G} I={I}", lambda M=M, K=K, G=G, N=N: gv2(
            torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV), **args_v2(M, K, G, N)),
            expect_shape=(M, I))
    for M, K, G, I in [(64, 2048, 2, 512), (32, 7168, 2, 2048), (64, 2048, 2, 2048)]:
        N = 2 * I
        case("gmm_v1", f"M={M} K={K} G={G} I={I}", lambda M=M, K=K, G=G, N=N: gv1(
            torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV), **args_v1(M, K, G, N)),
            expect_shape=(M, I))


# ============================== hc ==============================
def sweep_hc():
    from triton_ops.hc_post import hc_post
    from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms

    def post(B, H, D, dt=torch.bfloat16):
        torch.manual_seed(0)
        x = (torch.rand(B, D, device=DEV) * 2 - 1).mul(0.4).to(dt)
        res = (torch.rand(B, H, D, device=DEV) * 2 - 1).mul(0.4).to(dt)
        p = ((torch.rand(B, H, device=DEV) * 2 - 1).mul(0.4)).to(dt)
        comb = ((torch.rand(B, H, H, device=DEV) * 2 - 1).mul(0.4)).to(dt)
        return (x, res, p, comb)

    for B, H, D in [(4096, 4, 512), (4096, 4, 7168), (2048, 4, 7168), (4096, 4, 1001), (4096, 4, 2048)]:
        case("hc_post", f"B={B} H={H} D={D}", lambda B=B, H=H, D=D: hc_post(*post(B, H, D)),
             expect_shape=(B, H, D))
    for shape in [(4096, 4, 512), (4096, 4, 7168), (2048, 4, 7168), (33, 4, 100)]:
        case("hc_pre_inv_rms", f"shape={shape}",
             lambda s=shape: hc_pre_inv_rms((torch.randn(*s, device=DEV) * 0.5).to(torch.bfloat16), 1e-6),
             expect_shape=shape)


# ============================== rms_norm_dynamic_quant ==============================
def sweep_rmsdq():
    from triton_ops.rms_norm_dynamic_quant import rms_norm_dynamic_quant

    def args(R, D, dt=torch.bfloat16):
        torch.manual_seed(0)
        x = (torch.randn(R, D, device=DEV) * 0.5).to(dt)
        g = torch.rand(D, device=DEV, dtype=torch.float32) + 0.5
        return (x, g)

    for R, D in [(4096, 7168), (4096, 512), (33, 1000), (1, 7168), (4096, 4096), (2048, 2048)]:
        case("rms_norm_dynamic_quant", f"R={R} D={D}",
             lambda R=R, D=D: rms_norm_dynamic_quant(*args(R, D)))


# ============================== dequant_swiglu_quant ==============================
def sweep_dsq():
    from triton_ops.dequant_swiglu_quant import dequant_swiglu_quant

    def go(M, H, G=None, swiglu_mode=0):
        torch.manual_seed(9)
        x = torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8
        if G:
            ws = torch.rand(G, 2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
            gi = torch.tensor([M // G] * G, dtype=torch.int64, device=DEV)
            gi[-1] = M - (M // G) * (G - 1)
        else:
            ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
            gi = None
        a = torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25
        return dequant_swiglu_quant(x, ws, a, None, None, None, gi, True, "dynamic",
                                    torch.int8, "rint", -1, swiglu_mode,
                                    0.0 if swiglu_mode else None,
                                    1.0 if swiglu_mode else None,
                                    0.0 if swiglu_mode else None)

    for M, H in [(4096, 2048), (1, 2048), (33, 2048), (1024, 4608), (4096, 512), (65536, 512)]:
        case("dequant_swiglu_quant", f"M={M} H={H}", lambda M=M, H=H: go(M, H), expect_shape=(M, H))
    case("dequant_swiglu_quant", "M=4096 H=2048 G=8", lambda: go(4096, 2048, G=8), expect_shape=(4096, 2048))


# ============================== moe_gating ==============================
def sweep_mgating():
    from triton_ops.moe_gating_top_k import moe_gating_top_k
    from triton_ops.moe_gating_top_k_hash import moe_gating_top_k_hash

    for N, E, k in [(4096, 256, 8), (32, 257, 8), (32, 300, 8), (4096, 512, 8), (1, 256, 8), (256, 1024, 8)]:
        def go(N=N, E=E, k=k):
            torch.manual_seed(0)
            x = (torch.randn(N, E, device=DEV) * 0.5).to(torch.bfloat16)
            bias = (torch.randn(E, device=DEV) * 0.1).to(torch.bfloat16)
            return moe_gating_top_k(x, k, k_group=8, group_count=8, group_select_mode=0,
                                    renorm=0, norm_type=0, out_flag=False,
                                    routed_scaling_factor=1.0, eps=1e-20, bias=bias)
        case("moe_gating_top_k", f"N={N} E={E} k={k} G=8", go, expect_shape=(N, k))
    for N, E, k in [(4096, 256, 8), (32, 257, 8), (32, 300, 8)]:
        def goh(N=N, E=E, k=k):
            torch.manual_seed(0)
            x = (torch.randn(N, E, device=DEV) * 0.5).to(torch.bfloat16)
            bias = (torch.randn(E, device=DEV) * 0.1).to(torch.bfloat16)
            return moe_gating_top_k_hash(x, bias, None, None, k=k, k_group=8, group_count=8,
                                         renorm=0, norm_type=0, out_flag=False,
                                         routed_scaling_factor=1.0, eps=1e-20)
        case("moe_gating_top_k_hash", f"N={N} E={E} k={k} G=8", goh, expect_shape=(N, k))


# ============================== mla_preprocess ==============================
def sweep_mla():
    from triton_ops.mla_preprocess import mla_preprocess
    import inspect
    sig = [p for p in inspect.signature(mla_preprocess).parameters]
    print(f"  mla_preprocess 形参: {sig}", flush=True)
    case("mla_preprocess", "signature-only", lambda: torch.zeros(1, device=DEV))


GROUPS = {"gmm": sweep_gmm, "hc": sweep_hc, "rmsdq": sweep_rmsdq,
          "dsq": sweep_dsq, "mgating": sweep_mgating, "mla": sweep_mla}

if __name__ == "__main__":
    print(f"DEV={DEV} current={torch.npu.current_device()} torch={torch.__version__}", flush=True)
    only = os.environ.get("COV_ONLY", "all")
    for name, fn in GROUPS.items():
        if only not in ("all", name):
            continue
        print(f"\n===== sweep group: {name} =====", flush=True)
        try:
            fn()
        except Exception:
            print("  GROUP FAILED:\n" + traceback.format_exc()[-900:], flush=True)
    print("\n===== 汇总 =====", flush=True)
    print("status counts:", dict(Counter(r["status"] for r in RESULTS)), flush=True)
    for r in RESULTS:
        if r["status"] != "OK":
            print(f"  {r['status']:16s} {r['op']:26s} {r['case']:34s} {r['detail'][:90]}", flush=True)
    print(f"TOTAL={len(RESULTS)} BAD={sum(1 for r in RESULTS if r['status'] != 'OK')}", flush=True)
    print("COVSWEEPDONE", flush=True)
