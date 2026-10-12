"""验证 grouped_matmul_swiglu_quant_weight_nz_tensor_list 的 Triton 实现。

调用点风格：参数名与 `_repos/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/
grouped_matmul_swiglu_quant_torch_adpt.h` L62-70 一致
（x / weight[] / weight_scale[] / x_scale / group_list / bias / offset / swiglu_limit）。

参考实现（fp64）：
    for g in groups:
        rows = [start_g, end_g)
        C   = (x[rows] * x_scale[rows,None]) @ (W_g * ws_g[None,:])
        gate, up = C[:, :N/2], C[:, N/2:]
        sw  = (gate * sigmoid(gate)) * up          # swiglu_limit>0 时 clamp gate
        mx  = max|sw| per row
        y   = rint_fp16(sw*127/mx) → int8
        y_scale = mx/127

判据：y（int8）与参考**完全一致**（舍入链路已逐步复刻；若出现 ±1 会打印差异）；
      y_scale fp32 ≤ 2e-6 × max(1,|ref|)。
      负例：bias/offset 非空必须抛 NotImplementedError。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.grouped_matmul_swiglu_quant_weight_nz_tensor_list import (  # noqa: E402
    grouped_matmul_swiglu_quant,
    grouped_matmul_swiglu_quant_weight_nz,
    grouped_matmul_swiglu_quant_weight_nz_tensor_list,
)

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


# ============ 数据点2：tl.dot 在各 shape（含极小、含不满足最小 tile）下的孤立实测 ============
@triton.jit
def _p_dot_shape(A, B, C, M, K, N, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    m = tl.arange(0, BM)
    k = tl.arange(0, BK)
    n = tl.arange(0, BN)
    a = tl.load(A + m[:, None] * K + k[None, :],
                mask=(m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float16)
    b = tl.load(B + k[:, None] * N + n[None, :],
                mask=(k[:, None] < K) & (n[None, :] < N), other=0.0).to(tl.float16)
    c = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + m[:, None] * BN + n[None, :], c,
             mask=(m[:, None] < BM) & (n[None, :] < BN))


# ============ 数据点1/3：结构化 kernel 逐段 store（tl.dot 版 vs 逐元素归约版） ============
@triton.jit
def _p_struct(X, W, EPS, S1, S2, S3, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
              HALF: tl.constexpr, USE_DOT: tl.constexpr):
    m = tl.arange(0, BM)
    k = tl.arange(0, BK)
    n = tl.arange(0, BN)
    x = tl.load(X + m[:, None] * BK + k[None, :]).to(tl.float16)
    w1 = tl.load(W + k[:, None] * BN + n[None, :]).to(tl.float16)
    if USE_DOT:
        c = tl.dot(x, w1, out_dtype=tl.float32)
    else:
        c = tl.zeros([BM, BN], dtype=tl.float32)
        for kk in tl.static_range(BK):
            xc = tl.sum(tl.where(k[None, :] == kk, x.to(tl.float32), 0.0), axis=1)
            wc = tl.sum(tl.where(k[:, None] == kk, w1.to(tl.float32), 0.0), axis=0)
            c += xc[:, None] * wc[None, :]
    tl.store(S1 + m[:, None] * BN + n[None, :], c)      # 段1 matmul
    g = tl.where(n[None, :] < HALF, c, 0.0)
    u = tl.where(n[None, :] >= HALF, c, 0.0)
    sw = (g / (1.0 + tl.exp(-g))) * u
    tl.store(S2 + m[:, None] * BN + n[None, :], sw)     # 段2 swiglu
    mx = tl.max(tl.abs(sw), axis=1)
    q = sw * (127.0 / tl.maximum(mx, EPS))
    tl.store(S3 + m[:, None] * BN + n[None, :], q)      # 段3 量化前


def probe():
    emit("== 数据点2: tl.dot 各 shape 孤立实测（fp16 输入, fp32 累加）==")
    for (M, K, N) in [(2, 8, 4), (4, 8, 8), (8, 8, 8), (16, 16, 16), (2, 16, 16),
                      (16, 16, 2), (2, 2, 2), (1, 16, 16)]:
        BM = max(2, 1 << (M - 1).bit_length())
        BK = max(2, 1 << (K - 1).bit_length())
        BN = max(2, 1 << (N - 1).bit_length())
        a = torch.randn(M, K, device=DEV).to(torch.float16)
        b = torch.randn(K, N, device=DEV).to(torch.float16)
        c = torch.zeros(BM, BN, device=DEV, dtype=torch.float32)
        try:
            _p_dot_shape[(1,)](a, b, c, M, K, N, BM=BM, BK=BK, BN=BN)
            torch.npu.synchronize()
            err = (c[:M, :N].cpu().double() - a.cpu().double() @ b.cpu().double()).abs().max().item()
            emit(f"  M={M:2d} K={K:2d} N={N:2d} tile={BM}x{BK}x{BN}  err={err:.3e}  "
                 f"{'OK' if err < 1e-2 else 'ZERO/WRONG'}")
        except Exception as e:  # noqa: BLE001
            emit(f"  M={M:2d} K={K:2d} N={N:2d}  FAIL: {type(e).__name__}: {str(e)[:70]}")
    emit()
    emit("== 数据点1/3: 结构化 kernel 逐段 store（tl.dot vs 逐元素归约）==")
    BM, BK, BN, HALF = 16, 16, 16, 8
    torch.manual_seed(0)
    x = torch.randn(BM, BK, device=DEV).to(torch.float16)
    w = torch.randn(BK, BN, device=DEV).to(torch.float16)
    cd = x.cpu().double() @ w.cpu().double()
    for use_dot in (True, False):
        s1 = torch.zeros(BM, BN, device=DEV)
        s2 = torch.zeros(BM, BN, device=DEV)
        s3 = torch.zeros(BM, BN, device=DEV)
        _p_struct[(1,)](x, w, 1e-30, s1, s2, s3, BM=BM, BK=BK, BN=BN, HALF=HALF,
                        USE_DOT=use_dot)
        torch.npu.synchronize()
        e1 = (s1.cpu().double() - cd).abs().max().item()
        g, u = cd[:, :HALF], cd[:, HALF:]
        sw = (g * torch.sigmoid(g)) * u
        e2 = (s2.cpu().double()[:, :HALF] - sw).abs().max().item()
        tag = "tl.dot   " if use_dot else "逐元素归约"
        emit(f"  [{tag}] 段1(matmul) err={e1:.3e}  段2(swiglu) err={e2:.3e}  "
             f"段1非零={int((s1.abs()>0).sum())}/{BM*BN}  段3非零={int((s3.abs()>0).sum())}/{BM*BN}")
    emit()


def _rint_fp16(x):
    q16 = x.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref(x, w, ws, xs, ends, swiglu_limit=0.0):
    xd = x.detach().cpu().double()
    W = w.detach().cpu().double()
    WS = ws.detach().cpu().double()
    XS = xs.detach().cpu().double()
    M, K = xd.shape
    G, _, N = W.shape
    I = N // 2
    ends_l = [int(e) for e in ends.detach().cpu().tolist()]
    y = torch.zeros(M, I, dtype=torch.int64)
    ys = torch.zeros(M, dtype=torch.float64)
    prev = 0
    for g in range(G):
        s, e = prev, ends_l[g]
        prev = e
        if e <= s:
            continue
        rows = slice(s, e)
        C = (xd[rows] * XS[rows, None]) @ (W[g] * WS[g][None, :])
        gate, up = C[:, :I], C[:, I:]
        if swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
        sw = (gate * torch.sigmoid(gate)) * up
        mx = sw.abs().amax(dim=1)
        y[s:e] = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64)
        ys[s:e] = mx / 127.0
    return y, ys


def run_case(name, M, K, G, N, swiglu_limit=0.0, seed=0, alias=None):
    torch.manual_seed(seed)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    # group_list = 累计结束偏移（cumsum 语义）
    per = max(1, M // G)
    ends = torch.tensor([min(M, per * (i + 1)) for i in range(G)],
                        dtype=torch.int64, device=DEV)
    ends[-1] = M

    fn = {"list": grouped_matmul_swiglu_quant_weight_nz_tensor_list,
          "nz": grouped_matmul_swiglu_quant_weight_nz,
          "alias": grouped_matmul_swiglu_quant}[alias or "list"]
    if (alias or "list") == "list":
        y, ys, yoff = fn(x, weight=[w[i] for i in range(G)],
                         weight_scale=[ws[i] for i in range(G)],
                         x_scale=xs, group_list=ends, swiglu_limit=swiglu_limit)
    else:
        y, ys, yoff = fn(x, weight=w, weight_scale=ws, x_scale=xs,
                         group_list=ends, swiglu_limit=swiglu_limit)
    torch.npu.synchronize()
    ry, rys = ref(x, w, ws, xs, ends, swiglu_limit)

    ey = (y.detach().cpu().long() - ry).abs().max().item()
    es = (ys.detach().cpu().double() - rys).abs().max().item()
    fs = 2e-6 * max(1.0, rys.abs().max().item())
    ok = ey == 0 and es <= fs
    emit(f"  {name:56s} y_diff={ey} scale_err={es:.3e}(<={fs:.1e})  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


def expect_raise(name, fn):
    try:
        fn()
        emit(f"  {name:56s} 未报错  FAIL")
        return False
    except NotImplementedError as e:
        emit(f"  {name:56s} 已拒绝: {str(e)[:44]}...  PASS")
        return True


emit("=" * 118)
emit("grouped_matmul_swiglu_quant_weight_nz_tensor_list（调用点风格签名）: Triton vs 参考")
emit("=" * 118)
probe()

allok = True
allok &= run_case("tensor_list M=32 K=64 G=2 N=64 int8", 32, 64, 2, 64)
allok &= run_case("tensor_list M=16 K=32 G=3 N=32 int8", 16, 32, 3, 32)
allok &= run_case("tensor_list M=13 K=17 G=2 N=10 (非整除)", 13, 17, 2, 10)
allok &= run_case("tensor_list M=24 K=48 G=4 N=48 swiglu_limit=8", 24, 48, 4, 48,
                  swiglu_limit=8.0)
allok &= run_case("nz 单张量版 M=16 K=32 G=2 N=32", 16, 32, 2, 32, alias="nz")
allok &= run_case("grouped_matmul_swiglu_quant 别名 M=16 K=32 G=2 N=32", 16, 32, 2, 32,
                  alias="alias")

emit()
emit("-- 负例：不许静默忽略参数 --")
torch.manual_seed(0)
_x = torch.randint(-30, 31, (8, 16), dtype=torch.int8, device=DEV)
_w = torch.randint(-30, 31, (2, 16, 16), dtype=torch.int8, device=DEV)
_ws = torch.rand(2, 16, device=DEV) * 0.01
_xs = torch.rand(8, device=DEV) * 0.01
_gl = torch.tensor([4, 8], dtype=torch.int64, device=DEV)
_base = dict(x=_x, weight=[_w[0], _w[1]], weight_scale=[_ws[0], _ws[1]],
             x_scale=_xs, group_list=_gl)
allok &= expect_raise("bias 非空 → NotImplementedError",
                      lambda: grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                          **_base, bias=torch.zeros(16, device=DEV)))
allok &= expect_raise("offset 非空 → NotImplementedError",
                      lambda: grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                          **_base, offset=torch.zeros(1, device=DEV)))

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")
emit("注：group_list 按「累计结束偏移(cumsum)」解释；weight 在 AscendC 侧是 FRACTAL_NZ，")
emit("    本 Triton 实现使用其逻辑等价的 [K, N] 行主序布局。")
emit("    已核对源码：grouped_matmul_swiglu_quant 是 grouped_matmul_swiglu_quant_weight_nz 的纯别名；")
emit("    二者在 _repos/vllm-ascend/csrc/gmm/ 下**无独立 AscendC 内核**，故复用同一实现。")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/gmm_swiglu_quant_test.txt", "w", encoding="utf-8").write(txt + "\n")
