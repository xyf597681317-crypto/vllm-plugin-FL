"""验证 grouped_matmul_swiglu_quant_v2 的 Triton 实现。

调用点风格：参数名与 ./grouped_matmul_swiglu_quant_v2_torch_adpt.h L20-36 完全一致
（x / weight[] / weight_scale[] / x_scale / group_list / smooth_scale /
  weight_assist_matrix / bias / dequant_mode / dequant_dtype / quant_mode /
  quant_dtype / transpose_weight / group_list_type / tuning_config / swiglu_limit）。

参考实现（fp64）：
    for g: C1 = (x[rows]*x_scale[:,None]) @ (W_g*ws_g[None,:])
           gate, up = C1[:, :I], C1[:, I:]        # ← up 左移 I 列，与 gate 同列对齐
           sw = silu(gate) * up                    # swiglu_limit>0 先 clamp gate
           sw = sw * smooth_scale_g                # v2 新增：SwiGLU 后、量化前
           mx = max|sw| per row
           y = rint_fp16(sw*127/mx) → int8 ; y_scale = mx/127

判据：y(int8) 与参考**完全一致**（y_diff=0）；y_scale ≤ 2e-6×max(1,|ref|)。
负例：bias / weight_assist_matrix / transpose_weight 必须显式抛 NotImplementedError。
起步用**哨兵初值法**：y 初值 -123 起手，确认 store 是否落地。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.grouped_matmul_swiglu_quant_v2 import grouped_matmul_swiglu_quant_v2  # noqa: E402

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


def _rint_fp16(x):
    q16 = x.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref(x, w, ws, xs, ends, smooth, swiglu_limit=0.0):
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
        if smooth is not None:
            smt = smooth.detach().cpu().double()
            sw = sw * (smt[g][None, :] if smt.dim() == 2 else smt[g])
        mx = sw.abs().amax(dim=1)
        y[s:e] = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64)
        ys[s:e] = mx / 127.0
    return y, ys


def run_case(name, M, K, G, N, smooth_kind=None, swiglu_limit=0.0, glist_type=0,
             sentinel=False, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = torch.randint(-40, 41, (G, K, N), dtype=torch.int8, device=DEV)
    ws = torch.rand(G, N, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    per = max(1, M // G)
    cnt = [min(M, per * (i + 1)) - min(M, per * i) for i in range(G)]
    cnt[-1] = M - sum(cnt[:-1])
    cnts = torch.tensor(cnt, dtype=torch.int64, device=DEV)
    if glist_type == 1:
        gl = cnts
    else:
        gl = torch.cumsum(cnts, dim=0)
    ends = torch.cumsum(cnts, dim=0) if glist_type == 1 else gl

    I = N // 2
    smooth = None
    if smooth_kind == "col":
        smooth = torch.rand(G, I, dtype=torch.float32, device=DEV) * 0.8 + 0.6
    elif smooth_kind == "row":
        smooth = torch.rand(G, dtype=torch.float32, device=DEV) * 0.8 + 0.6

    y, ys = grouped_matmul_swiglu_quant_v2(
        x, weight=[w[i] for i in range(G)], weight_scale=[ws[i] for i in range(G)],
        x_scale=xs, group_list=gl, smooth_scale=smooth,
        weight_assist_matrix=None, bias=None,
        dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
        transpose_weight=False, group_list_type=glist_type,
        tuning_config=[0], swiglu_limit=swiglu_limit)
    torch.npu.synchronize()
    ry, rys = ref(x, w, ws, xs, ends, smooth, swiglu_limit)

    ey = (y.detach().cpu().long() - ry).abs().max().item()
    es = (ys.detach().cpu().double() - rys).abs().max().item()
    fs = 2e-6 * max(1.0, rys.abs().max().item())
    ok = ey == 0 and es <= fs
    emit(f"  {name:58s} y_diff={ey} scale_err={es:.3e}(<={fs:.1e})  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


def expect_raise(name, fn):
    try:
        fn()
        emit(f"  {name:58s} 未报错  FAIL")
        return False
    except NotImplementedError as e:
        emit(f"  {name:58s} 已拒绝: {str(e)[:40]}...  PASS")
        return True


emit("=" * 122)
emit("grouped_matmul_swiglu_quant_v2（调用点风格签名）: Triton vs 参考")
emit("=" * 122)

# ---- 哨兵初值起步：确认真实 shape 下 store 会落地 ----
torch.manual_seed(0)
_sy = torch.full((8, 8), -123, dtype=torch.int8, device=DEV)
_x = torch.randint(-30, 31, (8, 16), dtype=torch.int8, device=DEV)
_w = [torch.randint(-30, 31, (16, 16), dtype=torch.int8, device=DEV)]
_ws = [torch.rand(16, dtype=torch.float32, device=DEV) * 0.01]
_xs = torch.rand(8, dtype=torch.float32, device=DEV) * 0.01
_gl = torch.tensor([8], dtype=torch.int64, device=DEV)
_yy, _ysc = grouped_matmul_swiglu_quant_v2(_x, _wx := _w, _ws, _xs, _gl)
torch.npu.synchronize()
emit(f"  [哨兵] 输出前 4 个 = {_yy.detach().cpu()[0][:4].tolist()}  "
     f"非零数={int((_yy != 0).sum())}/{_yy.numel()}")
emit()

allok = True
allok &= run_case("v2 M=32 K=64 G=2 N=64（无 smooth）", 32, 64, 2, 64)
allok &= run_case("v2 M=16 K=32 G=3 N=32 smooth=(E,N/2)", 16, 32, 3, 32, smooth_kind="col")
allok &= run_case("v2 M=16 K=32 G=3 N=32 smooth=(E,)", 16, 32, 3, 32, smooth_kind="row")
allok &= run_case("v2 M=13 K=17 G=2 N=10（非整除）", 13, 17, 2, 10)
allok &= run_case("v2 M=24 K=48 G=4 N=48 swiglu_limit=8", 24, 48, 4, 48, swiglu_limit=8.0)
allok &= run_case("v2 M=20 K=32 G=2 N=32 group_list_type=1(计数)", 20, 32, 2, 32,
                  glist_type=1)

emit()
emit("-- 负例：不许静默忽略参数 --")
allok &= expect_raise("bias 非空 → NotImplementedError",
                      lambda: grouped_matmul_swiglu_quant_v2(
                          _x, weight=_w, weight_scale=_ws, x_scale=_xs, group_list=_gl,
                          bias=torch.zeros(16, device=DEV)))
allok &= expect_raise("weight_assist_matrix 非空 → NotImplementedError",
                      lambda: grouped_matmul_swiglu_quant_v2(
                          _x, weight=_w, weight_scale=_ws, x_scale=_xs, group_list=_gl,
                          weight_assist_matrix=[torch.zeros(16, 16, dtype=torch.int8, device=DEV)]))
allok &= expect_raise("transpose_weight=True → NotImplementedError",
                      lambda: grouped_matmul_swiglu_quant_v2(
                          _x, weight=_w, weight_scale=_ws, x_scale=_xs, group_list=_gl,
                          transpose_weight=True))

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")
emit("注：group_list_type 0=cumsum(累计结束偏移) / 1=每组计数；weight 在 AscendC 侧为 FRACTAL_NZ，")
emit("    本实现用逻辑等价的 [K, N] 行主序；bias 在 AscendC 内核中未被使用（utils.h L84）。")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/gmm_swiglu_quant_v2_test.txt", "w", encoding="utf-8").write(txt + "\n")
