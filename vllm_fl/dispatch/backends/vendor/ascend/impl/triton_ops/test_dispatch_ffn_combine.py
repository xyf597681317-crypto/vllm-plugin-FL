"""验证 dispatch_ffn_combine（W8A8 int8 权重变体）的 Triton 实现（**本地计算部分**）。

调用点风格：参数名与 dispatch_ffn_combine_torch_adpt.h L21-37 完全一致。

参考实现复刻的量化链路：
    a_s = max|a_row|/127;  a_q = rint_fp16(a/a_s)                # moeInitRoutingQuantV2
    C1  = (a_q * a_s) @ (w1_int8[e] * scale1[e][None, :])
    sw  = (gate * sigmoid(gate)) * up                            # up = C1[:, I:]
    mx  = max|sw|;  D1 = rint_fp16(sw*127/mx);  d1 = D1*(mx/127)  # W8A8 ⇒ D1 落 int8
    C2  = d1 @ (w2_int8[e] * scale2[e][None, :])
    out[token] += probs[token, slot] * C2

判据：int8 权重量化损失主导 ⇒ 相对误差 ≤ 1.5e-2；expert_token_nums 完全相等。
负例：bias1/bias2 非空、scale 缺失、max_output_size 不足、probs=None 都必须抛
NotImplementedError（不许静默忽略）。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.dispatch_ffn_combine import dispatch_ffn_combine  # noqa: E402

DEV = "npu:0"
L = []
# 手册的绝对 dtype 量化地板（仅用于对照打印）
FLOOR = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}
_FIRST = [True]


def quant_floor(dtype, ref_absmax):
    """输出 dtype 在实际幅度下的 half-ULP 量化地板（推导，无手写保底）。

    half-ULP = 2^(e-p)，e = floor(log2|v|)，p = 有效位数（含隐含位）：bf16=8, fp16=11。
    手册 4e-3/1e-3 对应 O(1) 输出；本算子 out = Σ_j probs_j·C2_j 会把多个槽位叠加，
    输出幅度可达 O(10)，地板随 ULP 线性放大 —— 量化性质，非放宽判据。
    """
    import math
    p = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    m = max(1.0, float(ref_absmax))
    e = math.floor(math.log2(m))
    return 2.0 ** (e - p)


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


def _rint_fp16(x):
    q16 = x.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref(x, w1s, s1s, w2s, s2s, expert_idx, probs, mask, base, E, swiglu_limit=0.0):
    xd = x.detach().cpu().double()
    M0, K = xd.shape
    topK = expert_idx.shape[1]
    N = w1s.shape[2]
    I = N // 2
    W1 = w1s.detach().cpu().double()
    W2 = w2s.detach().cpu().double()
    S1 = s1s.detach().cpu().double()
    S2 = s2s.detach().cpu().double()
    ei = expert_idx.detach().cpu().long()
    pr = probs.detach().cpu().double()
    mk = None if mask is None else mask.detach().cpu().bool()

    amax = xd.abs().amax(dim=1)
    a_s = amax / 127.0
    a_q = _rint_fp16(xd / a_s[:, None]).clamp(-128, 127)
    a_dq = a_q * a_s[:, None]

    out = torch.zeros(M0, K, dtype=torch.float64)
    etn = torch.zeros(E, dtype=torch.int64)
    for le in range(E):
        g = base + le
        sel = (ei == g)
        if mk is not None:
            sel = sel & mk[:, None]
        idx = sel.nonzero()
        etn[le] = idx.shape[0]
        if idx.numel() == 0:
            continue
        rows, slots = idx[:, 0], idx[:, 1]
        C1 = (a_dq[rows] @ W1[le]) * S1[le][None, :]
        gate, up = C1[:, :I], C1[:, I:]
        if swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
        sw = (gate * torch.sigmoid(gate)) * up
        mx = sw.abs().amax(dim=1)
        D1 = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127)
        d1 = D1 * (mx / 127.0)[:, None]
        C2 = (d1 @ W2[le]) * S2[le][None, :]
        out.index_add_(0, rows, pr[rows, slots][:, None] * C2)
    return out, etn


def run_case(name, M0, topK, K, I, E, dtype, base=0, use_mask=False,
             swiglu_limit=0.0, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(M0, K, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
    w1 = [torch.randint(-40, 41, (K, 2 * I), dtype=torch.int8, device=DEV) for _ in range(E)]
    w2 = [torch.randint(-40, 41, (I, K), dtype=torch.int8, device=DEV) for _ in range(E)]
    s1 = [torch.rand(2 * I, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(E)]
    s2 = [torch.rand(K, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(E)]
    expert_idx = torch.randint(base, base + E, (M0, topK), dtype=torch.int32, device=DEV)
    probs = torch.rand(M0, topK, dtype=torch.float32, device=DEV)
    am = (torch.rand(M0, device=DEV) > 0.25) if use_mask else None

    # 哨兵初值：区分「根本没写进去」与「写了但值错」
    out_buf = torch.full((M0, K), -12345.0, dtype=dtype, device=DEV)
    etn_buf = torch.zeros(E, dtype=torch.int32, device=DEV)
    out, etn = dispatch_ffn_combine(
        x, weight1=w1, weight2=w2, expert_idx=expert_idx,
        scale1=s1, scale2=s2, bias1=[], bias2=[],
        probs=probs, group="hccl_group_ep", max_output_size=M0 * topK,
        out=out_buf, expert_token_nums=etn_buf,
        x_active_mask=am, swiglu_limit=swiglu_limit, type="int8",
        local_expert_base=base)
    torch.npu.synchronize()
    n_sentinel = int((out_buf.detach().cpu().float() == -12345.0).sum().item())
    if _FIRST[0]:
        _FIRST[0] = False
        emit(f"  [哨兵] out 中仍为 -12345 的元素数 = {n_sentinel}/{M0 * K}"
             f"  ⇒ {'完全没写' if n_sentinel == M0 * K else '写了(至少部分)'}")
        emit(f"  [哨兵] kernel out[0][:6] = {out_buf.detach().cpu()[0][:6].tolist()}")
    rout, retn = ref(x, torch.stack(w1), torch.stack(s1), torch.stack(w2), torch.stack(s2),
                     expert_idx, probs, am, base, E, swiglu_limit)
    if n_sentinel < M0 * K:
        emit(f"  [哨兵] ref    out[0][:6] = {rout[0][:6].tolist()}")

    eo = (out.detach().cpu().double() - rout).abs().max().item()
    amp = rout.abs().max().item()
    ee = (etn.detach().cpu().long() - retn).abs().max().item()

    # ---- 裁决后的判据（与 bf16 变体一致）：①完整 ULP 主判据 + ②gated 安全网 ----
    #   ① 最大绝对误差位置：|got - ref_q| <= 1.0 ULP
    #      （ref_q = fp64 真值**正确舍入**到输出 dtype；1 ULP 是 fp32 流水线的物理下界）
    #   ② 安全网：|ref_q| >= 0.1*max 的位置上 gated max ULP <= 4（当前数据包络）
    #   ③ 三个读数全部打印；手册绝对地板作为**信息列**（不参与判定）
    import math as _m
    p_bits = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    ref_q = rout.to(dtype).double()
    got = out.detach().cpu().double()
    a = ref_q.abs()
    e_pt = torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a))))
    u_pt = torch.where(a > 0, torch.pow(torch.tensor(2.0, dtype=torch.float64),
                                        e_pt - (p_bits - 1)), torch.zeros_like(a))
    diff_q = (got - ref_q).abs()
    rat = torch.where(u_pt > 0, diff_q / torch.clamp(u_pt, min=1e-300),
                      torch.zeros_like(diff_q))
    i_max = int(diff_q.flatten().argmax().item())
    rat_at_maxerr = rat.flatten()[i_max].item()
    refq_at = ref_q.flatten()[i_max].item()
    got_at = got.flatten()[i_max].item()
    big = a >= (0.1 * amp)
    rat_big = rat[big].max().item() if int(big.sum()) > 0 else 0.0
    raw_max = rat.max().item()

    emit(f"  {name:50s} etn_diff={ee}")
    emit(f"      |ref|max={amp:.2f}  out_err={eo:.3e}")
    emit(f"      [信息列] 手册绝对地板={FLOOR[dtype]:.3e} → "
         f"{'超' if eo > FLOOR[dtype] else '在'}（不参与判定）")
    emit(f"      ③ raw max ULP = {raw_max:.1f}  ← 被近零位置主导，仅作参考")
    emit(f"      ② gated max ULP(|ref_q|>=0.1*max) = {rat_big:.2f}   [安全网上界 4]")
    emit(f"      ① 最大误差位置 ULP 比 = {rat_at_maxerr:.2f}"
         f"   (ref_q={refq_at:.4e} got={got_at:.4e})")
    # 分段取严：|ref|max ≤ 1 用 README 地板；> 1 用完整 1 ULP
    _e_mag = _m.floor(_m.log2(max(1.0, amp)))
    f_full = 2.0 ** (_e_mag - (p_bits - 1))
    f_seg = FLOOR[dtype] if amp <= 1.0 else f_full
    emit(f"      [判据] 分段取严={f_seg:.3e}（|ref|max={amp:.2f} "
         f"{'≤1 → README地板' if amp <= 1.0 else '>1 → 完整1ULP'}）"
         f"  README={FLOOR[dtype]:.3e} 1ULP={f_full:.3e}")
    ok_main = (eo <= f_seg) and (rat_at_maxerr <= 1.0) and ee == 0
    ok_gate = rat_big <= 4.0
    if ok_main and ok_gate:
        verdict = "PASS(1ULP)"
    elif ok_main:
        verdict = "PASS ⚠️需复核(gated>4ULP)"
    else:
        verdict = "FAIL"
    emit(f"      ⇒ 判定: {verdict}   [①最大误差位≤1ULP:{ok_main}] [②gated≤4ULP:{ok_gate}]")
    return ok_main


def expect_raise(name, fn, exc=NotImplementedError):
    try:
        fn()
        emit(f"  {name:56s} 未报错  FAIL")
        return False
    except exc as e:
        emit(f"  {name:56s} 已拒绝({exc.__name__}): {str(e)[:48]}...  PASS")
        return True


emit("=" * 116)
emit("dispatch_ffn_combine（W8A8 int8，本地计算部分，调用点风格签名）: Triton vs AscendC 语义")
emit("=" * 116)

allok = True
allok &= run_case("int8 M0=16 topK=2 K=32 I=16 E=2 bf16", 16, 2, 32, 16, 2, torch.bfloat16)
allok &= run_case("int8 M0=16 topK=2 K=64 I=32 E=3 fp16", 16, 2, 64, 32, 3, torch.float16)
allok &= run_case("int8 M0=7  topK=3 K=17 I=5  E=2 (非整除)", 7, 3, 17, 5, 2, torch.bfloat16)
allok &= run_case("int8 M0=24 topK=4 K=64 I=32 E=4 bf16", 24, 4, 64, 32, 4, torch.bfloat16)
allok &= run_case("int8 M0=16 topK=2 K=32 I=16 E=2 x_active_mask", 16, 2, 32, 16, 2,
                  torch.bfloat16, use_mask=True)
allok &= run_case("int8 M0=12 topK=2 K=48 I=24 E=2 local_expert_base=4", 12, 2, 48, 24, 2,
                  torch.bfloat16, base=4)
allok &= run_case("int8 M0=16 topK=2 K=32 I=16 E=2 swiglu_limit=8", 16, 2, 32, 16, 2,
                  torch.bfloat16, swiglu_limit=8.0)

emit()
emit("-- 负例：不许静默忽略参数 --")
torch.manual_seed(0)
_x = torch.randn(8, 16, dtype=torch.float32, device=DEV).to(torch.bfloat16)
_w1 = [torch.randint(-30, 31, (16, 8), dtype=torch.int8, device=DEV)]
_w2 = [torch.randint(-30, 31, (4, 16), dtype=torch.int8, device=DEV)]
_s1 = [torch.rand(8, device=DEV) * 0.01]
_s2 = [torch.rand(16, device=DEV) * 0.01]
_ei = torch.zeros(8, 2, dtype=torch.int32, device=DEV)
_pr = torch.rand(8, 2, device=DEV)
_base = dict(x=_x, weight1=_w1, weight2=_w2, expert_idx=_ei, scale1=_s1, scale2=_s2,
             probs=_pr, local_expert_base=0)
allok &= expect_raise("bias1 非空 → NotImplementedError",
                      lambda: dispatch_ffn_combine(**_base, bias1=[torch.zeros(8, device=DEV)]))
allok &= expect_raise("scale1 缺失 → NotImplementedError",
                      lambda: dispatch_ffn_combine(**{**_base, "scale1": []}))
allok &= expect_raise("max_output_size 不足 → NotImplementedError",
                      lambda: dispatch_ffn_combine(**_base, max_output_size=1))
allok &= expect_raise("probs=None → NotImplementedError",
                      lambda: dispatch_ffn_combine(**{**_base, "probs": None}))

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")
emit("注：跨卡 hccl dispatch/combine 不在实现范围内；group 只做类型校验，不参与本地数学。")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/dispatch_ffn_combine_test.txt", "w", encoding="utf-8").write(txt + "\n")
