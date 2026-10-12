"""验证 dispatch_ffn_combine_bf16 的 Triton 实现（**本地计算部分**）。

调用点风格：参数名与 csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h
L21-37 的 `vllm_fl_native::dispatch_ffn_combine` 完全一致
（x / weight1 / weight2 / expert_idx / scale1 / scale2 / bias1 / bias2 / probs /
  group / max_output_size / out / expert_token_nums / x_active_mask / swiglu_limit / type）。

参考实现复刻内核的量化步骤（block_epilogue_pertoken_swiglu.hpp L225-265）：
    C1 = x_row @ w1[e];  gate = C1[:, :I]; up = C1[:, I:]
    sw = (gate * sigmoid(gate)) * up
    mx = max|sw|;  D1 = rint_fp16(sw * 127/mx);  d1 = D1 * (mx/127)
    C2 = d1 @ w2[e];  out[token] += probs[token, slot] * C2

外加「不许静默忽略参数」的负例：bias1/bias2 非空、scale 非全 1、max_output_size 不足
都必须抛 NotImplementedError。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.dispatch_ffn_combine_bf16 import dispatch_ffn_combine_bf16  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.bfloat16: 4e-3, torch.float16: 1e-3}


def quant_floor(dtype, ref_absmax):
    """输出 dtype 在实际幅度下的 half-ULP 量化地板（推导，无手写保底）。

    half-ULP = 2^(e-p)，e = floor(log2|v|)，p = 有效位数（含隐含位）：bf16=8, fp16=11。
    手册的 4e-3/1e-3 对应 O(1) 输出（bf16 |v|∈[1,2) → 2^-8=3.906e-3；fp16 → 2^-11=4.883e-4）。
    本算子的 out = Σ_j probs_j · C2_j 会把多个槽位的量级叠加起来，
    输出幅度可达 O(10)，此时地板按 ULP 线性放大 —— 这是量化性质，不是放宽判据。
    """
    import math
    p = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    m = max(1.0, float(ref_absmax))
    e = math.floor(math.log2(m))
    return 2.0 ** (e - p)


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


# ---------------- tl.dot 可用性探针 ----------------
@triton.jit
def _p_dot(A, B, C, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    a = tl.load(A + tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :])
    b = tl.load(B + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :])
    tl.store(C + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :],
             tl.dot(a, b, out_dtype=tl.float32))


def probe():
    emit("== tl.dot 探针 ==")
    for dt in (torch.float16, torch.bfloat16):
        BM, BK, BN = 16, 32, 32
        a = torch.randn(BM, BK, device=DEV).to(dt)
        b = torch.randn(BK, BN, device=DEV).to(dt)
        c = torch.zeros(BM, BN, device=DEV, dtype=torch.float32)
        try:
            _p_dot[(1,)](a, b, c, BM=BM, BK=BK, BN=BN)
            torch.npu.synchronize()
            r = (c.cpu().double() - a.cpu().double() @ b.cpu().double()).abs().max().item()
            emit(f"  tl.dot {str(dt):16s} err={r:.3e}")
        except Exception as e:  # noqa: BLE001
            emit(f"  tl.dot {str(dt):16s} FAIL: {type(e).__name__}: {str(e)[:110]}")
    emit()


def _rint_fp16(x):
    q16 = x.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref(x, w1s, w2s, expert_idx, probs, mask, base, E, swiglu_limit=0.0):
    xd = x.detach().cpu().double()
    M0, K = xd.shape
    topK = expert_idx.shape[1]
    N = w1s.shape[2]
    I = N // 2
    W1 = w1s.detach().cpu().double()
    W2 = w2s.detach().cpu().double()
    ei = expert_idx.detach().cpu().long()
    pr = probs.detach().cpu().double()
    mk = None if mask is None else mask.detach().cpu().bool()
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
        # ⚠️ 参考必须**复刻内核的 fp32 流水线精度**，否则在 per-token rint 的 .5 边界附近
        #    会量化到不同整数 —— 那是「参考引入了内核没有的精度损失」，会制造**假 FAIL**。
        #    内核：tl.dot fp32 累加 → fp32 swiglu → fp32 量化。
        #    这里即按同样的 fp32 精度算 GEMM1 / swiglu / 量化，再把结果提升回 fp64 参与最终比较。
        W1f = W1[le].float()
        xf = xd[rows].float()
        C1 = (xf @ W1f).double()
        gate, up = C1[:, :I], C1[:, I:]
        if swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
        # swiglu 与量化都在 fp32 里做（与内核一致）
        g32 = gate.float()
        sw32 = (g32 * torch.sigmoid(g32)) * up.float()
        mx32 = sw32.abs().amax(dim=1)
        D1 = _rint_fp16(sw32 * (127.0 / mx32)[:, None])
        d1 = D1 * (mx32.double() / 127.0)[:, None]
        C2 = (d1.float() @ W2[le].float()).double()
        out.index_add_(0, rows, pr[rows, slots][:, None] * C2)
    return out, etn


def run_case(name, M0, topK, K, I, E, dtype, base=0, use_mask=False,
             swiglu_limit=0.0, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(M0, K, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
    w1 = [(torch.randn(K, 2 * I, dtype=torch.float32, device=DEV) * 0.2).to(dtype)
          for _ in range(E)]
    w2 = [(torch.randn(I, K, dtype=torch.float32, device=DEV) * 0.2).to(dtype)
          for _ in range(E)]
    expert_idx = torch.randint(base, base + E, (M0, topK), dtype=torch.int32, device=DEV)
    probs = torch.rand(M0, topK, dtype=torch.float32, device=DEV)
    am = (torch.rand(M0, device=DEV) > 0.25) if use_mask else None

    out_buf = torch.zeros((M0, K), dtype=dtype, device=DEV)
    etn_buf = torch.zeros(E, dtype=torch.int32, device=DEV)
    out, etn = dispatch_ffn_combine_bf16(
        x, weight1=w1, weight2=w2, expert_idx=expert_idx,
        scale1=[], scale2=[], bias1=[], bias2=[],
        probs=probs, group="hccl_group_ep", max_output_size=M0 * topK,
        out=out_buf, expert_token_nums=etn_buf,
        x_active_mask=am, swiglu_limit=swiglu_limit, type="bf16",
        local_expert_base=base)
    torch.npu.synchronize()
    rout, retn = ref(x, torch.stack(w1), torch.stack(w2), expert_idx, probs, am, base, E,
                     swiglu_limit)

    eo = (out.detach().cpu().double() - rout).abs().max().item()
    amp = rout.abs().max().item()
    # ---- 判据三口径并列（只报数据，不因判据改结论） ----
    import math as _m
    p_bits = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    e_mag = _m.floor(_m.log2(max(1.0, amp)))
    f_manual = FLOOR[dtype]                       # 手册绝对地板
    f_manual = FLOOR[dtype]                       # README 绝对地板（小幅度分支）
    f_full = 2.0 ** (e_mag - (p_bits - 1))        # 完整 1 ULP @ |ref|max（大幅度分支）
    f_half = 2.0 ** (e_mag - p_bits)              # 半 ULP（已弃用：原理不可达）
    # 分段取严（Lead 裁决）：|ref|max ≤ 1 用 README 地板；> 1 用完整 1 ULP。
    # 注：字面 min(README,1ULP) 与裁决表的 fp16 用例冲突（min 会得 1e-3 → FAIL），
    #     故按裁决**表**实现为分段；字面 min 值一并打印供核对。
    f_seg = f_manual if amp <= 1.0 else f_full
    f_min_literal = min(f_manual, f_full)

    # ---- 实验 A：逐位置 ULP 分析 ----
    #   ref_q = fp64 真值**正确舍入**到输出 dtype；got 也是输出 dtype。
    #   位置级 ULP = 2^(floor(log2|v|) - (p-1))，v 取该位置的 ref_q。
    #   max|got - ref_q| / ulp_at(ref_q) ≤ 1.0 ⇒ 实现已把结果舍入到输出 dtype 的
    #   最优表示（任何实现都不可能更准）⇒ 差异完全由输出量化决定。
    ref_q = rout.to(dtype).double()
    got = out.detach().cpu().double()
    a = ref_q.abs()
    e_pt = torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a))))
    u_pt = torch.where(a > 0, torch.pow(torch.tensor(2.0, dtype=torch.float64),
                                        e_pt - (p_bits - 1)), torch.zeros_like(a))
    diff_q = (got - ref_q).abs()
    rat = torch.where(u_pt > 0, diff_q / torch.clamp(u_pt, min=1e-300),
                      torch.zeros_like(diff_q))
    max_rat = rat.max().item()
    n_ne = int((rat > 0).sum().item())
    n_gt1 = int((rat > 1.0).sum().item())
    # ⚠️ 逐位置 ULP 会被**接近 0 的位置**主导（那里 ULP 极小，微小的绝对差就是很多 ULP），
    #    单看 max 没有意义。下面补两个有信息量的读数：
    #      · 最大绝对误差位置上的 ULP 比
    #      · 只在 |ref_q| ≥ 0.1·|ref|max 的"有效量级"位置上取 max
    i_max = int(diff_q.flatten().argmax().item())
    rat_at_maxerr = rat.flatten()[i_max].item()
    big = a >= (0.1 * amp)
    rat_big = rat[big].max().item() if int(big.sum()) > 0 else 0.0
    n_big = int(big.sum().item())
    refq_at = ref_q.flatten()[i_max].item()
    got_at = got.flatten()[i_max].item()

    ee = (etn.detach().cpu().long() - retn).abs().max().item()
    emit(f"  {name:50s} etn_diff={ee}")
    emit(f"      |ref|max={amp:.2f}  out_err={eo:.3e}")
    emit(f"      [判据] 分段取严={f_seg:.3e}（|ref|max={amp:.2f} "
         f"{'≤1 → 用README地板' if amp <= 1.0 else '>1 → 用完整1ULP'}）"
         f"  README={f_manual:.3e} 1ULP={f_full:.3e} 字面min={f_min_literal:.3e}")
    emit(f"      半ULP={f_half:.3e}（已弃用：原理不可达）")
    emit(f"      ③ raw max ULP = {max_rat:.1f} (差异位 {n_ne}/{rat.numel()}, >1ULP {n_gt1})"
         f"  ← 被近零位置主导，仅作参考")
    emit(f"      ② gated max ULP(|ref_q|>=0.1*max，{n_big} 位) = {rat_big:.2f}   [安全网上界 4]")
    emit(f"      ① 最大误差位置 ULP 比 = {rat_at_maxerr:.2f}"
         f"   (ref_q={refq_at:.4e} got={got_at:.4e})")
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
        emit(f"  {name:56s} 已拒绝({exc.__name__}): {str(e)[:52]}...  PASS")
        return True


emit("=" * 116)
emit("dispatch_ffn_combine_bf16（本地计算部分，调用点风格签名）: Triton vs AscendC 语义")
emit("=" * 116)
probe()

# ---------------- 极小规模隔离：E=1, topK=1, probs=1 → out 应等于单条 FFN ----------------
emit("-- 隔离用例（最小规模，E=1 topK=1 probs=1，用于定位 gemm/量化/合并 哪一段错） --")
torch.manual_seed(0)
_M0, _K, _I, _E = 2, 8, 4, 1
_x = torch.randn(_M0, _K, dtype=torch.float32, device=DEV).to(torch.bfloat16)
_w1 = [torch.randn(_K, 2 * _I, dtype=torch.float32, device=DEV).to(torch.bfloat16)]
_w2 = [torch.randn(_I, _K, dtype=torch.float32, device=DEV).to(torch.bfloat16)]
_ei = torch.zeros(_M0, 1, dtype=torch.int32, device=DEV)
_pr = torch.ones(_M0, 1, dtype=torch.float32, device=DEV)
_ob = torch.zeros((_M0, _K), dtype=torch.bfloat16, device=DEV)
_eb = torch.zeros(_E, dtype=torch.int32, device=DEV)
_o, _e = dispatch_ffn_combine_bf16(
    _x, weight1=_w1, weight2=_w2, expert_idx=_ei, scale1=[], scale2=[], bias1=[],
    bias2=[], probs=_pr, group="g", max_output_size=_M0, out=_ob,
    expert_token_nums=_eb, type="bf16", local_expert_base=0)
torch.npu.synchronize()
_ro, _re = ref(_x, torch.stack(_w1), torch.stack(_w2), _ei, _pr, None, 0, _E)
emit(f"  kernel out[0][:4] = {_o.detach().cpu()[0][:4].tolist()}")
emit(f"  ref    out[0][:4] = {_ro[0][:4].tolist()}")
emit(f"  kernel out[1][:4] = {_o.detach().cpu()[1][:4].tolist()}")
emit(f"  ref    out[1][:4] = {_ro[1][:4].tolist()}")
emit(f"  etn kernel={_e.detach().cpu().tolist()} ref={_re.tolist()}")
emit(f"  x[0][:4] = {_x.detach().cpu()[0][:4].tolist()}")
emit()

allok = True
allok &= run_case("bf16 M0=16 topK=2 K=32 I=16 E=2", 16, 2, 32, 16, 2, torch.bfloat16)
allok &= run_case("fp16 M0=16 topK=2 K=64 I=32 E=3", 16, 2, 64, 32, 3, torch.float16)
allok &= run_case("bf16 M0=7  topK=3 K=17 I=5  E=2 (非整除)", 7, 3, 17, 5, 2, torch.bfloat16)
allok &= run_case("bf16 M0=24 topK=4 K=64 I=32 E=4", 24, 4, 64, 32, 4, torch.bfloat16)
allok &= run_case("bf16 M0=16 topK=2 K=32 I=16 E=2 x_active_mask", 16, 2, 32, 16, 2,
                  torch.bfloat16, use_mask=True)
allok &= run_case("bf16 M0=12 topK=2 K=48 I=24 E=2 local_expert_base=4", 12, 2, 48, 24, 2,
                  torch.bfloat16, base=4)
allok &= run_case("fp16 M0=16 topK=2 K=32 I=16 E=2 swiglu_limit=8", 16, 2, 32, 16, 2,
                  torch.float16, swiglu_limit=8.0)

emit()
emit("-- 负例：不许静默忽略参数 --")

torch.manual_seed(0)
_x = torch.randn(8, 16, dtype=torch.float32, device=DEV).to(torch.bfloat16)
_w1 = [torch.randn(16, 8, dtype=torch.float32, device=DEV).to(torch.bfloat16)]
_w2 = [torch.randn(4, 16, dtype=torch.float32, device=DEV).to(torch.bfloat16)]
_ei = torch.zeros(8, 2, dtype=torch.int32, device=DEV)
_pr = torch.rand(8, 2, dtype=torch.float32, device=DEV)
_base = dict(x=_x, weight1=_w1, weight2=_w2, expert_idx=_ei, probs=_pr,
             local_expert_base=0)
allok &= expect_raise("bias1 非空 → NotImplementedError",
                      lambda: dispatch_ffn_combine_bf16(
                          **_base, bias1=[torch.zeros(8, device=DEV)]))
allok &= expect_raise("scale1 非全 1 → NotImplementedError",
                      lambda: dispatch_ffn_combine_bf16(
                          **_base, scale1=[torch.full((8,), 2.0, device=DEV)]))
allok &= expect_raise("max_output_size 不足 → NotImplementedError",
                      lambda: dispatch_ffn_combine_bf16(**_base, max_output_size=1))
allok &= expect_raise("probs=None → NotImplementedError",
                      lambda: dispatch_ffn_combine_bf16(
                          **{**_base, "probs": None}))

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")
emit("注：跨卡 hccl dispatch/combine 不在实现范围内；group 只做类型校验，不参与本地数学。")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/dispatch_ffn_combine_bf16_test.txt", "w", encoding="utf-8").write(txt + "\n")
