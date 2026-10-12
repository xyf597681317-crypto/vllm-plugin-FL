"""验证 recurrent_gated_delta_rule 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.recurrent_gated_delta_rule import recurrent_gated_delta_rule  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def case(*a, **kw):
    """跑单个用例；编译/运行异常也要记录，不能让进程挂掉丢结果。"""
    global allok, nexc
    try:
        allok &= run_case(*a, **kw)
    except Exception as e:  # noqa: BLE001
        nexc += 1
        emit(f"  {a[0]:46s} EXCEPTION(编译器/运行时) {type(e).__name__}: "
             f"{str(e)[:100].replace(chr(10), ' ')}")
        allok = False


def ref(query, key, value, beta, state, cu, ssm, g, gk, nacc, scale):
    """fp64 参考：严格按 AscendC 内核顺序（exp(g) 衰减 -> delta -> 更新 -> 输出）。"""
    q = query.detach().cpu().double()
    k = key.detach().cpu().double()
    v = value.detach().cpu().double()
    b = beta.detach().cpu().double()
    gg = None if g is None else g.detach().cpu().double()
    gkk = None if gk is None else gk.detach().cpu().double()

    T, NK, DK = q.shape
    NV = v.shape[1]
    DV = v.shape[2]
    S = state.shape[0]
    fin = state.detach().cpu().double().clone()
    out = torch.zeros(T, NV, DV, dtype=torch.float64)

    for bi in range(len(cu) - 1):
        seq0, seq1 = int(cu[bi]), int(cu[bi + 1])
        if seq1 <= seq0:
            continue
        st = seq0 + (int(nacc[bi]) - 1 if nacc is not None else 0)
        slot = int(ssm[st])
        for h in range(NV):
            hk = h // (NV // NK)
            cur = fin[slot, h].clone()
            for t in range(seq0, seq1):
                if gg is not None:
                    cur = cur * torch.exp(gg[t, h])
                if gkk is not None:
                    cur = cur * torch.exp(gkk[t, h])[None, :]
                d = v[t, h] - cur @ k[t, hk]
                d = d * b[t, h]
                cur = cur + d[:, None] * k[t, hk][None, :]
                out[t, h] = cur @ (q[t, hk] * scale)
                fin[int(ssm[t]), h] = cur
    return out, fin


def run_case(name, T, B, NK, NV, DK, DV, state_dtype=torch.bfloat16,
             use_g=True, use_gk=True, use_nacc=False, scale=1.0, seed=0):
    torch.manual_seed(seed)
    dev = DEV

    def U(*shape, lo=0.2):
        return ((torch.rand(*shape, dtype=torch.float32, device=dev) * 2 - 1) * lo)

    query = U(T, NK, DK).to(torch.bfloat16)
    key = U(T, NK, DK).to(torch.bfloat16)
    value = U(T, NV, DV).to(torch.bfloat16)
    beta = U(T, NV).to(torch.bfloat16)

    seqlen = max(1, T // B)
    cu = torch.tensor([min(b * seqlen, T) for b in range(B)] + [T], dtype=torch.int32, device=dev)
    # 每个 batch 用独立的状态槽
    ssm = torch.arange(T, dtype=torch.int32, device=dev) * NV  # 槽号递增，避免跨 batch 冲突
    S = int(ssm.max().item()) + NV
    state = U(S, NV, DV, DK).to(state_dtype)

    g = (U(T, NV, lo=0.1) - 0.05).to(torch.float32) if use_g else None
    gk = (U(T, NV, DK, lo=0.1) - 0.05).to(torch.float32) if use_gk else None
    nacc = None
    if use_nacc:
        nacc = torch.clamp(torch.randint(1, seqlen + 1, (B,), dtype=torch.int32, device=dev), 1, seqlen)

    out, fin = recurrent_gated_delta_rule(query, key, value, beta, state, cu, ssm,
                                          g, gk, nacc, scale)
    torch.npu.synchronize()
    rout, rfin = ref(query, key, value, beta, state, cu.cpu(), ssm.cpu(), g, gk,
                     None if nacc is None else nacc.cpu(), scale)

    eo = (out.detach().cpu().double() - rout).abs().max().item()
    ef = (fin.detach().cpu().double() - rfin).abs().max().item()
    fo = FLOOR[value.dtype]
    ff = FLOOR[state_dtype]
    ok = eo <= fo and ef <= ff
    emit(f"  {name:46s} out_err={eo:.2e}(<= {fo:.0e}) state_err={ef:.2e}(<= {ff:.0e})  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 90)
emit("recurrent_gated_delta_rule: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 90)

allok = True
nexc = 0


def fp32_state_explicitly_rejected():
    """state=fp32 是**已声明不支持**的特化（后端编译器缺陷）：必须显式报错，绝不能静默算错。

    PASS 条件：抛 NotImplementedError（而不是 MLIRCompilationError、不是静默返回错误结果）。
    """
    global allok
    torch.manual_seed(0)
    T, B, NK, NV, DK, DV = 32, 2, 2, 2, 32, 32

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    query = U(T, NK, DK).to(torch.bfloat16)
    key = U(T, NK, DK).to(torch.bfloat16)
    value = U(T, NV, DV).to(torch.bfloat16)
    beta = U(T, NV).to(torch.bfloat16)
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=DEV)
    ssm = torch.arange(T, dtype=torch.int32, device=DEV) * NV
    state = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.float32)   # ← fp32
    g = U(T, NV, lo=0.1) - 0.05
    try:
        recurrent_gated_delta_rule(query, key, value, beta, state, cu, ssm, g)
    except NotImplementedError as e:
        emit(f"  {'state=fp32 → 显式 NotImplementedError（已声明不支持）':46s} "
             f"PASS：{str(e)[:46].replace(chr(10), ' ')}")
        return True
    except Exception as e:  # noqa: BLE001
        nexc += 1
        emit(f"  {'state=fp32 → 显式 NotImplementedError':46s} "
             f"抛出 {type(e).__name__}（期望 NotImplementedError）  FAIL")
        allok = False
        return False
    nexc += 1
    emit(f"  {'state=fp32 → 显式 NotImplementedError':46s} "
         f"未报错（静默算错，危险！）  FAIL")
    allok = False
    return False


fp32_state_explicitly_rejected()
case("T=32 B=2 NK=2 NV=2 K=32 V=32 g+gk bf16", 32, 2, 2, 2, 32, 32)
case("T=32 B=2 NK=2 NV=2 K=32 V=32 仅 g", 32, 2, 2, 2, 32, 32, use_gk=False)
case("T=32 B=2 NK=2 NV=2 K=32 V=32 无 g/gk", 32, 2, 2, 2, 32, 32,
     use_g=False, use_gk=False)
case("T=16 B=1 NK=1 NV=2 K=128 V=128 GQA scale=0.1", 16, 1, 1, 2, 128, 128,
     scale=0.1)
case("T=32 B=4 NK=2 NV=2 K=64 V=48 非2幂V", 32, 4, 2, 2, 64, 48)
case("T=32 B=2 NK=2 NV=2 K=32 V=32 num_accepted", 32, 2, 2, 2, 32, 32, use_nacc=True)
case("T=7  B=2 NK=2 NV=2 K=32 V=32 短序列", 7, 2, 2, 2, 32, 32)


# ---- 调用点风格：全关键字（vllm_ascend/ops/gdn.py L556-567 的传参形式，scale 走别名）----
def callsite_scale_alias():
    global allok
    torch.manual_seed(3)
    T, B, NK, NV, DK, DV = 32, 2, 2, 2, 32, 32

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    query = U(T, NK, DK).to(torch.bfloat16)
    key = U(T, NK, DK).to(torch.bfloat16)
    value = U(T, NV, DV).to(torch.bfloat16)
    beta = U(T, NV).to(torch.bfloat16)
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=DEV)
    ssm = torch.arange(T, dtype=torch.int32, device=DEV) * NV
    state = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.bfloat16)
    g = U(T, NV, lo=0.1) - 0.05
    sc = float(DK) ** -0.5
    out, fin = recurrent_gated_delta_rule(
        query, key, value,
        state=state, g=g, beta=beta, scale=sc,
        actual_seq_lengths=cu, ssm_state_indices=ssm,
    )
    torch.npu.synchronize()
    rout, rfin = ref(query, key, value, beta, state, cu.cpu(), ssm.cpu(), g, None, None, sc)
    eo = (out.detach().cpu().double() - rout).abs().max().item()
    ef = (fin.detach().cpu().double() - rfin).abs().max().item()
    ok = eo <= 4e-3 and ef <= 4e-3
    emit(f"  {'调用点风格：全关键字 + scale=K**-0.5':46s} out_err={eo:.2e} state_err={ef:.2e} "
         f"(<= 4e-03)  {'PASS' if ok else 'FAIL'}")
    return ok


try:
    allok &= callsite_scale_alias()
except Exception as e:  # noqa: BLE001
    nexc += 1
    emit(f"  调用点风格 scale EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    allok = False

emit()
emit(f"总体: {'全部 PASS ✅' if allok else f'存在 FAIL ❌ (其中 {nexc} 项为编译器异常)'}")

txt = "\n".join(L)
open("/root/dsh_work/recurrent_gated_delta_rule_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
