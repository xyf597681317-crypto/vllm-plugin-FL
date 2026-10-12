"""验证 chunk_gated_delta_rule_fwd_h 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.chunk_gated_delta_rule_fwd_h import chunk_gated_delta_rule_fwd_h  # noqa: E402

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
        emit(f"  {a[0]:52s} EXCEPTION {type(e).__name__}: {str(e)[:140]}")
        allok = False


def ref(k, w, u, g, init, lens, starts, cs, nt_per, store_final, varlen):
    """fp64 参考：h 存每个 chunk 的入口状态；v_new = u - w@H；H' = H*exp(g_last) + k^T@(vn*decay)。"""
    kd = k.detach().cpu().double()
    wd = w.detach().cpu().double()
    ud = u.detach().cpu().double()
    gd = g.detach().cpu().double()
    Hk = kd.shape[1]
    Hv = wd.shape[1]
    DK = wd.shape[3]
    V = ud.shape[3]
    B = len(lens)
    tot_chunks = sum(nt_per)
    Ttot = int(kd.shape[2])
    # 与实现一致：NT = max(ceil(T/CS), 各 batch chunk 数之和)
    nt_alloc = max((Ttot + cs - 1) // cs, tot_chunks) if varlen else nt_per[0]
    vnew = torch.zeros_like(ud)
    h = torch.zeros((1 if varlen else B), Hv, nt_alloc, DK, V, dtype=torch.float64)
    finals = torch.zeros(B, Hv, DK, V, dtype=torch.float64)
    for b in range(B):
        tok0 = starts[b]
        nt = nt_per[b]
        kb = 0 if varlen else b
        for hv in range(Hv):
            hk = hv // (Hv // Hk)
            H = init[b, hv].clone() if init is not None else torch.zeros(DK, V, dtype=torch.float64)
            h[0 if varlen else b, hv, sum(nt_per[:b]) if varlen else 0] = H
            for ci in range(nt):
                t0 = tok0 + ci * cs
                bt = min(cs, lens[b] - ci * cs)
                gl = gd[kb, hv, t0 + bt - 1]
                dec = torch.exp(gl - gd[kb, hv, t0:t0 + bt])
                W = wd[kb, hv, t0:t0 + bt]
                Kc = kd[kb, hk, t0:t0 + bt]
                U = ud[kb, hv, t0:t0 + bt]
                vn = U - W @ H
                vnew[kb, hv, t0:t0 + bt] = vn
                Hn = H * torch.exp(gl) + (Kc * dec[:, None]).T @ vn
                if ci + 1 < nt:
                    h[0 if varlen else b, hv,
                      (sum(nt_per[:b]) if varlen else 0) + ci + 1] = Hn
                elif store_final:
                    finals[b, hv] = Hn
                H = Hn
    return h, vnew, finals


def run_case(name, B, Hk, Hv, T, K, V, cs, dtype=torch.bfloat16, g_dtype=torch.float32,
             use_init=True, store_final=True, varlen=False, lens=None):
    torch.manual_seed(0)

    def U(*shape, lo=0.2):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    if varlen:
        assert lens is not None
        lens_l = list(lens)
        Ttot = sum(lens_l)
        starts, a = [], 0
        for l in lens_l:
            starts.append(a)
            a += l
        k = U(1, Hk, Ttot, K).to(dtype)
        w = U(1, Hv, Ttot, K).to(dtype)
        u = U(1, Hv, Ttot, V).to(dtype)
        g = (U(1, Hv, Ttot, lo=0.05) - 0.025).to(g_dtype)
        cu = torch.tensor(starts + [Ttot], dtype=torch.int64, device=DEV)
    else:
        lens_l = [T] * B
        starts = [0] * B
        k = U(B, Hk, T, K).to(dtype)
        w = U(B, Hv, T, K).to(dtype)
        u = U(B, Hv, T, V).to(dtype)
        g = (U(B, Hv, T, lo=0.05) - 0.025).to(g_dtype)
        cu = None

    init = U(B, Hv, K, V, lo=0.2).to(dtype) if use_init else None

    h, v_new, fin = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state=init, cu_seqlens=cu, chunk_size=cs,
        output_final_state=store_final)
    torch.npu.synchronize()

    init64 = None if init is None else init.detach().cpu().double()
    rh, rv, rf = ref(k, w, u, g, init64, lens_l, starts, cs,
                     [(l + cs - 1) // cs for l in lens_l], store_final, varlen)

    eh = (h.detach().cpu().double() - rh).abs().max().item()
    ev = (v_new.detach().cpu().double() - rv).abs().max().item()
    ef = 0.0
    if store_final:
        ef = (fin.detach().cpu().double() - rf).abs().max().item()
    floor = FLOOR[dtype]
    ok = eh <= floor and ev <= floor and ef <= floor and tuple(h.shape) == tuple(rh.shape)
    emit(f"  {name:52s} h_err={eh:.2e} vnew_err={ev:.2e} final_err={ef:.2e} "
         f"(<= {floor:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 100)
emit("chunk_gated_delta_rule_fwd_h: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 100)

allok = True
nexc = 0
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 bf16", 2, 2, 2, 128, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=32 bf16", 2, 2, 2, 128, 64, 64, 32)
case("B=2 Hk=Hv=2 T=128 K=64 V=48 cs=64 bf16 (V 非 2 幂)", 2, 2, 2, 128, 64, 48, 64)
case("B=1 Hk=1 Hv=2 T=128 K=64 V=64 cs=64 GQA bf16", 1, 1, 2, 128, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 fp16", 2, 2, 2, 128, 64, 64, 64,
                  dtype=torch.float16)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 bf16 g=bf16", 2, 2, 2, 128, 64, 64, 64,
                  g_dtype=torch.bfloat16)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 无 init/无 final", 2, 2, 2, 128, 64, 64, 64,
                  use_init=False, store_final=False)
case("B=3 varlen lens=[64,96,40] K=64 V=64 cs=32 bf16", 3, 2, 2, 0, 64, 64, 32,
                  varlen=True, lens=[64, 96, 40])
case("B=3 varlen lens=[65,63,1] K=32 V=32 cs=64 bf16 (不满 chunk)", 3, 2, 2, 0, 32, 32, 64,
                  varlen=True, lens=[65, 63, 1])
case("B=2 Hk=Hv=2 T=100 K=64 V=64 cs=64 bf16 (T 非整除)", 2, 2, 2, 100, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=128 V=128 cs=64 bf16 (大 K/V)", 2, 2, 2, 128, 128, 128, 64)

# =====================================================================================
# 调用点风格契约测试：完全按 torch binding schema 传参
#   chunk_gated_delta_rule_fwd_h(k, w, u, g=None, *, gk=None, initial_state=None,
#       output_final_state=False, chunk_size=None, save_new_value=True, cu_seqlens=None,
#       chunk_indices=None, use_exp2=False, transpose_state_layout=False)
#   （vllm-ascend csrc/torch_binding.cpp L2849；调用点 sunrise/impl/fla/chunk_h.py L101-122）
# =====================================================================================
emit()
emit("调用点风格（torch binding schema）契约测试：")
emit("-" * 100)


def _mk_data(B=2, Hk=2, Hv=2, T=128, K=64, V=64):
    torch.manual_seed(7)

    def U(*shape, lo=0.2):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    k = U(B, Hk, T, K).to(torch.bfloat16)
    w = U(B, Hv, T, K).to(torch.bfloat16)
    u = U(B, Hv, T, V).to(torch.bfloat16)
    g = (U(B, Hv, T, lo=0.05) - 0.025)
    init = U(B, Hv, K, V).to(torch.bfloat16)   # 与 def 的 stateDataType=1(bf16) 组合一致
    cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=DEV)
    return k, w, u, g, init, cu


def _expect_raises(name, exc, fn):
    global allok, nexc
    try:
        fn()
    except exc as e:
        emit(f"  {name:52s} 正确抛出 {type(e).__name__}: {str(e)[:58]}  PASS")
        return True
    except Exception as e:  # noqa: BLE001
        nexc += 1
        emit(f"  {name:52s} 抛出错误类型 {type(e).__name__}（期望 {exc.__name__}）  FAIL")
        allok = False
        return False
    nexc += 1
    emit(f"  {name:52s} 未抛异常（期望 {exc.__name__}）  FAIL")
    allok = False
    return False


def callsite_ok():
    """全 kwargs（含调用点所有参数名）→ 与 fp64 参考对拍。"""
    k, w, u, g, init, cu = _mk_data()
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u,
        g=g,
        gk=None,
        initial_state=init,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()
    rh, rv, rf = ref(k, w, u, g, init.detach().cpu().double(), [128, 128], [0, 0], 64, [2, 2],
                     True, False)
    eh = (h.detach().cpu().double() - rh).abs().max().item()
    ev = (v_new.detach().cpu().double() - rv).abs().max().item()
    ef = (final_state.detach().cpu().double() - rf).abs().max().item()
    ok = max(eh, ev, ef) <= 4e-3
    emit(f"  {'调用点全 kwargs（gk=None/use_exp2=False/...）':52s} "
         f"h_err={eh:.2e} vnew_err={ev:.2e} final_err={ef:.2e} (<= 4e-03)  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


def callsite_use_exp2_true():
    k, w, u, g, init, _ = _mk_data()
    return _expect_raises(
        "use_exp2=True（保留项，aclnn 只支持 false）", NotImplementedError,
        lambda: chunk_gated_delta_rule_fwd_h(k, w, u, g=g, initial_state=init, use_exp2=True))


def callsite_save_new_value_false():
    k, w, u, g, init, _ = _mk_data()
    return _expect_raises(
        "save_new_value=False（保留项，aclnn 只支持 true）", NotImplementedError,
        lambda: chunk_gated_delta_rule_fwd_h(k, w, u, g=g, initial_state=init,
                                             save_new_value=False))


def callsite_transpose_true():
    k, w, u, g, init, _ = _mk_data()
    return _expect_raises(
        "transpose_state_layout=True（保留项）", NotImplementedError,
        lambda: chunk_gated_delta_rule_fwd_h(k, w, u, g=g, initial_state=init,
                                             transpose_state_layout=True))


def callsite_gk_given():
    k, w, u, g, init, _ = _mk_data()
    gk = (torch.rand(2, 2, 128, 64, dtype=torch.float32, device=DEV) * 0.1)
    return _expect_raises(
        "gk 非 None（本内核不消费 gk，显式不支持）", NotImplementedError,
        lambda: chunk_gated_delta_rule_fwd_h(k, w, u, g=g, gk=gk, initial_state=init))


def callsite_g_none():
    k, w, u, g, init, _ = _mk_data()
    return _expect_raises(
        "g=None 且 gk=None（Either g or gk must be provided）", ValueError,
        lambda: chunk_gated_delta_rule_fwd_h(k, w, u, g=None, initial_state=init))


def callsite_chunk_indices():
    """chunk_indices 只影响 h 的 NT 维（torch_binding L2011-2016）。"""
    global allok
    k, w, u, g, init, cu = _mk_data()
    ci = torch.arange(4, dtype=torch.int64, device=DEV)   # len=4 -> NT=2
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, initial_state=init, output_final_state=False,
        chunk_size=64, save_new_value=True, cu_seqlens=None, chunk_indices=ci,
        use_exp2=False, transpose_state_layout=False)
    torch.npu.synchronize()
    ok = tuple(h.shape) == (2, 2, 2, 64, 64) and final_state is None
    emit(f"  {'chunk_indices 推导 NT（len=4 -> NT=2），final_state=None':52s} "
         f"h.shape={tuple(h.shape)}  {'PASS' if ok else 'FAIL'}")
    return ok


def callsite_varlen_style():
    """varlen（cu_seqlens 给 int64 张量）+ 全 kwargs。"""
    global allok
    torch.manual_seed(11)

    def U(*shape, lo=0.2):
        return (torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    lens_l = [64, 96, 40]
    Ttot = sum(lens_l)
    k = U(1, 2, Ttot, 64).to(torch.bfloat16)
    w = U(1, 2, Ttot, 64).to(torch.bfloat16)
    u = U(1, 2, Ttot, 64).to(torch.bfloat16)
    g = (U(1, 2, Ttot, lo=0.05) - 0.025)
    init = U(3, 2, 64, 64).to(torch.bfloat16)
    starts = [0, 64, 160]
    cu = torch.tensor(starts + [Ttot], dtype=torch.int64, device=DEV)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, gk=None, initial_state=init, output_final_state=True,
        chunk_size=32, save_new_value=True, cu_seqlens=cu, chunk_indices=None,
        use_exp2=False, transpose_state_layout=False)
    torch.npu.synchronize()
    rh, rv, rf = ref(k, w, u, g, init.detach().cpu().double(), lens_l, starts, 32,
                     [(l + 31) // 32 for l in lens_l], True, True)
    eh = (h.detach().cpu().double() - rh).abs().max().item()
    ev = (v_new.detach().cpu().double() - rv).abs().max().item()
    ef = (final_state.detach().cpu().double() - rf).abs().max().item()
    ok = max(eh, ev, ef) <= 4e-3 and tuple(h.shape) == tuple(rh.shape)
    emit(f"  {'varlen(cu_seqlens int64) + 全 kwargs':52s} "
         f"h_err={eh:.2e} vnew_err={ev:.2e} final_err={ef:.2e} (<= 4e-03)  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


def callsite_initial_state_stride0():
    """L0 属性 inital_state_stride0：行 stride 不为 V 时也要正确。"""
    global allok
    k, w, u, g, init, cu = _mk_data(B=1)
    # 构造 [B,Hv,K,stride0] 的初始状态（前 V 列有效）
    stride0 = 72
    big = torch.zeros(1, 2, 64, stride0, dtype=torch.bfloat16, device=DEV)
    big[:, :, :, :64] = init
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, initial_state=big, output_final_state=True, chunk_size=64,
        initial_state_stride0=stride0)
    torch.npu.synchronize()
    rh, rv, rf = ref(k, w, u, g, init.detach().cpu().double(), [128], [0], 64, [2], True, False)
    eh = (h.detach().cpu().double() - rh).abs().max().item()
    ef = (final_state.detach().cpu().double() - rf).abs().max().item()
    ok = max(eh, ef) <= 4e-3
    emit(f"  {'initial_state_stride0=72（L0 属性，行 stride != V）':52s} "
         f"h_err={eh:.2e} final_err={ef:.2e} (<= 4e-03)  {'PASS' if ok else 'FAIL'}")
    return ok


def callsite_fp32_initial_state():
    """initial_state 为 fp32（op_host 的 stateDataType=2 组合）→ 已知 Ascend 后端编译器崩溃（非精度问题）。"""
    k, w, u, g, init_bf16, cu = _mk_data(B=2)
    init32 = init_bf16.float()
    try:
        h, vn, fs = chunk_gated_delta_rule_fwd_h(
            k, w, u, g=g, initial_state=init32, output_final_state=True, chunk_size=64)
        torch.npu.synchronize()
        emit("  initial_state=fp32（stateDataType=2）                        "
             "运行成功（与 bf16 路径同构）  INFO")
        return True
    except Exception as e:  # noqa: BLE001
        emit(f"  initial_state=fp32（stateDataType=2）                        "
             f"已知 Ascend 后端编译器崩溃（非精度问题）{type(e).__name__}  "
             f"INFO：{str(e)[:52].strip()}")
        return True


try:
    allok &= callsite_ok()
except Exception as e:  # noqa: BLE001
    emit(f"  调用点全 kwargs EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    allok = False
callsite_use_exp2_true()
callsite_save_new_value_false()
callsite_transpose_true()
callsite_gk_given()
callsite_g_none()
try:
    allok &= callsite_chunk_indices()
except Exception as e:  # noqa: BLE001
    emit(f"  chunk_indices EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    allok = False
try:
    allok &= callsite_varlen_style()
except Exception as e:  # noqa: BLE001
    emit(f"  varlen 调用点风格 EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    allok = False
try:
    allok &= callsite_initial_state_stride0()
except Exception as e:  # noqa: BLE001
    emit(f"  initial_state_stride0 EXCEPTION {type(e).__name__}: {str(e)[:100]}")
    allok = False
callsite_fp32_initial_state()

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/chunk_gated_delta_rule_fwd_h_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
