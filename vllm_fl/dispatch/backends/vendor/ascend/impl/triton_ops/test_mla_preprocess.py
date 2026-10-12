"""验证 mla_preprocess 的 Triton 实现 vs AscendC 语义（no_quant 路径，fp64 参考）。"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.mla_preprocess import mla_preprocess  # noqa: E402
from triton_ops.mla_preprocess_int8 import (  # noqa: E402
    _quant_asymm_kernel, _mm_i8_kernel, _dequant_kernel)

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def _rnd(x, dt):
    """fp64 -> 目标 dtype（CAST_RINT / round-to-nearest-even），再回 fp64。"""
    return x.to(dt).double()


def _rope_half(x, c, s, HALF):
    """out[i] = c[i]*x[i] - s[i]*x[i+HALF]; out[HALF+i] = s[HALF+i]*x[i] + c[HALF+i]*x[HALF+i]"""
    xa, xb = x[..., :HALF], x[..., HALF:]
    ca, cb = c[..., :HALF], c[..., HALF:]
    sa, sb = s[..., :HALF], s[..., HALF:]
    return torch.cat([ca * xa - sa * xb, sb * xa + cb * xb], dim=-1)


def ref(hs, wdqkv, g1, wuq, g2, wuk, cos, sin, slot, dt, Dq, Dkv, Dnope, Drd, HN, eps, cache_mode):
    Dkvr = Dkv + Drd
    HD = Dnope + Drd
    HALF = Drd // 2
    N = hs.shape[0]
    hs = hs.detach().cpu().double()
    wd = wdqkv.detach().cpu().double()
    g1 = g1.detach().cpu().double()
    wq = wuq.detach().cpu().double()
    g2 = g2.detach().cpu().double()
    wk = wuk.detach().cpu().double()
    cs = cos.detach().cpu().double()
    sn = sin.detach().cpu().double()
    sl = slot.detach().cpu().to(torch.int64)

    s3 = _rnd(hs @ wd.t(), dt)                     # [N, Dkvr+Dq]
    s3q = s3[:, Dkvr:]
    s3kv = s3[:, :Dkvr]

    ss = (s3q * s3q).sum(dim=1) / Dq
    q_lora = _rnd(s3q / torch.sqrt(ss + eps).unsqueeze(1) * g1, dt)
    s2 = _rnd(q_lora @ wq.t(), dt)                 # [N, HN*HD]
    s2v = s2.reshape(N, HN, HD)
    q_nope = _rnd(torch.einsum('thk,hkn->thn', s2v[:, :, :Dnope], wk), dt)
    q_rope = _rnd(_rope_half(s2v[:, :, Dnope:], cs.unsqueeze(1), sn.unsqueeze(1), HALF), dt)

    kvss = (s3kv[:, :Dkv] * s3kv[:, :Dkv]).sum(dim=1) / Dkv
    kv = _rnd(s3kv[:, :Dkv] / torch.sqrt(kvss + eps).unsqueeze(1) * g2, dt)
    krope = _rnd(_rope_half(s3kv[:, Dkv:], cs, sn, HALF), dt)
    return q_nope, q_rope, kv, krope, sl


def run_case(name, N, hidden, Dq, Dkv, Dnope, Drd, HN, dt, cache_mode, ns, seed=0, callsite=False):
    torch.manual_seed(seed)
    Dkvr = Dkv + Drd
    HD = Dnope + Drd
    dev = DEV

    hs = (torch.randn(N, hidden, dtype=torch.float32, device=dev) * 1.0).to(dt)
    wdqkv = (torch.randn(Dkvr + Dq, hidden, dtype=torch.float32, device=dev) / math.sqrt(hidden)).to(dt)
    g1 = (torch.rand(Dq, dtype=torch.float32, device=dev) * 0.5 + 0.5).to(dt)
    # 0.5 倍：把 q_nope / q_rope 的输出量级压到 <2，使「输出 dtype 量化误差」(半 ulp)
    # 远小于验收地板（bf16 4e-3 / fp16 1e-3），避免 bf16 半 ulp 顶到地板上。
    wuq = (torch.randn(HN * HD, Dq, dtype=torch.float32, device=dev) / math.sqrt(Dq) * 0.5).to(dt)
    g2 = (torch.rand(Dkv, dtype=torch.float32, device=dev) * 0.5 + 0.5).to(dt)
    wuk = (torch.randn(HN, Dnope, Dkv, dtype=torch.float32, device=dev) / math.sqrt(Dnope) * 0.5).to(dt)
    th = torch.rand(N, Drd, dtype=torch.float32, device=dev) * 6.28
    cos = torch.cos(th).to(dt)
    sin = torch.sin(th).to(dt)
    slot = torch.arange(N, dtype=torch.int32, device=dev) * 3 + 1

    split = (cache_mode == "krope_ctkv")
    qline = Dkv if split else (Dkv + Drd)
    q = torch.zeros((N, HN, qline), dtype=dt, device=dev)
    q2 = torch.zeros((N * HN, Drd), dtype=dt, device=dev)
    kvc0 = torch.zeros((ns, Dkv if split else Dkvr), dtype=dt, device=dev)
    kvc1 = torch.zeros((ns, Drd), dtype=dt, device=dev)

    if callsite:
        # 位置参数顺序 + 关键字参数名完全对齐 device_op.py:349（同一份输入）
        rq, rkvc0, rq2, rkvc1 = mla_preprocess(
            hs, wdqkv, None, g1, None, wuq, None, g2, cos, sin, wuk, None, None, slot,
            quant_scale0=None, quant_offset0=None, bias0=None,
            quant_scale1=None, quant_offset1=None, bias1=None,
            ctkv_scale=None, q_nope_scale=None,
            cache_mode=cache_mode, quant_mode="no_quant",
            q_out0=q, kv_cache_out0=kvc0, q_out1=q2, kv_cache_out1=kvc1,
            enable_inner_out=False,
            q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
            qk_rope_head_dim=Drd, head_num=HN)
    else:
        rq, rkvc0, rq2, rkvc1 = mla_preprocess(
            hs, wdqkv, gamma1=g1, wuq=wuq, gamma2=g2, wuk=wuk, cos=cos, sin=sin,
            slot_mapping=slot,
            q_out0=q, kv_cache_out0=kvc0, q_out1=q2, kv_cache_out1=kvc1,
            q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
            qk_rope_head_dim=Drd, head_num=HN, num_slots=ns, cache_mode=cache_mode)
    torch.npu.synchronize()

    rqn, rqr, rkv, rkr, rsl = ref(hs, wdqkv, g1, wuq, g2, wuk, cos, sin, slot, dt,
                                  Dq, Dkv, Dnope, Drd, HN, 1e-6, cache_mode)

    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dt]

    def mx(a, b):
        return (a.detach().cpu().double() - b).abs().max().item()

    # q 的 nope 部分
    e_nope = mx(rq[:, :, :Dkv], rqn)
    errs = [("q_nope", e_nope)]
    if split:
        e_rq = mx(rq2.reshape(N, HN, Drd), rqr)
        e_kv = mx(rkvc0[rsl], rkv)
        e_kr = mx(rkvc1[rsl], rkr)
    else:
        e_rq = mx(rq[:, :, Dkv:Dkv + Drd], rqr)
        e_kv = mx(rkvc0[rsl, :Dkv], rkv)
        e_kr = mx(rkvc0[rsl, Dkv:Dkvr], rkr)
    errs += [("q_rope", e_rq), ("kv_ctkv", e_kv), ("k_rope", e_kr)]

    ok = all(e <= floor for _, e in errs)
    txt = "  ".join(f"{k}={v:.3e}" for k, v in errs)
    emit(f"  {name:52s} {txt}  (<= {floor:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 108)
emit("mla_preprocess: Triton vs AscendC 语义（no_quant 路径，fp64 参考）")
emit("=" * 108)

allok = True
allok &= run_case("小配置 bf16 krope_ctkv N=6", 6, 128, 128, 64, 64, 32, 4, torch.bfloat16, "krope_ctkv", 64)
allok &= run_case("小配置 bf16 krope_ctkv N=5 (非整除)", 5, 128, 128, 64, 64, 32, 4, torch.bfloat16, "krope_ctkv", 64)
allok &= run_case("小配置 fp16 krope_ctkv N=7", 7, 128, 128, 64, 64, 32, 4, torch.float16, "krope_ctkv", 64)
allok &= run_case("小配置 bf16 kvcache    N=6", 6, 128, 128, 64, 64, 32, 4, torch.bfloat16, "kvcache", 64)
allok &= run_case("真实维度 bf16 krope_ctkv N=4 (Dq=1536,Dkv=512,HD=192,HN=128)",
                  4, 256, 1536, 512, 128, 64, 128, torch.bfloat16, "krope_ctkv", 32)
allok &= run_case("真实维度 fp16 kvcache    N=3",
                  3, 256, 1536, 512, 128, 64, 128, torch.float16, "kvcache", 32)

emit()
emit("--- 调用点风格（参数名/顺序对齐 device_op.py:349）与未实现参数显式报错 ---")


def case_callsite():
    """用真实调用点的位置参数顺序 + 关键字参数名调用（与上一个 N=6 用例**同一份输入**）。"""
    return run_case("callsite 位置参数顺序对齐 device_op.py:349", 6, 128, 128, 64, 64, 32, 4,
                    torch.bfloat16, "krope_ctkv", 64, callsite=True)


def case_reject():
    """未实现的量化参数必须显式 NotImplementedError，不能静默忽略。"""
    N, hidden, Dq, Dkv, Dnope, Drd, HN = 2, 64, 64, 64, 64, 32, 4
    dev, dt = DEV, torch.bfloat16
    hs = torch.zeros(N, hidden, device=dev).to(dt)
    s = torch.zeros(1, device=dev).to(dt)
    checks = [
        ("quant_scale0 非空", dict(quant_scale0=s)),
        ("bias0(int32) 非空", dict(bias0=torch.zeros(1, dtype=torch.int32, device=dev))),
        ("ctkv_scale 非空", dict(ctkv_scale=s)),
        ("q_nope_scale 非空", dict(q_nope_scale=s)),
        ("cache_mode=int8_nzcache 缺 ctkv_scale", dict(cache_mode="int8_nzcache")),
        ("quant_mode=per_token_quant_symm", dict(quant_mode="per_token_quant_symm")),
        ("enable_inner_out=True", dict(enable_inner_out=True)),
        ("deq_scale_qkv 非空", dict(deq_scale_qkv=torch.zeros(1, dtype=torch.float32, device=dev))),
        ("decode_k_nope 非空", dict(decode_k_nope=s)),
        ("beta1 非空", dict(beta1=s)),
    ]
    allok2 = True
    for nm, kw in checks:
        try:
            mla_preprocess(hs, torch.zeros(Dkv + Drd + Dq, hidden, device=dev).to(dt),
                           gamma1=s, wuq=s, gamma2=s, wuk=s, cos=s, sin=s,
                           slot_mapping=torch.zeros(N, dtype=torch.int32, device=dev),
                           q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
                           qk_rope_head_dim=Drd, head_num=HN, **kw)
            emit(f"  [拒绝检查] {nm:48s} 未报错 ❌ FAIL")
            allok2 = False
        except NotImplementedError as ex:
            first = str(ex).split("\n")[0][:70]
            emit(f"  [拒绝检查] {nm:48s} NotImplementedError ✅  {first}")
        except ValueError as ex:
            emit(f"  [拒绝检查] {nm:48s} ValueError ✅  {str(ex)[:60]}")
        except Exception as ex:  # noqa: BLE001
            emit(f"  [拒绝检查] {nm:48s} 抛了 {type(ex).__name__} ❌ {str(ex)[:60]}")
            allok2 = False
    return allok2


def _q_i8(x, s, o):
    """fp64 模拟 per_tensor_quant_asymm：saturate_i8(round_f16(x/s + o))。"""
    v = (x / s).to(torch.float16).double() + o
    return torch.clamp(torch.round(v), -128, 127)


def case_int8_primitives():
    """int8 三个原语：量化 int8 逐位一致 / W8A8 int32 逐位一致 / 反量化 ≤2e-6。"""
    dev = DEV
    ok = True
    M, D = 16, 256
    torch.manual_seed(3)
    x = torch.randn(M, D, device=dev) * 2.0
    s = torch.tensor([0.02], device=dev)
    o = torch.zeros(1, dtype=torch.int8, device=dev)
    y = torch.empty(M, D, dtype=torch.int8, device=dev)
    _quant_asymm_kernel[(1,)](x, y, s, o, M, D, BLK_M=16, BLK_D=256)
    ref = _q_i8(x.detach().cpu().double(), s.detach().cpu().double().item(), 0.0)
    nd = int((y.detach().cpu().to(torch.int16) != ref.to(torch.int16)).sum())
    emit(f"  [原语] 激活量化 int8            等值不一致数={nd}/{M*D}  {'✅' if nd == 0 else '❌'}")
    ok &= (nd == 0)
    M2, K2, N2 = 16, 256, 128
    torch.manual_seed(4)
    a = torch.randint(-128, 128, (M2, K2), dtype=torch.int8, device=dev)
    w = torch.randint(-128, 128, (N2, K2), dtype=torch.int8, device=dev)
    c = torch.empty(M2, N2, dtype=torch.int32, device=dev)
    _mm_i8_kernel[(1, 1)](a, w, c, M2, K2, N2, BLK_M=16, BLK_N=128, BLK_K=128)
    cr = (a.detach().cpu().double() @ w.detach().cpu().double().t())
    nd2 = int((c.detach().cpu().to(torch.int64) != cr.to(torch.int64)).sum())
    emit(f"  [原语] W8A8 int32 累加           等值不一致数={nd2}/{M2*N2}  {'✅' if nd2 == 0 else '❌'}")
    ok &= (nd2 == 0)
    bias = torch.randint(-1000, 1000, (N2,), dtype=torch.int32, device=dev)
    # descale 取小量级：让 |out| ~ 1，从而 fp32 输出的绝对误差可被手册地板 2e-6 直接判定
    dsc = torch.rand(N2, device=dev) * 1e-6
    out = torch.empty(M2, N2, dtype=torch.float32, device=dev)
    _dequant_kernel[(1, 1)](c, bias, dsc, out, M2, N2, BLK_M=16, BLK_N=128)
    dr = (c.detach().cpu().double() + bias.detach().cpu().double().unsqueeze(0)) \
        * dsc.detach().cpu().double().unsqueeze(0)
    e3 = (out.detach().cpu().double() - dr).abs().max().item()
    emit(f"  [原语] 反量化 (C+bias)*descale    err={e3:.3e}(<= 2e-06,手册绝对口径)  "
         f"{'✅' if e3 <= 2e-6 else '❌'}")
    ok &= (e3 <= 2e-6)
    return ok


def case_int8_e2e(name, N, hidden, Dq, Dkv, Dnope, Drd, HN, ns, cache_mode):
    """int8 量化端到端：per_tensor_quant_asymm + krope_ctkv/nzcache。"""
    dev, dt = DEV, torch.bfloat16
    Dkvr, HD = Dkv + Drd, Dnope + Drd
    HALF = Drd // 2
    torch.manual_seed(5)
    hs = torch.randn(N, hidden, device=dev).to(dt)
    wdqkv = torch.randint(-128, 128, (Dkvr + Dq, hidden), dtype=torch.int8, device=dev)
    wuq = torch.randint(-128, 128, (HN * HD, Dq), dtype=torch.int8, device=dev)
    gamma1 = (torch.rand(Dq, device=dev) * 0.5 + 0.5).to(dt)
    gamma2 = (torch.rand(Dkv, device=dev) * 0.5 + 0.5).to(dt)
    wuk = (torch.randn(HN, Dnope, Dkv, device=dev) / math.sqrt(Dnope) * 0.5).to(dt)
    th = torch.rand(N, Drd, device=dev) * 6.28
    cos, sin = torch.cos(th).to(dt), torch.sin(th).to(dt)
    slot = torch.arange(N, dtype=torch.int32, device=dev) * 3 + 1
    s0 = torch.tensor([0.05], device=dev).to(dt)
    s1q = torch.tensor([0.02], device=dev).to(dt)
    o0 = torch.zeros(1, dtype=torch.int8, device=dev)
    o1 = torch.zeros(1, dtype=torch.int8, device=dev)
    des0 = torch.rand(Dkvr + Dq, device=dev) * 0.002 + 0.001
    des1 = torch.rand(HN * HD, device=dev) * 0.002 + 0.001
    b0 = torch.zeros(Dkvr + Dq, dtype=torch.int32, device=dev)
    b1 = torch.zeros(HN * HD, dtype=torch.int32, device=dev)
    nz = (cache_mode == "nzcache")
    nout = (ns + 127) // 128
    kvc0 = (torch.zeros(nout * 128 * Dkv, dtype=dt, device=dev) if nz
            else torch.zeros((ns, Dkv), dtype=dt, device=dev))
    kvc1 = (torch.zeros(nout * 128 * Drd, dtype=dt, device=dev) if nz
            else torch.zeros((ns, Drd), dtype=dt, device=dev))
    q = torch.zeros((N, HN, Dkv), dtype=dt, device=dev)
    q2 = torch.zeros((N * HN, Drd), dtype=dt, device=dev)
    r = mla_preprocess(
        hs, wdqkv, des0, gamma1, None, wuq, des1, gamma2, cos, sin, wuk, None, None, slot,
        quant_scale0=s0, quant_offset0=o0, bias0=b0,
        quant_scale1=s1q, quant_offset1=o1, bias1=b1,
        cache_mode=cache_mode, quant_mode="per_tensor_quant_asymm",
        q_out0=q, kv_cache_out0=kvc0, q_out1=q2, kv_cache_out1=kvc1,
        q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
        qk_rope_head_dim=Drd, head_num=HN)
    torch.npu.synchronize()

    hd = hs.detach().cpu().double()
    wd = wdqkv.detach().cpu().double()
    wq = wuq.detach().cpu().double()
    g1 = gamma1.detach().cpu().double()
    g2 = gamma2.detach().cpu().double()
    wk = wuk.detach().cpu().double()
    csd, snd = cos.detach().cpu().double(), sin.detach().cpu().double()
    s1i = _q_i8(hd, s0.detach().cpu().double().item(), 0.0)
    c1 = torch.round(s1i @ wd.t())
    s3 = ((c1 + b0.detach().cpu().double().unsqueeze(0))
          * des0.detach().cpu().double().unsqueeze(0)).to(dt).double()
    ql = s3[:, Dkvr:]
    ql = (ql / torch.sqrt((ql * ql).sum(-1, keepdim=True) / Dq + 1e-6) * g1).to(dt).double()
    s1bi = _q_i8(ql, s1q.detach().cpu().double().item(), 0.0)
    c2 = torch.round(s1bi @ wq.t())
    s4 = ((c2 + b1.detach().cpu().double().unsqueeze(0))
          * des1.detach().cpu().double().unsqueeze(0)).to(dt).double()
    s4v = s4.reshape(N, HN, HD)
    qn = torch.einsum('thk,hkn->thn', s4v[:, :, :Dnope], wk).to(dt).double()
    qr = _rope_half(s4v[:, :, Dnope:], csd.unsqueeze(1), snd.unsqueeze(1), HALF).to(dt).double()
    kv = s3[:, :Dkv]
    kv = (kv / torch.sqrt((kv * kv).sum(-1, keepdim=True) / Dkv + 1e-6) * g2).to(dt).double()
    kr = _rope_half(s3[:, Dkv:Dkvr], csd, snd, HALF).to(dt).double()

    def mx(a_, b_):
        return (a_.detach().cpu().double() - b_).abs().max().item()

    el = [("q_nope", mx(r[0][:, :, :Dkv], qn)), ("q_rope", mx(r[2].reshape(N, HN, Drd), qr))]
    if nz:
        def nz_off(L):
            n = torch.arange(L)
            return (n // 16) * 2048 + (n % 16)
        ref_kv = torch.zeros(nout * 128 * Dkv, dtype=torch.float64)
        ref_kr = torch.zeros(nout * 128 * Drd, dtype=torch.float64)
        for t in range(N):
            sl = int(slot[t]); ou, inn = sl // 128, sl % 128
            ref_kv[ou * 128 * Dkv + inn * 16 + nz_off(Dkv)] = kv[t]
            ref_kr[ou * 128 * Drd + inn * 16 + nz_off(Drd)] = kr[t]
        el += [("kv_nz", mx(r[1].reshape(-1), ref_kv)),
               ("krope_nz", mx(r[3].reshape(-1), ref_kr))]
    else:
        el += [("kv_ctkv", mx(r[1][slot], kv)), ("k_rope", mx(r[3][slot], kr))]
    floor = 4e-3
    ok = all(e <= floor for _, e in el)
    txt = "  ".join(f"{k}={v:.3e}" for k, v in el)
    emit(f"  {name:52s} {txt}  (<= {floor:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


allok &= case_callsite()
allok &= case_reject()

emit()
emit("--- int8 per_tensor_quant_asymm（量化原语 + 端到端） ---")
allok &= case_int8_primitives()
allok &= case_int8_e2e("int8 asymm krope_ctkv N=6 (Dq=128,Dkv=64)", 6, 128, 128, 64, 64, 32, 4, 64,
                       "krope_ctkv")
allok &= case_int8_e2e("int8 asymm nzcache    N=6 (NZ 分块写回)", 6, 128, 128, 64, 64, 32, 4, 64,
                       "nzcache")


def case_int8_nzcache(name, N=6, hidden=128, Dq=128, Dkv=64, Dnope=64, Drd=32, HN=4, ns=64):
    """int8_nzcache：kv 段用 ctkv_scale 量化成 int8(C0=32 NZ) + q_nope 用 per-head
    q_nope_scale 量化成 int8(EinSumQuant)。int8 结果要求逐位一致。"""
    dev, dt = DEV, torch.bfloat16
    Dkvr, HD, HALF = Dkv + Drd, Dnope + Drd, Drd // 2
    torch.manual_seed(6)
    hs = torch.randn(N, hidden, device=dev).to(dt)
    wdqkv = torch.randint(-128, 128, (Dkvr + Dq, hidden), dtype=torch.int8, device=dev)
    wuq = torch.randint(-128, 128, (HN * HD, Dq), dtype=torch.int8, device=dev)
    gamma1 = (torch.rand(Dq, device=dev) * 0.5 + 0.5).to(dt)
    gamma2 = (torch.rand(Dkv, device=dev) * 0.5 + 0.5).to(dt)
    wuk = (torch.randn(HN, Dnope, Dkv, device=dev) / math.sqrt(Dnope) * 0.5).to(dt)
    th = torch.rand(N, Drd, device=dev) * 6.28
    cos, sin = torch.cos(th).to(dt), torch.sin(th).to(dt)
    slot = torch.arange(N, dtype=torch.int32, device=dev) * 3 + 1
    s0 = torch.tensor([0.05], device=dev).to(dt)
    s1q = torch.tensor([0.02], device=dev).to(dt)
    o0 = torch.zeros(1, dtype=torch.int8, device=dev)
    o1 = torch.zeros(1, dtype=torch.int8, device=dev)
    des0 = torch.rand(Dkvr + Dq, device=dev) * 0.002 + 0.001
    des1 = torch.rand(HN * HD, device=dev) * 0.002 + 0.001
    b0 = torch.zeros(Dkvr + Dq, dtype=torch.int32, device=dev)
    b1 = torch.zeros(HN * HD, dtype=torch.int32, device=dev)
    ctkv = torch.tensor([0.01], device=dev).to(dt)
    qns = (torch.rand(HN, device=dev) * 0.02 + 0.01).to(dt)
    nout = (ns + 127) // 128
    kvc0 = torch.zeros(nout * 128 * Dkv, dtype=torch.int8, device=dev)
    kvc1 = torch.zeros(nout * 128 * Drd, dtype=dt, device=dev)
    q = torch.zeros((N, HN, Dkv), dtype=torch.int8, device=dev)
    q2 = torch.zeros((N * HN, Drd), dtype=dt, device=dev)
    r = mla_preprocess(
        hs, wdqkv, des0, gamma1, None, wuq, des1, gamma2, cos, sin, wuk, None, None, slot,
        quant_scale0=s0, quant_offset0=o0, bias0=b0,
        quant_scale1=s1q, quant_offset1=o1, bias1=b1,
        ctkv_scale=ctkv, q_nope_scale=qns,
        cache_mode="int8_nzcache", quant_mode="per_tensor_quant_asymm",
        q_out0=q, kv_cache_out0=kvc0, q_out1=q2, kv_cache_out1=kvc1,
        q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
        qk_rope_head_dim=Drd, head_num=HN)
    # inner_out：NaN 哨兵初值 → 必须被写入，且与参考 q_lora 逐位一致
    io = torch.full((N, Dq), float("nan"), dtype=dt, device=dev)
    mla_preprocess(hs, wdqkv, des0, gamma1, None, wuq, des1, gamma2, cos, sin, wuk, None, None,
                   slot, quant_scale0=s0, quant_offset0=o0, bias0=b0, quant_scale1=s1q,
                   quant_offset1=o1, bias1=b1, ctkv_scale=ctkv, q_nope_scale=qns,
                   cache_mode="int8_nzcache", quant_mode="per_tensor_quant_asymm",
                   q_out0=q, kv_cache_out0=kvc0, q_out1=q2, kv_cache_out1=kvc1,
                   q_lora_rank=Dq, kv_lora_rank=Dkv, qk_nope_head_dim=Dnope,
                   qk_rope_head_dim=Drd, head_num=HN,
                   enable_inner_out=True, inner_out=io)
    torch.npu.synchronize()

    hd = hs.detach().cpu().double()
    wd, wq_ = wdqkv.detach().cpu().double(), wuq.detach().cpu().double()
    g1 = gamma1.detach().cpu().double()
    g2 = gamma2.detach().cpu().double()
    wk = wuk.detach().cpu().double()
    csd, snd = cos.detach().cpu().double(), sin.detach().cpu().double()
    cti = ctkv.detach().cpu().double().item()
    qni = qns.detach().cpu().double()

    def sat8(x):
        return torch.clamp(torch.round(x.to(torch.float16).double()), -128, 127)

    s1i = _q_i8(hd, s0.detach().cpu().double().item(), 0.0)
    c1 = torch.round(s1i @ wd.t())
    s3 = ((c1 + b0.detach().cpu().double().unsqueeze(0))
          * des0.detach().cpu().double().unsqueeze(0)).to(dt).double()
    ql = s3[:, Dkvr:]
    ql = (ql / torch.sqrt((ql * ql).sum(-1, keepdim=True) / Dq + 1e-6) * g1).to(dt).double()
    s1bi = _q_i8(ql, s1q.detach().cpu().double().item(), 0.0)
    c2 = torch.round(s1bi @ wq_.t())
    s4 = ((c2 + b1.detach().cpu().double().unsqueeze(0))
          * des1.detach().cpu().double().unsqueeze(0)).to(dt).double()
    s4v = s4.reshape(N, HN, HD)
    mm3 = torch.einsum('thk,hkn->thn', s4v[:, :, :Dnope], wk).to(dt).double()
    qn8 = sat8(mm3 * qni[None, :, None])                       # per-head 乘性 scale
    kv = s3[:, :Dkv]
    # ⚠️ int8 路径下内核把 **fp32** 的 rmsnorm 结果直接量化（不写回 bf16）⇒ 参考不能 .to(dt)，
    #    否则引入 0.4% 的 bf16 舍入，远大于 fp16 步长，会翻转 ~5% 的 int8 值。
    kv = kv / torch.sqrt((kv * kv).sum(-1, keepdim=True) / Dkv + 1e-6) * g2
    kv8 = sat8(kv * (1.0 / cti))                               # ctkv_scale 取倒数
    kr = _rope_half(s3[:, Dkv:Dkvr], csd, snd, HALF).to(dt).double()

    n_io = int((~torch.isnan(io)).sum().item())
    e_io = (io.detach().cpu().double() - ql).abs().max().item()
    ok_io = (n_io == N * Dq) and (e_io == 0.0)
    emit(f"  {'[inner_out] NaN初值→被写入 & 与参考 q_lora 逐位一致':50s} "
         f"written={n_io}/{N*Dq} err={e_io:.3e}  {'✅' if ok_io else '❌'}")
    n_q = int((r[0].detach().cpu().to(torch.int16) - qn8.to(torch.int16)).abs().sum().item())
    ref8 = torch.zeros(nout * 128 * Dkv, dtype=torch.int64)
    ref_kr = torch.zeros(nout * 128 * Drd, dtype=torch.float64)
    mask8 = torch.zeros(nout * 128 * Dkv, dtype=torch.bool)
    for t in range(N):
        sl = int(slot[t]); ou, inn = sl // 128, sl % 128
        n = torch.arange(Dkv)
        offi = ou * 128 * Dkv + (n // 32) * 4096 + inn * 32 + (n % 32)
        ref8[offi] = kv8[t].to(torch.int64)
        mask8[offi] = True
        n2 = torch.arange(Drd)
        ref_kr[ou * 128 * Drd + inn * 16 + (n2 // 16) * 2048 + (n2 % 16)] = kr[t]
    got8 = r[1].detach().cpu().to(torch.int64)
    diff = (got8 - ref8).abs()[mask8]
    n_exact = int((diff == 0).sum().item())
    n_tot = int(mask8.sum().item())
    maxd = int(diff.max().item()) if n_tot else 0
    # ---- 判定翻转是否 =「平局点两侧」而非「舍入规则不同」----
    # (1) 双方都是 round-half-even：kernel 用 (r>0.5?+1 : r<0.5?0 : +odd) 的 RNE 模拟；
    #     参考用 torch.round（round-half-to-even）。规则一致。
    # (2) 若翻转只发生在「量化前值落在 .5 平局点、fp32 与 fp64 分居两侧」，则翻转点的
    #     |frac-0.5| 应为 1e-7 量级；若规则不同，翻转会散布在任意 frac 上。
    bad = (diff != 0).nonzero().flatten()
    frac_dev = -1.0
    if bad.numel() > 0:
        pre16 = (kv * (1.0 / cti)).to(torch.float16).double()
        fr = pre16 - torch.floor(pre16)
        frac_dev = (fr[(bad // Dkv), (bad % Dkv)] - 0.5).abs().max().item()
    n_kv = n_tot - n_exact
    e_kr = (r[3].detach().cpu().double() - ref_kr).abs().max().item()
    rint_ok = (maxd <= 1) and (frac_dev < 0 or frac_dev < 1e-5)
    ok = (n_q == 0) and rint_ok and (e_kr <= 4e-3) and ok_io
    emit(f"  {name:52s} q_nope_i8逐位一致={N*HN*Dkv-n_q}/{N*HN*Dkv} "
         f"kv_i8逐位一致={n_exact}/{n_tot}(差{n_kv},最大{maxd}LSB,"
         f"翻转点|frac-0.5|max={frac_dev:.2e}→RINT一致) "
         f"krope_err={e_kr:.3e}(<=4e-03)  {'PASS' if ok else 'FAIL'}")
    return ok


emit()
emit("--- int8_nzcache（ctkv_scale int8 KV + q_nope_scale EinSumQuant） ---")
allok &= case_int8_nzcache("int8_nzcache N=6 (C0=32 KV int8 + per-head q scale)")

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
try:
    open("/root/dsh_work/mla_preprocess_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
