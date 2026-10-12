"""验证 chunk_fwd_o 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.chunk_fwd_o import chunk_fwd_o  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def ref(q, k, v, h, g, scale, cs, pairs, starts, lens, h_stride, varlen):
    """fp64 参考：o = scale*(exp(g)⊙(q@H) + (mask⊙exp(g_i-g_j)⊙(q@kᵀ))@v)，逐 chunk。"""
    qd = q.detach().cpu().double()
    kd = k.detach().cpu().double()
    vd = v.detach().cpu().double()
    hd = h.detach().cpu().double()
    gd = g.detach().cpu().double()
    B, Hk, T, K = qd.shape
    Hv, V = vd.shape[1], vd.shape[3]
    o = torch.zeros(B, Hv, T, V, dtype=torch.float64)
    for c, (bi, ci) in enumerate(pairs):
        t0 = starts[bi] + ci * cs
        bt = min(cs, lens[bi] - ci * cs)
        if bt <= 0:
            continue
        hb = 0 if varlen else bi
        ab = 0 if varlen else bi        # varlen 时 q/k/v/g 的 batch 维恒为 0
        for hv in range(Hv):
            hk = hv // (Hv // Hk)
            Qc = qd[ab, hk, t0:t0 + bt]
            Kc = kd[ab, hk, t0:t0 + bt]
            Vc = vd[ab, hv, t0:t0 + bt]
            gc = gd[ab, hv, t0:t0 + bt]
            Hc = hd[hb, hv, ci if not varlen else c]
            idx = torch.arange(bt)
            causal = idx[:, None] >= idx[None, :]
            dg = gc[:, None] - gc[None, :]
            A = (Qc @ Kc.T) * torch.exp(torch.where(causal, dg, torch.zeros_like(dg)))
            A = torch.where(causal, A, torch.zeros_like(A))
            o[ab, hv, t0:t0 + bt] = scale * (torch.exp(gc)[:, None] * (Qc @ Hc) + A @ Vc)
    return o


def run_case(name, B, Hk, Hv, T, K, V, cs, dtype=torch.bfloat16, scale=1.0,
             varlen=False, lens=None, give_chunk_indices=False, seed=0):
    torch.manual_seed(seed)

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    if varlen:
        lens_l = list(lens)
        Ttot = sum(lens_l)
        q = U(1, Hk, Ttot, K).to(dtype)
        k = U(1, Hk, Ttot, K).to(dtype)
        v = U(1, Hv, Ttot, V).to(dtype)
        g = (U(1, Hv, Ttot, lo=0.2) - 0.1)
        starts, acc = [], 0
        for l in lens_l:
            starts.append(acc)
            acc += l
        nt_per = [(l + cs - 1) // cs for l in lens_l]
        tot_chunks = sum(nt_per)
        cu = torch.tensor(starts + [Ttot], dtype=torch.int64, device=DEV)
        h = U(1, Hv, tot_chunks, K, V).to(dtype)
        pairs = []
        for bi in range(len(lens_l)):
            for c in range(nt_per[bi]):
                pairs.append((bi, c))
        ci = None
        if give_chunk_indices:
            flat = [x for p in pairs for x in p]
            ci = torch.tensor(flat, dtype=torch.int64, device=DEV)
        h_stride = tot_chunks
    else:
        lens_l = [T] * B
        q = U(B, Hk, T, K).to(dtype)
        k = U(B, Hk, T, K).to(dtype)
        v = U(B, Hv, T, V).to(dtype)
        g = (U(B, Hv, T, lo=0.2) - 0.1)
        nt = (T + cs - 1) // cs
        h = U(B, Hv, nt, K, V).to(dtype)
        starts = [0] * B
        cu = None
        ci = None
        pairs = [(b, c) for b in range(B) for c in range(nt)]
        h_stride = nt

    o = chunk_fwd_o(q, k, v, h, scale, g=g, cu_seqlens=cu, chunk_indices=ci, chunk_size=cs)
    torch.npu.synchronize()
    r = ref(q, k, v, h, g, scale, cs, pairs, starts, lens_l, h_stride, varlen)
    err = (o.detach().cpu().double() - r).abs().max().item()
    amax = r.abs().max().item()
    floor = FLOOR[dtype]
    ok = err <= floor and tuple(o.shape) == (
        (1, Hv, sum(lens_l), V) if varlen else (B, Hv, T, V))
    emit(f"  {name:52s} err={err:.3e} (<= {floor:.0e}) rel={err / max(amax, 1e-9):.1e} "
         f"|o|max={amax:.2f}  {'PASS' if ok else 'FAIL'}")
    return ok


def case(*a, **kw):
    global allok, nexc
    try:
        allok &= run_case(*a, **kw)
    except Exception as e:  # noqa: BLE001
        nexc += 1
        emit(f"  {a[0]:52s} EXCEPTION {type(e).__name__}: {str(e)[:110]}")
        allok = False


def expect_raises(name, exc, fn):
    global allok, nexc
    try:
        fn()
    except exc as e:
        emit(f"  {name:52s} 正确抛出 {type(e).__name__}  PASS")
        return
    except Exception as e:  # noqa: BLE001
        nexc += 1
        emit(f"  {name:52s} 抛出 {type(e).__name__}（期望 {exc.__name__}）  FAIL")
        allok = False
        return
    nexc += 1
    emit(f"  {name:52s} 未抛异常（期望 {exc.__name__}）  FAIL")
    allok = False


emit("=" * 96)
emit("chunk_fwd_o: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 96)

allok = True
nexc = 0
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 bf16", 2, 2, 2, 128, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=32 bf16", 2, 2, 2, 128, 64, 64, 32)
case("B=1 Hk=1 Hv=2 T=128 K=64 V=64 cs=64 GQA bf16", 1, 1, 2, 128, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=48 cs=64 bf16 (V 非 2 幂)", 2, 2, 2, 128, 64, 48, 64)
case("B=2 Hk=Hv=2 T=100 K=64 V=64 cs=64 bf16 (T 非整除)", 2, 2, 2, 100, 64, 64, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 fp16", 2, 2, 2, 128, 64, 64, 64, dtype=torch.float16)
case("B=2 Hk=Hv=2 T=128 K=128 V=128 cs=64 bf16 (大 K/V)", 2, 2, 2, 128, 128, 128, 64)
case("B=2 Hk=Hv=2 T=128 K=64 V=64 cs=64 bf16 scale=0.125", 2, 2, 2, 128, 64, 64, 64, scale=0.125)
case("B=3 varlen lens=[64,96,40] K=64 V=64 cs=32 bf16", 3, 2, 2, 0, 64, 64, 32,
     varlen=True, lens=[64, 96, 40])
case("B=3 varlen lens=[65,63,1] K=32 V=32 cs=64 bf16 (不满 chunk)", 3, 2, 2, 0, 32, 32, 64,
     varlen=True, lens=[65, 63, 1])
case("B=3 varlen + 显式 chunk_indices", 3, 2, 2, 0, 64, 64, 32,
     varlen=True, lens=[64, 96, 40], give_chunk_indices=True)
case("B=1 Hk=Hv=2 T=128 K=64 V=64 cs=128 bf16 (大 chunk)", 1, 2, 2, 128, 64, 64, 128)

emit()
emit("调用点风格（torch binding schema: chunk_fwd_o_vllm）契约测试：")
emit("-" * 96)


def callsite_ok():
    global allok
    torch.manual_seed(5)

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    B, Hk, Hv, T, K, V, cs = 2, 2, 2, 128, 64, 64, 64
    q = U(B, Hk, T, K).to(torch.bfloat16)
    k = U(B, Hk, T, K).to(torch.bfloat16)
    v = U(B, Hv, T, V).to(torch.bfloat16)
    g = (U(B, Hv, T, lo=0.2) - 0.1)
    nt = (T + cs - 1) // cs
    h = U(B, Hv, nt, K, V).to(torch.bfloat16)
    o = chunk_fwd_o(q, k, v, h, 0.25, g=g, g_gamma=None, cu_seqlens=None,
                    chunk_indices=None, chunk_size=cs, transpose_state_layout=False)
    torch.npu.synchronize()
    pairs = [(b, c) for b in range(B) for c in range(nt)]
    r = ref(q, k, v, h, g, 0.25, cs, pairs, [0] * B, [T] * B, nt, False)
    err = (o.detach().cpu().double() - r).abs().max().item()
    ok = err <= 4e-3
    emit(f"  {'调用点全 kwargs（scale 位置参数 + g/kwargs）':52s} err={err:.3e} (<= 4e-03)  "
         f"{'PASS' if ok else 'FAIL'}")
    allok &= ok


callsite_ok()

emit()
emit("显式拒绝（不静默忽略）：")
emit("-" * 96)


def _mk():
    torch.manual_seed(6)

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    B, Hk, Hv, T, K, V, cs = 2, 2, 2, 64, 64, 64, 64
    q = U(B, Hk, T, K).to(torch.bfloat16)
    k = U(B, Hk, T, K).to(torch.bfloat16)
    v = U(B, Hv, T, V).to(torch.bfloat16)
    g = (U(B, Hv, T, lo=0.2) - 0.1)
    h = U(B, Hv, 1, K, V).to(torch.bfloat16)
    return q, k, v, h, g


q_, k_, v_, h_, g_ = _mk()
expect_raises("g_gamma 非 None（binding L2064 丢弃它）", NotImplementedError,
              lambda: chunk_fwd_o(q_, k_, v_, h_, 1.0, g=g_,
                                  g_gamma=torch.zeros(2, 2, 64, device=DEV)))
expect_raises("transpose_state_layout=True", NotImplementedError,
              lambda: chunk_fwd_o(q_, k_, v_, h_, 1.0, g=g_, transpose_state_layout=True))
expect_raises("g=None（aclnn g must not be nullptr）", ValueError,
              lambda: chunk_fwd_o(q_, k_, v_, h_, 1.0, g=None))

emit()
emit(f"总体: {'全部 PASS ✅' if allok else f'存在 FAIL ❌ (其中 {nexc} 项为异常)'}")

txt = "\n".join(L)
open("/root/dsh_work/chunk_fwd_o_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
