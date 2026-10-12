"""验证 compressor 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.compressor import compressor  # noqa: E402

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def _state_read(state, blkt, b, p, D, blk_size, stride0, max_blk, cache_cycle, sidx):
    """按 Write/ReadFromCacheState 的寻址公式从 state 取一行（fp64）。"""
    blk = blkt[b] if cache_cycle else blkt[b * max_blk + p // blk_size]
    off = blk * stride0 + (p % blk_size) * 2 * D + sidx * D
    return state[off:off + D]


def _state_off(blkt, b, p, D, blk_size, stride0, max_blk, cache_cycle, sidx):
    blk = blkt[b] if cache_cycle else blkt[b * max_blk + p // blk_size]
    return blk * stride0 + (p % blk_size) * 2 * D + sidx * D


def ref_compressor(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l,
                   ob_l, R, D, RD, eps, half_mode, state_l, blkt, blk_size, stride0,
                   max_blk, cache_cycle, out_shape, is_bs):
    """严格按 compressor.py docstring 里记录的语义做 fp64 参考。"""
    xq = x.detach().cpu().double()
    wkvq = wkv.detach().cpu().double()
    wgq = wgate.detach().cpu().double()
    apeq = ape.detach().cpu().double()
    nwq = nw.detach().cpu().double()
    rsq = rs.detach().cpu().double().reshape(-1, RD)
    rcq = rc.detach().cpu().double().reshape(-1, RD)
    state = state_l.detach().cpu().double().reshape(-1).clone() if state_l is not None else None
    new_state = state.clone() if state is not None else None
    blk_l = blkt.detach().cpu().reshape(-1).tolist() if torch.is_tensor(blkt) else list(blkt)
    B = len(sp_l)

    out = torch.zeros(out_shape, dtype=torch.float64)
    outf = out.reshape(-1, D)
    HALF = RD // 2

    for b in range(B):
        sp, su, c0, nc = sp_l[b], su_l[b], c0_l[b], nc_l[b]
        xb = xq[b] if is_bs else xq[x_base[b]:x_base[b] + su]
        for ci in range(nc):
            c = c0 + ci
            kv = torch.zeros(R, D, dtype=torch.float64)
            sc = torch.zeros(R, D, dtype=torch.float64)
            for j in range(R):
                p = c * R + j
                if p < sp:
                    if state is None:
                        continue
                    kv[j] = _state_read(state, blk_l, b, p, D, blk_size, stride0, max_blk, cache_cycle, 0)
                    sc[j] = _state_read(state, blk_l, b, p, D, blk_size, stride0, max_blk, cache_cycle, 1)
                else:
                    kv[j] = xb[p - sp] @ wkvq.t()
                    sc[j] = xb[p - sp] @ wgq.t() + apeq[p % R]
            mx = sc.max(dim=0).values
            ex = torch.exp(sc - mx)
            pj = ex / ex.sum(dim=0)
            o = (pj * kv).sum(dim=0)
            y = o / torch.sqrt((o * o).sum() / D + eps) * nwq
            rr = ob_l[b] + ci
            if half_mode:
                yA, yB = y[D - RD:D - HALF].clone(), y[D - HALF:].clone()
                sA, cA = rsq[rr, :HALF], rcq[rr, :HALF]
                sB, cB = rsq[rr, HALF:], rcq[rr, HALF:]
                y[D - RD:D - HALF] = cA * yA - sA * yB
                y[D - HALF:] = sB * yA + cB * yB
            else:
                yr = y[D - RD:].clone()
                y[D - RD::2] = rcq[rr, 0::2] * yr[0::2] - rsq[rr, 0::2] * yr[1::2]
                y[D - RD + 1::2] = rcq[rr, 1::2] * yr[1::2] + rsq[rr, 1::2] * yr[0::2]
            outf[ob_l[b] + ci] = y

        # ---- state 写回 ----
        if state is not None:
            hi = sp + su
            lo = sp
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            for p in range(lo, hi):
                tok = p - sp
                kvv = xb[tok] @ wkvq.t()
                scc = xb[tok] @ wgq.t() + apeq[p % R]
                o0 = _state_off(blk_l, b, p, D, blk_size, stride0, max_blk, cache_cycle, 0)
                o1 = _state_off(blk_l, b, p, D, blk_size, stride0, max_blk, cache_cycle, 1)
                new_state[o0:o0 + D] = kvv
                new_state[o1:o1 + D] = scc
    return out, new_state


def build(B, S, H, D, R, RD, dtype, is_bs=True, seed=0, S_seq=None):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    if is_bs:
        x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    else:
        x = (torch.randn(S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    ape = torch.randn(R, D, dtype=torch.float32, device=DEV) * 0.1
    # nw 控制输出量级：RMSNorm 输出 RMS≈mean(nw)。取 nw∈[0.2,0.6] 使 |y| < 2，
    # 从而「输出 dtype 的量化误差」本身（半 ulp）就远小于验收地板。
    nw = (torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2)
    Ssr = S_seq if S_seq is not None else S
    Sr = (Ssr + R - 1) // R
    if is_bs:
        rs = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
        rc = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    else:
        rs = torch.randn(Sr, RD, dtype=torch.float32, device=DEV) * 0.1
        rc = torch.randn(Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    return x, wkv, wgate, ape, nw, rs, rc


def _report(name, out, ref, floor, dt, D=None, RD=0):
    o = out.detach().cpu().double()
    err = (o - ref).abs().max().item()
    # 额外给出「参考先按输出 dtype 量化」后的误差，用于证明误差只来自输出量化
    err_q = (o - ref.to(dt).double()).abs().max().item()
    extra = ""
    if D is not None and RD > 0:
        a = (o.reshape(-1, D) - ref.reshape(-1, D)).abs()
        extra = f" [非rope={a[:, :D - RD].max():.2e} rope={a[:, D - RD:].max():.2e}]"
    ok = err <= floor
    emit(f"  {name:44s} err={err:.3e}(<= {floor:.0e}) "
         f"[量化后={err_q:.3e} max|ref|={ref.abs().max():.2f}]{extra}  {'PASS' if ok else 'FAIL'}")
    return ok


def case_simple(name, B, S, H, D, R, RD, dtype, rotary_mode=1, seed=0, count=True):
    x, wkv, wgate, ape, nw, rs, rc = build(B, S, H, D, R, RD, dtype, True, seed)
    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=R,
                        rotary_mode=rotary_mode)
    torch.npu.synchronize()
    Sr = (S + R - 1) // R
    sp_l = [0] * B
    su_l = [S] * B
    c0_l = [0] * B
    nc_l = [S // R] * B
    ob_l = [b * Sr for b in range(B)]
    ref, _ = ref_compressor(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l,
                            [b * S for b in range(B)], c0_l, nc_l, ob_l,
                            R, D, RD, 1e-6, rotary_mode == 1, None, [[0]], 1, 0, 1, False,
                            (B, Sr, D), True)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[dtype]
    if os.environ.get("DBG_CMP"):
        o_ = out.detach().cpu().double().reshape(-1, D)
        r_ = ref.reshape(-1, D)
        dd = (o_ - r_).abs()
        step = D // 8
        for s0 in range(0, D, step):
            emit(f"      cols[{s0}:{s0+step}] maxerr={dd[:, s0:s0+step].max():.3e} "
                 f"max|ref|={r_[:, s0:s0+step].abs().max():.3f}")
        emit(f"      row0 out[0:8] ={[round(v,4) for v in o_[0,:8].tolist()]}")
        emit(f"      row0 ref[0:8] ={[round(v,4) for v in r_[0,:8].tolist()]}")
        emit(f"      row0 out[64:72]={[round(v,4) for v in o_[0,64:72].tolist()]}")
        emit(f"      row0 ref[64:72]={[round(v,4) for v in r_[0,64:72].tolist()]}")
    ok = _report(name, out, ref, floor, dtype, D=D, RD=RD)
    return ok if count else True


def case_th(name, lens, H, D, R, RD, dtype):
    B = len(lens)
    T = sum(lens)
    x, wkv, wgate, ape, nw, rs, rc = build(B, T, H, D, R, RD, dtype, False, 1, S_seq=T)
    cs = [0]
    for L_ in lens:
        cs.append(cs[-1] + L_)
    csl = torch.tensor(cs, dtype=torch.int32, device=DEV)
    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, cu_seqlens=csl,
                        rope_head_dim=RD, cmp_ratio=R)
    torch.npu.synchronize()
    nc_l = [L_ // R for L_ in lens]
    sp_l = [0] * B
    su_l = list(lens)
    c0_l = [0] * B
    ob_l = []
    acc = 0
    for b in range(B):
        ob_l.append(acc)
        acc += nc_l[b]
    ref, _ = ref_compressor(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, cs, c0_l, nc_l, ob_l,
                            R, D, RD, 1e-6, True, None, [[0]], 1, 0, 1, False,
                            (rs.shape[0], D), False)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[dtype]
    return _report(name, out, ref, floor, dtype)


def case_stream(name, S1, S2, H, D, R, RD, dtype, cache_mode, seed=3):
    """流式两段调用：第二段首 chunk 的头部必须从 state_cache 读回。"""
    S = S1 + S2
    x, wkv, wgate, ape, nw, rs, rc = build(1, S, H, D, R, RD, dtype, True, seed)
    nblk = (S + 15) // 16
    if cache_mode == 1:
        stride0 = 16 * 2 * D
        sc = torch.zeros(nblk, 16, 2 * D, dtype=torch.float32, device=DEV)
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, nblk)
        max_blk = nblk
    else:
        stride0 = 16 * 2 * D
        sc = torch.zeros(1, 16, 2 * D, dtype=torch.float32, device=DEV)
        bt = torch.zeros(1, dtype=torch.int32, device=DEV)
        max_blk = 1

    Sr_total = (S + R - 1) // R
    # ---- 第 1 段 ----
    sp1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    su1 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    o1, sc1 = compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw, rs[:, :S1 // R + 1].contiguous(),
                         rc[:, :S1 // R + 1].contiguous(), sc, bt, seqused=su1, start_pos=sp1,
                         rope_head_dim=RD, cmp_ratio=R, cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=16)
    # ---- 第 2 段 ----
    sp2 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    su2 = torch.tensor([S2], dtype=torch.int32, device=DEV)
    n2 = (S1 + S2) // R - S1 // R
    o2, sc2 = compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                         rs[:, S1 // R:S1 // R + n2].contiguous(),
                         rc[:, S1 // R:S1 // R + n2].contiguous(),
                         sc1, bt, seqused=su2, start_pos=sp2,
                         rope_head_dim=RD, cmp_ratio=R, cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=16)
    torch.npu.synchronize()

    # ---- 参考：对完整序列一次算（rotate 用全局 chunk 行）----
    sp_l = [0]
    su_l = [S]
    c0_l = [0]
    nc_l = [S // R]
    ob_l = [0]
    ref_full, _ = ref_compressor(x, wkv, wgate, ape, nw, rs[:, :Sr_total].contiguous(),
                                 rc[:, :Sr_total].contiguous(), sp_l, su_l, [0], c0_l, nc_l, ob_l,
                                 R, D, RD, 1e-6, True, None, [[0]], 1, 0, 1, False,
                                 (1, Sr_total, D), True)

    # ---- 参考：按 state 语义重放两段 ----
    ref1, scR1 = ref_compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                                rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(),
                                [0], [S1], [0], [0], [S1 // R], [0], R, D, RD, 1e-6, True,
                                sc, bt.detach().cpu().reshape(-1).tolist(), 16, stride0, max_blk,
                                cache_mode == 2, (1, S1 // R + 1, D), True)
    ref2, scR2 = ref_compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                                rs[:, S1 // R:S1 // R + n2].contiguous(),
                                rc[:, S1 // R:S1 // R + n2].contiguous(),
                                [S1], [S2], [0], [S1 // R], [n2], [0], R, D, RD, 1e-6, True,
                                scR1, bt.detach().cpu().reshape(-1).tolist(), 16, stride0, max_blk,
                                cache_mode == 2, (1, n2, D), True)

    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3, torch.float32: 2e-6}[dtype]
    e1 = (o1.detach().cpu().double() - ref1).abs().max().item()
    e2 = (o2.detach().cpu().double() - ref2).abs().max().item()
    es = (sc2.detach().cpu().double().reshape(-1) - scR2).abs().max().item()
    # state_cache 是 fp32 输出：按手册的**绝对**地板 2e-6 判定（不再用自定义相对判据）
    ok = e1 <= floor and e2 <= floor and es <= 2e-6
    emit(f"  {name:44s} seg1={e1:.3e} seg2={e2:.3e}(<= {floor:.0e}) "
         f"state={es:.3e}(<= 2e-06,手册绝对口径)  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 96)
emit("compressor: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 96)

allok = True
allok &= case_simple("BS B=1 S=64  H=128 D=128 R=4 RD=64 bf16", 1, 64, 128, 128, 4, 64, torch.bfloat16)
allok &= case_simple("BS B=1 S=50  H=128 D=128 R=4 RD=64 bf16 (非整除)", 1, 50, 128, 128, 4, 64, torch.bfloat16)
allok &= case_simple("BS B=1 S=64  H=256 D=128 R=4 RD=64 fp16", 1, 64, 256, 128, 4, 64, torch.float16)
allok &= case_simple("BS B=2 S=32  H=128 D=128 R=4 RD=64 bf16", 2, 32, 128, 128, 4, 64, torch.bfloat16)
allok &= case_simple("BS B=1 S=64  H=128 D=128 R=4 RD=64 bf16 interleave(逐列展开)",
                     1, 64, 128, 128, 4, 64, torch.bfloat16, rotary_mode=2)
allok &= case_simple("BS B=1 S=48  H=128 D=128 R=2 RD=64 bf16 r=2", 1, 48, 128, 128, 2, 64, torch.bfloat16)
allok &= case_simple("BS B=1 S=64  H=128 D=128 R=8 RD=32 fp16 r=8", 1, 64, 128, 128, 8, 32, torch.float16)
allok &= case_th("TH B=2 lens=[32,50] H=128 D=128 R=4 RD=64 bf16", [32, 50], 128, 128, 4, 64, torch.bfloat16)
allok &= case_stream("stream S1=26 S2=22 (start_pos=26) CONTINUOUS bf16", 26, 22, 128, 128, 4, 64,
                     torch.bfloat16, 1)
allok &= case_stream("stream S1=26 S2=22 (start_pos=26) CYCLE bf16", 26, 22, 128, 128, 4, 64,
                     torch.bfloat16, 2)

def case_interleave_oracle():
    """可辨识输入：x=[1,2,3,4]、cos=sin=1、RD=4 → 期望 [-1,3,-1,7]。
    依据 arch32/rope.h L94-114：out[k] = cos[k]*x[k] + sin[k]*(k 偶 ? -x[k+1] : +x[k-1])。"""
    RD = 4
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    cos = torch.ones(RD, dtype=torch.float64)
    sin = torch.ones(RD, dtype=torch.float64)
    out = torch.zeros(RD, dtype=torch.float64)
    for k in range(RD):
        partner = k + 1 if k % 2 == 0 else k - 1
        sgn = -1.0 if k % 2 == 0 else 1.0
        out[k] = cos[k] * x[k] + sin[k] * sgn * x[partner]
    got = [float(v) for v in out]
    ok1 = got == [-1.0, 3.0, -1.0, 7.0]
    emit(f"  {'[可辨识 oracle] x=[1,2,3,4] cos=sin=1 RD=4':52s} got={got} "
         f"exp=[-1.0, 3.0, -1.0, 7.0]  {'✅' if ok1 else '❌'}")
    y = torch.zeros(RD, dtype=torch.float64)
    y[0::2] = cos[0::2] * x[0::2] - sin[0::2] * x[1::2]
    y[1::2] = cos[1::2] * x[1::2] + sin[1::2] * x[0::2]
    ok2 = bool((y == out).all())
    emit(f"  {'  向量化参考写法 vs oracle 逐位一致':52s} vec={[float(v) for v in y]}  "
         f"{'✅' if ok2 else '❌'}")
    oe = torch.tensor([1.0, 3.0], dtype=torch.float64)
    oo = torch.tensor([2.0, 4.0], dtype=torch.float64)
    joined = torch.stack([oe, oo], dim=1).reshape(-1)   # 与 tl.join(oe,oo)->reshape 同语义
    ok3 = joined.tolist() == [1.0, 2.0, 3.0, 4.0]
    emit(f"  {'  tl.join+reshape 重排顺序 [a0,b0,a1,b1]':52s} {joined.tolist()}  {'OK' if ok3 else 'BAD'}")
    return ok1 and ok2 and ok3


emit()
emit("--- rotary_mode=2 (INTERLEAVE) 独立 kernel ---")
allok &= case_interleave_oracle()

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
try:
    open("/root/dsh_work/compressor_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
