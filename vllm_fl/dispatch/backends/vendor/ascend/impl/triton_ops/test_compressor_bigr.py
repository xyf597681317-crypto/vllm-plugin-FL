"""验证 compressor 的 `coff=1` + `cmp_ratio > 16`（生产 r=128）路径。

* `ref_A` = **online / streaming softmax 视角**（行分块 running max/sum/acc，即实现所用的算法）
* `ref_B` = **一次性 softmax 视角**（上游 vLLM Triton kernel `fused_compress_quant_cache.py:156-211`
  在 `OVERLAP=0` 时的语义：整窗 R 行 + `other=-inf / 0.0`）
两者结构独立 ⇒ 交叉校验。另有手算 oracle 判定「行数 = R 而不是 BM」。
"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.compressor import compressor  # noqa: E402

DEV = "npu:0"
L = []
MIN_SCORE = -1.0e30
BM = 16


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def _st_blk(blkt, b, p, blk_size, max_blk, cache_cycle):
    return blkt[b] if cache_cycle else blkt[b * max_blk + p // blk_size]


def _state(b, p, blk_l, blk_size, stride0, max_blk, cache_cycle, is_score, D, st):
    blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
    base = blk * stride0 + (p % blk_size) * (2 * D) + (D if is_score else 0)
    return st[base:base + D]


def _rope(y, rr, rsq, rcq, D, RD, half_mode):
    HALF = RD // 2
    if half_mode:
        yA, yB = y[D - RD:D - HALF].clone(), y[D - HALF:].clone()
        y[D - RD:D - HALF] = rcq[rr, :HALF] * yA - rsq[rr, :HALF] * yB
        y[D - HALF:] = rsq[rr, HALF:] * yA + rcq[rr, HALF:] * yB
    else:
        yr = y[D - RD:].clone()
        y[D - RD::2] = rcq[rr, 0::2] * yr[0::2] - rsq[rr, 0::2] * yr[1::2]
        y[D - RD + 1::2] = rcq[rr, 1::2] * yr[1::2] + rsq[rr, 1::2] * yr[0::2]
    return y


def _prep(x, wkv, wgate, ape, nw, rs, rc, state_l, blkt, RD):
    return (x.detach().cpu().double(), wkv.detach().cpu().double(),
            wgate.detach().cpu().double(), ape.detach().cpu().double(),
            nw.detach().cpu().double(), rs.detach().cpu().double().reshape(-1, RD),
            rc.detach().cpu().double().reshape(-1, RD),
            state_l.detach().cpu().double().reshape(-1).clone() if state_l is not None else None,
            blkt.detach().cpu().reshape(-1).tolist() if torch.is_tensor(blkt) else list(blkt))


def _rows_A(xq, wkvq, wgq, apeq, state, blk_l, b, sp, c, R, D, blk_size, stride0, max_blk,
            cache_cycle, xb):
    """online 视角：逐 16 行块维护 running max/sum/acc。"""
    run_max = torch.full((D,), MIN_SCORE, dtype=torch.float64)
    run_sum = torch.zeros(D, dtype=torch.float64)
    run_acc = torch.zeros(D, dtype=torch.float64)
    for m0 in range(0, R, BM):
        blk_sc = torch.full((BM, D), MIN_SCORE, dtype=torch.float64)
        blk_kv = torch.zeros(BM, D, dtype=torch.float64)
        good = torch.zeros(BM, dtype=torch.bool)
        for i in range(BM):
            j = m0 + i
            if j >= R:
                continue
            p = c * R + j
            if p < 0 or p >= sp + (xb.shape[0] if xb is not None else 0):
                continue
            if p < sp:
                if state is None:
                    continue
                blk_kv[i] = _state(b, p, blk_l, blk_size, stride0, max_blk, cache_cycle, False, D, state)
                blk_sc[i] = _state(b, p, blk_l, blk_size, stride0, max_blk, cache_cycle, True, D, state)
            else:
                blk_kv[i] = xb[p - sp] @ wkvq.t()
                blk_sc[i] = xb[p - sp] @ wgq.t() + apeq[j % R]
            good[i] = True
        s = torch.where(good[:, None], blk_sc, torch.tensor(MIN_SCORE, dtype=torch.float64))
        new_max = torch.maximum(run_max, s.max(dim=0).values)
        alpha = torch.exp(run_max - new_max)
        e = torch.where(good[:, None], torch.exp(s - new_max), torch.zeros((), dtype=torch.float64))
        run_acc = run_acc * alpha + (e * blk_kv).sum(dim=0)
        run_sum = run_sum * alpha + e.sum(dim=0)
        run_max = new_max
    return run_acc / run_sum


def _rows_B(xq, wkvq, wgq, apeq, state, blk_l, b, sp, c, R, D, blk_size, stride0, max_blk,
            cache_cycle, xb):
    """一次性视角（上游 Triton 直译，OVERLAP=0）：整窗 R 行 + softmax。"""
    sc = torch.full((R, D), float("-inf"), dtype=torch.float64)
    kv = torch.zeros(R, D, dtype=torch.float64)
    n = xb.shape[0] if xb is not None else 0
    for j in range(R):
        p = c * R + j
        if p < 0 or p >= sp + n:
            continue
        if p < sp:
            if state is None:
                continue
            kv[j] = _state(b, p, blk_l, blk_size, stride0, max_blk, cache_cycle, False, D, state)
            sc[j] = _state(b, p, blk_l, blk_size, stride0, max_blk, cache_cycle, True, D, state)
        else:
            kv[j] = xb[p - sp] @ wkvq.t()
            sc[j] = xb[p - sp] @ wgq.t() + apeq[j % R]
    mx = sc.max(dim=0).values
    ex = torch.exp(sc - mx)
    pj = ex / ex.sum(dim=0)
    return (pj * kv).sum(dim=0)


def _refs(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l,
          R, D, RD, eps, half_mode, state_l, blkt, blk_size, stride0, max_blk,
          cache_cycle, out_shape, is_bs):
    xq, wkvq, wgq, apeq, nwq, rsq, rcq, state, blk_l = _prep(x, wkv, wgate, ape, nw, rs, rc,
                                                             state_l, blkt, RD)
    outA = torch.zeros(out_shape, dtype=torch.float64)
    outB = torch.zeros(out_shape, dtype=torch.float64)
    oA, oB = outA.reshape(-1, D), outB.reshape(-1, D)
    stateA = state.clone() if state is not None else None
    B = len(sp_l)
    for b in range(B):
        sp, su, c0, nc = sp_l[b], su_l[b], c0_l[b], nc_l[b]
        xb = xq[b] if is_bs else xq[x_base[b]:x_base[b] + su]
        for ci in range(nc):
            c = c0 + ci
            rr = ob_l[b] + ci
            for fn, dst in ((_rows_A, oA), (_rows_B, oB)):
                o = fn(xq, wkvq, wgq, apeq, state, blk_l, b, sp, c, R, D, blk_size, stride0,
                       max_blk, cache_cycle, xb)
                y = o / torch.sqrt((o * o).sum() / D + eps) * nwq
                dst[rr] = _rope(y, rr, rsq, rcq, D, RD, half_mode)
        if state is not None:
            hi, lo = sp + su, sp
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            for p in range(lo, hi):
                tok = p - sp
                blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
                base = blk * stride0 + (p % blk_size) * (2 * D)
                stateA[base:base + D] = xb[tok] @ wkvq.t()
                stateA[base + D:base + 2 * D] = xb[tok] @ wgq.t() + apeq[p % R]
    return outA, outB, stateA


def build(B, S, H, D, R, RD, dtype, is_bs=True, seed=0, T=None):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    if is_bs:
        x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    else:
        x = (torch.randn(T, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    ape = torch.randn(R, D, dtype=torch.float32, device=DEV) * 0.1
    nw = (torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2)
    Sseq = S if is_bs else T
    Sr = (Sseq + R - 1) // R
    if is_bs:
        rs = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
        rc = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    else:
        rs = torch.randn(Sr, RD, dtype=torch.float32, device=DEV) * 0.1
        rc = torch.randn(Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    return x, wkv, wgate, ape, nw, rs, rc


def _report(name, out, refA, refB, floor, dt, D, RD):
    o = out.detach().cpu().double().reshape(-1, D)
    refA = refA.detach().cpu().double().reshape(-1, D)
    refB = refB.detach().cpu().double().reshape(-1, D)
    eA = (o - refA).abs().max().item()
    eB = (o - refB).abs().max().item()
    xchk = (refA - refB).abs().max().item()
    okq = (o - refA.to(dt).double()).abs().max().item()
    ok = eA <= floor and eB <= floor and xchk <= 1e-12
    emit(f"  {name:46s} errA={eA:.3e} errB={eB:.3e} |A-B|={xchk:.2e} "
         f"[量化后={okq:.3e} max|ref|={refA.abs().max():.2f}]  {'PASS' if ok else 'FAIL'}")
    return ok


def case(name, B, S, H, D, R, RD, dtype, rotary_mode=2, seed=0, is_bs=True, lens=None):
    if is_bs:
        x, wkv, wgate, ape, nw, rs, rc = build(B, S, H, D, R, RD, dtype, True, seed)
        Sr = rs.shape[1]
        ob_l = [b * Sr for b in range(B)]
        lens_l = [S] * B
        x_base = [b * S for b in range(B)]
        out_shape = (B, Sr, D)
        kwargs = {}
    else:
        T = sum(lens)
        x, wkv, wgate, ape, nw, rs, rc = build(len(lens), 0, H, D, R, RD, dtype, False, seed, T)
        cs = [0]
        for L_ in lens:
            cs.append(cs[-1] + L_)
        kwargs = {"cu_seqlens": torch.tensor(cs, dtype=torch.int32, device=DEV)}
        lens_l = list(lens)
        x_base = cs[:-1]
        nc_l0 = [L_ // R for L_ in lens]
        ob_l = []
        acc = 0
        for v in nc_l0:
            ob_l.append(acc)
            acc += v
        out_shape = (rs.shape[0], D)

    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=R, coff=1,
                        rotary_mode=rotary_mode, **kwargs)
    torch.npu.synchronize()
    sp_l = [0] * len(lens_l)
    su_l = list(lens_l)
    c0_l = [0] * len(lens_l)
    nc_l = [L_ // R for L_ in lens_l]
    rA, rB, _ = _refs(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l,
                      R, D, RD, 1e-6, rotary_mode == 1, None, [[0]], 1, 0, 1, False,
                      out_shape, is_bs)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dtype]
    return _report(name, out, rA, rB, floor, dtype, D, RD)


def case_oracle():
    """手算 oracle：判定「softmax 覆盖的是 R=128 行，而不是 16 行块」。

    r=128, D=512, RD=64, H=128(=R), S=384（3 个 chunk）：
      x[t,h] = 1 当 h == t % H（one-hot，H=R ⇒ 每个 chunk 内 h == 局部行号 j）
      wkv[d,h] = h+1  、wgate ≡ 0  、ape[j,d] = +j (d < D/2) / -j (d ≥ D/2)
      cos ≡ 1、sin ≡ 0（RoPE 恒等）、norm_weight ≡ 1
    ⇒ kv_{j,d} = j+1；score_{j,d} = ±j
    ⇒ o_d = M(σ)，M(σ) = Σ_j softmax(σ j)_j · (j+1)      （σ = +1 前半区 / −1 后半区）
    ⇒ y[0]/y[D-1] = M(+)/M(−)   （RMSNorm 的公共标量约掉）
    """
    R, D, RD, H, S = 128, 512, 64, 128, 384
    dev = DEV
    t = torch.arange(S, device=dev)
    x = torch.zeros(1, S, H, dtype=torch.float32, device=dev)
    x[0, t, t % H] = 1.0
    x = x.to(torch.bfloat16)
    hh = torch.arange(H, device=dev, dtype=torch.float32) + 1.0
    wkv = hh[None, :].expand(D, H).to(torch.bfloat16).contiguous()
    wgate = torch.zeros(D, H, dtype=torch.bfloat16, device=dev)
    j = torch.arange(R, device=dev, dtype=torch.float32)
    ape = torch.zeros(R, D, dtype=torch.float32, device=dev)
    ape[:, :D // 2] = j[:, None]
    ape[:, D // 2:] = -j[:, None]
    nw = torch.ones(D, dtype=torch.float32, device=dev)
    Sr = S // R
    rs = torch.zeros(1, Sr, RD, dtype=torch.float32, device=dev)
    rc = torch.ones(1, Sr, RD, dtype=torch.float32, device=dev)

    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=R, coff=1,
                        rotary_mode=2)
    torch.npu.synchronize()
    o = out.detach().cpu().double().reshape(-1, D)

    q = torch.arange(R, dtype=torch.float64)
    def M(sign):
        w = torch.exp(sign * q)
        w = w / w.sum()
        return float((w * (q + 1)).sum())
    exp_ratio = M(1.0) / M(-1.0)
    # 反例：若只用了 16 行（BM 块），比值应为多少
    def M16(sign):
        w = torch.exp(sign * q[:BM])
        w = w / w.sum()
        return float((w * (q[:BM] + 1)).sum())
    bad_ratio = M16(1.0) / M16(-1.0)

    got = [float(o[c, 0] / o[c, D - 1]) for c in range(Sr)]
    ok = all(abs(v - exp_ratio) / exp_ratio < 0.02 for v in got)
    emit(f"  {'[oracle] y[0]/y[511] 应 = M(+)/M(-)':46s} got={[round(v, 3) for v in got]} "
         f"exp={exp_ratio:.3f}  {'✅' if ok else '❌'}")
    emit(f"  {'  反例判别（若只用 16 行 ⇒ 比值应为）':46s} bad={bad_ratio:.3f} "
         f"（实测 {got[0]:.3f} 远离 {bad_ratio:.3f} ⇒ 覆盖 {R} 行）"
         f"  {'✅' if abs(got[0] - bad_ratio) / bad_ratio > 0.2 else '❌'}")
    return ok


def case_contract():
    R, D, RD, H = 128, 512, 64, 64
    x, wkv, wgate, ape, nw, rs, rc = build(1, 256, H, D, R, RD, torch.bfloat16, True, 0)
    ok = True
    for tag, args, exc in (
        ("wkv.shape[0] != headDim", dict(wkv=wkv[:D // 2].contiguous()), ValueError),
        ("ape.shape != [r, headDim]", dict(ape=ape[:, :D // 2].contiguous()), ValueError),
        ("cmp_ratio 非 2 的幂 (96)", dict(cmp_ratio=96), ValueError),
    ):
        kw = dict(x=x, wkv=wkv, wgate=wgate, ape=ape, norm_weight=nw,
                  rope_sin=rs, rope_cos=rc, rope_head_dim=RD, cmp_ratio=R, coff=1)
        kw.update(args)
        try:
            compressor(**kw)
            torch.npu.synchronize()
            emit(f"  {'[契约] ' + tag:46s} 未报错  ❌")
            ok = False
        except exc as e:
            emit(f"  {'[契约] ' + tag:46s} 已报错 ✅  {str(e)[:64]}")
    return ok


def case_stream(name, S1, S2, H, D, R, RD, dtype, cache_mode, seed=3):
    S = S1 + S2
    x, wkv, wgate, ape, nw, rs, rc = build(1, S, H, D, R, RD, dtype, True, seed)
    # ⚠️ ring 槽位数必须 >= R（否则同一槽位被多个位置覆盖，写入顺序决定内容 ⇒ 并行内核
    #    无法复现顺序语义）。R=128 时用 block_size=128。
    BLKS = max(128, R)
    nblk = (S + BLKS - 1) // BLKS
    stride0 = BLKS * 2 * D
    if cache_mode == 1:
        sc = torch.zeros(nblk, BLKS, 2 * D, dtype=torch.float32, device=DEV)
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, nblk)
        max_blk = nblk
    else:
        sc = torch.zeros(1, BLKS, 2 * D, dtype=torch.float32, device=DEV)
        bt = torch.zeros(1, dtype=torch.int32, device=DEV)
        max_blk = 1
    sp1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    su1 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    o1, sc1 = compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                         rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(),
                         sc, bt, seqused=su1, start_pos=sp1, rope_head_dim=RD, cmp_ratio=R,
                         coff=1, rotary_mode=2, cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=BLKS)
    sp2 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    su2 = torch.tensor([S2], dtype=torch.int32, device=DEV)
    n2 = (S1 + S2) // R - S1 // R
    o2, sc2 = compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                         rs[:, S1 // R:S1 // R + n2].contiguous(),
                         rc[:, S1 // R:S1 // R + n2].contiguous(),
                         sc1, bt, seqused=su2, start_pos=sp2, rope_head_dim=RD, cmp_ratio=R,
                         coff=1, rotary_mode=2, cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=BLKS)
    torch.npu.synchronize()

    c1 = ([0], [S1], [0], [0], [S1 // R], [0], R, D, RD, 1e-6, False, sc,
          bt.detach().cpu().reshape(-1).tolist(), BLKS, stride0, max_blk, cache_mode == 2,
          (1, S1 // R + 1, D), True)
    rA1, rB1, scR1 = _refs(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                           rs[:, :S1 // R + 1].contiguous(),
                           rc[:, :S1 // R + 1].contiguous(), *c1)
    c2 = ([S1], [S2], [0], [S1 // R], [n2], [0], R, D, RD, 1e-6, False, scR1,
          bt.detach().cpu().reshape(-1).tolist(), BLKS, stride0, max_blk, cache_mode == 2,
          (1, n2, D), True)
    rA2, rB2, scR2 = _refs(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                           rs[:, S1 // R:S1 // R + n2].contiguous(),
                           rc[:, S1 // R:S1 // R + n2].contiguous(), *c2)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dtype]
    e1 = max((o1.detach().cpu().double() - rA1).abs().max().item(),
             (o1.detach().cpu().double() - rB1).abs().max().item())
    e2 = max((o2.detach().cpu().double() - rA2).abs().max().item(),
             (o2.detach().cpu().double() - rB2).abs().max().item())
    es = (sc2.detach().cpu().double().reshape(-1) - scR2).abs().max().item()
    xchk = max((rA1 - rB1).abs().max().item(), (rA2 - rB2).abs().max().item())
    ok = e1 <= floor and e2 <= floor and es <= 2e-6
    emit(f"  {name:46s} seg1={e1:.3e} seg2={e2:.3e}(<= {floor:.0e}) "
         f"state={es:.3e}(<= 2e-06) |A-B|={xchk:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 112)
emit("compressor coff=1 & cmp_ratio>16（生产 r=128/headDim=512）：online softmax vs 两个独立 fp64 参考")
emit("=" * 112)

allok = True
emit()
emit("--- ① 可辨识 oracle（判定 softmax 覆盖 R=128 行）---")
allok &= case_oracle()
emit()
emit("--- ② 契约负例 ---")
allok &= case_contract()
emit()
emit("--- ③ 数值对照（errA=online视角 errB=一次性视角）---")
allok &= case("BS B=1 S=512 D=512 R=128 RD=64 bf16 interleave", 1, 512, 128, 512, 128, 64,
              torch.bfloat16, 2)
allok &= case("BS B=1 S=512 D=512 R=128 RD=64 fp16 half", 1, 512, 128, 512, 128, 64,
              torch.float16, 1, seed=1)
allok &= case("BS B=1 S=384 D=512 R=128 RD=64 bf16 (非整除)", 1, 384, 128, 512, 128, 64,
              torch.bfloat16, 2, seed=2)
allok &= case("TH B=2 lens=[256,300] D=512 R=128 RD=64 bf16", 0, 0, 128, 512, 128, 64,
              torch.bfloat16, 2, seed=4, is_bs=False, lens=[256, 300])
emit()
emit("--- ④ 流式两段（带 state_cache）—— 三段式 K1 投影/K2 纯向量池化/K3 写 state ---")
allok &= case_stream("stream S1=300 S2=212 (start_pos=300) CONTINUOUS", 300, 212, 128, 512, 128, 64,
                     torch.bfloat16, 1)
allok &= case_stream("stream S1=300 S2=212 (start_pos=300) CYCLE", 300, 212, 128, 512, 128, 64,
                     torch.bfloat16, 2)
emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
try:
    open("/root/dsh_work/zeta/bigr_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
