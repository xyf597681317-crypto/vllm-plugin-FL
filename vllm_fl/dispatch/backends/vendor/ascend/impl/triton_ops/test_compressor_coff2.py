"""验证 compressor 的 Triton `coff=2 (COFF::OVERLAP)` 路径。

方法论（与 test_compressor.py 一致 + Lead 追加要求）：
1. **两个结构上独立的 fp64 参考**：
   * `ref_A` —— 按 AscendC `MulReduceSumbaseVF(coff,cmpRatio,...)` / `DuplicateFirstBlock` 的
     「两块（前 chunk 取 [0,D) / 当前 chunk 取 [D,2D)）各自归约后合并」写法；
   * `ref_B` —— 按上游 vLLM 官方 Triton kernel
     (`fused_compress_quant_cache.py:156-211`) 的「2r 行窗口 + head_offset 取通道 + 逐列 softmax」
     写法（**逐行直译**，含 `other=-inf / 0.0` 的 mask 语义）。
   两者若语义理解有偏差，不会同时错成一样 ⇒ 交叉校验有效。
2. **可辨识输入 oracle**（手算期望值）：直接判定「窗口是否 2r」「通道是否分半」「首块无历史」。
3. 输出 dtype 量化地板：bf16 4e-3 / fp16 1e-3（同 test_compressor.py）。
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
MIN_SCORE = -1.0e30   # 与实现一致（避免 -inf 在 Ascend 上退化）
NEG_INF = float("-inf")


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


# ---------------------------------------------------------------------------
# state_cache 寻址（compressor_block_vec.h Read/WriteFromCacheState L727-829）
#   coff=2: state 行宽 = 2*coff*D = 4D，行 stride = 4D
#           [ kv(2D) | score+ape(2D) ]，kv 内再分 [0,D) / [D,2D) 两个通道半区
# ---------------------------------------------------------------------------
def _st_blk(blkt, b, p, blk_size, max_blk, cache_cycle):
    if cache_cycle:
        return blkt[b]
    return blkt[b * max_blk + p // blk_size]


def _bit(A, w, r, c, d):
    return (A[:, None] == torch.arange(r, device=A.device)[None, :]) & \
           (w >= c) & (w < c + d)


def ref_A(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l,
          R, D, RD, eps, half_mode, state_l, blkt, blk_size, stride0, max_blk,
          cache_cycle, out_shape, is_bs):
    """参考 A：两块归约后合并（AscendC 视角）。"""
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
    DC = 2 * D
    out = torch.zeros(out_shape, dtype=torch.float64)
    outf = out.reshape(-1, D)
    HALF = RD // 2

    def rd_state(b, p, half_off, is_score):
        blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
        base = blk * stride0 + (p % blk_size) * (2 * DC) + (2 * D if is_score else 0) + half_off
        return state[base:base + D]

    for b in range(B):
        sp, su, c0, nc = sp_l[b], su_l[b], c0_l[b], nc_l[b]
        xb = xq[b] if is_bs else xq[x_base[b]:x_base[b] + su]
        for ci in range(nc):
            c = c0 + ci
            # --- 前一 chunk：通道 [0, D) ---
            kvP = torch.zeros(R, D, dtype=torch.float64)
            scP = torch.full((R, D), MIN_SCORE, dtype=torch.float64)
            for j in range(R):
                p = (c - 1) * R + j
                if p < 0:
                    continue                       # 无历史：score=-inf 且 kv=0
                if p < sp:
                    kvP[j] = rd_state(b, p, 0, False)
                    scP[j] = rd_state(b, p, 0, True)
                else:
                    kvP[j] = xb[p - sp] @ wkvq[:D].t()
                    scP[j] = xb[p - sp] @ wgq[:D].t() + apeq[j % R, :D]
            # --- 当前 chunk：通道 [D, 2D) ---
            kvC = torch.zeros(R, D, dtype=torch.float64)
            scC = torch.full((R, D), MIN_SCORE, dtype=torch.float64)
            for j in range(R):
                p = c * R + j
                if p < 0:
                    continue
                if p < sp:
                    kvC[j] = rd_state(b, p, D, False)
                    scC[j] = rd_state(b, p, D, True)
                else:
                    kvC[j] = xb[p - sp] @ wkvq[D:].t()
                    scC[j] = xb[p - sp] @ wgq[D:].t() + apeq[j % R, D:]
            mx = torch.maximum(scP.max(dim=0).values, scC.max(dim=0).values)
            eP = torch.exp(scP - mx)
            eC = torch.exp(scC - mx)
            den = eP.sum(dim=0) + eC.sum(dim=0)
            o = (eP * kvP).sum(dim=0) / den + (eC * kvC).sum(dim=0) / den
            y = o / torch.sqrt((o * o).sum() / D + eps) * nwq
            rr = ob_l[b] + ci
            if half_mode:
                yA, yB = y[D - RD:D - HALF].clone(), y[D - HALF:].clone()
                y[D - RD:D - HALF] = rcq[rr, :HALF] * yA - rsq[rr, :HALF] * yB
                y[D - HALF:] = rsq[rr, HALF:] * yA + rcq[rr, HALF:] * yB
            else:
                yr = y[D - RD:].clone()
                y[D - RD::2] = rcq[rr, 0::2] * yr[0::2] - rsq[rr, 0::2] * yr[1::2]
                y[D - RD + 1::2] = rcq[rr, 1::2] * yr[1::2] + rsq[rr, 1::2] * yr[0::2]
            outf[ob_l[b] + ci] = y

        # ---- state 写回：kv 全宽 2D（列 [0,2D)）、score+ape 全宽 2D（列 [2D,4D)）----
        if state is not None:
            hi, lo = sp + su, sp
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            for p in range(lo, hi):
                tok = p - sp
                blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
                base = blk * stride0 + (p % blk_size) * (2 * DC)
                new_state[base:base + D] = xb[tok] @ wkvq[:D].t()
                new_state[base + D:base + 2 * D] = xb[tok] @ wkvq[D:].t()
                new_state[base + 2 * D:base + 3 * D] = xb[tok] @ wgq[:D].t() + apeq[p % R, :D]
                new_state[base + 3 * D:base + 4 * D] = xb[tok] @ wgq[D:].t() + apeq[p % R, D:]
    return out, new_state


def ref_B(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l,
          R, D, RD, eps, half_mode, state_l, blkt, blk_size, stride0, max_blk,
          cache_cycle, out_shape, is_bs):
    """参考 B：**逐行直译**上游 vLLM 官方 Triton kernel。

    对应 fused_compress_quant_cache.py:156-211::

        start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
        tokens = arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
        head_offset = (tokens >= COMPRESS_RATIO) * HEAD_SIZE
        score = load(row_base + STATE_WIDTH + block, other=-inf); softmax(dim=0)
        kv    = load(row_base + block, other=0.0)
    """
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
    DC = 2 * D
    W = 2 * R                                  # (1 + OVERLAP) * COMPRESS_RATIO
    out = torch.zeros(out_shape, dtype=torch.float64)
    outf = out.reshape(-1, D)
    HALF = RD // 2

    for b in range(B):
        sp, su, c0, nc = sp_l[b], su_l[b], c0_l[b], nc_l[b]
        xb = xq[b] if is_bs else xq[x_base[b]:x_base[b] + su]
        for ci in range(nc):
            c = c0 + ci
            position = (c + 1) * R - 1          # 窗口末端
            start = position - W + 1
            tokens = torch.arange(W)
            pos = start + tokens
            mask_pos = pos >= 0
            head_offset = (tokens >= R).to(torch.int64) * D
            kv = torch.zeros(W, D, dtype=torch.float64)
            sc = torch.full((W, D), NEG_INF, dtype=torch.float64)
            for j in range(W):
                p = int(pos[j])
                if not bool(mask_pos[j]):
                    continue                     # other = -inf (score) / 0.0 (kv)
                ho = int(head_offset[j])
                if p < sp:
                    blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
                    base = blk * stride0 + (p % blk_size) * (2 * DC) + ho
                    kv[j] = state[base:base + D]
                    sc[j] = state[base + 2 * D:base + 2 * D + D]
                else:
                    kv[j] = xb[p - sp] @ wkvq[ho:ho + D].t()
                    sc[j] = xb[p - sp] @ wgq[ho:ho + D].t() + apeq[j % R, ho:ho + D]
            mx = sc.max(dim=0).values
            ex = torch.exp(sc - mx)              # -inf 行 → 0
            pj = ex / ex.sum(dim=0)
            o = (pj * kv).sum(dim=0)
            y = o / torch.sqrt((o * o).sum() / D + eps) * nwq
            rr = ob_l[b] + ci
            if half_mode:
                yA, yB = y[D - RD:D - HALF].clone(), y[D - HALF:].clone()
                y[D - RD:D - HALF] = rcq[rr, :HALF] * yA - rsq[rr, :HALF] * yB
                y[D - HALF:] = rsq[rr, HALF:] * yA + rcq[rr, HALF:] * yB
            else:
                yr = y[D - RD:].clone()
                y[D - RD::2] = rcq[rr, 0::2] * yr[0::2] - rsq[rr, 0::2] * yr[1::2]
                y[D - RD + 1::2] = rcq[rr, 1::2] * yr[1::2] + rsq[rr, 1::2] * yr[0::2]
            outf[ob_l[b] + ci] = y

        if state is not None:
            hi, lo = sp + su, sp
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            for p in range(lo, hi):
                tok = p - sp
                blk = _st_blk(blk_l, b, p, blk_size, max_blk, cache_cycle)
                base = blk * stride0 + (p % blk_size) * (2 * DC)
                new_state[base:base + D] = xb[tok] @ wkvq[:D].t()
                new_state[base + D:base + 2 * D] = xb[tok] @ wkvq[D:].t()
                new_state[base + 2 * D:base + 3 * D] = xb[tok] @ wgq[:D].t() + apeq[p % R, :D]
                new_state[base + 3 * D:base + 4 * D] = xb[tok] @ wgq[D:].t() + apeq[p % R, D:]
    return out, new_state


# ---------------------------------------------------------------------------
# 数据构造
# ---------------------------------------------------------------------------
def build(B, S, H, D, R, RD, dtype, is_bs=True, seed=0, T=None):
    """coff=2：wkv/wgate 形状 [2D, H]，ape 形状 [R, 2D]。"""
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    if is_bs:
        x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    else:
        x = (torch.randn(T, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
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
    a = (o - refA).abs()
    extra = ""
    if RD and D > RD:
        extra = f" [非rope={a[:, :D - RD].max():.2e} rope={a[:, D - RD:].max():.2e}]"
    ok = eA <= floor and eB <= floor and xchk <= 1e-12
    emit(f"  {name:46s} errA={eA:.3e} errB={eB:.3e} |A-B|={xchk:.2e} "
         f"[量化后={okq:.3e} max|ref|={refA.abs().max():.2f}]{extra}  {'PASS' if ok else 'FAIL'}")
    return ok


def case(name, B, S, H, D, R, RD, dtype, rotary_mode=2, seed=0, is_bs=True, T=None, lens=None):
    if is_bs:
        x, wkv, wgate, ape, nw, rs, rc = build(B, S, H, D, R, RD, dtype, True, seed)
        Sr = rs.shape[1]
        csl = None
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
        csl = torch.tensor(cs, dtype=torch.int32, device=DEV)
        lens_l = list(lens)
        x_base = cs[:-1]
        nc_l = [L_ // R for L_ in lens]
        ob_l = []
        acc = 0
        for L_ in nc_l:
            ob_l.append(acc)
            acc += L_
        out_shape = (rs.shape[0], D)
        kwargs = {"cu_seqlens": csl}

    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=R, coff=2,
                        rotary_mode=rotary_mode, **kwargs)
    torch.npu.synchronize()
    R_ = R
    sp_l = [0] * len(lens_l)
    su_l = list(lens_l)
    c0_l = [0] * len(lens_l)
    nc_l = [L_ // R_ for L_ in lens_l]
    common = (R_, D, RD, 1e-6, rotary_mode == 1, None, [[0]], 1, 0, 1, False, out_shape, is_bs)
    rA, _ = ref_A(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)
    rB, _ = ref_B(x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dtype]
    return _report(name, out, rA, rB, floor, dtype, D if is_bs else D, RD)


# ---------------------------------------------------------------------------
# 可辨识 oracle
# ---------------------------------------------------------------------------
def case_oracle():
    """手算 oracle：判定「窗口 = 2r」「通道分半」「首块无历史」三件事。

    构造（r=4, D=128, H=16）：
      x ≡ 1；wgate ≡ 0；ape ≡ 0 ⇒ score ≡ 0 ⇒ softmax 在每个存在的行上**均匀**；
      wkv 的第 0 半区（列 [0,D)）全 1；第 1 半区（列 [D,2D)）第 0 列 = 3、其余 = 1。
      cos ≡ 1、sin ≡ 0 ⇒ RoPE 为恒等；norm_weight ≡ 1。

    于是（kv 取的就是该行所属半区的投影值）：
      首块（c=0，窗前 r 行无历史 ⇒ 只有后 r 行存在）: o[0]=3, o[d>0]=1  ⇒ y[0]/y[1] = 3
      其余（全窗 2r 行）                    : o[0]=(1+3)/2=2, o[d>0]=1 ⇒ y[0]/y[1] = 2
    """
    R, D, RD, H, S = 4, 128, 64, 16, 24
    dev = DEV
    x = torch.ones(1, S, H, dtype=torch.bfloat16, device=dev)
    a = torch.ones(2 * D, dtype=torch.float32, device=dev)
    a[D] = 3.0
    wkv = (a[:, None].expand(2 * D, H) / H).to(torch.bfloat16).contiguous()
    wgate = torch.zeros(2 * D, H, dtype=torch.bfloat16, device=dev)
    ape = torch.zeros(R, 2 * D, dtype=torch.float32, device=dev)
    nw = torch.ones(D, dtype=torch.float32, device=dev)
    Sr = S // R
    rs = torch.zeros(1, Sr, RD, dtype=torch.float32, device=dev)
    rc = torch.ones(1, Sr, RD, dtype=torch.float32, device=dev)

    out, _ = compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=R, coff=2,
                        rotary_mode=2)
    torch.npu.synchronize()
    o = out.detach().cpu().double().reshape(-1, D)
    ratios = [float(o[c, 0] / o[c, 1]) for c in range(Sr)]
    exp = [3.0] + [2.0] * (Sr - 1)
    ok = all(abs(ratios[i] - exp[i]) < 0.05 for i in range(Sr))
    emit(f"  {'[oracle] y[0]/y[1]: 首块=3 其余=2（窗口2r+通道分半）':46s} "
         f"got={[round(v, 3) for v in ratios]} exp={exp}  {'✅' if ok else '❌'}")
    # 反例判据：若通道不分半 ⇒ 全 1；若窗口只有 r 行 ⇒ 全部为 3
    naive_all1 = all(abs(v - 1.0) < 0.05 for v in ratios)
    naive_all3 = all(abs(v - 3.0) < 0.05 for v in ratios)
    emit(f"  {'  反例判别（全1=未分半 / 全3=窗口未扩展）':46s} "
         f"全1={naive_all1} 全3={naive_all3}  ⇒ {'正确区分' if ok and not naive_all1 and not naive_all3 else '未区分'}")
    return ok


# ---------------------------------------------------------------------------
# shape 契约（compressor_tiling.cpp:829-852）
# ---------------------------------------------------------------------------
def case_shape_contract():
    R, D, RD, H = 4, 128, 64, 16
    x, wkv, wgate, ape, nw, rs, rc = build(1, 8, H, D, R, RD, torch.bfloat16, True, 0)
    ok = True
    for tag, args in (
        ("wkv.shape[0] != coff*headDim", dict(wkv=wkv[:D].contiguous())),
        ("ape.shape != [r, coff*headDim]", dict(ape=ape[:, :D].contiguous())),
        ("norm_weight 长度 != headDim", dict(norm_weight=torch.ones(D // 2, device=DEV))),
    ):
        kw = dict(x=x, wkv=wkv, wgate=wgate, ape=ape, norm_weight=nw,
                  rope_sin=rs, rope_cos=rc, rope_head_dim=RD, cmp_ratio=R, coff=2)
        kw.update(args)
        try:
            compressor(**kw)
            torch.npu.synchronize()
            emit(f"  {'[契约] ' + tag:46s} 未报错  ❌")
            ok = False
        except (ValueError, NotImplementedError) as e:
            emit(f"  {'[契约] ' + tag:46s} 已报错 ✅  {str(e)[:70]}")
    # 非法 (r, coff, D) 组合
    try:
        compressor(x, wkv, wgate, ape, nw, rs, rc, rope_head_dim=RD, cmp_ratio=8, coff=2)
        torch.npu.synchronize()
        emit(f"  {'[契约] (r=8,coff=2) 非合法组合':46s} 未报错  ❌")
        ok = False
    except (ValueError, NotImplementedError) as e:
        emit(f"  {'[契约] (r=8,coff=2) 非合法组合':46s} 已报错 ✅  {str(e)[:70]}")
    return ok


# ---------------------------------------------------------------------------
# 流式两段（state_cache 跨调用）：首段写 state，第二段的窗前 r 行从 state 读回
# ---------------------------------------------------------------------------
def case_stream(name, S1, S2, H, D, R, RD, dtype, cache_mode, seed=3):
    S = S1 + S2
    x, wkv, wgate, ape, nw, rs, rc = build(1, S, H, D, R, RD, dtype, True, seed)
    nblk = (S + 15) // 16
    stride0 = 16 * 4 * D                       # 行宽 4D
    if cache_mode == 1:
        sc = torch.zeros(nblk, 16, 4 * D, dtype=torch.float32, device=DEV)
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, nblk)
        max_blk = nblk
    else:
        sc = torch.zeros(1, 16, 4 * D, dtype=torch.float32, device=DEV)
        bt = torch.zeros(1, dtype=torch.int32, device=DEV)
        max_blk = 1

    Sr_total = (S + R - 1) // R
    sp1 = torch.zeros(1, dtype=torch.int32, device=DEV)
    su1 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    o1, sc1 = compressor(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                         rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(),
                         sc, bt, seqused=su1, start_pos=sp1,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=16)
    sp2 = torch.tensor([S1], dtype=torch.int32, device=DEV)
    su2 = torch.tensor([S2], dtype=torch.int32, device=DEV)
    n2 = (S1 + S2) // R - S1 // R
    o2, sc2 = compressor(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                         rs[:, S1 // R:S1 // R + n2].contiguous(),
                         rc[:, S1 // R:S1 // R + n2].contiguous(),
                         sc1, bt, seqused=su2, start_pos=sp2,
                         rope_head_dim=RD, cmp_ratio=R, coff=2, rotary_mode=2,
                         cache_mode=cache_mode,
                         state_cache_stride_dim0=stride0, block_size=16)
    torch.npu.synchronize()

    common1 = ([0], [S1], [0], [0], [S1 // R], [0], R, D, RD, 1e-6, False, sc,
               bt.detach().cpu().reshape(-1).tolist(), 16, stride0, max_blk,
               cache_mode == 2, (1, S1 // R + 1, D), True)
    rA1, scR1 = ref_A(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                      rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(), *common1)
    rB1, scQ1 = ref_B(x[:, :S1].contiguous(), wkv, wgate, ape, nw,
                      rs[:, :S1 // R + 1].contiguous(), rc[:, :S1 // R + 1].contiguous(), *common1)
    common2 = ([S1], [S2], [0], [S1 // R], [n2], [0], R, D, RD, 1e-6, False, scR1,
               bt.detach().cpu().reshape(-1).tolist(), 16, stride0, max_blk,
               cache_mode == 2, (1, n2, D), True)
    rA2, scR2 = ref_A(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                      rs[:, S1 // R:S1 // R + n2].contiguous(),
                      rc[:, S1 // R:S1 // R + n2].contiguous(), *common2)
    rB2, _ = ref_B(x[:, S1:].contiguous(), wkv, wgate, ape, nw,
                   rs[:, S1 // R:S1 // R + n2].contiguous(),
                   rc[:, S1 // R:S1 // R + n2].contiguous(), *common2)

    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dtype]
    e1 = max((o1.detach().cpu().double() - rA1).abs().max().item(),
             (o1.detach().cpu().double() - rB1).abs().max().item())
    e2 = max((o2.detach().cpu().double() - rA2).abs().max().item(),
             (o2.detach().cpu().double() - rB2).abs().max().item())
    es = (sc2.detach().cpu().double().reshape(-1) - scR2).abs().max().item()
    xchk = max((scR1 - scQ1).abs().max().item(), (rA1 - rB1).abs().max().item(),
               (rA2 - rB2).abs().max().item())
    ok = e1 <= floor and e2 <= floor and es <= 2e-6
    emit(f"  {name:46s} seg1={e1:.3e} seg2={e2:.3e}(<= {floor:.0e}) "
         f"state={es:.3e}(<= 2e-06) |A-B|={xchk:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
emit("=" * 112)
emit("compressor coff=2 (COFF::OVERLAP)：Triton vs 两个独立 fp64 参考")
emit("=" * 112)

allok = True

emit()
emit("--- ① 可辨识 oracle（手算期望值）---")
allok &= case_oracle()

emit()
emit("--- ② shape 契约（compressor_tiling.cpp:829-852 / :958-964）---")
allok &= case_shape_contract()

emit()
emit("--- ③ 数值对照（errA/errB 分别对两个独立参考）---")
allok &= case("BS B=1 S=24  H=64 D=128 R=4 RD=64 bf16 interleave", 1, 24, 64, 128, 4, 64,
              torch.bfloat16, 2)
allok &= case("BS B=1 S=24  H=64 D=128 R=4 RD=64 fp16 half", 1, 24, 64, 128, 4, 64,
              torch.float16, 1, seed=1)
allok &= case("BS B=1 S=22  H=64 D=128 R=4 RD=64 bf16 (非整除)", 1, 22, 64, 128, 4, 64,
              torch.bfloat16, 2, seed=2)
allok &= case("BS B=2 S=16  H=64 D=128 R=4 RD=64 bf16", 2, 16, 64, 128, 4, 64,
              torch.bfloat16, 2, seed=4)
allok &= case("BS B=1 S=4   H=64 D=128 R=4 RD=64 bf16 首块无历史", 1, 4, 64, 128, 4, 64,
              torch.bfloat16, 2, seed=5)
allok &= case("TH B=2 lens=[16,28] H=64 D=128 R=4 RD=64 bf16", 0, 0, 64, 128, 4, 64,
              torch.bfloat16, 2, seed=6, is_bs=False, lens=[16, 28])
allok &= case("BS B=1 S=24  H=64 D=512 R=4 RD=64 bf16 (D=512)", 1, 24, 64, 512, 4, 64,
              torch.bfloat16, 2, seed=7)

emit()
emit("--- ④ 流式两段（state_cache 跨调用）---")
allok &= case_stream("stream S1=18 S2=14 (start_pos=18) CONTINUOUS bf16", 18, 14, 64, 128, 4, 64,
                     torch.bfloat16, 1)
allok &= case_stream("stream S1=18 S2=14 (start_pos=18) CYCLE bf16", 18, 14, 64, 128, 4, 64,
                     torch.bfloat16, 2)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
try:
    open("/root/dsh_work/zeta/coff2_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
