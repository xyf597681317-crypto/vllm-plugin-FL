"""chunk_fwd_o —— AscendC → Triton

来源：csrc/ascend/moe/chunk_fwd_o/（vllm-ascend 侧同源：csrc/moe/chunk_fwd_o_vllm/）
接口（op_host/chunk_fwd_o_def.cpp L25-83 + aclnn_chunk_fwd_o.cpp L51-71/110-124）：
    Input : q   [B, Hk, T, K]      bf16/fp16
            k   [B, Hk, T, K]      bf16/fp16
            v   [B, Hv, T, V]      bf16/fp16
            h   [B, Hv, NT, K, V]（varlen 时 [1, Hv, ΣNT, K, V]）bf16/fp16
                —— 即 chunk_gated_delta_rule_fwd_h 产出的「每个 chunk 的入口状态」
            g   [B, Hv, T]         fp32/bf16/fp16（REQUIRED；log 空间）
            cu_seqlens     [N+1] int64 OPTIONAL
            chunk_offsets / chunk_indices [2*numChunks] int64 OPTIONAL（每 chunk 一对
                           (batchIdx, chunkInBatch)，scheduler L150-151 的读法）
    Output: o   [B, Hv, T, V]      同 q 的 dtype
    Attr  : scale（REQUIRED，默认 1.0）、chunk_size（REQUIRED，默认 64）

    torch 侧签名（vllm-ascend csrc/torch_binding.cpp L2048-2069 / L2854）：
        chunk_fwd_o_vllm(q, k, v, h, scale, *, g=None, g_gamma=None, cu_seqlens=None,
                         chunk_indices=None, chunk_size=None, transpose_state_layout=False)
        binding 里 `(void)g_gamma; (void)transpose_state_layout;` → 两者被显式忽略（L2064-2065）

offset 推导（op_kernel/arch22/gemm/block/block_scheduler_gdn_fwd_o.hpp L162-166）：
    qkOffset = (shapeBatch*Hk*seqlen + hk*seqlen + tokenOffset + ci*CS) * K      # q/k: [B,Hk,T,K]
    ovOffset = (shapeBatch*Hv*seqlen + hv*seqlen + tokenOffset + ci*CS) * V      # v/o: [B,Hv,T,V]
    gOffset  =  shapeBatch*Hv*seqlen + hv*seqlen + tokenOffset + ci*CS           # g:   [B,Hv,T]
    hOffset  = (shapeBatch*Hv*numChunks + hv*numChunks + chunkIdx) * K * V       # h:   [B,Hv,NT,K,V]
    （chunkIdx 为全局 chunk 序号；varlen 下 tokenOffset=cu_seqlens[b]、ci 来自 chunk_offsets 表）

公式（cube1=Q@Kᵀ → qkmask epilogue → cube2=Q@H → cube3=A@V → output epilogue）：
    A = Q @ Kᵀ                                   # cube1（blockTokens×blockTokens）
    A = A ⊙ exp(g_i - g_j) ⊙ causal(i>=j)        # qkmask epilogue L218-240（再 cast 回输入 dtype）
    oH = Q @ H                                   # cube2
    o  = scale * ( exp(g) ⊙ oH + A @ V )         # output epilogue L219-225：
                                                 #   Mul(h, exp(g)) → Add(a) → Muls(scale)
    ==> o[b,h,t,:] = scale * ( exp(g[t]) * (q[t] @ H[b,h,c(t)]) + Σ_{s≤t, 同 chunk} exp(g_t-g_s)(q_t·k_s) v_s )
    其中 hk = h // (Hv/Hk)，c(t) = t 所在 chunk。

实现：**一个 program 处理一个 chunk**（无需跨 chunk 递推 → 天然避开「循环内 2-D store」的后端坑，
见 后端崩溃最小复现.md）。每个 program 用 3 次 tl.dot。
数值细节：先把 (g_i-g_j) 做因果掩码再 exp（避免上三角出现无意义的大指数→inf），
与参考实现的「先 exp 再掩码」数学等价。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["chunk_fwd_o"]


@triton.jit
def _chunk_fwd_o_kernel(
    Q, K, V, H, G, O, CH_B, CH_CI, CH_T0, CH_LEN, CH_HI,
    T, Hk, Hv, Kd, Vd, SCALE, H_STRIDE,
    BLK_T: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_V: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_c = tl.program_id(2)

    b = tl.load(CH_B + pid_c).to(tl.int32)
    ci = tl.load(CH_CI + pid_c).to(tl.int32)
    tok0 = tl.load(CH_T0 + pid_c).to(tl.int32)
    bt = tl.load(CH_LEN + pid_c).to(tl.int32)
    hb = tl.load(CH_HI + pid_c)                     # h 的 batch 基址（非 varlen=b*Hv*NT，varlen=0）

    hk = pid_h // (Hv // Hk)

    t_off = tl.arange(0, BLK_T)
    tmask = t_off < bt
    k_off = tl.arange(0, BLK_K)
    kmask = k_off < Kd
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < Vd

    qbase = (b * Hk + hk) * T + tok0
    vbase = (b * Hv + pid_h) * T + tok0
    q = tl.load(Q + (qbase + t_off[:, None]) * Kd + k_off[None, :],
                mask=tmask[:, None] & kmask[None, :], other=0.0)
    kk = tl.load(K + (qbase + t_off[:, None]) * Kd + k_off[None, :],
                 mask=tmask[:, None] & kmask[None, :], other=0.0)
    vv = tl.load(V + (vbase + t_off[:, None]) * Vd + v_off[None, :],
                 mask=tmask[:, None] & vmask[None, :], other=0.0)
    gg = tl.load(G + vbase + t_off, mask=tmask, other=0.0).to(tl.float32)

    hptr = H + ((hb + pid_h * H_STRIDE + ci) * Kd + k_off[:, None]) * Vd + v_off[None, :]
    hh = tl.load(hptr, mask=kmask[:, None] & vmask[None, :], other=0.0)

    # A = Q @ Kᵀ，再做 exp(g_i-g_j) ⊙ causal
    a = tl.dot(q, tl.trans(kk)).to(tl.float32)                     # [BLK_T, BLK_T]
    dg = gg[:, None] - gg[None, :]
    causal = t_off[:, None] >= t_off[None, :]
    dg = tl.where(causal & tmask[:, None] & tmask[None, :], dg, 0.0)
    a = a * tl.exp(dg)
    a = tl.where(causal & tmask[:, None] & tmask[None, :], a, 0.0)

    oh = tl.dot(q, hh).to(tl.float32)                              # [BLK_T, BLK_V]
    o = (tl.exp(gg)[:, None] * oh + tl.dot(a.to(vv.dtype), vv).to(tl.float32)) * SCALE

    tl.store(O + (vbase + t_off[:, None]) * Vd + v_off[None, :],
             o.to(O.dtype.element_ty), mask=tmask[:, None] & vmask[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float = 1.0,
    *,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
) -> torch.Tensor:
    """等价于 AscendC 的 ChunkFwdO（chunk 内注意力输出）。

    签名与 torch binding `chunk_fwd_o_vllm`（vllm-ascend csrc/torch_binding.cpp L2854）一致。

    Args:
        q/k: [B, Hk, T, K]；v: [B, Hv, T, V]；h: [B, Hv, NT, K, V]
        scale: 位置参数（attr scale）
        g: [B, Hv, T]（log 空间）—— **必填**（aclnn L57 "g must not be nullptr" / def 里 g 是 REQUIRED）
        g_gamma: 不支持（binding L2064 `(void)g_gamma;` 直接丢弃 → 传了会静默算错，故显式报错）
        cu_seqlens: [N+1] int64 可选（varlen）
        chunk_indices: [2*numChunks] int64 可选（每 chunk 一对 (batchIdx, chunkInBatch)）；
                       不给时按 cu_seqlens/shape 自动推导（等价于 FLA prepare_chunk_indices）
        chunk_size / transpose_state_layout

    Returns:
        o: [B, Hv, T, V]，dtype 同 q
    """
    if g_gamma is not None:
        raise NotImplementedError(
            "g_gamma 不支持：torch binding L2064 对它是 `(void)g_gamma;`（直接丢弃），"
            "op_host def 也没有该输入；传了会被静默忽略从而算错，故显式拒绝。")
    if transpose_state_layout:
        raise NotImplementedError(
            "transpose_state_layout=True 不支持：binding L2065 `(void)transpose_state_layout;`，"
            "op_host def 无该属性（h 恒为 [B,Hv,NT,K,V] row-major）。")
    if g is None:
        raise ValueError("g 必填：aclnn_chunk_fwd_o_vllm.cpp L57 'g must not be nullptr'，"
                         "op_host def 里 g 是 REQUIRED 输入。")
    assert q.shape == k.shape, f"q/k 形状不一致 {tuple(q.shape)} vs {tuple(k.shape)}"
    B, Hk, T, Kd = (int(s) for s in q.shape)
    Hv = int(v.shape[1])
    Vd = int(v.shape[3])
    assert Hv % Hk == 0, f"Hv={Hv} 必须是 Hk={Hk} 的整数倍"
    assert tuple(int(s) for s in v.shape) == (B, Hv, T, Vd), \
        f"v shape {tuple(v.shape)} != {(B, Hv, T, Vd)}"
    assert h.dim() == 5 and int(h.shape[1]) == Hv and int(h.shape[3]) == Kd and int(h.shape[4]) == Vd, \
        f"h shape {tuple(h.shape)} 应为 [B,Hv,NT,{Kd},{Vd}]"

    is_varlen = cu_seqlens is not None
    if is_varlen:
        cu_host = cu_seqlens.contiguous().to(torch.int64).cpu().tolist()
        nb = len(cu_host) - 1
        lens = [max(0, int(cu_host[i + 1] - cu_host[i])) for i in range(nb)]
        nt_per = [ceil_div(l, chunk_size) for l in lens]
        tot_chunks = sum(nt_per)
        if chunk_indices is not None:
            ci_host = chunk_indices.contiguous().to(torch.int64).cpu().tolist()
            assert len(ci_host) == 2 * tot_chunks, \
                f"chunk_indices 长度 {len(ci_host)} != 2*numChunks={2 * tot_chunks}"
            pairs = [(int(ci_host[2 * c]), int(ci_host[2 * c + 1])) for c in range(tot_chunks)]
        else:
            pairs = []
            for bi in range(nb):
                for c in range(nt_per[bi]):
                    pairs.append((bi, c))
        h_stride = tot_chunks
        h_base = [0] * tot_chunks
        tot_tok = sum(lens)
    else:
        nt = ceil_div(T, chunk_size)
        tot_chunks = B * nt
        pairs = [(b, c) for b in range(B) for c in range(nt)]
        h_stride = nt
        h_base = [b * Hv * nt for b in range(B) for _ in range(nt)]
        lens = [T] * B
        tot_tok = T

    starts = []
    if is_varlen:
        acc = 0
        for l in lens:
            starts.append(acc)
            acc += l
    else:
        starts = [0] * B

    ch_b, ch_ci, ch_t0, ch_len, ch_hi = [], [], [], [], []
    for c, (bi, ci) in enumerate(pairs):
        bi = min(max(bi, 0), max(len(lens) - 1, 0))
        t0 = starts[bi] + ci * chunk_size
        bt = min(chunk_size, max(0, lens[bi] - ci * chunk_size))
        # varlen 时 q/k/v/g 的形状是 [1, H, T_total, D]，scheduler L162-165 用 shapeBatchIdx=0，
        # 批内偏移由 tokenOffset(=cu_seqlens[b]) 承担 → 这里必须填 0，不能填真实 batch 号
        ch_b.append(0 if is_varlen else bi)
        # h 的 chunk 下标：varlen 用**全局** chunk 序号（scheduler L164 的 chunkIdx），
        # 非 varlen 用批内 chunk 序号（chunkOffset=0，hv*numChunks 由 H_STRIDE 承担）
        ch_ci.append(c if is_varlen else ci)
        ch_t0.append(t0)
        ch_len.append(bt)
        ch_hi.append(h_base[c] if c < len(h_base) else 0)

    dev = q.device

    def _t(vals, dt=torch.int32):
        return torch.tensor(vals, dtype=dt, device=dev)

    o = torch.zeros((B, Hv, T, Vd), dtype=q.dtype, device=dev)
    BLK_T = max(16, min(next_pow2(chunk_size), 128))
    BLK_K = next_pow2(Kd)
    # tl.dot 要求 N>=16；UB=192KB，A 为 [BLK_T,BLK_T] fp32，BLK_V 取 64 以内
    BLK_V = max(16, min(next_pow2(Vd), 64))
    grid = (Hv, ceil_div(Vd, BLK_V), tot_chunks)
    _chunk_fwd_o_kernel[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), h.contiguous(), g.contiguous(),
        o, _t(ch_b), _t(ch_ci), _t(ch_t0), _t(ch_len), _t(ch_hi, torch.int64),
        tot_tok if is_varlen else T, Hk, Hv, Kd, Vd, float(scale), h_stride,
        BLK_T=BLK_T, BLK_K=BLK_K, BLK_V=BLK_V,
    )
    return o
