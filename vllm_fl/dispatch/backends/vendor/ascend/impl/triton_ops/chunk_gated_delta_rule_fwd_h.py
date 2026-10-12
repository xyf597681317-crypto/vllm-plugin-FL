"""chunk_gated_delta_rule_fwd_h —— AscendC → Triton

来源：csrc/ascend/moe/chunk_gated_delta_rule_fwd_h/
接口（op_host/chunk_gated_delta_rule_fwd_h_def.cpp L23-94 + _tiling.cpp L59-167）：
    Input : k             [B, Hk, T, K]      bf16/fp16     （DIM: 0=batch,1=head,2=seq,3=dim）
            w             [B, Hv, T, K]      bf16/fp16
            u             [B, Hv, T, V]      bf16/fp16
            g             [B, Hv, T]         fp32 / bf16 / fp16（log 空间）
            inital_state  [B, Hv, K, stride0]  OPTIONAL
            cu_seqlens    [N+1] int64        OPTIONAL（varlen）
            chunk_indices (未使用)
    Output: h             [B, Hv, NT, K, V]     k 的 dtype
            v_new         [B, Hv, T, V]         k 的 dtype
            final_state   [B, Hv, K, V]         state dtype（output_final_state=true 时）
    Attr  : output_final_state(bool, 默认 false), chunk_size(int, 默认 64),
            inital_state_stride0(int，inital_state 的行 stride)

offset 推导（op_kernel/arch22/gemm/block/block_scheduler_gdn_fwd_h.hpp L213-229）：
    hSrcOffset = (shapeBatch*Hv*totalChunks + hv*totalChunks + chunkOffset + ci) * K * V   # L217
    hDstOffset = hSrcOffset + K*V                                                          # L218
    uvOffset   = (shapeBatch*Hv*totalTokens + hv*totalTokens + tokenOffset + ci*CS) * V    # L219
    wOffset    = (shapeBatch*Hv*totalTokens + hv*totalTokens + tokenOffset + ci*CS) * K    # L221
    wkOffset   = (shapeBatch*Hk*totalTokens + hk*totalTokens + tokenOffset + ci*CS) * K    # L220
    gOffset    =  shapeBatch*Hv*totalTokens + hv*totalTokens + tokenOffset + ci*CS         # L222
    finalState = (batchIdx*Hv + hv) * K * V                                                # L216
    hk = hv / (Hv/Hk)（L212）；blockTokens = 末 chunk 为真实 token 数（L225）
    ==> w/u/v_new/g 按 **v-head** 排布，k 按 k-head 排布

公式（op_kernel/arch22/gemm/kernel/gdn_fwd_h_kernel.hpp L259-301 + 两个 epilogue）：
    # 每 (batch, v-head) 独立，chunk 间串行；H = [K, V] 状态
    H = inital_state[b, hv]                                   # L315-360 写入 h[...,0]
    h[b, hv, 0] = cast(H)
    for ci in range(NT):
        bt     = blockTokens(ci)
        g_last = g[b, hv, chunk_start + bt - 1]               # vnew-epilogue L179（mActual-1）
        decay  = exp(g_last - g[chunk])                       # L184-191 Duplicate/Sub/Exp
        v_work = W_ci @ H                                     # cube1 L261-277
        vn     = U_ci - v_work                                # L216 Sub（fp32）
        v_new[chunk] = cast(vn)                               # L218 / L221 CAST_RINT
        vupd   = cast(vn * decay)                             # L224 Mul + L226 Cast（bf16 工作区）
        h_work = K_ci^T @ vupd                                # cube2 L285-297
        Hnew   = H * exp(g_last) + h_work                     # update-epilogue L126-143
        h[b, hv, ci+1] = cast(Hnew)             (ci+1 < NT)   # hDstOffset
        else if output_final_state: final_state[b,hv] = cast(Hnew)   # L145-156
        H = Hnew

    * h 的语义 = **每个 chunk 的入口状态**：h[...,0] 来自 inital_state，最后一个槽由 ci=NT-2 写入；
      最后一个 chunk 之后的状态只进 final_state。

实现说明 / 语义假设：
  0) **后端 workaround（重要）**：本内核的 state 落盘（h / final_state）原本写成「chunk 循环内的
     2-D tl.store」，会触发 Ascend bishengir 在 ConvertLinalgRToBinary 阶段的 LLVM crash
     （fp32 state 时必崩，见 _analysis/triton_ops/后端崩溃最小复现.md）。现改成**等价的 1-D store**：
     值用 `tl.reshape` 展平（已实测保序），偏移用算术式重算（`i1 // BLK_V`、`i1 % BLK_V`）。
     修复后 fp32 initial_state 路径可用，21/21 用例（含数值对拍）全部 PASS。
     注意：偏移张量**不能**用 `tl.reshape(广播结果)`，那样会错位（实测数值错）。
  1) H 全程保留在 fp32 寄存器，只在写 h/v_new/final_state 时按输出 dtype 取整。AscendC 会把 H
     以输入 dtype 存回 h 再读（每个 chunk 一次 bf16/fp16 往返），本实现省掉这次往返，属于同等
     量级的 dtype 舍入（对 fp64 参考只会更准）。
  2) h 更新用的衰减因子取「本 chunk 最后一个真实 token」的 g（= g[bt-1]）。AscendC 的 update
     epilogue L113 读的是 `gInput[chunkSize-1]`，末 chunk 不满时该位置可能越界；vnew epilogue
     L179 用的是 mActual-1（= bt-1），本实现与后者保持一致。
  3) 矩阵乘用「按 token 展开 + tl.sum 归约」实现（不依赖 tl.dot）：
     v_work_i = Σ_k W[i,k]*H[k,:]，h_work = Σ_i k_i ⊗ vupd_i，与 GEMM 数学等价。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["chunk_gated_delta_rule_fwd_h"]


@triton.jit
def _cgdr_fwd_h_kernel(
    KPTR, WPTR, UPTR, GPTR, INIT, HOUT, VNEW, FINAL,
    W_BASE, U_BASE, G_BASE, K_BASE, H_BASE, SEQ_LEN, CH_OFF,
    TOT_TOK, NTOT, HV, HK, DK, DV, CS, STRIDE0,
    HAS_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_V: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_n = tl.program_id(2)

    lens = tl.load(SEQ_LEN + pid_n).to(tl.int32)
    if lens <= 0:
        return
    nchunks = (lens + CS - 1) // CS

    wb = tl.load(W_BASE + pid_n)
    ub = tl.load(U_BASE + pid_n)
    gb = tl.load(G_BASE + pid_n)
    kb = tl.load(K_BASE + pid_n)
    hb = tl.load(H_BASE + pid_n)

    hk = pid_h // (HV // HK)

    k_off = tl.arange(0, BLK_K)
    kmask = k_off < DK
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < DV
    tm = kmask[:, None] & vmask[None, :]

    if HAS_INIT:
        H = tl.load(
            INIT + ((pid_n * HV + pid_h) * DK + k_off[:, None]) * STRIDE0 + v_off[None, :],
            mask=tm, other=0.0,
        ).to(tl.float32)
    else:
        H = tl.zeros((BLK_K, BLK_V), dtype=tl.float32)

    h_hv_base = hb + pid_h * NTOT
    # workaround（Ascend bishengir bug）：循环内「2-D tl.store」会让 bishengir 在
    # ConvertLinalgRToBinary 阶段 LLVM crash（fp32 值无 dtype 转换时必崩）。state 落盘改成
    # 等价的 1-D store：值用 tl.reshape 展平（保序，已实测），偏移用算术式重算。
    # 注意：不能对「广播算出来的偏移张量」做 reshape（会错位）。详见 后端崩溃最小复现.md
    i1 = tl.arange(0, BLK_K * BLK_V)
    kk_i = i1 // BLK_V
    vv_i = pid_v * BLK_V + (i1 % BLK_V)
    off1 = kk_i * DV + vv_i
    fm1 = (kk_i < DK) & (vv_i < DV)
    tl.store(HOUT + h_hv_base * DK * DV + off1,
             tl.reshape(H.to(HOUT.dtype.element_ty), [BLK_K * BLK_V]), mask=fm1)

    gptr = GPTR + gb + pid_h * TOT_TOK
    wptr = WPTR + (wb + pid_h * TOT_TOK) * DK
    kptr = KPTR + (kb + hk * TOT_TOK) * DK

    ci = 0
    while ci < nchunks:
        tok = ci * CS
        bt = tl.minimum(CS, lens - tok)

        g_last = tl.load(gptr + tok + bt - 1).to(tl.float32)

        up = UPTR + (ub + pid_h * TOT_TOK + tok) * DV + v_off     # 读 u
        vp = VNEW + (ub + pid_h * TOT_TOK + tok) * DV + v_off     # 写 v_new
        hwork = tl.zeros((BLK_K, BLK_V), dtype=tl.float32)

        i = 0
        while i < bt:
            wi = tl.load(wptr + (tok + i) * DK + k_off, mask=kmask, other=0.0).to(tl.float32)
            kk = tl.load(kptr + (tok + i) * DK + k_off, mask=kmask, other=0.0).to(tl.float32)
            ui = tl.load(up + i * DV, mask=vmask, other=0.0).to(tl.float32)
            gi = tl.load(gptr + tok + i).to(tl.float32)
            di = tl.exp(g_last - gi)
            vw = tl.sum(wi[:, None] * H, axis=0)          # W_i @ H -> [BLK_V]
            vn = ui - vw
            tl.store(vp + i * DV, vn.to(VNEW.dtype.element_ty), mask=vmask)
            hwork += kk[:, None] * (vn * di)[None, :]
            i += 1

        Hnew = H * tl.exp(g_last) + hwork

        if ci + 1 < nchunks:
            tl.store(HOUT + (h_hv_base + ci + 1) * DK * DV + off1,
                     tl.reshape(Hnew.to(HOUT.dtype.element_ty), [BLK_K * BLK_V]), mask=fm1)
        else:
            if STORE_FINAL:
                tl.store(FINAL + (pid_n * HV + pid_h) * DK * DV + off1,
                         tl.reshape(Hnew.to(FINAL.dtype.element_ty), [BLK_K * BLK_V]), mask=fm1)

        H = Hnew
        ci += 1


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    initial_state_stride0: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """等价于 AscendC 的 ChunkGatedDeltaRuleFwdH（跨 chunk 状态递推）。

    **签名与调用点/torch binding 完全一致**（vllm-ascend csrc/torch_binding.cpp L1981-1994 与
    L2849 的 schema：k, w, u 位置参数，其余 keyword-only，默认值同）。

    Args:
        k: [B, Hk, T, K]（varlen 时 [1, Hk, T_total, K]）
        w: [B, Hv, T, K]
        u: [B, Hv, T, V]
        g: [B, Hv, T]（log 空间标量门控；本实现要求非 None）
        gk: [B, Hv, T, K] —— **显式不支持**，见下
        initial_state: [B, Hv, K, initial_state_stride0] 可选（op_host 里拼写为 inital_state）
        output_final_state: 是否返回 final_state（False 时第三个返回值为 None）
        chunk_size: 默认 64
        save_new_value: 仅支持 True（见下）
        cu_seqlens: [N+1] int64 可选（varlen）
        chunk_indices: 可选；仅用于推导 h 的 NT 维（torch_binding L2011-2016：NT = len//2）
        use_exp2 / transpose_state_layout: 仅支持默认值（见下）
        initial_state_stride0: L0 属性 inital_state_stride0，默认 = V

    Returns:
        (h [B,Hv,NT,K,V], v_new [同 u], final_state [N,Hv,K,V] 或 None)

    参数语义（全部有源码依据，不支持的显式报错，绝不静默忽略）：
      * save_new_value  : aclnn L195-196 "reserved and only true is supported" → False 抛 NotImplementedError
      * use_exp2        : aclnn L197-198 "reserved and only false is supported" → True 抛 NotImplementedError
                          （注：vllm-ascend 的 arch22/arch35 KDA 版本用 exp2(g*LN2) 路径，
                            本次迁移的 AscendC 源码 csrc/ascend/moe/chunk_gated_delta_rule_fwd_h 无该分支）
      * transpose_state_layout: aclnn L199-200 同上 → True 抛 NotImplementedError
      * gk              : 本次迁移的 AscendC 内核（op_kernel/arch22 + scheduler/epilogue）只消费
                          3D 标量门控 g，从未 load gk；vllm-ascend 的新版 KDA 内核才逐 K 使用 gk。
                          数值语义无法从本仓库源码确定 → gk 非 None 抛 NotImplementedError
                          （上层调用点 sunrise/impl/fla/chunk_h.py L117 同样在 gk 非 None 时走 fallback）
      * g 为 None       : aclnn L188-189 "Either g or gk must be provided."，而 gk 又不支持 → ValueError
      * chunk_indices   : 只影响 NT（h 的分配形状）；内核 scheduler 解析后未使用
    """
    assert k.dim() == 4 and w.dim() == 4 and u.dim() == 4, "k/w/u 必须是 4 维 [B,H,T,D]"

    # ---- 调用点契约校验（对应 aclnn CheckReservedOptions / CheckGateOptionalNonNull / CheckGkParams）----
    if not save_new_value:
        raise NotImplementedError(
            "save_new_value=False 不支持：aclnn_chunk_gated_delta_rule_fwd_h.cpp L195-196 "
            "'save_new_value is reserved and only true is supported.'")
    if use_exp2:
        raise NotImplementedError(
            "use_exp2=True 不支持：aclnn L197-198 'use_exp2 is reserved and only false is supported.'；"
            "本次迁移的 AscendC 源码（csrc/ascend/moe/chunk_gated_delta_rule_fwd_h/op_kernel）没有 exp2 分支")
    if transpose_state_layout:
        raise NotImplementedError(
            "transpose_state_layout=True 不支持：aclnn L199-200 "
            "'transpose_state_layout is reserved and only false is supported.'")
    if gk is not None:
        raise NotImplementedError(
            "gk 不支持：本次迁移的 AscendC 内核只消费 3D 标量门控 g（scheduler gOffset 无 K 维），"
            "从未 load gk；仅 vllm-ascend 的 arch22/arch35 KDA 版本逐 K 使用 gk，其数值语义无法从本源码确定。"
            "调用点 sunrise/impl/fla/chunk_h.py L117 在 gk 非 None 时同样走 fallback。")
    if g is None:
        raise ValueError("g 不能为 None：aclnn L188-189 'Either g or gk must be provided.'，且 gk 不支持")

    Hk = int(k.shape[1])
    DK = int(w.shape[3])
    assert int(k.shape[3]) == DK, f"k/w 的 K 维不一致: {k.shape[3]} vs {DK}"
    Hv = int(w.shape[1])
    V = int(u.shape[3])
    assert int(u.shape[1]) == Hv, "w/u 的 head 数不一致"
    assert Hv % Hk == 0, f"Hv={Hv} 必须是 Hk={Hk} 的整数倍"
    assert int(g.shape[0]) == int(k.shape[0]) or cu_seqlens is not None, "g 的 batch 维与 k 不一致"

    g = g.contiguous()
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        B = int(k.shape[0])
        lens = [int(k.shape[2])] * B
        starts = [0] * B
        sb = list(range(B))
        tot_tok = int(k.shape[2])
    else:
        cu_host = cu_seqlens.contiguous().to(torch.int64).cpu().tolist()
        B = len(cu_host) - 1
        lens = [max(0, int(cu_host[b + 1] - cu_host[b])) for b in range(B)]
        starts = [int(cu_host[b]) for b in range(B)]
        sb = [0] * B
        tot_tok = sum(lens)

    nt_per = [ceil_div(l, chunk_size) for l in lens]
    tot_chunks = sum(nt_per)

    # h 的分配形状按 torch_binding L2011-2016/L2018 推导：NT = len(chunk_indices)//2 或 ceil(T/CS)。
    # 注意 varlen 下 Σ ceil(len_b/CS) 可能 **大于** ceil(Σlen_b/CS)（如 lens=[65,63,1], CS=64：
    # 4 > 3），此时 binding 的 NT 会小于内核实际写入的槽数，故取两者较大值以免越界。
    if chunk_indices is not None:
        nt_bind = len(chunk_indices) // 2
        assert nt_bind > 0, f"chunk_indices 长度 {len(chunk_indices)} 非法（需 2*NT）"
    else:
        nt_bind = ceil_div(int(k.shape[2]), int(chunk_size))
    nt_needed = tot_chunks if is_varlen else (nt_per[0] if nt_per else 0)
    nt_alloc = max(nt_bind, nt_needed)

    if not is_varlen:
        assert len(set(nt_per)) <= 1, "非 varlen 模式下各 batch 的 chunk 数必须一致"
        nt = nt_per[0] if B > 0 else 0
        h_shape = (B, Hv, nt_alloc, DK, V)
        # 非 varlen（scheduler L217）：offset = (b*Hv + hv)*NT + ci，NT 为每 batch 的 chunk 数
        h_stride = nt
        h_base = [b * Hv * nt for b in range(B)]
    else:
        h_shape = (1, Hv, nt_alloc, DK, V)
        # varlen（scheduler L217/200）：shapeBatch=0，offset = hv*totalChunks + chunkOffset + ci
        h_stride = tot_chunks
        h_base = [sum(nt_per[:b]) for b in range(B)]

    dev = k.device
    # 与 torch_binding L2018-2019 一致：h/v_new 都是 zeros（未写入的槽保持 0）
    h = torch.zeros(h_shape, dtype=k.dtype, device=dev)
    v_new = torch.zeros(u.shape, dtype=u.dtype, device=dev)

    use_init = initial_state is not None
    stride0 = int(initial_state_stride0) if initial_state_stride0 else V
    init = initial_state.contiguous() if use_init else torch.empty(1, dtype=k.dtype, device=dev)
    final_dtype = initial_state.dtype if use_init else k.dtype
    n_final = B
    final = torch.empty((n_final, Hv, DK, V), dtype=final_dtype, device=dev)

    # 每 batch 的元素基址（kernel 内用 batch 内局部 token 下标）
    g_base = [sb[b] * Hv * tot_tok + starts[b] for b in range(B)]
    u_base = [sb[b] * Hv * tot_tok + starts[b] for b in range(B)]
    w_base = [sb[b] * Hv * tot_tok + starts[b] for b in range(B)]
    k_base = [sb[b] * Hk * tot_tok + starts[b] for b in range(B)]
    choff = [sum(nt_per[:b]) for b in range(B)]

    def _t(vals):
        return torch.tensor(vals, dtype=torch.int64, device=dev)

    BLK_K = next_pow2(DK)
    # UB=192KB：H/hwork/[BLK_K,BLK_V] 临时量都要驻留，V 分块取 32 以免 bishengir ub overflow
    BLK_V = min(next_pow2(V), 32)
    grid = (Hv, ceil_div(V, BLK_V), max(B, 1))
    _cgdr_fwd_h_kernel[grid](
        k.contiguous(), w.contiguous(), u.contiguous(), g, init, h, v_new, final,
        _t(w_base), _t(u_base), _t(g_base), _t(k_base), _t(h_base), _t(lens), _t(choff),
        tot_tok, h_stride, Hv, Hk, DK, V, int(chunk_size), stride0,
        HAS_INIT=bool(use_init),
        STORE_FINAL=bool(output_final_state),
        BLK_K=BLK_K, BLK_V=BLK_V,
    )
    return h, v_new, (final if output_final_state else None)
