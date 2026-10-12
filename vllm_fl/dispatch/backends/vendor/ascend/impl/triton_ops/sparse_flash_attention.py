"""sparse_flash_attention —— AscendC → Triton（支持 BSND / TND query + BSND / PA_BSND KV）

来源：csrc/ascend/attention/sparse_flash_attention/
接口（op_host/sparse_flash_attention_def.cpp L23-88）：
    Input : query      fp16/bf16   layout_query="BSND" → [B,S1,N1,512]
                                   layout_query="TND"  → [T,N1,512]（各序列 token 拼接）
            key        fp16/bf16   layout_kv="BSND"    → [B,S2,N2,512]
                                   layout_kv="PA_BSND" → [numBlocks, blockSize, N2, 512]
            value      同 key
            sparse_indices int32   BSND → [B,S1,N2,K]；TND → [T,N2,K]
            query_rope fp16/bf16   与 query 同 layout，最后一维 64
            key_rope   同 key
            block_table int32 [B, maxBlockNumPerBatch]（PA 必填）
            actual_seq_lengths_query [B] 或 [B+1]（累积）
            actual_seq_lengths_kv    [B] 或 [B+1]（累积）
    Attr  : scale_value, sparse_block_size(1), sparse_mode(3 下三角),
            attention_mode(2), return_softmax_lse(False), pre/next_tokens(INT64_MAX)

公式出处：
  op_kernel/arch35/sparse_flash_attention_service_vector_mla.h
    L790  dSizeRope = 64                          → KV 行 = [512 dense | 64 rope]
    L791  softmaxScale = scaleValue
    L470-529 在线 softmax；L572 LastDivNew：输出 = acc / sum
    L266-310 keyRopeGm[keyOffset * 64] / blkTableSrcStride：
        PA 分页寻址由 block_table 把逻辑块映射到物理块
  ⇒ MLA 稀疏注意力核心：
        j 取自 sparse_indices[b, s1, n2, :]（-1 无效）
        kv_row(j) = BSND: b*S2 + j
                    PA_BSND: block_table[b, j//blockSize]*blockSize + j%blockSize
        qk    = ( q·kv[j] + q_rope·kv_rope[j] ) * scale_value
        下三角：j <= s1 + (kvLen[b] - qLen[b])
        softmax 只在被选中的位置；out = Σ p·v / Σ p

layout 归一化（host 侧只做**布局簿记**，核心数学仍在 Triton）：
    - TND：由 actual_seq_lengths_query 得到每个 token 的 (batch, 序列内位置) 与每 batch 的 q 长度
      （等价于 cu_seqlens 的 searchsorted；vLLM 同样在 host 侧预计算 seq_idx）
    - PA_BSND：分页寻址在 **kernel 内**用 block_table 完成

**已实现的布局组合**：query∈{BSND,TND} × kv∈{BSND,PA_BSND}。
**简化点**：不做多核 Cube/Vector 流水；两趟算法（先算分落 GM，再归一化加权 V）；
          sparse_block_size>1 按"每个 index 展开成连续 bs 个 KV 行"实现。

后端坑（910_93 + BiShengIR，实测见 `_probe_sfa_ilv.py`）：
  `sparse_block_size > 1` 时，两趟算法中间 scratch `SB` 若写成**交织布局**
  `SB[row, slot, sub]`（store 偏移 `slot*BS + sub`，即 1-D 步长=BS 的跨步写 + 掩码），
  BiShengIR 会在 `InterleaveOptimization.cpp:635` 的
  `InterleaveStatusWithMaskOptimization` 断言失败 → **SIGABRT（RC=134，编译期崩溃）**：
      Assertion `llvm::isa<Attribute>(extractSizes.back())' failed
  触发条件是「**跨步 + 掩码**」两者同时具备（去掉掩码、改成连续、或改成 2-D [BLK,BS]
  都不触发，均实测）。`SB` 是内部 scratch，故这里直接改成**块状布局** `SB[row, sub, slot]`
  （store 连续、无跨步），`_sfa_out_kernel` 的 lane→(slot, sub) 映射同步改为 sub-major；
  求和/取最大值与 lane 顺序无关，所以数值语义不变。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_flash_attention"]

_NEG = -1.0e30
_D_ROPE = 64


@triton.jit
def _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2,
            PA: tl.constexpr, vld):
    """把逻辑 KV 位置 j 解析成分页/连续 KV cache 里的**行起始元素偏移**。

        BSND    : row = b*S2 + j
        PA_BSND : row = block_table[b, j//BLOCK_SIZE]*BLOCK_SIZE + (j % BLOCK_SIZE)
    返回 (row * N2 + n2) * D，即该 (row, head n2) 行在 KV cache 中的元素偏移。
    """
    if PA:
        page = tl.load(BT + b * MAX_BLOCKS + j // BLOCK_SIZE, mask=vld, other=0)
        kvrow = page * BLOCK_SIZE + (j % BLOCK_SIZE)
    else:
        kvrow = b * S2 + j
    return (kvrow * N2 + n2) * D


@triton.jit
def _sfa_score_kernel(
    Q, QR, KV, KVR, IDX, SB, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS, K_IDX, N_TOK,
    BLOCK_SIZE, MAX_BLOCKS, S2,
    SCALE,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr, DR: tl.constexpr,
    BLK_D: tl.constexpr, BLK_R: tl.constexpr,
    BLK_K: tl.constexpr, BS: tl.constexpr,
):
    row = tl.program_id(0)                 # 一行 = 一个 (token, q head)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    dr = tl.arange(0, BLK_R)
    drm = dr < DR

    q = tl.load(Q + row * D + d, mask=dm, other=0.0).to(tl.float32)
    qr = tl.load(QR + row * DR + dr, mask=drm, other=0.0).to(tl.float32)
    idx_base = (t * N2 + n2) * K_IDX

    kvlen = tl.load(KVLEN + b)
    if CAUSAL:
        j_hi = s1 + (kvlen - tl.load(QLEN + b))
    else:
        j_hi = S2 - 1

    for c in range(0, tl.cdiv(K_IDX, BLK_K)):
        slot = c * BLK_K + tl.arange(0, BLK_K)
        sm = slot < K_IDX
        i0 = tl.load(IDX + idx_base + slot, mask=sm, other=-1)
        for sub in tl.static_range(BS):
            j = i0 * BS + sub
            vld = sm & (i0 >= 0) & (j >= 0) & (j <= j_hi) & (j < kvlen)
            base = _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2, PA, vld)
            k = tl.load(KV + base[:, None] + d[None, :],
                        mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
            baser = _kv_row(KVR, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, DR, S2, PA, vld)
            kr = tl.load(KVR + baser[:, None] + dr[None, :],
                         mask=vld[:, None] & drm[None, :], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) + tl.sum(kr * qr[None, :], axis=1)
            s = tl.where(vld, s * SCALE, NEG)
            # ⚠️ SB 的内部布局是 **块状** `SB[row, sub, slot]`（sub 在外、slot 在内），
            # 不能写成交织的 `slot*BS + sub`：
            # 「1-D 跨步(步长=BS) + 掩码」的 store 会让 BiShengIR 在
            # `InterleaveOptimization.cpp:635 InterleaveStatusWithMaskOptimization`
            # 断言失败并 **SIGABRT（RC=134，编译期崩溃）**：
            #     Assertion `llvm::isa<Attribute>(extractSizes.back())' failed
            # 最小复现见 `_probe_sfa_ilv.py`：同一 kernel 里
            #   stride2_masked → RC=134（复现）；stride2_nomask → OK（**去掉掩码就不触发**）；
            #   contig_masked  → OK（改成连续）；twod → OK（改成 2-D [BLK,BS]）。
            # SB 只是本算子内部 scratch，布局可自由选择，所以这里改为连续写；
            # `_sfa_out_kernel` 里 lane→(slot, sub) 的映射同步改成 sub-major。
            tl.store(SB + row * (K_IDX * BS) + sub * K_IDX + slot, s, mask=sm)


@triton.jit
def _sfa_out_kernel(
    KV, SB, IDX, OUT, LSE_MAX, LSE_SUM, TOK_B, TOK_S1, BT,
    N1, N2, GS, K_IDX, N_TOK,
    BLOCK_SIZE, MAX_BLOCKS, S2, LSE_B_STRIDE, LSE_N2_STRIDE,
    RETURN_LSE: tl.constexpr,
    PA: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    total = K_IDX * BS
    ar = tl.arange(0, BLK_ALL)
    sv = tl.load(SB + row * total + ar, mask=ar < total, other=NEG)
    mx = tl.max(sv, axis=0)
    p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
    lse = tl.sum(p, axis=0)

    acc = tl.zeros([BLK_D], dtype=tl.float32)
    for c in range(0, tl.cdiv(total, BLK_K)):
        lane = c * BLK_K + tl.arange(0, BLK_K)
        sm = lane < total
        pc = tl.load(SB + row * total + lane, mask=sm, other=NEG)
        pc = tl.where(sm, tl.exp(pc - mx), 0.0)
        # SB 是**块状**布局 SB[row, sub, slot]（与 _sfa_score_kernel 的 store 一致）：
        # sub 是高位、slot 是低位。改回交织布局会触发 FLIR 编译期崩溃，见那边的注释。
        slot = lane % K_IDX
        sub = lane // K_IDX
        i0 = tl.load(IDX + (t * N2 + n2) * K_IDX + slot, mask=sm, other=-1)
        j = i0 * BS + sub
        vld = sm & (i0 >= 0) & (j >= 0) & (j < S2)
        base = _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2, PA, vld)
        v = tl.load(KV + base[:, None] + d[None, :],
                    mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(vld, pc, 0.0)[:, None] * v, axis=0)

    res = tl.where(lse > 0.0, acc / lse, tl.zeros([BLK_D], dtype=tl.float32))
    tl.store(OUT + row * D + d, res.to(OUT.dtype.element_ty), mask=dm)
    if RETURN_LSE:
        # 两种 layout 统一成 off = b*LSE_B_STRIDE + n2*LSE_N2_STRIDE + s1*GS + (n1%GS)
        # BSND: [B,N2,S1,gs] → (N2*S1*gs, S1*gs)；TND: [N2,T,gs] → (0, T*gs)
        tl.store(LSE_MAX + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), mx)
        tl.store(LSE_SUM + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), lse)


def _as_cumulative(seq, B, name):
    """把 actual_seq_lengths 归一化成累积形式（长度 B+1 视为已是 cu_seqlens）。"""
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B + 1:
        return s
    if s.numel() == B:
        return torch.cat([torch.zeros(1, dtype=torch.int64), s.cumsum(0)])
    raise ValueError(f"{name} 长度必须是 B={B} 或 B+1={B + 1}，实际 {s.numel()}")


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    scale_value: float = 1.0,
    sparse_block_size: int = 1,
    sparse_mode: int = 3,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    attention_mode: int = 2,
    pre_tokens: int | None = None,
    next_tokens: int | None = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 SparseFlashAttention。

    调用点签名（vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py L682-698）：
        npu_sparse_flash_attention(
            query=..., key=..., value=..., sparse_indices=..., scale_value=...,
            sparse_block_size=1, block_table=..., actual_seq_lengths_query=...,
            actual_seq_lengths_kv=..., query_rope=..., key_rope=...,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2)
    该组合（TND + PA_BSND）已实现。

    Returns:
        (attention_out，softmax_max，softmax_sum)；后两者在 return_softmax_lse=False 时为空张量
    """
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_query={layout_query!r} 未实现（支持 BSND/TND）")
    if layout_kv not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_kv={layout_kv!r} 未实现（支持 BSND/PA_BSND）")
    if int(attention_mode) != 2:
        raise NotImplementedError(f"attention_mode={attention_mode} 未实现（op 默认且仅实现 2）")
    for nm, v in (("pre_tokens", pre_tokens), ("next_tokens", next_tokens)):
        if v is not None and int(v) != (1 << 63) - 1:
            raise NotImplementedError(f"{nm}={v} 未实现（只支持 INT64_MAX）")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode 只支持 0 或 3，实际 {sparse_mode}")
    pa = layout_kv in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    # ---------------- layout 归一化 ----------------
    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND 必须提供 actual_seq_lengths_query（cu_seqlens）")
        T, N1, D = query.shape
        cu_q = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu_q.numel() < 2:
            raise ValueError("TND 下 actual_seq_lengths_query 必须是 cu_seqlens（长度 B+1）")
        B = cu_q.numel() - 1
        if int(cu_q[-1]) != T:
            raise ValueError(f"actual_seq_lengths_query 末项 {int(cu_q[-1])} != T={T}")
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    dev = query.device
    if pa:
        if block_table is None:
            raise ValueError("layout_kv=PA_BSND 必须提供 block_table")
        num_blocks, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        Bk, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key 的 D 与 query 不一致: {Dk} vs {D}")
    if tuple(value.shape) != tuple(key.shape):
        raise ValueError("value 必须与 key 同 shape")

    cu_kv = _as_cumulative(actual_seq_lengths_kv, B, "actual_seq_lengths_kv")
    if cu_kv is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dr = _D_ROPE
    if query_rope is None or key_rope is None:
        raise ValueError("本实现要求提供 query_rope / key_rope（MLA 的 64 维 rope 部分）")
    if key_rope.shape[:-1] != key.shape[:-1] or key_rope.shape[-1] != dr:
        raise ValueError(f"key_rope 形状应为 {tuple(key.shape[:-1]) + (dr,)}")
    if query_rope.shape[:-1] != query.shape[:-1] or query_rope.shape[-1] != dr:
        raise ValueError(f"query_rope 形状应为 {tuple(query.shape[:-1]) + (dr,)}")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} 必须能被 N2={N2} 整除")
    if sparse_indices.dtype != torch.int32:
        raise ValueError("sparse_indices 必须是 int32")
    bs = int(sparse_block_size)
    if bs < 1:
        raise ValueError("sparse_block_size 必须 >= 1")

    gs = N1 // N2
    k_idx = sparse_indices.shape[-1]
    # 展平成 [n_tokens, ...]
    q2 = query.contiguous().reshape(T, N1, D)
    qr2 = query_rope.contiguous().reshape(T, N1, dr)
    idx2 = sparse_indices.contiguous().reshape(T, N2, k_idx)
    kv = key.contiguous()
    kvr = key_rope.contiguous()
    val = value.contiguous()
    bt = block_table.contiguous().to(torch.int32) if pa else key.contiguous()
    tok_b = tok_b.to(dev)
    tok_s1 = tok_s1.to(dev)
    kvlen_d = kvlen_b.to(dev)
    qlen_d = qlen_b.to(dev)

    n_rows = T * N1
    sb_buf = torch.empty((n_rows, k_idx * bs), dtype=torch.float32, device=dev)
    out = torch.empty((n_rows, D), dtype=query.dtype, device=dev)

    blk_d = next_pow2(D)
    blk_r = next_pow2(dr)
    blk_k = 16
    _sfa_score_kernel[(n_rows,)](
        q2, qr2, kv, kvr, idx2, sb_buf, tok_b, tok_s1, kvlen_d, qlen_d, bt,
        N1, N2, gs, k_idx, T,
        block_size, max_blocks, S2, float(scale_value),
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        D=D, DR=dr, BLK_D=blk_d, BLK_R=blk_r, BLK_K=blk_k, BS=bs,
    )

    if tnd:
        lse_shape = (N2, T, gs)
    else:
        lse_shape = (B, N2, S1, gs)
    if return_softmax_lse:
        lse_max = torch.empty(lse_shape, dtype=torch.float32, device=dev)
        lse_sum = torch.empty(lse_shape, dtype=torch.float32, device=dev)
    else:
        lse_max = torch.empty(0, dtype=torch.float32, device=dev)
        lse_sum = torch.empty(0, dtype=torch.float32, device=dev)

    total = k_idx * bs
    lse_b_stride = 0 if tnd else (N2 * S1 * gs)
    lse_n2_stride = (T * gs) if tnd else (S1 * gs)
    _sfa_out_kernel[(n_rows,)](
        val, sb_buf, idx2, out, lse_max, lse_sum, tok_b, tok_s1, bt,
        N1, N2, gs, k_idx, T,
        block_size, max_blocks, S2, lse_b_stride, lse_n2_stride,
        RETURN_LSE=bool(return_softmax_lse), PA=bool(pa), NEG=_NEG,
        D=D, BLK_D=blk_d, BLK_K=blk_k, BLK_ALL=next_pow2(total), BS=bs,
    )
    if tnd:
        out = out.reshape(T, N1, D)
    else:
        out = out.reshape(B, S1, N1, D)
    return out, lse_max, lse_sum
