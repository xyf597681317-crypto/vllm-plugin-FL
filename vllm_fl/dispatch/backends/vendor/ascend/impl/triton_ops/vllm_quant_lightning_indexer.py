"""vllm_quant_lightning_indexer —— AscendC → Triton

来源：csrc/ascend/attention/vllm_quant_lightning_indexer/
接口（op_host/vllm_quant_lightning_indexer_def.cpp L23-81，arch32/910B 配置）：
    Input : query               int8   [B, S1, N1, 128]   (layout_query="BSND")
            key                 int8   [B, S2, N2, 128]   (layout_key="BSND")
            weights             fp16   [B, S1, N1]
            query_dequant_scale fp16   [B, S1, N1]
            key_dequant_scale   fp16   [B, S2]            (逐 key 位置)
            actual_seq_lengths_query/key, block_table, metadata  (可选)
    Output: sparse_indices int32 [B, S1, N2, sparse_count]   ← InferShape L64-75
            sparse_values  fp32  [B, S1, N2, sparse_count]   ← 仅 return_values=True；否则空张量(L76-83)
    Attr  : query/key_quant_mode=0(per-token-head), sparse_count(默认2048),
            sparse_mode=3(只算下三角), pre_tokens/next_tokens=INT64_MAX, cmp_ratio=1,
            return_values=False, stride, scale_stride

常量约束（op_host/vllm_quant_lightning_indexer_tiling.cpp）：
    - head_dim 只能是 128（L549 HEAD_DIM_LIMIT）
    - gSize = n1Size_ / n2Size_ 必须是 G_SIZE_LIMIT（L483-484）
    - weights 的 N1 维必须等于 n1Size_（L720），即按 **q head** 索引
    - sparse_mode ∈ {0, 3}（L249），pre/next_tokens 必须为 INT64_MAX（L252-257）

公式出处（op_kernel/arch35/vf/quant_lightning_indexer_vector1.h）：
    MulWeightAndReduceSum L329-353（fp32 版）与 L206-280：
        regW    = weights * qScale                      # L243 / L331 逐 (s1,qhead)
        for g in 0..gSize-1:  Relu在cube随路做          # L341 注释
            MulAddDst(sum, qk[g], w_brc[g])             # L181-185 WeightedAccum：先 Relu 再 MulAdd
        sum = sum * kScale                              # L352-353
    arch32 同一公式见 op_kernel/arch32/quant_lightning_indexer_service_vector.h
        L265 Mul(weights, weights, qScale) ; L360 Mul(mm1Res, mm1Res, kScale)

    ⇒ 核心计算（**语义等价版本**）：
        QK[s1, n2, g, s2] = Σ_d query[s1, n2*gSize+g, d] * key[s2, n2, d]      (int8 点积，int32 精确)
        wq[s1, g']        = weights[s1, g'] * query_dequant_scale[s1, g']      (g' = n2*gSize+g)
        score[s1, n2, s2] = key_dequant_scale[s2] * Σ_g relu(QK[g, s2]) * wq[g]
        sparse_mode=3（下三角）：score = -inf  当 s2 > s1 + (S2 - S1)
                                  （tiling 内核 L327: cuRealAcSeq = actS2SizeOrig - actS1Size + cuS1BeginIdx）
        top-k：按 score 降序取 sparse_count 个 key 位置；不足则填 -1
               （INVALID_IDX = -1，见 arch32/quant_lightning_indexer_common.h L79）

    排序精度的说明：AscendC 的 topk 把 score 转成 **bfloat16** 再转 sortable key 后排序
    （L273-277 FloatToSortableKey<bfloat16_t>），即排序按 bf16 精度比较。本实现在 fp32 上比较，
    对"最优 K 个得分之间有明显间隔"的数据两者一致（测试里会打印间隔来佐证）。

**简化点**（相对 AscendC 完整实现）：
    1. 只支持 layout_query/layout_key = "BSND"（不支持 TND / PA_BSND 分页 KV）
    2. 不支持 cmp_ratio != 1 的压缩、不支持 actual_seq_lengths 变长（按满长度 BSND 处理）
    3. 不做多核 Cube/Vector 流水，不融合；score 矩阵落一次 GM（两趟 kernel：打分 → topk）
    4. topk 用逐轮 argmax 实现（O(K) 轮），同分时取 **较小下标**（AscendC 未定义同分次序）
    5. query/key 的 int8 点积用 fp32 广播乘加实现（精确：int8 乘积和 < 2^24），未用 Cube int8 matmul
    6. sparse_values 在无效槽位（索引 -1）上输出 0.0；AscendC 未定义该处的值
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["vllm_quant_lightning_indexer"]

_NEG = -1.0e30


@triton.jit
def _qli_score_kernel(
    Q, K, W, QS, KS, SC, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS,
    BLOCK_SIZE, MAX_BLOCKS, S2, CMP_RATIO,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    HD: tl.constexpr,
    BLK_S2: tl.constexpr,
    BLK_D: tl.constexpr,
):
    row = tl.program_id(0)              # 一行 = 一个 (token, n2)
    n2 = row % N2
    t = row // N2
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)

    h0 = n2 * GS
    d = tl.arange(0, BLK_D)
    dm = d < HD
    q_base = (t * N1 + h0) * HD
    w_base = t * N1 + h0

    kvlen = tl.load(KVLEN + b)
    if CAUSAL:
        # 对齐 AscendC：cuRealAcSeq = (count + (actS2Orig - actS1 + s1)) / cmpRatio
        j_hi = (s1 + 1 + (kvlen - tl.load(QLEN + b))) // CMP_RATIO - 1
    else:
        j_hi = S2 - 1

    for c in range(0, tl.cdiv(S2, BLK_S2)):
        s2 = c * BLK_S2 + tl.arange(0, BLK_S2)
        s2m = s2 < S2
        vld = s2m & (s2 <= j_hi) & (s2 < kvlen)
        if PA:
            page = tl.load(BT + b * MAX_BLOCKS + s2 // BLOCK_SIZE, mask=vld, other=0)
            kvrow = page * BLOCK_SIZE + (s2 % BLOCK_SIZE)
        else:
            kvrow = b * S2 + s2
        k_off = (kvrow * N2 + n2)[:, None] * HD + d[None, :]
        kblk = tl.load(K + k_off, mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)

        acc = tl.zeros([BLK_S2], dtype=tl.float32)
        for g in range(0, GS):
            qg = tl.load(Q + q_base + g * HD + d, mask=dm, other=0.0).to(tl.float32)
            wq = tl.load(W + w_base + g).to(tl.float32) * tl.load(QS + w_base + g).to(tl.float32)
            qk = tl.sum(kblk * qg[None, :], axis=1)          # int8 点积（fp32 精确）
            acc += tl.where(qk > 0.0, qk, 0.0) * wq          # Relu + 加权累加
        ksc = tl.load(KS + kvrow, mask=vld, other=0.0).to(tl.float32)
        sc = tl.where(vld, acc * ksc, NEG)
        tl.store(SC + row * S2 + s2, sc, mask=s2m)


@triton.jit
def _qli_topk_kernel(
    SC, IDX, VAL, S2,
    K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BLK_K: tl.constexpr,
    WRITE_VAL: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLK_ALL)
    s = tl.load(SC + pid * S2 + off, mask=off < S2, other=NEG)

    ar = tl.arange(0, BLK_K)
    oi = tl.full([BLK_K], -1, tl.int32)
    ov = tl.zeros([BLK_K], dtype=tl.float32)
    for k in tl.static_range(K):
        mx = tl.max(s, axis=0)
        lane = tl.min(tl.where(s >= mx, off, BLK_ALL))    # 同分取较小下标
        sel = tl.where(mx > NEG, lane, -1)
        oi = tl.where(ar == k, sel, oi)
        if WRITE_VAL:
            ov = tl.where(ar == k, tl.where(mx > NEG, mx, 0.0), ov)
        s = tl.where(off == lane, NEG, s)

    m = ar < K
    tl.store(IDX + pid * K + ar, oi, mask=m)
    if WRITE_VAL:
        tl.store(VAL + pid * K + ar, ov, mask=m)


def vllm_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    return_value: bool = False,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    pre_tokens: int | None = None,
    next_tokens: int | None = None,
    cmp_ratio: int = 1,
    return_values: bool | None = None,
    out_dtype: torch.dtype = torch.int32,
):
    """等价于 AscendC 的 VllmQuantLightningIndexer（BSND，语义等价核心版本）。

    调用点签名（vllm_fl/attention/ascend/dsa_v1.py L2658-2678）：
        npu_vllm_quant_lightning_indexer(
            query=..., key=..., weights=..., query_dequant_scale=..., key_dequant_scale=...,
            actual_seq_lengths_query=..., actual_seq_lengths_key=..., block_table=...,
            metadata=..., query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND", sparse_count=...,
            sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
            cmp_ratio=4, return_value=False)
    ⚠️ 真实调用点是 **TND + PA_BSND + cmp_ratio=4**，本核心版本只实现 BSND + cmp_ratio=1；
       传不支持的取值会显式抛 NotImplementedError（不静默忽略）。

    Args:
        query:               int8 [B, S1, N1, 128]
        key:                 int8 [B, S2, N2, 128]
        weights:             fp16/bf16/fp32 [B, S1, N1]
        query_dequant_scale: 同 weights
        key_dequant_scale:   [B, S2] 或 [S2]
        sparse_count:        top-k 个数 K
        sparse_mode:         0（全算）或 3（下三角，AscendC 默认）
        return_value:        是否返回 sparse_values（fp32 分数）；False 时返回空张量
        layout_query/layout_key: 只支持 "BSND"
        actual_seq_lengths_query/key, block_table, metadata: 只支持 None
        query_quant_mode/key_quant_mode: 只支持 0（per-token-head）
        pre_tokens/next_tokens: 只支持 None 或 INT64_MAX
        cmp_ratio: 只支持 1
        return_values: return_value 的别名（向后兼容）

    Returns:
        (sparse_indices [B,S1,N2,K] int32, sparse_values [B,S1,N2,K] fp32 或 torch.empty(0))
    """
    if return_values is not None:
        return_value = return_values
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_query={layout_query!r} 未实现（支持 BSND/TND）")
    if layout_key not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_key={layout_key!r} 未实现（支持 BSND/PA_BSND）")
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError(f"metadata 必须是 int32 描述符张量，实际 {metadata.dtype}")
    for nm, v in (("query_quant_mode", query_quant_mode), ("key_quant_mode", key_quant_mode)):
        if int(v) != 0:
            raise NotImplementedError(f"{nm}={v} 未实现（tiling 只接受 0: per-token-head）")
    for nm, v in (("pre_tokens", pre_tokens), ("next_tokens", next_tokens)):
        if v is not None and int(v) != (1 << 63) - 1:
            raise NotImplementedError(f"{nm}={v} 未实现（tiling 只接受 INT64_MAX）")
    return_values = bool(return_value)
    pa = layout_key in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    # ---------------- layout 归一化（host 只做布局簿记，核心数学在 Triton）----------------
    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND 必须提供 actual_seq_lengths_query（cu_seqlens）")
        T, N1, D = query.shape
        cu_q = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu_q.numel() == 0:
            raise ValueError("layout_query=TND 的 actual_seq_lengths_query 不能为空")
        # 两种约定都要接受：
        #   * AscendC / 单测： ``[0, a1, a1+a2, ...]``（长度 B+1，带前导 0）
        #   * 生产调用点    ： ``dsa_v1.py`` L2653/L2664 传 ``query_start_loc[1:]``，
        #                     即 ``[a1, a1+a2, ...]``（长度 B，**无**前导 0）
        # 单序列 decode 时生产只传 1 个元素（``[1]``），旧实现在这里直接抛
        # ValueError，worker 死在集合通信里、其它 rank 报 507035。
        if int(cu_q[0]) != 0:
            cu_q = torch.cat([torch.zeros(1, dtype=torch.int64), cu_q])
        if int(cu_q[-1]) != T:
            raise ValueError(
                f"TND 下 actual_seq_lengths_query 必须是 cu_seqlens（长度 B 或 B+1、末项={T}），"
                f"实际末项={int(cu_q[-1])}"
            )
        B = cu_q.numel() - 1
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    if pa:
        if block_table is None:
            raise ValueError("layout_key=PA_BSND 必须提供 block_table")
        num_blocks, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        Bk, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key 的 D 与 query 不一致: {Dk} vs {D}")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key 必须是 int8")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} 必须能被 N2={N2} 整除")
    gs = N1 // N2
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode 只支持 0 或 3，实际 {sparse_mode}")
    # weights/query_dequant_scale 逐 q head：BSND 传 [B,S1,N1]，TND 传 [T,N1]，统一展成 [T,N1]
    if weights.numel() != T * N1 or query_dequant_scale.numel() != T * N1:
        raise ValueError(
            f"weights/query_dequant_scale 元素数必须是 T*N1={T * N1}"
            f"（BSND→[B,S1,N1]，TND→[T,N1]），实际 {tuple(weights.shape)} / "
            f"{tuple(query_dequant_scale.shape)}")
    k = int(sparse_count)
    if k <= 0:
        raise ValueError("sparse_count 必须 > 0")

    sq_kv = actual_seq_lengths_key
    if sq_kv is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        s = sq_kv.detach().to(torch.int64).cpu().flatten()
        if s.numel() == B:
            kvlen_b = s.to(torch.int32)
        elif s.numel() == B + 1:
            kvlen_b = (s[1:] - s[:-1]).to(torch.int32)
        else:
            raise ValueError(f"actual_seq_lengths_key 长度必须是 B={B} 或 B+1，实际 {s.numel()}")
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dev = query.device
    q2 = query.contiguous().reshape(T, N1, D)
    w2 = weights.contiguous().reshape(T, N1)
    qs2 = query_dequant_scale.contiguous().reshape(T, N1)
    kv = key.contiguous()
    bt = block_table.contiguous().to(torch.int32) if pa else key.contiguous()
    ks = key_dequant_scale.contiguous()
    if pa:
        # PA 下 key_dequant_scale 按 (物理块, 块内偏移) 排布（GetKeyScale: kScaleGm[blockId*scaleStride+off]）
        # 生产调用点会多带一个尾部 1 维（如 (21321, 32, 1)）——squeeze 掉即可，
        # 底层寻址只看 [numBlocks, blockSize]，多这一维不影响语义。
        while ks.dim() > 2 and ks.shape[-1] == 1:
            ks = ks.squeeze(-1)
        if ks.dim() != 2 or ks.shape[1] != block_size:
            raise ValueError(f"PA 下 key_dequant_scale 应为 [numBlocks, blockSize]={(num_blocks, block_size)}"
                             f"（尾部单例维可省略），实际 {tuple(key_dequant_scale.shape)}")
        ks = ks.contiguous()
    else:
        if ks.dim() == 1:
            ks = ks.unsqueeze(0).expand(B, -1).contiguous()
        if tuple(ks.shape) != (B, S2):
            raise ValueError(f"key_dequant_scale 必须是 [B,S2] 或 [S2]，实际 {tuple(key_dequant_scale.shape)}")

    n_rows = T * N2
    blk_d = next_pow2(D)
    # BLK_S2 受 UB(192KB) 限制：BLK_S2*BLK_D 的 fp32 中间张量要放得下（128x128 会 ub overflow）
    blk_s2 = min(64, next_pow2(S2))
    scores = torch.empty((n_rows, S2), dtype=torch.float32, device=dev)
    _qli_score_kernel[(n_rows,)](
        q2, kv, w2, qs2, ks, scores,
        tok_b.to(dev), tok_s1.to(dev), kvlen_b.to(dev), qlen_b.to(dev), bt,
        N1, N2, gs,
        block_size, max_blocks, S2, int(cmp_ratio),
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        HD=D, BLK_S2=blk_s2, BLK_D=blk_d,
    )
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=dev)
    blk_k = next_pow2(k)
    blk_all = next_pow2(S2)
    # topk 核按 pid*K 步长写 VAL，所以缓冲区必须是 [n_rows, k]
    vals = torch.empty((n_rows, k), dtype=torch.float32, device=dev)
    _qli_topk_kernel[(n_rows,)](
        scores, idx, vals, S2,
        K=k, BLK_ALL=blk_all, BLK_K=blk_k,
        WRITE_VAL=bool(return_values), NEG=_NEG,
    )
    if return_values:
        out_v = vals.view(n_rows, k).to(torch.float32)
    else:
        out_v = torch.empty(0, dtype=torch.float32, device=dev)
    if tnd:
        return idx.view(T, N2, k), out_v.view(T, N2, k) if return_values else out_v
    return idx.view(B, S1, N2, k), (out_v.view(B, S1, N2, k) if return_values else out_v)
