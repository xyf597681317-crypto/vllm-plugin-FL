"""sparse_attention_score —— AscendC → Triton

来源：csrc/ascend/attention/sparse_attention_score/
接口（op_host/sparse_attention_score_def.cpp L19-72）：
    Input : query [T, N, D]                 fp16/bf16/fp8_e4m3fn
            key   [numBlocks, blockSize, kvHeads, D]
            value [numBlocks, blockSize, kvHeads, D]
            selectIdx     [kvHeads, maxQSeqlen, topK]  int32
            blockTable    [B, maxBlocksPerBatch]       int32
            selectNumIdx  [kvHeads, maxQSeqlen]        int32 (可选；fp8 时必填)
            actualSeqLengths   [B] int32 (可选)
            actualSeqLengthsKv [B] int32 (可选)
            qDequantScale [B, numHeads, ceil(maxQSeqlen/blockSize)] fp32 (可选)
            kDequantScale [B, kvHeads, maxBlocksPerBatch]           fp32 (可选)
            vDequantScale [B, kvHeads, maxBlocksPerBatch]           fp32 (可选)
    Output: attentionOut [T, N, D]（dtype 同 query；fp8 输入时为 fp16）
            softmaxLse   [T, N, 1] fp32 (可选)
    Attr  : numKeyValueHeads=1, scaleValue=0.0(→1/sqrt(D)), blockSize=128,
            topK=16, innerPrecise=4

公式（Regular 路径：op_kernel/arch35/sparse_attention_score_kernel_arch35.h L112-289）
    for taskIdx in [0, T*kvHeads):                     # 每个 (qToken, kvHead) 一个任务
        qToken  = taskIdx // kvHeads ; kvHead = taskIdx % kvHeads
        groupSize = N // kvHeads ; qHeadStart = kvHead * groupSize
        G  = query[qToken, qHeadStart : qHeadStart+groupSize, :]        # [groupSize, D]
        batchIdx / qTokenInBatch 由 actualSeqLengths 前缀和决定（L124-135）
        validTopK = selectNumIdx[kvHead, qToken] if 存在 else topK
        kvSeqlen  = actualSeqLengthsKv[batchIdx] ; qSeqlen = actualSeqLengths[batchIdx]
        historyLen = kvSeqlen - qSeqlen
        lastLogicalBlockId = (historyLen + qTokenInBatch) // blockSize
        lastBlockTileSize  = (historyLen + qTokenInBatch) %  blockSize + 1     # 因果
        for i in range(min(validTopK, topK)):          # L160-169
            logicalId = selectIdx[kvHead, qToken, i] ; if logicalId < 0: continue
            physId = blockTable[batchIdx, logicalId]
            tile   = blockSize if logicalId != lastLogicalBlockId else lastBlockTileSize
            K_b = key[physId][:, kvHead, :] ; S = (G @ K_b^T) * scaleValue
            S[.., j >= tile] = -inf
            逐块 online softmax（EpilogueAtlasA2OnlineSoftmax，scaleValue 在 L538-541 乘到 S）
            O += P @ value[physId][:, kvHead, :]
        attentionOut[qToken, qHeadStart + r, :] = O / l
        softmaxLse  [qToken, qHeadStart + r, 0] = m + log(l)

fp8 FullQuant 路径（op_kernel/arch35/sparse_attention_score_kernel_arch35_full_quant.h L146-344）
    与 Regular 路径的差别（**以代码为准**）：
      * 反量化在**矩阵乘结果**上做：deqScalar 传给 blockMmadQK / blockMmadPV，
        由 TileCopy 在 L0C→UB 拷贝时按 PER_TENSOR 粒度缩放
        （sparse_attention_score_kernel_interface.cpp L204-207 / L221-224 的
         `ScaleGranularity::PER_TENSOR`），即
             S = (Q_fp8 · K_fp8^T) * (qDequantScale * kDequantScale)   # L251-263
             O = (P · V_fp8)          * vDequantScale                  # L315-326
      * 三个 scale 的**索引与形状**（L210-213 / L251-254 / L315-318）：
             maxQBlocksPerBatch = ceil(maxQSeqlen / blockSize)
             logicalQBlock      = (historyLen + qTokenInBatch) // blockSize
             sq = qDequantScale[batchIdx * numHeads * maxQBlocksPerBatch
                                + qHeadIdx * maxQBlocksPerBatch + logicalQBlock]
                                             # 形状 [B, numHeads, maxQBlocksPerBatch]
             sk = kDequantScale[batchIdx * kvHeads * maxBlocksPerBatch
                                + kvHeadIdx * maxBlocksPerBatch + logicalId]
             sv = vDequantScale[batchIdx * kvHeads * maxBlocksPerBatch
                                + kvHeadIdx * maxBlocksPerBatch + logicalId]
                                             # 形状 [B, kvHeads, maxBlocksPerBatch]
        即：Q 侧按 (batch, qHead, qBlock) 粒度，K/V 侧按 (batch, kvHead, logicalBlock) 粒度，
        都是 **per-tensor（每 tile 一个标量）**，没有 per-token/per-head 的通道级缩放。
      * fp8 时 `effectiveGroupSize = 1`（L126-128：per-head dequant scale 无法打包多 head），
        逐 head 处理；本实现按 GQA 组一次处理（数学等价，见「简化说明」第 5 条）。
      * P（softmax 概率）以 **ElementP = InDtype = fp8_e4m3fn** 写入 L1 再参与 PV
        （kernel_interface.cpp L189/L213；epilogue
         block_epilogue_online_softmax_arch35_reg_low_prec.hpp L41/L81
         `FULL_QUANT_FP8 = IsSameType<ElementOutput, fp8_e4m3fn_t>`，
         经 `ComputeExpSubSum16FP8` L483-598 以 CAST_RINT 转 e4m3）。
        本实现默认复现该 e4m3 量化（`p_fp8=True`，用 `tl.float8e4nv`）；
        若后端不支持 fp8 dtype 会自动退化为 fp16 并打印告警。
      * fp8 路径**不回写 softmaxLse**（FullQuant 内核没有 LSE 输出张量）。

简化说明（明确未覆盖/取舍的部分）
    1. 只支持 `saca_regular` 与 `full_quant(fp8)` 两条路径；arch22（910B/910C 的 fp16/bf16
       分支）与 arch35 的数学语义一致，本实现按 arch35 描述。
    2. `innerPrecise` 属性未使用（内核走同一路径）。
    3. LSE：Regular 内核用 LseMode::NONE，不回写 softmaxLse；本实现仍输出数学定义的
       LSE = m + log(l)（显式记录的扩展）；fp8 路径同。
    4. `numKeyValueHeads` <= 0 时从 key.shape[2] 推断。
    5. fp8 时按整个 GQA 组（G 个 qHead）一次算，而不是 head-by-head；
       `S = (Q·K^T)*(sq*sk)` 对同一 (batch, qBlock) 下同组的每个 head 都要乘各自的 sq，
       本实现按行（head）取各自的 sq，与 head-by-head 完全等价。
    6. fp8 输入在核内先无损升到 bf16（e4m3 ⊂ bf16，转换精确）再做 fp32 累加的 dot，
       与 AscendC「fp8 立方核乘 + fp32 累加，再乘 deqScalar」在数值上一致。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_attention_score"]

_MIN_SCORE = -1.0e30  # ⚠️ 不用 float("-inf")（Ascend 上会退化）

# fp8 概率（P）量化是否可用：首次尝试失败后置 False（后端不支持 fp8 dtype 时）
_P_FP8_OK = True


@triton.jit
def _sasa_kernel(
    Q, K, V, SEL, BT, SELNUM, ACTQ, ACTKV, QSC, KSC, VSC, OUT, LSE,
    Q_BATCH, Q_INBATCH,
    KVH, MQB, MQ, N, D, BS, SCALE, MAXQBLK,
    TOP_K: tl.constexpr,
    BLK_G: tl.constexpr, BLK_D: tl.constexpr, BLK_BS: tl.constexpr,
    HAS_SELNUM: tl.constexpr, WRITE_LSE: tl.constexpr,
    IS_FP8: tl.constexpr, P_FP8: tl.constexpr,
    MIN_SCORE: tl.constexpr,
):
    task = tl.program_id(0)
    qToken = task // KVH
    h = task % KVH

    G = N // KVH
    gstart = h * G
    rg = tl.arange(0, BLK_G)
    rd = tl.arange(0, BLK_D)
    gm = rg < G
    dm = rd < D

    # Q: [G, D]
    qraw = tl.load(Q + qToken * N * D + (gstart + rg)[:, None] * D + rd[None, :],
                   mask=gm[:, None] & dm[None, :], other=0.0)
    if IS_FP8:
        q = qraw.to(tl.bfloat16)          # e4m3 -> bf16 无损
    else:
        q = qraw

    batchIdx = tl.load(Q_BATCH + qToken)
    qInBatch = tl.load(Q_INBATCH + qToken)
    kvSeqlen = tl.load(ACTKV + batchIdx)
    qSeqlen = tl.load(ACTQ + batchIdx)
    hist = kvSeqlen - qSeqlen
    lastLogical = (hist + qInBatch) // BS
    lastTile = (hist + qInBatch) % BS + 1

    if HAS_SELNUM:
        vtopk = tl.load(SELNUM + h * MQ + qToken)
    else:
        vtopk = TOP_K

    # fp8: Q 的反量化 scale 按 (batch, qHead, qBlock) 取（L210-213）
    sq = tl.zeros((BLK_G,), dtype=tl.float32) + 1.0
    if IS_FP8:
        logicalQBlock = (hist + qInBatch) // BS
        qs_off = batchIdx * N * MAXQBLK + (gstart + rg) * MAXQBLK + logicalQBlock
        sq = tl.load(QSC + qs_off, mask=gm, other=1.0)

    sel_base = h * MQ * TOP_K + qToken * TOP_K
    bt_base = batchIdx * MQB

    rb = tl.arange(0, BLK_BS)
    m_i = tl.full((BLK_G,), MIN_SCORE, dtype=tl.float32)
    l_i = tl.zeros((BLK_G,), dtype=tl.float32)
    o_i = tl.zeros((BLK_G, BLK_D), dtype=tl.float32)

    for i in range(TOP_K):
        logical = tl.load(SEL + sel_base + i, mask=i < vtopk, other=-1)
        active = (i < vtopk) & (logical >= 0)
        phys = tl.load(BT + bt_base + logical, mask=active, other=0)
        tile = tl.where(logical == lastLogical, lastTile, BS)
        koff = phys * (BS * KVH * D) + h * D
        km = active & (rb[:, None] < tile) & dm[None, :]
        kraw = tl.load(K + koff + rb[:, None] * (KVH * D) + rd[None, :], mask=km, other=0.0)
        vraw = tl.load(V + koff + rb[:, None] * (KVH * D) + rd[None, :], mask=km, other=0.0)
        if IS_FP8:
            k = kraw.to(tl.bfloat16)
            vv = vraw.to(tl.bfloat16)
            # K/V 反量化 scale 按 (batch, kvHead, logicalBlock) 取（L251-254 / L315-318）
            kv_off = batchIdx * KVH * MQB + h * MQB + logical
            ksc = tl.load(KSC + kv_off, mask=active, other=1.0)
            vsc = tl.load(VSC + kv_off, mask=active, other=1.0)
        else:
            k = kraw
            vv = vraw
            ksc = 1.0
            vsc = 1.0

        if IS_FP8:
            # S = (Q·K^T) * (sq * sk) * scaleValue   （L251-263 + epilogue L538-541）
            s = tl.dot(q, tl.trans(k)) * (SCALE * ksc) * sq[:, None]
        else:
            s = tl.dot(q, tl.trans(k)) * SCALE
        s = tl.where(active & (rb[None, :] < tile) & gm[:, None], s, MIN_SCORE)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if P_FP8:
            pd = p.to(tl.float8e4nv).to(tl.bfloat16)     # ElementP = fp8_e4m3fn（CAST_RINT）
        elif IS_FP8:
            pd = p.to(tl.bfloat16)
        else:
            pd = p.to(Q.dtype.element_ty)
        o_i = o_i * alpha[:, None] + tl.dot(pd, vv) * vsc
        m_i = m_new

    has_tile = m_i > (MIN_SCORE * 0.5)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o_i = tl.where(has_tile[:, None], o_i / l_safe[:, None], 0.0)
    olm = gm[:, None] & dm[None, :]
    tl.store(OUT + qToken * N * D + (gstart + rg)[:, None] * D + rd[None, :],
             o_i.to(OUT.dtype.element_ty), mask=olm)
    if WRITE_LSE:
        lse = tl.where(has_tile, m_i + tl.log(l_safe), 0.0)
        tl.store(LSE + qToken * N + gstart + rg, lse, mask=gm)


def _launch(query, key, value, select_idx, block_table, selnum_c, actq_c, actkv_c,
            qsc, ksc, vsc, out, lse, qb_c, qi_c, KVH, MQB, maxQ, Nh, D, BS, scale,
            maxqblk, topK, has_selnum, write_lse, is_fp8, BLK_G, BLK_D, BLK_BS, p_fp8):
    _sasa_kernel[(int(query.shape[0]) * KVH,)](
        query, key, value, select_idx, block_table, selnum_c, actq_c, actkv_c,
        qsc, ksc, vsc, out, lse, qb_c, qi_c,
        KVH, MQB, maxQ, Nh, D, BS, scale, maxqblk,
        TOP_K=topK, BLK_G=BLK_G, BLK_D=BLK_D, BLK_BS=BLK_BS,
        HAS_SELNUM=has_selnum, WRITE_LSE=write_lse,
        IS_FP8=is_fp8, P_FP8=p_fp8, MIN_SCORE=_MIN_SCORE,
    )


def sparse_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    select_idx: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    select_num_idx: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    q_dequant_scale: torch.Tensor | None = None,
    k_dequant_scale: torch.Tensor | None = None,
    v_dequant_scale: torch.Tensor | None = None,
    num_key_value_heads: int | None = None,
    scale_value: float = 0.0,
    block_size: int = 128,
    top_k: int = 16,
    inner_precise: int = 4,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    write_lse: bool = True,
    p_fp8: bool = True,
    fp8_mode: bool | None = None,
    # ---- AscendC 原名别名（真实调用点用这些名字），显式传入时覆盖同名 snake_case 参数 ----
    selectIdx: torch.Tensor | None = None,
    blockTable: torch.Tensor | None = None,
    selectNumIdx: torch.Tensor | None = None,
    actualSeqLengths: torch.Tensor | None = None,
    actualSeqLengthsKv: torch.Tensor | None = None,
    qDequantScale: torch.Tensor | None = None,
    kDequantScale: torch.Tensor | None = None,
    vDequantScale: torch.Tensor | None = None,
    numKeyValueHeads: int | None = None,
    scaleValue: float | None = None,
    blockSize: int | None = None,
    topK: int | None = None,
    innerPrecise: int | None = None,
    attentionOut: torch.Tensor | None = None,
    softmaxLse: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 sparse_attention_score（fp16/bf16 Regular + fp8 FullQuant）。

    同时接受 snake_case 与 AscendC 原名（驼峰）两套参数名：
    `selectIdx/blockTable/selectNumIdx/actualSeqLengths/actualSeqLengthsKv/
     qDequantScale/kDequantScale/vDequantScale/numKeyValueHeads/scaleValue/
     blockSize/topK/innerPrecise/attentionOut/softmaxLse`。

    fp8_e4m3fn 输入时：query/key/value 为 fp8，输出 attentionOut 为 fp16
    （op_host/sparse_attention_score_infershape.cpp L91-96）；三个 dequant scale
    的形状见模块 docstring。

    Returns:
        (attentionOut [T, N, D], softmaxLse [T, N, 1] fp32)
    """
    # ---- 别名回填 ----
    if selectIdx is not None:
        select_idx = selectIdx
    if blockTable is not None:
        block_table = blockTable
    if selectNumIdx is not None:
        select_num_idx = selectNumIdx
    if actualSeqLengths is not None:
        actual_seq_lengths = actualSeqLengths
    if actualSeqLengthsKv is not None:
        actual_seq_lengths_kv = actualSeqLengthsKv
    if qDequantScale is not None:
        q_dequant_scale = qDequantScale
    if kDequantScale is not None:
        k_dequant_scale = kDequantScale
    if vDequantScale is not None:
        v_dequant_scale = vDequantScale
    if numKeyValueHeads is not None:
        num_key_value_heads = numKeyValueHeads
    if scaleValue is not None:
        scale_value = scaleValue
    if blockSize is not None:
        block_size = blockSize
    if topK is not None:
        top_k = topK
    if innerPrecise is not None:
        inner_precise = innerPrecise
    if attentionOut is not None:
        out = attentionOut
    if softmaxLse is not None:
        lse = softmaxLse
    if select_idx is None or block_table is None:
        raise ValueError("sparse_attention_score 需要 select_idx/selectIdx 与 "
                         "block_table/blockTable")
    global _P_FP8_OK
    # fp8 通路选择：默认按输入 dtype 自动判定；也可用 fp8_mode 显式指定
    # （用于本机无法在 NPU 上落地 fp8 张量时，用「已量化到 e4m3 的 bf16 张量」验证
    #   反量化数学：e4m3 ⊂ bf16，二者数值完全一致）
    is_fp8 = (query.dtype == torch.float8_e4m3fn) if fp8_mode is None else bool(fp8_mode)
    T, Nh, D = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
    BS = int(block_size)
    BSz = int(key.shape[1])
    KVH = int(num_key_value_heads) if num_key_value_heads else int(key.shape[2])
    if KVH <= 0:
        KVH = int(key.shape[2])
    Bt, MQB = int(block_table.shape[0]), int(block_table.shape[1])
    maxQ = int(select_idx.shape[1])
    topK = int(top_k)
    if int(select_idx.shape[2]) != topK:
        raise ValueError(f"topK 属性({topK}) 必须等于 selectIdx 最后一维({int(select_idx.shape[2])})")
    # qDequantScale 的 dim2 长度（head 维 stride）：内核按
    #   offset = batchIdx*numHeads*maxQBlocksPerBatch + qHead*maxQBlocksPerBatch + logicalQBlock
    # 取值（full_quant L210-213），而 logicalQBlock 用的是**绝对位置**块号
    # (historyLen + qTokenInBatch)//blockSize，因此 dim2 必须覆盖该上界。
    # 优先取传入张量的实际 dim2，否则按 KV 序列长度上界构造（见下方 act_kv 之后）。

    scale = float(scale_value)
    if abs(scale) < 1e-9 and D > 0:
        scale = 1.0 / math.sqrt(float(D))

    act_q = (actual_seq_lengths.detach().cpu().to(torch.int64).tolist()
             if actual_seq_lengths is not None else [T])
    act_kv = (actual_seq_lengths_kv.detach().cpu().to(torch.int64).tolist()
              if actual_seq_lengths_kv is not None else act_q)

    if q_dequant_scale is not None and q_dequant_scale.dim() == 3:
        maxqblk = int(q_dequant_scale.shape[2])
    else:
        # logicalQBlock 是**绝对位置**块号，其上界由 KV 序列长度决定（不是 maxQSeqlen）
        maxqblk = max(1, max(ceil_div(int(v), BS) for v in act_kv) if act_kv else 1)

    q_batch, q_inb = [], []
    for b in range(Bt):
        for t in range(int(act_q[b]) if b < len(act_q) else 0):
            q_batch.append(b)
            q_inb.append(t)
    if len(q_batch) < T:
        q_batch += [Bt - 1] * (T - len(q_batch))
        q_inb += [0] * (T - len(q_inb))
    q_batch = q_batch[:T]
    q_inb = q_inb[:T]

    dev = query.device
    if out is None:
        out = torch.zeros((T, Nh, D), dtype=(torch.float16 if is_fp8 else query.dtype), device=dev)
    if lse is None:
        lse = torch.zeros((T, Nh, 1), dtype=torch.float32, device=dev)

    G = Nh // KVH
    BLK_G = max(16, next_pow2(G))
    BLK_D = max(16, next_pow2(D))
    BLK_BS = max(16, next_pow2(BSz))
    assert topK <= 16, "topK 必须 <= 16（见 op_host tiling MAX_TOP_K）"

    q_c = query.contiguous()
    k_c = key.contiguous()
    v_c = value.contiguous()
    sel_c = select_idx.to(torch.int32).contiguous()
    bt_c = block_table.to(torch.int32).contiguous()
    selnum_c = (select_num_idx.to(torch.int32).contiguous() if select_num_idx is not None
                else torch.zeros(1, dtype=torch.int32, device=dev))
    actq_c = torch.tensor(act_q, dtype=torch.int32, device=dev)
    actkv_c = torch.tensor(act_kv, dtype=torch.int32, device=dev)
    qb_c = torch.tensor(q_batch, dtype=torch.int32, device=dev)
    qi_c = torch.tensor(q_inb, dtype=torch.int32, device=dev)

    def _sc(t, shape):
        # 传入 None 时按内核真实寻址范围构造全 1 scale（等价「不反量化」）
        if t is None:
            return torch.ones(shape, dtype=torch.float32, device=dev).reshape(-1).contiguous()
        return t.to(torch.float32).reshape(-1).contiguous()

    qsc = _sc(q_dequant_scale, (Bt, Nh, maxqblk))
    ksc = _sc(k_dequant_scale, (Bt, KVH, MQB))
    vsc = _sc(v_dequant_scale, (Bt, KVH, MQB))

    args = (q_c, k_c, v_c, sel_c, bt_c, selnum_c, actq_c, actkv_c, qsc, ksc, vsc,
            out, lse, qb_c, qi_c, KVH, MQB, maxQ, Nh, D, BS, scale, maxqblk,
            topK, bool(select_num_idx is not None), bool(write_lse), bool(is_fp8),
            BLK_G, BLK_D, BLK_BS)
    if is_fp8 and p_fp8 and _P_FP8_OK:
        try:
            _launch(*args, p_fp8=True)
            return out, lse
        except Exception as e:  # 后端不支持 fp8 dtype：退化为 fp16 概率
            _P_FP8_OK = False
            import warnings
            warnings.warn(f"sparse_attention_score: fp8 概率量化不可用，退化为 fp16: {e}")
    _launch(*args, p_fp8=False)
    return out, lse
