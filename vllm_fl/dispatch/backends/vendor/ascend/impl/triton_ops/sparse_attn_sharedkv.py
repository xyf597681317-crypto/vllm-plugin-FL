"""sparse_attn_sharedkv —— AscendC → Triton（支持 BSND/TND query + BSND/PA_ND KV）

来源：csrc/ascend/attention/sparse_attn_sharedkv/
接口（op_host/sparse_attn_sharedkv_def.cpp L23-111）：
    Input : q          fp16/bf16  layout_q="BSND" → [B,S1,N1,D]；"TND" → [T,N1,D]
            ori_kv     fp16/bf16  未压缩 KV（滑窗用）
                       layout_kv="BSND"  → [B,S2o,N2,D]
                       layout_kv="PA_ND" → [numBlocks, blockSize, N2, D] + ori_block_table
            cmp_kv     fp16/bf16  压缩 KV（稀疏挑选用），同上两种 layout + cmp_block_table
            cmp_sparse_indices int32  BSND→[B,S1,N2,K]；TND→[T,N2,K]（-1 无效）
            cu_seqlens_q / cu_seqlens_ori_kv / cu_seqlens_cmp_kv / seqused_q / seqused_kv
            sinks      fp32 [N1]
    Output: attn_out 同 q；softmax_lse fp32
    Attr  : softmax_scale, cmp_ratio, ori_mask_mode(4), cmp_mask_mode(3),
            ori_kv_stride(0), cmp_kv_stride(0), ori_win_left(127), ori_win_right(0),
            layout_q("BSND"), layout_kv("PA_ND"), return_softmax_lse(False)

运行模式（op_kernel/sparse_attn_sharedkv_common.h L28-32）：
    SWA_MODE=0（滑窗）/ SCFA_MODE=1（滑窗 + 压缩稀疏）/ CFA_MODE=2

公式出处（op_kernel/arch32/）：
  sparse_attn_sharedkv_swa_kernel.h L690-692（scfa_kernel.h L716-718 同）：
        s2Start = s1EndIdx - oriWinLeft ; s2End = s1EndIdx + oriWinRight
        ⇒ ori KV 滑窗 = [s1_abs - ori_win_left, s1_abs + ori_win_right]，
          s1_abs = s1 + (kvLen_ori - qLen)（对齐到因果位置）
  sparse_attn_sharedkv_swa_block_vector.h：
        L325 Muls(mmRes, mmRes, softmaxScale) → logits = (q·kv) * softmax_scale
        L268-293 sinks 是逐 q head 的标量，Brcb 广播到整行
        L295-307 WholeReduceMax 把 sink 一起纳入 max
        L687  sink 的 exp 只进 softmax 分母
  ⇒ 语义等价核心：
        logits = [ (q·ori_kv[j]) for j in 滑窗 ] ⊕ [ (q·cmp_kv[t]) for t in cmp_sparse_indices ]
        logits *= softmax_scale
        m   = max( max(logits), sinks[n1] )
        p   = exp(logits - m)
        lse = Σ p + exp(sinks[n1] - m)          # sink 只贡献分母，不贡献 V
        out = (Σ p · v) / lse

layout 归一化（host 只做布局簿记，核心数学在 Triton）：
    TND：由 cu_seqlens_q 得到每 token 的 (batch, 序列内位置)，由 cu_seqlens_*/seqused_* 得到每 batch 的 kv 长度
    PA_ND：分页寻址在 **kernel 内**用 block_table 完成
           ori: page=ori_block_table[b, j//bs]; row=page*bs + j%bs（cmp 同理）

**简化点 / 未实现（不支持的取值都显式抛 NotImplementedError）**：
    1. ori_kv_stride / cmp_kv_stride != 0 未实现（非 0 表示块间非紧凑排布）
    2. metadata 未使用（多核调度描述符，不携带数学语义）
    3. 不做多核 Cube/Vector 流水；两趟算法（先算 logits 落 GM，再归一化加权 V）
    4. cmp_mask_mode=3 的下三角在**原始序列位置**上折叠 cmp_ratio：
       n_valid_cmp = (s1_abs_orig + 1) // cmp_ratio，
       s1_abs_orig = s1 + (kvLen_ori - qLen)（无 ori 分支时用 kvLen_cmp*cmp_ratio 还原）
       出处：swa_kernel.h L405 / scfa_kernel.h L373,589,600 的
             thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio；
       tiling 校验 cmp_ratio ∈ {4, 128}（sparse_attn_sharedkv_tiling.cpp L1267）
    5. ori 分支的位置由滑窗决定，不接受外部 ori_sparse_indices

## ⚠️ 曾经的错误（后来人请引以为戒，勿删）
本文件的 cmp 下三角掩码**一度被改成"不折算 cmp_ratio"**（`j <= s1 + (kvLen_cmp - qLen)`）。
原因：加入 cmp 掩码后 BSND 用例回归，而**我的测试参考当时没有折叠 cmp_ratio** ——
我误判为"实现错了"，于是**改实现去迁就一个错误的参考**，结果测试全绿、实现却是错的。

后来用 4 条独立证据裁定"参考才是错的"（裁定优先级：仓库自带 golden > 内核原文 > 自写参考）：
  ① `benchmarks/prepare_indexer_indices.py` L22-27（仓库自带 golden，非本文件作者所写）：
         `visible = (positions + 1) // compress_ratio` 且哨兵用 -1
  ② 本算子内核 `sparse_attn_sharedkv_swa_kernel.h` L405（scfa_kernel.h L373/589/600 同）：
         `thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio`
  ③ 物理含义 `scfa_block_vector.h` L552：cmp cache 行数 = `kvSeqSize / cmpRatio`
     （每 cmpRatio 个原始 token 压成 1 行）
  ④ 合法取值 `sparse_attn_sharedkv_tiling.cpp` L1267：`cmp_ratio ∈ {4, 128}`
     （我最初测试用的 2 是**非法值**，即使 PASS 也不代表生产）

**结论/规范**：回归出现时先判断"实现错还是参考错"，**绝不能为了让测试变绿而改实现**；
写参考时**优先抄仓库自带的 golden**，而不是凭自己理解写。

## ⚠️ TND+PA 数值错误的根因（2026-10-12 定位并修复，勿回退）
**症状**：BSND 8/8 PASS，TND+PA 4/4 FAIL，fp16 err≈1.46e-01（阈值 1e-3），
bf16 err≈3.65e-01，另有 `out=512.0` 之类的 fp16 饱和值。

**根因**：块表寻址里，**负下标没有被 clamp 住**。滑窗在序列开头会出现
`j = base_o + lane < 0` 的 lane（`base_o = s1_abs - win_left` 可能是负数），
旧的 `_row_of` 写成 `j_safe = tl.maximum(j, 0)`，但**该 clamp 没有传播到
`j_safe // BLOCK_SIZE` / `j_safe % BLOCK_SIZE`**：实测这些 lane 读到的
不是逻辑第 0 行，而是**别的行**（甚至完全越界的地址）。证据（设备实测）：

* 逐列对比内部 score 缓冲 vs 参考 logits：只有 `j<0` 的列错
  （如 `j=-59..-56` 的 4 列），`j>=0` 的列误差 ~1e-7；
* 身份块表（页号=0,1,2,3）下仍是同样 4 列错 ⇒ 与块表内容无关；
* 改写前 score 误差 |max| 随模块增大到 1.5e+02~1.9e+03，输出出现 NaN/793/512；
* 反向验证：`j>=0` 的 lane 全部与参考一致 ⇒ 数学部分没错，只是寻址。

**修法（本文件当前实现）**：让**进入取模/除法/地址运算的下标恒为非负**，
mask 只用非负量表达：

* 滑窗：`row0 = max(base_o, 0)`（地址基准）、`shift = max(-base_o, 0)`（只进 mask）、
  `cnt = min(N_ORI, kvlen - row0)`；有效 lane 为 `shift <= lane < shift + cnt`，
  地址行 = `row0 + min(max(lane - shift, 0), cnt - 1)`。
  恒等式 `row0 + (lane-shift) + shift == base_o + lane` 保证数学不变。
* 稀疏 cmp：非法/越界索引（含 -1）**回落到行 0**，只用 `vld` 决定数值，
  `tl.where(vld, i0, 0)` 之后才允许进 `_kv_elem_off`。
* score 缓冲：pad 列回落到最后一个合法列（同理，避免越界地址）。
* `_kv_elem_off` 只接受**已保证合法**的行号（内部仍保留 min/max 兜底）。

**为什么不能"只加 clamp"**：`tl.maximum(j, 0)` 这种就地 clamp 在 Ascend 后端上
对后续 `//`、`%` 不生效（实测），而"把负值从寻址里彻底拿掉"才可靠。
"""

from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_attn_sharedkv"]

_NEG = -1.0e30

# 调试槽位：仅当 SASKV_DUMP_SCORES=1 时由 sparse_attn_sharedkv 写入中间 score 缓冲，
# 供 _probe_saskv_*.py 做数值定位；生产路径不读不写、零开销。
_DBG: dict = {}


@triton.jit
def _kv_elem_off(KV, BT, b, kvrow, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2,
                 PA: tl.constexpr):
    """KV cache 里 (行 kvrow, head n2, 元素 0) 的线性偏移。

    ``kvrow`` 必须是**逻辑行号**（PA 下 = 块表页号*blockSize + 块内行），
    且调用方保证它落在 ``[0, 该 batch 的有效行数)``。绝不接受负数/巨大值：
    Ascend 后端上「负数取模/除法」与「被 mask 掉的地址」都会出事（见文件头）。
    """
    if PA:
        bidx = tl.minimum(tl.maximum(kvrow // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
        page = tl.maximum(tl.load(BT + b * MAX_BLOCKS + bidx), 0)
        row = page * BLOCK_SIZE + (kvrow % BLOCK_SIZE)
    else:
        row = b * S2 + kvrow
    return (row * N2 + n2) * D


@triton.jit
def _saskv_score_kernel(
    Q, OKV, CKV, CIDX, SB, TOK_B, TOK_S1, KVLEN_O, KVLEN_C, QLEN, BTO, BTC,
    N1, N2, GS,
    BLOCK_SIZE_O, MAX_BLOCKS_O, S2O, BLOCK_SIZE_C, MAX_BLOCKS_C, S2C, CMP_RATIO,
    SCALE, WIN_LEFT,
    N_ORI: tl.constexpr,
    K_IDX: tl.constexpr,
    PA_O: tl.constexpr,
    PA_C: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    ROW_MODE: tl.constexpr,
):
    row = tl.program_id(0)                 # 一行 = 一个 (token, q head)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    q = tl.load(Q + row * D + d, mask=dm, other=0.0).to(tl.float32)
    total = N_ORI + K_IDX

    # ---- ori KV 滑窗 ----
    # ⚠️ ROW_MODE=1 用**逐行 1-D load** 取代 [BLK_K, BLK_D] 的 2-D 间接寻址 tile。
    #    设备实测（TND+PA，B2 qlen[5,4] kvo[64,48] kvc[24,16]）：
    #    t=0 的 ori 段 lane 68..71（j=0..3，窗口里最早的 4 个有效行）数值错，
    #    lane 72+ 精确；t=5 时 lane 83,84 对而 85..87 = 0/任意值、88+ 全对。
    #    ⇒ 2-D tile 形态在**窗口最初几个有效 lane** 上丢写/读脏，与 BACKEND_NOTES 坑 1 同族
    #    （2-D tile + 间接寻址）。逐行 1-D 形态是已由 `_kv_cache_kernel` 验证的可靠写法。
    if N_ORI > 0:
        kvlo = tl.load(KVLEN_O + b)
        base_o = s1 + (kvlo - tl.load(QLEN + b)) - WIN_LEFT
        if ROW_MODE:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ji = base_o + li
                    vi = (ji >= 0) & (ji < kvlo) & (li < N_ORI)
                    off_i = _kv_elem_off(OKV, BTO, b, tl.where(vi, ji, 0), n2, BLOCK_SIZE_O,
                                         MAX_BLOCKS_O, N2, D, S2O, PA_O)
                    kvi = tl.load(OKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    tl.store(SB + row * total + li,
                             tl.where(vi, tl.sum(kvi * q, axis=0) * SCALE, NEG),
                             mask=li < N_ORI)
        else:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                lane = c * BLK_K + tl.arange(0, BLK_K)
                window = lane < N_ORI
                # 逐 lane 用 where 消除负下标：负值根本不进入取址（见文件头「根因」）
                j = base_o + lane
                vld = (j >= 0) & (j < kvlo)
                rw = tl.where(vld, j, 0)
                off = _kv_elem_off(OKV, BTO, b, rw, n2, BLOCK_SIZE_O, MAX_BLOCKS_O,
                                   N2, D, S2O, PA_O)
                kv = tl.load(OKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                sc = tl.sum(kv * q[None, :], axis=1) * SCALE
                tl.store(SB + row * total + lane, tl.where(vld, sc, NEG), mask=window)

    # ---- cmp KV 稀疏挑选 ----
    if K_IDX > 0:
        kvlc = tl.load(KVLEN_C + b)
        # cmp_mask_mode=3（下三角）：**在原始序列位置上折叠 cmp_ratio**。
        # 出处：sparse_attn_sharedkv_swa_kernel.h L405（scfa_kernel.h L373/589/600 同）
        #     thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
        # s1EndIdx 是**原始序列**里的绝对位置（同一变量也用于 ori 滑窗，见 L692），
        # 因此用 ori 的 kv 长度对齐；无 ori 分支时用 cmp 长度 × cmp_ratio 还原。
        if N_ORI > 0:
            ref_len = tl.load(KVLEN_O + b)
        else:
            ref_len = kvlc * CMP_RATIO
        n_valid = (s1 + 1 + (ref_len - tl.load(QLEN + b))) // CMP_RATIO
        if ROW_MODE:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ii = tl.load(CIDX + (t * N2 + n2) * K_IDX + li, mask=li < K_IDX, other=-1)
                    vi = (ii >= 0) & (ii < tl.minimum(kvlc, n_valid)) & (li < K_IDX)
                    off_i = _kv_elem_off(CKV, BTC, b, tl.where(vi, ii, 0), n2, BLOCK_SIZE_C,
                                         MAX_BLOCKS_C, N2, D, S2C, PA_C)
                    kvi = tl.load(CKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    tl.store(SB + row * total + N_ORI + li,
                             tl.where(vi, tl.sum(kvi * q, axis=0) * SCALE, NEG),
                             mask=li < K_IDX)
        else:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                slot = c * BLK_K + tl.arange(0, BLK_K)
                window = slot < K_IDX
                i0 = tl.load(CIDX + (t * N2 + n2) * K_IDX + slot, mask=window, other=-1)
                # 非法索引（-1 / 越界）回落到行 0，只用 mask 决定数值：地址恒合法
                vld = (i0 >= 0) & (i0 < tl.minimum(kvlc, n_valid))
                off = _kv_elem_off(CKV, BTC, b, tl.where(vld, i0, 0), n2, BLOCK_SIZE_C,
                                   MAX_BLOCKS_C, N2, D, S2C, PA_C)
                kv = tl.load(CKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                sc = tl.sum(kv * q[None, :], axis=1) * SCALE
                tl.store(SB + row * total + N_ORI + slot, tl.where(vld, sc, NEG), mask=window)


@triton.jit
def _saskv_out_kernel(
    OKV, CKV, SB, CIDX, SINK, OUT, LSE_MAX, LSE_SUM,
    TOK_B, TOK_S1, KVLEN_O, KVLEN_C, QLEN, BTO, BTC,
    N1, N2, GS,
    BLOCK_SIZE_O, MAX_BLOCKS_O, S2O, BLOCK_SIZE_C, MAX_BLOCKS_C, S2C, CMP_RATIO,
    WIN_LEFT, LSE_B_STRIDE, LSE_N2_STRIDE,
    N_ORI: tl.constexpr,
    K_IDX: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    PA_O: tl.constexpr,
    PA_C: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    ROW_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_d = tl.program_id(1)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS
    # D 方向分块（第二个 grid 轴）：把 [BLK_K, BLK_D] KV tile 的 UB 占用压下来。
    # 注：曾改成「块内循环」以省 program 数，但 TND+PA 用例出现 NaN（数值不对），
    # 已回退到 grid 轴方案——grid 轴版本是单测全 PASS 验证过的。
    d = pid_d * BLK_D + tl.arange(0, BLK_D)
    dm = d < D

    total = N_ORI + K_IDX
    ar = tl.arange(0, BLK_ALL)
    # 地址 clamp：`ar` 可以到 BLK_ALL-1 > total，末行的 `row*total + ar` 会越过
    # SB 缓冲区（掩码挡不住地址计算，见文件头「根因」一节）。
    sv = tl.load(SB + row * total + tl.minimum(ar, total - 1), mask=ar < total, other=NEG)
    if HAS_SINK:
        sink = tl.load(SINK + n1)
        mx = tl.maximum(tl.max(sv, axis=0), sink)
        p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
        lse = tl.sum(p, axis=0) + tl.exp(sink - mx)      # sink 只进分母
    else:
        mx = tl.max(sv, axis=0)
        p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
        lse = tl.sum(p, axis=0)

    # ---- 头部标量只读一次（与 D 无关）----
    if N_ORI > 0:
        kvlo = tl.load(KVLEN_O + b)
        base_o = s1 + (kvlo - tl.load(QLEN + b)) - WIN_LEFT
    if K_IDX > 0:
        kvlc = tl.load(KVLEN_C + b)
        # 同 score kernel：cmp 下三角在**原始序列位置**上折叠 cmp_ratio
        # （swa_kernel.h L405 / scfa_kernel.h L373）
        if N_ORI > 0:
            ref_len = tl.load(KVLEN_O + b)
        else:
            ref_len = kvlc * CMP_RATIO
        n_valid = (s1 + 1 + (ref_len - tl.load(QLEN + b))) // CMP_RATIO

    acc = tl.zeros([BLK_D], dtype=tl.float32)

    # ⚠️ 与 score kernel 同为 ROW_MODE=1 的逐行 1-D load（见 score kernel 上方注释）。
    if N_ORI > 0:
        if ROW_MODE:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ji = base_o + li
                    vi = (ji >= 0) & (ji < kvlo) & (li < N_ORI)
                    off_i = _kv_elem_off(OKV, BTO, b, tl.where(vi, ji, 0), n2, BLOCK_SIZE_O,
                                         MAX_BLOCKS_O, N2, D, S2O, PA_O)
                    kvi = tl.load(OKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    pci = tl.load(SB + row * total + li, mask=li < N_ORI, other=NEG)
                    acc += tl.where(vi, tl.exp(pci - mx), 0.0) * kvi
        else:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                lane = c * BLK_K + tl.arange(0, BLK_K)
                window = lane < N_ORI
                j = base_o + lane
                vld = (j >= 0) & (j < kvlo)
                rw = tl.where(vld, j, 0)
                pc = tl.load(SB + row * total + lane, mask=window, other=NEG)
                pc = tl.where(vld, tl.exp(pc - mx), 0.0)
                off = _kv_elem_off(OKV, BTO, b, rw, n2, BLOCK_SIZE_O, MAX_BLOCKS_O,
                                   N2, D, S2O, PA_O)
                kv = tl.load(OKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(pc[:, None] * kv, axis=0)

    if K_IDX > 0:
        if ROW_MODE:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ii = tl.load(CIDX + (t * N2 + n2) * K_IDX + li, mask=li < K_IDX, other=-1)
                    vi = (ii >= 0) & (ii < tl.minimum(kvlc, n_valid)) & (li < K_IDX)
                    off_i = _kv_elem_off(CKV, BTC, b, tl.where(vi, ii, 0), n2, BLOCK_SIZE_C,
                                         MAX_BLOCKS_C, N2, D, S2C, PA_C)
                    kvi = tl.load(CKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    pci = tl.load(SB + row * total + N_ORI + li, mask=li < K_IDX, other=NEG)
                    acc += tl.where(vi, tl.exp(pci - mx), 0.0) * kvi
        else:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                slot = c * BLK_K + tl.arange(0, BLK_K)
                window = slot < K_IDX
                i0 = tl.load(CIDX + (t * N2 + n2) * K_IDX + slot, mask=window, other=-1)
                vld = (i0 >= 0) & (i0 < tl.minimum(kvlc, n_valid))
                pc = tl.load(SB + row * total + N_ORI + slot, mask=window, other=NEG)
                pc = tl.where(vld, tl.exp(pc - mx), 0.0)
                off = _kv_elem_off(CKV, BTC, b, tl.where(vld, i0, 0), n2, BLOCK_SIZE_C,
                                   MAX_BLOCKS_C, N2, D, S2C, PA_C)
                kv = tl.load(CKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(pc[:, None] * kv, axis=0)

    res = tl.where(lse > 0.0, acc / lse, tl.zeros([BLK_D], dtype=tl.float32))
    tl.store(OUT + row * D + d, res.to(OUT.dtype.element_ty), mask=dm)

    if RETURN_LSE:
        if pid_d == 0:      # LSE 与 D 无关，只写一次
            tl.store(LSE_MAX + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), mx)
            tl.store(LSE_SUM + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), lse)


def _lens_per_batch(seq, B, name) -> torch.Tensor | None:
    """[B] 每 batch 长度 或 [B+1] cu_seqlens → 每 batch 长度（int32 CPU）。"""
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B:
        return s.to(torch.int32)
    if s.numel() == B + 1:
        return (s[1:] - s[:-1]).to(torch.int32)
    raise ValueError(f"{name} 长度必须是 B={B} 或 B+1={B + 1}，实际 {s.numel()}")


def sparse_attn_sharedkv(
    q: torch.Tensor,
    ori_kv: torch.Tensor | None = None,
    cmp_kv: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    softmax_scale: float = 1.0,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    cmp_ratio: int = 1,
    return_softmax_lse: bool = False,
    ori_sparse_indices: torch.Tensor | None = None,
    ori_block_table: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_kv: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_kv_stride: int = 0,
    cmp_kv_stride: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
):
    """等价于 AscendC 的 SparseAttnSharedkv（BSND/TND × BSND/PA_ND）。

    Args:
        q: BSND [B,S1,N1,D] / TND [T,N1,D]
        ori_kv/cmp_kv: BSND [B,S2*,N2,D] / PA_ND [numBlocks, blockSize, N2, D]
        cmp_sparse_indices: BSND [B,S1,N2,K] / TND [T,N2,K]（-1 无效）
        sinks: [N1] fp32
        cu_seqlens_q: layout_q="TND" 时必填（cu_seqlens，长度 B+1，末项 = T）
        cu_seqlens_ori_kv/cu_seqlens_cmp_kv: 各 batch 的 kv 长度（[B] 或 [B+1] cu_seqlens）
        seqused_q/seqused_kv: 各 batch 的 q/kv 使用长度（[B]）
        ori_block_table/cmp_block_table: layout_kv="PA_ND" 时必填
        layout_q/layout_kv: "BSND"/"TND" 与 "BSND"/"PA_ND"（"PA_BSND" 同义）
        ori_kv_stride/cmp_kv_stride: 只支持 0

    Returns:
        (attn_out 同 q, lse_max, lse_sum)
    """
    if layout_q not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_q={layout_q!r} 未实现（支持 BSND/TND）")
    if layout_kv not in ("BSND", "PA_ND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_kv={layout_kv!r} 未实现（支持 BSND/PA_ND）")
    if ori_sparse_indices is not None:
        raise NotImplementedError("ori_sparse_indices 未实现（ori 分支位置由滑窗决定）")
    if int(ori_kv_stride) != 0 or int(cmp_kv_stride) != 0:
        raise NotImplementedError("ori/cmp_kv_stride != 0 未实现（块间非紧凑排布）")
    if int(ori_mask_mode) != 4:
        raise NotImplementedError(f"ori_mask_mode={ori_mask_mode} 未实现（只支持 4: 滑窗）")
    if int(cmp_mask_mode) != 3:
        raise NotImplementedError(f"cmp_mask_mode={cmp_mask_mode} 未实现（只支持 3: 下三角）")
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError(f"metadata 必须是 int32 描述符张量，实际 {metadata.dtype}")
    if cmp_kv is None and ori_kv is None:
        raise ValueError("ori_kv 与 cmp_kv 至少要有一个")
    pa = layout_kv in ("PA_ND", "PA_BSND", "PA_BBND")
    tnd = layout_q == "TND"

    # ---------------- layout 归一化 ----------------
    if tnd:
        if cu_seqlens_q is None:
            raise ValueError("layout_q=TND 必须提供 cu_seqlens_q（长度 B+1，末项 = T）")
        T, N1, D = q.shape
        cu = cu_seqlens_q.detach().to(torch.int64).cpu().flatten()
        if cu.numel() < 2 or int(cu[-1]) != T:
            raise ValueError(f"cu_seqlens_q 必须是 cu_seqlens（长度 B+1、末项={T}）")
        B = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = q.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    N2 = (cmp_kv if cmp_kv is not None else ori_kv).shape[2]
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} 必须能被 N2={N2} 整除")
    gs = N1 // N2

    def _kv_geom(kv, bt, nm):
        """返回 (block_size, max_blocks, S2, bt_tensor)。"""
        if kv is None:
            return 1, 0, 0, q
        if pa:
            nb, bs, n2k, dk = kv.shape
            if bt is None:
                raise ValueError(f"layout_kv={layout_kv} 必须提供 {nm} 的 block_table")
            if n2k != N2 or dk != D:
                raise ValueError(f"{nm} 的 N2/D 与 q 不匹配: {tuple(kv.shape)} vs N2={N2},D={D}")
            return bs, bt.shape[1], bs * bt.shape[1], bt.contiguous().to(torch.int32)
        _b, s2, n2k, dk = kv.shape
        if n2k != N2 or dk != D:
            raise ValueError(f"{nm} 的 N2/D 与 q 不匹配: {tuple(kv.shape)} vs N2={N2},D={D}")
        return 1, 0, s2, q

    bso, mbo, S2O, BTO = _kv_geom(ori_kv, ori_block_table, "ori_block_table")
    bsc, mbc, S2C, BTC = _kv_geom(cmp_kv, cmp_block_table, "cmp_block_table")

    if ori_kv is not None and S2O <= 0:
        raise ValueError("ori_kv 的 kv 长度非法")
    if cmp_kv is not None and cmp_sparse_indices is None:
        raise ValueError("给了 cmp_kv 就必须给 cmp_sparse_indices")
    if sinks is not None and tuple(sinks.shape) != (N1,):
        raise ValueError(f"sinks 必须是 [N1]={(N1,)}")
    if sinks is not None and sinks.dtype != torch.float32:
        sinks = sinks.float()

    kvlo_b = _lens_per_batch(cu_seqlens_ori_kv, B, "cu_seqlens_ori_kv")
    kvlc_b = _lens_per_batch(cu_seqlens_cmp_kv, B, "cu_seqlens_cmp_kv")
    if kvlo_b is None:
        kvlo_b = _lens_per_batch(seqused_kv, B, "seqused_kv")
    if kvlo_b is None:
        kvlo_b = torch.full((B,), S2O, dtype=torch.int32)
    if kvlc_b is None:
        kvlc_b = torch.full((B,), S2C if cmp_kv is not None else 0, dtype=torch.int32)
    sq = _lens_per_batch(seqused_q, B, "seqused_q")
    if sq is not None:
        qlen_b = sq
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    n_ori = int(ori_win_left) + int(ori_win_right) + 1
    if ori_kv is None:
        n_ori = 0
    k_idx = cmp_sparse_indices.shape[-1] if cmp_sparse_indices is not None else 0
    if cmp_kv is None:
        k_idx = 0
    if n_ori + k_idx == 0:
        raise ValueError("ori_kv 与 cmp_kv 至少要有一个有效分支")

    dev = q.device
    q2 = q.contiguous().reshape(T, N1, D)
    okv = ori_kv.contiguous() if ori_kv is not None else q
    ckv = cmp_kv.contiguous() if cmp_kv is not None else q
    if cmp_sparse_indices is not None:
        cidx = cmp_sparse_indices.contiguous().reshape(T, N2, k_idx).to(torch.int32)
    else:
        cidx = torch.zeros(1, dtype=torch.int32, device=dev)

    n_rows = T * N1
    sb = torch.empty((n_rows, n_ori + k_idx), dtype=torch.float32, device=dev)
    out = torch.empty((n_rows, D), dtype=q.dtype, device=dev)
    blk_d = next_pow2(D)
    # [BLK_K, BLK_D] KV/score tile 的 UB 占用必须留在 192KB 以内：D=576 时
    # BLK_D=1024、BLK_K=16 会报 `ub overflow, requires 8691712 bits`（5.5 倍），
    # 这里允许按环境变量收缩（默认值在实测后定）。
    blk_k = int(_os.environ.get("SASKV_BLK_K", "8"))
    # ROW_MODE=1（默认）：逐行 1-D load，避开 Ascend 后端上
    # 「2-D tile + 间接寻址」在窗口最初几个有效 lane 丢写的问题（见两个 kernel 内注释）。
    # 置 SASKV_ROW_MODE=0 回到 2-D tile 形态，仅供 A/B 对照。
    row_mode = 0 if _os.environ.get("SASKV_ROW_MODE") == "0" else 1
    common = (tok_b.to(dev), tok_s1.to(dev), kvlo_b.to(dev), kvlc_b.to(dev), qlen_b.to(dev))
    geom = (bso, mbo, S2O, bsc, mbc, S2C, int(cmp_ratio))

    _saskv_score_kernel[(n_rows,)](
        q2, okv, ckv, cidx, sb, *common, BTO, BTC,
        N1, N2, gs, *geom,
        float(softmax_scale), int(ori_win_left),
        N_ORI=n_ori, K_IDX=k_idx,
        PA_O=bool(pa), PA_C=bool(pa), NEG=_NEG, D=D, BLK_D=blk_d, BLK_K=blk_k,
        ROW_MODE=row_mode,
    )
    if _os.environ.get("SASKV_DUMP_SCORES"):
        # 调试槽（见 _DBG 定义）：探针据此对比内部 score 与参考 logits
        _DBG["sb"] = sb
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
    sink_ptr = sinks if sinks is not None else q
    lse_b_stride = 0 if tnd else (N2 * S1 * gs)
    lse_n2_stride = (T * gs) if tnd else (S1 * gs)

    # D 方向分块：`_saskv_out_kernel` 的 [BLK_K, BLK_D] KV tile + int64 偏移向量
    # 在 BLK_D=next_pow2(D)=1024 时 UB 需求 8,691,712 bits（5.5 倍），
    # 只缩 BLK_K 降到 2,886,144 bits 仍超（1.8 倍）→ 按 D 分块。
    # 分块放在**块内循环**（grid 仍是 n_rows）：放 grid 轴会让 program 数×ceil(D/BLK_D)
    # 且每份重算 score，生产上整轮超时（RC=124）。
    blk_d_out = max(16, min(blk_d, int(_os.environ.get("SASKV_BLK_D_OUT", "128"))))
    _saskv_out_kernel[(n_rows, ceil_div(D, blk_d_out))](
        okv, ckv, sb, cidx, sink_ptr, out, lse_max, lse_sum,
        *common, BTO, BTC,
        N1, N2, gs, *geom,
        int(ori_win_left), lse_b_stride, lse_n2_stride,
        N_ORI=n_ori, K_IDX=k_idx,
        HAS_SINK=bool(sinks is not None), RETURN_LSE=bool(return_softmax_lse),
        PA_O=bool(pa), PA_C=bool(pa), NEG=_NEG, D=D, BLK_D=blk_d_out, BLK_K=blk_k,
        BLK_ALL=next_pow2(max(1, n_ori + k_idx)),
        ROW_MODE=row_mode,
    )
    if tnd:
        out = out.reshape(T, N1, D)
    else:
        out = out.reshape(B, S1, N1, D)
    return out, lse_max, lse_sum
