"""compressor 的 `coff=2 (COFF::OVERLAP)` 路径 —— 跨 chunk 重叠窗压缩。

================================================================================
语义来源（三个独立来源互相印证，**不是猜测**）
================================================================================
记 r = cmp_ratio、D = headDim（= norm_weight 长度）、DC = coff*D = 2D（投影宽度）。

1) **上游 vLLM 官方 Triton 参考（主参考，可读且语义完整）**
   `/vllm-workspace/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
   `:156-211` `_fused_kv_compress_norm_rope_insert_sparse_attn`::

       if (position + 1) % COMPRESS_RATIO != 0: return
       start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
       tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
       pos = start + tokens
       mask_pos = pos >= 0
       head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE   # ← 通道分半
       row_base = state_cache_ptr + block_numbers*stride0 + block_offsets*stride1 + head_offset
       score = load(row_base[:,None] + STATE_WIDTH + block, mask=combined_mask, other=-inf)
       score = tl.softmax(score, dim=0)
       kv    = load(row_base[:,None] + block,            mask=combined_mask, other=0.0)
       compressed_kv = tl.sum(kv * score, axis=0)
       compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO       # ← RoPE 行基址

   `save_partial_states.py:92-101`: state 行 = `[ kv(DC) | score+ape(DC) ]`，
   `ape_row = position % compress_ratio`，`state_width = DC`。

2) **AscendC 源码（用于校验边界行为）**
   * `op_kernel/arch35/compressor_comm.h:83-86`  `enum class COFF { DISABLE = 1, OVERLAP = 2 };`
   * `op_host/arch32/compressor_tiling.cpp:829-852`（**shape 契约**）
       `wkv.shape[0] == wgate.shape[0] == coff*headDim`
       `stateCache.shape[2] == 2*coff*headDim`
       `ape.shape == [cmpRatio, coff*headDim]`
   * 同上 `:958-964` 合法组合只有 {r=4,coff=2,D=512} / {4,2,128} / {128,1,512}
   * 同上 `:179-183` coff==2 时 workspace ×2，且额外分配 `vec1TailCacheSize = cmpRatio*headDim`
   * `op_kernel/arch35/compressor_block_vec.h:906-928`（ReadState「填充左边/右边」）
       「填充右边」(headHolderSeqCnt>0) 读列 `dStartIdx + (coff-1)*headDim` → 通道 `[D,2D)`
       「填充左边」读列 `dStartIdx`                                → 通道 `[0,D)`
       `isFirst = bStartPos + sIdx < cmpRatio_`
   * 同上 `:866-879` `DuplicateFirstBlock` —— **无历史时 score 填 `SOFTMAX_MIN_NUM`、kv 填 `FLOAT_ZERO`**
   * 同上 `:937` `ReduceSize = coff_ * cmpRatio_`（softmax 归约长度 = 2r）
   * 同上 `:946` `MulReduceSumbaseVF(kv, score, dst, coff_, cmpRatio_, ...)`
   * 同上 `:574-582` coff=2 的 UB 布局注释 `|---D_L---|------D_R-----|`
   * 同上 `:629,636` 只有 `COMP::coff == COFF::OVERLAP` 才启用跨 chunk 的 `SaveToWorkSpace/LoadFromWorkSpace`

3) **模型侧等价改写**（`vllm_fl/models/deepseek_v4_ascend.py:664-670 overlap_transform`）::

       new_tensor[:, :, ratio:]  = tensor[:, :, :, d:]      # 当前 group 的后半通道 → 窗的后 r 行
       new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # 前一 group 的前半通道 → 窗的前 r 行

   并与 `vllm/models/deepseek_v4/compressor.py:141-142`
   `coff = 1 + (compress_ratio == 4)` / `sliding_window = coff*compress_ratio`（= 8 = 2r）自洽。

================================================================================
实现的语义（与上游参考逐行对应）
================================================================================
对**绝对 chunk 序号 c**（chunk c 覆盖绝对位置 [c*r, (c+1)*r)），其压缩输出的窗口是
**前一个 chunk ∪ 当前 chunk** 共 2r 个输入位置：`p_j = (c-1)*r + j`，`j ∈ [0, 2r)`。
（上游：`position = (c+1)*r - 1` ⇒ `start = position - 2r + 1 = (c-1)*r`，`tokens = j`。）

* 窗内 `j <  r`（前一个 chunk）取投影通道 `[0, D)`（上游 `head_offset = 0`）；
* 窗内 `j >= r`（当前 chunk） 取投影通道 `[D, 2D)`（上游 `head_offset = D`）。

即::

    kv_j[d]    = (x[p_j] @ wkv^T)[half(j)*D + d]
    score_j[d] = (x[p_j] @ wgate^T)[half(j)*D + d] + ape[j % r, half(j)*D + d]
    o[d]       = sum_j softmax_j(score[:, d]) * kv_j[d]     # softmax 在 **2r 行**上按列
    y          = RMSNorm_D(o) * norm_weight                 # ← 不变
    out        = RoPE(y)                                    # ← 不变，行基址 = c*r

顺序（Lead 强调）：**先 Σ_{2r} kv*softmax(score)**，**再** RMSNorm / RoPE。

边界（`DuplicateFirstBlock` / `ReadState`）：
* `p_j < 0`（窗口越过 batch 开头，即 `p_abs < r` 时的前 r 行）⇒
  **score = -inf（`MIN_SCORE`）且 kv = 0**，两者都做才等价于"忽略该行"；
* `0 <= p_j < start_pos` ⇒ 从 `state_cache` 读回（post-ape / pre-softmax），
  同样按 `half(j)` 选择通道 `[0,D)` 或 `[D,2D)`。

⚠️ **coff=2 与 coff=1 在相同 (r, D) 下输出必然不同**（窗长 2r vs r + 通道分半），
   不是"只是 workspace 变大"。生产 C4 路径 `overlap = (compress_ratio == 4)` 恒为 True，
   即 coff=2 是该路径的**唯一**配置，不能只接受参数而不实现。

================================================================================
后端约束的落实
================================================================================
* 2r 行 softmax 用**一个 [M, BLK_D] 寄存器张量**（`M = max(16, next_pow2(2r))`）表达，
  **不用 `tl.reshape` / `tl.join`**（坑 2）。
* 所有 store 索引 = `tl.arange`（单位步长）或"标量 + 常量"（坑 1）；
  RoPE 复用 `compressor_interleave.py` 已验证的**逐列标量展开**写法。
* `BLK_H` 按 `BLK_D` 自适应，保证 4 个权重 tile 的总字节数受控（UB 192KB，坑 3）。
* **程序级掩码全部由 host 决定**：同一个 kernel 用 `FIRST` constexpr 启动两次 ——
  常规 `grid=(方案数,)`（BIDX/CTX 传每个 program 的 `(batch, chunk)`）+ 序列首块
  `grid=(首块数,)`（BIDX 传 batch 号）。**kernel 内没有任何"本 program 是否有效"的
  运行时标量掩码**，每个启动的 program 都是活跃的 —— 这条路径曾触发
  `507015 VEC instruction error: the ub address out of bounds`（前任 Plan B 的失败形态）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["compressor_coff2"]

_MIN_SCORE = -1.0e30  # ⚠️ 不用 float("-inf")（Ascend 上会退化），用大负数替代

# 支持的 (cmp_ratio, coff, headDim) 组合 —— compressor_tiling.cpp:958-964
_SUPPORTED = ((4, 2, 512), (4, 2, 128), (128, 1, 512))


# ---------------------------------------------------------------------------
# kernel 1: 每个 program 计算 1 个输出 chunk（重叠窗 softmax + 加权和 + rmsnorm + rope）
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_coff2_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, OB, BIDX, CTX,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
    M: tl.constexpr, EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HALF_MODE: tl.constexpr, HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
    FIRST: tl.constexpr,
):
    # ---- 程序标识：**完全由 host 决定**，kernel 内没有任何「本 program 是否有效」的
    #      运行时标量掩码 ⇒ 每个启动的 program 都是活跃的（这条路径曾触发
    #      `507015 VEC instruction error: the ub address out of bounds`）。
    # FIRST=False（常规）：grid=(方案数,)，BIDX/CTX 给出每个 program 的 (batch, chunk)；
    #   被服务的 (b,ct) 满足 ct<nc[b] 且不含「c0==0 的首块」。
    # FIRST=True（序列首块 c0==0）：grid=(首块数,)，batch 号来自 BIDX，chunk 恒为该序列的首块。
    #   首块窗口不含任何负位置（c0==0 ⇒ p_abs=rows≥0），只用当前 chunk 的 r 行、通道 [D,2D)，
    #   并且走**与常规 program 完全相同的代码**（同一个 jit 函数，只换 constexpr）
    #   ⇒ codegen 形态与已验证可用的常规 program 一致。
    #   为什么首块要单独走：「(c-1)*r 起的 2r 窗」在 c==0 时含整块无历史的负位置前缀，
    #   实测这种 program 的输出会随输出缓冲的分配/内容漂移（BACKEND_NOTES 坑 4）。
    pid = tl.program_id(0)
    b = tl.load(BIDX + pid)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    rows = tl.arange(0, M)
    if FIRST:
        c = c0
        sc = 0
        in_win = rows < R                       # 窗长 = r（只有当前 chunk ⇒ 无负位置）
        p_abs = c0 * R + rows                   # ≥ 0
        half = rows < R                         # 当前 chunk ⇒ 通道 [D, 2D)
    else:
        ct = tl.load(CTX + pid)
        c = c0 + ct
        sc = ct
        in_win = rows < 2 * R                   # 窗长 = 2r（前 chunk ∪ 当前 chunk）
        p_abs = (c - 1) * R + rows              # j < r → 前一个 chunk；j >= r → 当前 chunk
        half = rows >= R                        # 通道分半开关（上游 head_offset）

    real = in_win & (p_abs >= sp) & (p_abs < sp + su)
    head = in_win & (p_abs >= 0) & (p_abs < sp)  # 历史位置 → 从 state_cache 读
    have = real | head                          # 该行是否有真实数据；否则 score=-inf 且 kv=0

    tok = p_abs - sp
    xrow = xbase + tl.where(tok >= 0, tok, 0)

    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)
    halfoff = half[:, None] * D                 # 通道分半偏移（标量表达式，非索引向量）

    acc_kv = tl.zeros((M, BLK_D), dtype=tl.float32)
    acc_sc = tl.zeros((M, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = real[:, None] & hmask[None, :]
        xv = tl.load(X + xrow[:, None] * H + hh[None, :], mask=xm, other=0.0)
        wm = hmask[:, None] & dmask[None, :]
        # 左矩阵（投影通道 [0,D)）与右矩阵（投影通道 [D,2D)）；wkv/wgate 形状 [2D, H]
        wkv_a = tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0)
        wkv_b = tl.load(WKV + (D + d)[None, :] * H + hh[:, None], mask=wm, other=0.0)
        wgt_a = tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0)
        wgt_b = tl.load(WGATE + (D + d)[None, :] * H + hh[:, None], mask=wm, other=0.0)
        ka = tl.dot(xv, wkv_a)
        kb = tl.dot(xv, wkv_b)
        ga = tl.dot(xv, wgt_a)
        gb = tl.dot(xv, wgt_b)
        acc_kv += tl.where(half[:, None], kb, ka)
        acc_sc += tl.where(half[:, None], gb, ga)

    # --- 历史行：从 state_cache 读回；state 行 = [ kv(2D) | score+ape(2D) ] ---
    #     通道 = half*D + d（上游 row_base 里的 head_offset）
    #     ⚠️ 必须用「标量行地址 + 1-D 列掩码 load」：实测 [M,BLK_D] 2-D 掩码 load +
    #     由内存载入的 blk_id 参与寻址会把**有效 lane** 读成错值（BACKEND_NOTES 1.2 症状 B
    #     的同族形态）。证据：流式第二段里含 history 行的两个 chunk err=1.27/1.44，
    #     不含 history 行的两个 chunk 只有 bf16 地板 1.9e-3/2.5e-3。
    if HAS_STATE:
        p_safe = tl.where(p_abs >= 0, p_abs, 0)
        # ⚠️ 非 history 行也指向「块 b*MAX_BLK 的第 0 行」（一定在界内），
        # 并且 blk_id 的 load **不带掩码** ⇒ 去掉「2-D 掩码 load + 掩码间接寻址」这个组合
        # （BACKEND_NOTES 1.2 症状 B 的触发形态）。行是否真的覆盖由下面的 tl.where 决定。
        p_rd = tl.where(head, p_safe, 0)
        blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_rd // BLK_SIZE))
        blk_id = tl.load(BLKT + blk_idx)
        row_in = p_rd % BLK_SIZE
        soff = blk_id[:, None] * STRIDE0 + row_in[:, None] * (4 * D) + halfoff + d[None, :]
        # 所有 lane 的地址都落在界内（非 history 行指向块 b*MAX_BLK 的第 0 行），
        # 因此这两次 load **完全不带掩码** ⇒ 彻底避开「2-D 掩码 load」这条被怀疑的路径。
        # 行是否真的采用读回值由下面的 tl.where(head) 决定。
        # （BLK_D == D 由 _SUPPORTED 保证：(4,2,512)/(4,2,128) 都是 2 的幂。）
        acc_kv = tl.where(head[:, None], tl.load(STATE + soff), acc_kv)
        acc_sc = tl.where(head[:, None], tl.load(STATE + soff + 2 * D), acc_sc)

    # --- ape：行号 = p_abs % r（== rows % r），列 = half*D + d；只加在真正的计算行上 ---
    j = rows % R
    apev = tl.load(APE + j[:, None] * (2 * D) + halfoff + d[None, :],
                   mask=real[:, None] & dmask[None, :], other=0.0)
    acc_sc = tl.where(real[:, None], acc_sc + apev, acc_sc)

    # --- 2r 行逐列 softmax + 加权和 ---
    #     无数据的行：score → MIN_SCORE（softmax 权重 0）。其 kv 已是 0（未写），
    #     两项都成立 ⇒ 等价于忽略该行（对应 DuplicateFirstBlock 的 -inf / 0）。
    good = have[:, None] & dmask[None, :]
    s_masked = tl.where(good, acc_sc, MIN_SCORE)
    mx = tl.max(s_masked, axis=0)
    ex = tl.exp(s_masked - mx[None, :])
    den = tl.sum(ex, axis=0)
    pj = ex / tl.where(den > 0.0, den, 1.0)[None, :]
    o = tl.sum(pj * acc_kv, axis=0)

    # --- RMSNorm over D（不变）---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc
    # --- 非 rope 部分直接透传（单位步长索引）；没有程序级掩码（每个 program 都活跃）---
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=dmask & (d < D - RD))

    # --- RoPE（不变）：逐列标量展开，索引 = 标量 + 常量 ⇒ 绝对单位步长 ---
    ROFF: tl.constexpr = D - RD
    if HALF_MODE:
        hr: tl.constexpr = RD // 2
        for i in range(0, hr):
            va = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vb = tl.sum(tl.where(d == ROFF + hr + i, y, 0.0), axis=0)
            ca = tl.load(RCOS + orow * RD + i)
            sa = tl.load(RSIN + orow * RD + i)
            cb = tl.load(RCOS + orow * RD + hr + i)
            sb = tl.load(RSIN + orow * RD + hr + i)
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ca * va - sa * vb).to(CMPKV.dtype.element_ty))
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty))
    else:
        # out[i] = cos[i]*y[i] + sin[i]*(i 偶 ? -y[i+1] : +y[i-1])
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ci * vi - si * vp).to(CMPKV.dtype.element_ty))
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ci * vi + si * vp).to(CMPKV.dtype.element_ty))


# ---------------------------------------------------------------------------
# kernel 2: 把本次调用覆盖的绝对位置写回 state_cache
#   coff=2 的 state 行宽 = 2*DC = 4D：[ kv(2D) | score+ape(2D) ]
#   每 program 1 个位置；写回一律「标量地址 + 1-D 列掩码 store」
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_coff2_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, RB, RQ,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D_ST: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    # 每 program 处理 **1 个绝对位置**（标量行号）：batch / 行号由 host 的方案表给出，
    # 没有任何「本 program 是否有效」的掩码；4 次写回都是 **标量地址 + 1-D 列掩码 store**
    # （BACKEND_NOTES 1.3 的可靠写法）。旧版是 [P,BLK_D] 2-D 掩码 store + 由内存载入的
    # blk_id 参与寻址，实测 kv 两个半区被写错（score 半区却是 1e-8 精确，且错的位置逐次漂移）。
    pid = tl.program_id(0)
    b = tl.load(RB + pid)
    q = tl.load(RQ + pid)

    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    xbase = tl.load(X_BASE + b)

    p_abs = wlo + q                     # 标量绝对位置（host 保证在写回范围内）
    tok = p_abs - sp
    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx)    # 标量、无掩码（无需掩码 ⇒ 不会出现越界地址）
    base = blk_id * STRIDE0 + (p_abs % BLK_SIZE) * (4 * D)
    j = p_abs % R

    h = tl.arange(0, BLK_H)
    xtok = tl.where(tok >= 0, tok, 0)
    # D 轴分块：无跨列归约，可以安全分块以控制 UB
    for d0 in range(0, D, BLK_D_ST):
        d = d0 + tl.arange(0, BLK_D_ST)
        dmask = d < D
        kv_a = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        kv_b = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        sc_a = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        sc_b = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xv = tl.load(X + (xbase + xtok) * H + hh, mask=hmask, other=0.0).to(tl.float32)
            wm = hmask[:, None] & dmask[None, :]
            # 权重 tile 与旧版逐字相同（[BLK_H, BLK_D_ST]），只把「矩阵乘」换成
            # 「广播乘 + axis=0 归约」⇒ 结果仍是 1-D 列向量，store 全是一维。
            kv_a += tl.sum(tl.load(WKV + d[None, :] * H + hh[:, None],
                                   mask=wm, other=0.0).to(tl.float32) * xv[:, None], axis=0)
            kv_b += tl.sum(tl.load(WKV + (D + d)[None, :] * H + hh[:, None],
                                   mask=wm, other=0.0).to(tl.float32) * xv[:, None], axis=0)
            sc_a += tl.sum(tl.load(WGATE + d[None, :] * H + hh[:, None],
                                   mask=wm, other=0.0).to(tl.float32) * xv[:, None], axis=0)
            sc_b += tl.sum(tl.load(WGATE + (D + d)[None, :] * H + hh[:, None],
                                   mask=wm, other=0.0).to(tl.float32) * xv[:, None], axis=0)
        # ape 行号 = p_abs % r；两半分别加各自的 ape 列
        sc_a = sc_a + tl.load(APE + j * (2 * D) + d, mask=dmask, other=0.0)
        sc_b = sc_b + tl.load(APE + j * (2 * D) + D + d, mask=dmask, other=0.0)
        tl.store(STATE + base + d, kv_a, mask=dmask)                # kv   [0, D)
        tl.store(STATE + base + D + d, kv_b, mask=dmask)            # kv   [D, 2D)
        tl.store(STATE + base + 2 * D + d, sc_a, mask=dmask)        # score[2D, 3D)
        tl.store(STATE + base + 3 * D + d, sc_b, mask=dmask)        # score[3D, 4D)


# ---------------------------------------------------------------------------
# host 侧
# ---------------------------------------------------------------------------
def _as_int32(t, device):
    if t is None:
        return None
    return t.to(device=device, dtype=torch.int32).contiguous()


def _check_shape_contract(x, wkv, wgate, ape, norm_weight, state_cache, coff, R, D):
    """compressor_tiling.cpp:829-852 的 shape 契约，显式校验（不满足直接报错）。"""
    DC = coff * D
    tag = "compressor_tiling.cpp:829-852"
    if wkv.shape[0] != DC or wgate.shape[0] != DC:
        raise ValueError(
            f"[{tag}] wkv.shape[0]={wkv.shape[0]} / wgate.shape[0]={wgate.shape[0]} "
            f"必须等于 coff*headDim={DC}（coff={coff}, headDim={D}）")
    if wkv.shape[1] != int(x.shape[-1]) or wgate.shape[1] != int(x.shape[-1]):
        raise ValueError(
            f"[{tag}] wkv.shape[1]={wkv.shape[1]} / wgate.shape[1]={wgate.shape[1]} "
            f"必须等于 hidden={int(x.shape[-1])}")
    if tuple(ape.shape) != (R, DC):
        raise ValueError(f"[{tag}] ape.shape={tuple(ape.shape)} 必须等于 [cmpRatio, coff*headDim]"
                         f"=[{R}, {DC}]")
    if norm_weight.numel() != D:
        raise ValueError(f"norm_weight.numel()={norm_weight.numel()} 必须等于 headDim={D}")
    if state_cache is not None and state_cache.shape[-1] != 2 * DC:
        raise ValueError(
            f"[{tag}] stateCache.shape[-1]={state_cache.shape[-1]} 必须等于 2*coff*headDim={2 * DC}")


def compressor_coff2(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_cache: torch.Tensor | None = None,
    state_block_table: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    start_pos: torch.Tensor | None = None,
    rope_head_dim: int = 64,
    cmp_ratio: int = 4,
    coff: int = 2,
    norm_eps: float = 1e-6,
    rotary_mode: int = 2,
    cache_mode: int = 1,
    state_cache_stride_dim0: int = 0,
    block_size: int | None = None,
    out_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AscendC `compressor` 的 `coff == 2 (COFF::OVERLAP)` 路径。

    与 `compressor()` 同签名；语义见模块 docstring（重叠窗 2r + 通道分半）。
    """
    if int(coff) != 2:
        raise ValueError(f"compressor_coff2 只处理 coff==2，收到 coff={coff}")

    dev = x.device
    R = int(cmp_ratio)
    D = int(norm_weight.numel())
    RD = int(rope_head_dim)
    H = int(x.shape[-1])

    if (R, 2, D) not in _SUPPORTED:
        raise ValueError(
            f"(cmp_ratio={R}, coff=2, headDim={D}) 不是合法组合；"
            f"合法组合只有 {_SUPPORTED}（compressor_tiling.cpp:958-964）")
    if RD % 2 != 0 or ((RD // 2) & (RD // 2 - 1)) != 0:
        raise ValueError(f"rope_head_dim/2 必须是 2 的幂，实际 RD={RD}")
    HALF = RD // 2
    if D % HALF != 0:
        raise ValueError(f"headDim({D}) 必须能被 rope_head_dim/2({HALF}) 整除")

    BLK_D = next_pow2(D)
    if BLK_D > 512:
        raise NotImplementedError(
            f"coff=2 路径 BLK_D={BLK_D} 超出 UB 预算（生产 C4 为 headDim=128）；"
            f"headDim>512 需要 D 轴分块的两遍实现，未实现")
    M = max(16, next_pow2(2 * R))
    # 4 个权重 tile（BLK_H x BLK_D, bf16/fp16）总字节数 ≈ 64KB
    BLK_H = max(16, min(128, 8192 // BLK_D))

    _check_shape_contract(x, wkv, wgate, ape, norm_weight, state_cache, 2, R, D)

    is_bs = x.dim() == 3
    if is_bs:
        B, S = int(x.shape[0]), int(x.shape[1])
        Sr = int(rope_sin.shape[1])
        x_base = [b * S for b in range(B)]
        lens = [S] * B
        out_shape = (B, Sr, D)
    else:
        T = int(x.shape[0])
        if cu_seqlens is None:
            raise ValueError("TH 布局（x 为 [T,H]）需要提供 cu_seqlens")
        B = int(cu_seqlens.numel()) - 1
        csl = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        lens = [int(csl[b + 1] - csl[b]) for b in range(B)]
        x_base = [int(csl[b]) for b in range(B)]
        Sr = int(rope_sin.shape[0]) if out_rows is None else int(out_rows)
        out_shape = (Sr, D)

    sp_l, su_l = [], []
    for b in range(B):
        sp = int(start_pos[b].item()) if start_pos is not None else 0
        su = int(seqused[b].item()) if seqused is not None else lens[b]
        sp_l.append(sp)
        su_l.append(su)

    c0_l = [sp_l[b] // R for b in range(B)]
    nc_l = [(sp_l[b] + su_l[b]) // R - c0_l[b] for b in range(B)]

    ob_l = []
    acc = 0
    for b in range(B):
        if is_bs:
            ob_l.append(b * Sr)
        else:
            ob_l.append(acc)
            acc += nc_l[b]
    max_nc = max(nc_l) if nc_l else 0

    out = torch.zeros(out_shape, dtype=x.dtype, device=dev)
    x_c = x.contiguous()
    wkv_c = wkv.contiguous()
    wgate_c = wgate.contiguous()
    ape_c = ape.contiguous().to(torch.float32)
    nw_c = norm_weight.contiguous().to(torch.float32)
    rs_c = rope_sin.contiguous().to(torch.float32)
    rc_c = rope_cos.contiguous().to(torch.float32)
    if rs_c.dim() == 3:
        rs_c = rs_c.reshape(-1, RD)
        rc_c = rc_c.reshape(-1, RD)

    has_state = state_cache is not None
    if has_state:
        sc_in = state_cache.to(torch.float32)
        sc_out = sc_in.clone()
        if block_size is None:
            block_size = int(sc_in.shape[1]) if sc_in.dim() >= 2 else int(sc_in.numel() // (4 * D))
    else:
        sc_out = None
        block_size = 0

    if has_state and state_block_table is None:
        raise ValueError("使用 state_cache 时必须提供 state_block_table")

    blkt = _as_int32(state_block_table, dev) if has_state else torch.zeros(1, dtype=torch.int32, device=dev)
    max_blk = int(state_block_table.shape[1]) if (state_block_table is not None and state_block_table.dim() == 2) else 1

    dev_arrays = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    x_base_t = dev_arrays(x_base)
    sp_t = dev_arrays(sp_l)
    su_t = dev_arrays(su_l)
    c0_t = dev_arrays(c0_l)
    ob_t = dev_arrays(ob_l)
    # ---- host 侧决定「哪些 (batch, chunk) 真的要算」----
    # ① c0==0 的序列首块：窗口含无历史的负位置前缀，走 FIRST 专用形态（BACKEND_NOTES 坑 4）；
    # ② 其余全部 (b, ct)（ct < nc[b]）。
    # 被服务的方案逐个进 grid ⇒ kernel 内没有任何「本 program 是否有效」的运行时标量掩码
    # （`507015 VEC instruction error: ub address out of bounds` 就出在被整 program mask 掉的路径上）。
    first_b = [b for b in range(B) if c0_l[b] == 0 and nc_l[b] > 0]
    pairs = [(b, ct) for b in range(B) for ct in range(nc_l[b])
             if not (ct == 0 and c0_l[b] == 0)]
    bidx_t = dev_arrays([p[0] for p in pairs]) if pairs else dev_arrays([0])
    ctx_t = dev_arrays([p[1] for p in pairs]) if pairs else dev_arrays([0])
    first_t = dev_arrays(first_b) if first_b else dev_arrays([0])

    cache_cycle = int(cache_mode) == 2
    half_mode = int(rotary_mode) == 1
    if not half_mode and (D - RD) % RD != 0:
        raise NotImplementedError(
            f"compressor INTERLEAVE 需要 (D-RD)%%RD==0；D={D} RD={RD}")

    if pairs:
        _compressor_coff2_kernel[(len(pairs),)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            sc_out if has_state else out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, ob_t, bidx_t, ctx_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
            M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            FIRST=False,
        )

    # 序列首块（c0==0）：host 决定 grid（= 首块 batch 数）+ 传 batch 号数组，
    # 用「只含当前 chunk 的 r 行、无负位置」的干净窗口算（见 kernel 头部注释）。
    if first_b:
        _compressor_coff2_kernel[(len(first_b),)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            sc_out if has_state else out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, ob_t, first_t, ctx_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
            M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            FIRST=True,
        )

    if has_state:
        # 写回范围：CONTINUOUS = [sp, sp+su)；CYCLE = [max(sp, Trunc(sp+su,r)), sp+su)
        wlo_l, whi_l = [], []
        for b in range(B):
            hi = sp_l[b] + su_l[b]
            lo = sp_l[b]
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            wlo_l.append(lo)
            whi_l.append(max(hi, lo))
        # 方案表：每个要写回的位置一个 program（batch + 行内序号），
        # ⇒ state kernel 内不需要任何「本行是否要写」的掩码。
        rb_l, rq_l = [], []
        for b in range(B):
            for q in range(whi_l[b] - wlo_l[b]):
                rb_l.append(b)
                rq_l.append(q)
        if rb_l:
            BLK_D_ST = min(BLK_D, 128)
            wlo_t = dev_arrays(wlo_l)
            rb_t = dev_arrays(rb_l)
            rq_t = dev_arrays(rq_l)
            _compressor_coff2_state_kernel[(len(rb_l),)](
                x_c, wkv_c, wgate_c, ape_c, sc_out, blkt,
                x_base_t, sp_t, wlo_t, rb_t, rq_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D_ST=BLK_D_ST,
                CACHE_CYCLE=bool(cache_cycle),
            )

    return out, sc_out
