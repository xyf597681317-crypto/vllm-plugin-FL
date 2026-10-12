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
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
    M: tl.constexpr, EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HALF_MODE: tl.constexpr, HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    """每个 program 算 1 个输出 chunk（coff=2：重叠窗 2r + 通道分半）。

    ⚠️ **两块结构**（对齐 AscendC `MulReduceSumbaseVF(coff, cmpRatio, ...)` 的形态）：
      块0 = 前一个 chunk 的 r 行，投影通道 `[0, D)`；
      块1 = 当前 chunk   的 r 行，投影通道 `[D, 2D)`；
    两块**各自在自己的 [M] lane tile 内归约**（`tl.max`/`tl.sum` 沿 axis=0），再按
    `mx = max(m0, m1)`、`den = Σe0 + Σe1`、`o = Σ(e0·kv0) + Σ(e1·kv1)` 合并。
    **不再把 2r 行压成一个扁平窗 + 前缀 mask**：双 sentinel 实测证明，那种形态下
    ct=0（首块整块无历史）program 的结果会**随输出缓冲内容/分配漂移**
    （同输入：1.65→0.44、NaN→0.0），而 c≥1 逐位正确。
    空块以「整块 `have` 全 False ⇒ 整块 `e` 全 0」参与归约，**不做 lane 级兜底**。
    """
    b = tl.program_id(0)
    ct = tl.program_id(1)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    nc = tl.load(NC + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    ok = ct < nc
    c = c0 + ct
    sc = ct

    # 两块共用同一套 lane 编号；lane < R 是该块的真实行（其余是 padding lane）
    rows = tl.arange(0, M)
    jm = rows < R
    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)

    # 块0：绝对位置 [(c-1)r, c·r)，通道 [0, D)；块1：绝对位置 [c·r, (c+1)r)，通道 [D, 2D)
    p0 = (c - 1) * R + rows
    p1 = c * R + rows
    real0 = jm & (p0 >= sp) & (p0 < sp + su) & ok
    real1 = jm & (p1 >= sp) & (p1 < sp + su) & ok
    head0 = jm & (p0 >= 0) & (p0 < sp) & ok      # 历史位置 → 从 state_cache 读
    head1 = jm & (p1 >= 0) & (p1 < sp) & ok
    have0 = real0 | head0                        # 该行是否有真实数据；否则 score=MIN_SCORE 且 kv=0
    have1 = real1 | head1

    t0 = tl.where(real0, p0 - sp, 0)
    t1 = tl.where(real1, p1 - sp, 0)
    xr0 = xbase + t0
    xr1 = xbase + t1

    # 两块**顺序**处理 + online softmax：同一时刻只有一对 [M, BLK_D] 累加器存活
    #   m/l/acc = 当前 softmax 最大值 / 归一化和 / 加权 kv 和（各 [BLK_D]）
    # 这样既保留「两块独立归约后合并」的语义，UB 又与原实现同级（D=512 能编过）。
    m = tl.full((BLK_D,), MIN_SCORE, dtype=tl.float32)
    l = tl.zeros((BLK_D,), dtype=tl.float32)
    acc = tl.zeros((BLK_D,), dtype=tl.float32)

    for blk in range(2):
        if blk == 0:
            p = p0
            real = real0
            head = head0
            have = have0
            xr = xr0
            hoff = 0
        else:
            p = p1
            real = real1
            head = head1
            have = have1
            xr = xr1
            hoff = D

        kv = tl.zeros((M, BLK_D), dtype=tl.float32)
        scp = tl.zeros((M, BLK_D), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            wm = hmask[:, None] & dmask[None, :]
            wkv_b = tl.load(WKV + (hoff + d)[None, :] * H + hh[:, None], mask=wm, other=0.0)
            wgt_b = tl.load(WGATE + (hoff + d)[None, :] * H + hh[:, None], mask=wm, other=0.0)
            xv = tl.load(X + xr[:, None] * H + hh[None, :],
                         mask=real[:, None] & hmask[None, :], other=0.0)
            kv += tl.dot(xv, wkv_b)
            scp += tl.dot(xv, wgt_b)

        # --- 历史行：从 state_cache 读回；state 行 = [ kv(2D) | score+ape(2D) ] ---
        if HAS_STATE:
            q = tl.where(p >= 0, p, 0)
            blk_id = tl.load(BLKT + tl.where(CACHE_CYCLE, b, b * MAX_BLK + (q // BLK_SIZE)),
                             mask=head, other=0)
            soff = blk_id[:, None] * STRIDE0 + (q % BLK_SIZE)[:, None] * (4 * D) \
                + hoff + d[None, :]
            sm = head[:, None] & dmask[None, :]
            kv = tl.where(head[:, None], tl.load(STATE + soff, mask=sm, other=0.0), kv)
            scp = tl.where(head[:, None],
                           tl.load(STATE + soff + 2 * D, mask=sm, other=0.0), scp)

        # --- ape：行号 = 绝对位置 % r == rows；只加在「实算」行上 ---
        apev = tl.load(APE + rows[:, None] * (2 * D) + hoff + d[None, :],
                       mask=real[:, None] & dmask[None, :], other=0.0)
        scp += apev

        # --- 本块归约（空块 ⇒ have 全 False ⇒ e 全 0，整块不参与，不做 lane 级兜底）---
        good = have[:, None] & dmask[None, :]
        s = tl.where(good, scp, MIN_SCORE)
        m_new = tl.maximum(m, tl.max(s, axis=0))
        e = tl.where(good, tl.exp(s - m_new[None, :]), 0.0)
        alpha = tl.exp(m - m_new)                 # 旧累加随最大值变化的缩放
        l = l * alpha + tl.sum(e, axis=0)
        acc = acc * alpha + tl.sum(e * kv, axis=0)
        m = m_new

    o = acc / tl.where(l > 0.0, l, 1.0)

    # --- RMSNorm over D（不变）---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc
    # --- 非 rope 部分直接透传（单位步长索引）---
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

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
            tl.store(CMPKV + orow * D + ROFF + i, (ca * va - sa * vb).to(CMPKV.dtype.element_ty),
                     mask=ok)
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty), mask=ok)
    else:
        # out[i] = cos[i]*y[i] + sin[i]*(i 偶 ? -y[i+1] : +y[i-1])
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)


# ---------------------------------------------------------------------------
# kernel 2: 把本次调用覆盖的绝对位置写回 state_cache
#   coff=2 的 state 行宽 = 2*DC = 4D：[ kv(2D) | score+ape(2D) ]
#   4 次单位步长 store（两半分开），不使用任何偏移向量
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_coff2_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D_ST: tl.constexpr,
    P: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    whi = tl.load(WHI + b)
    xbase = tl.load(X_BASE + b)

    rows = t * P + tl.arange(0, P)
    p_abs = wlo + rows
    ok = p_abs < whi
    tok = p_abs - sp

    h = tl.arange(0, BLK_H)
    p_safe = tl.where(p_abs >= 0, p_abs, 0)
    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx, mask=ok, other=0)
    base = blk_id[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (4 * D)
    j = p_abs % R

    # D 轴分块：state kernel 无跨列归约，可以安全分块以控制 UB
    for d0 in range(0, D, BLK_D_ST):
        d = d0 + tl.arange(0, BLK_D_ST)
        dmask = d < D
        sm = ok[:, None] & dmask[None, :]
        kv_a = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        kv_b = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        sc_a = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        sc_b = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xm = ok[:, None] & hmask[None, :]
            xv = tl.load(X + (xbase + tl.where(tok >= 0, tok, 0))[:, None] * H + hh[None, :],
                         mask=xm, other=0.0)
            wm = hmask[:, None] & dmask[None, :]
            kv_a += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            kv_b += tl.dot(xv, tl.load(WKV + (D + d)[None, :] * H + hh[:, None], mask=wm, other=0.0))
            sc_a += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            sc_b += tl.dot(xv, tl.load(WGATE + (D + d)[None, :] * H + hh[:, None], mask=wm, other=0.0))
        # ape 行号 = p_abs % r；两半分别加各自的 ape 列
        sc_a = sc_a + tl.load(APE + j[:, None] * (2 * D) + d[None, :], mask=sm, other=0.0)
        sc_b = sc_b + tl.load(APE + j[:, None] * (2 * D) + D + d[None, :], mask=sm, other=0.0)
        tl.store(STATE + base + d[None, :], kv_a, mask=sm)                # kv   [0, D)
        tl.store(STATE + base + D + d[None, :], kv_b, mask=sm)            # kv   [D, 2D)
        tl.store(STATE + base + 2 * D + d[None, :], sc_a, mask=sm)        # score[2D, 3D)
        tl.store(STATE + base + 3 * D + d[None, :], sc_b, mask=sm)        # score[3D, 4D)


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
    nc_t = dev_arrays(nc_l)
    ob_t = dev_arrays(ob_l)

    cache_cycle = int(cache_mode) == 2
    half_mode = int(rotary_mode) == 1
    if not half_mode and (D - RD) % RD != 0:
        raise NotImplementedError(
            f"compressor INTERLEAVE 需要 (D-RD)%%RD==0；D={D} RD={RD}")

    if max_nc > 0:
        grid = (B, max_nc)
        _compressor_coff2_kernel[grid](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            sc_out if has_state else out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
            M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
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
        max_w = max((whi_l[b] - wlo_l[b]) for b in range(B)) if B else 0
        if max_w > 0:
            BLK_D_ST = min(BLK_D, 128)
            # 4 个 [P, BLK_D_ST] fp32 累加器 ≈ 64KB
            P = max(16, min(64, 4096 // BLK_D_ST))
            wlo_t = dev_arrays(wlo_l)
            whi_t = dev_arrays(whi_l)
            _compressor_coff2_state_kernel[(B, ceil_div(max_w, P))](
                x_c, wkv_c, wgate_c, ape_c, sc_out, blkt,
                x_base_t, sp_t, wlo_t, whi_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D_ST=BLK_D_ST, P=P,
                CACHE_CYCLE=bool(cache_cycle),
            )

    return out, sc_out
