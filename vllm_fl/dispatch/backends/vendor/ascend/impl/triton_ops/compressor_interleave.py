"""compressor 的 `rotary_mode=2 (INTERLEAVE)` 专用实现。

从 compressor.py 拆出（主模块已接近 q.py 的 ~24KB 单次发送上限）。

================================================================================
INTERLEAVE 的排查历史与根因（第 25 条坑）
================================================================================
**约定**（权威：910C 实跑 arch32，`op_kernel/arch32/rope.h` L94-114 `RotaryPosEmb<INTERLEAVE>`）：
    gatherOffset[k] = k^1（相邻列互换）; Muls(-1) 只作用于偶位; cos 按自然顺序读
    ⇒ out[k] = cos[k]*x[k] + sin[k]*(k 偶 ? -x[k+1] : +x[k-1])
      cos/sin 表宽 = col = RD（不是 D）
（arch35 `vf/vf_rope.h` L96-138 `InterleaveModeRopeVF` 同语义，只是用 DeInterleave/Interleave 指令。）
可辨识 oracle 已在 test_compressor.py 中通过：x=[1,2,3,4]、cos=sin=1、RD=4 → [-1,3,-1,7]。

**走过的弯路（4 个变体，误差逐位不变）**：
| 变体 | rope 实现 | 结果 |
|---|---|---|
| v1 | `tl.sum(tl.where(...,axis=1))` + 错的 cos 索引 | err 1.642 |
| v2 | `tl.split` + 修正索引 | err 3.235e-01，非rope 3.24e-01 |
| v3 | 换成 HALF 同款 `tl.sum(tl.where(...))` | **逐位不变** |
| v4 | **独立 kernel 体、无 constexpr** | **逐位不变** |

**根因（判别实验确定）**：把 rope 的两行 `tl.store` 删掉后，**非 rope 区立刻从 3.24e-01 变成 3.90e-03**
（= HALF 同配置的 3.895e-03）。
⇒ **v2/v3/v4 的 rope store 用了非单位步长的索引向量 `2*c2` / `2*c2+1`，写错地址、越界覆盖了非 rope 列。**
   Ascend 后端对**非单位步长的 masked store 索引向量**处理有误（与坑 12「`tl.reshape` 造偏移会错位」同类：
   **偏移/索引向量必须单位步长或算术式重算**）。
   这解释了"3 种 rope 实现给出完全相同的错值"——**错在地址，不在值**。

**修复**：先把 rope 段算成**自然顺序的 [RD] 向量**，再用**单位步长** `tl.arange(0, RD)` store。
   约束 `(D-RD) % RD == 0` 且 `RD | BLK_D`（不满足时主模块抛 NotImplementedError，不静默走错分支）。

**方法论**（已固化）：**改动 X 不改变结果 ⇒ X 不是原因，但 X 的【副作用】（如 store 地址）可能是。**
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["_compressor_interleave_kernel"]


@triton.jit
def _compressor_interleave_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr,
    M: tl.constexpr, EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
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
    rows = tl.arange(0, M)
    d = tl.arange(0, BLK_D)
    dmask = d < D
    p_abs = c * R + rows
    in_chunk = rows < R
    real = in_chunk & (p_abs >= sp) & ok
    head = in_chunk & (p_abs < sp) & ok
    tok = p_abs - sp
    xrow = xbase + tl.where(tok >= 0, tok, 0)

    h = tl.arange(0, BLK_H)
    acc_kv = tl.zeros((M, BLK_D), dtype=tl.float32)
    acc_sc = tl.zeros((M, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = real[:, None] & hmask[None, :]
        xv = tl.load(X + xrow[:, None] * H + hh[None, :], mask=xm, other=0.0)
        woff = d[None, :] * H + hh[:, None]
        wm = hmask[:, None] & dmask[None, :]
        acc_kv += tl.dot(xv, tl.load(WKV + woff, mask=wm, other=0.0))
        acc_sc += tl.dot(xv, tl.load(WGATE + woff, mask=wm, other=0.0))

    if HAS_STATE:
        blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
        blk_id = tl.load(BLKT + blk_idx, mask=head, other=0)
        row_in = p_abs % BLK_SIZE
        soff = blk_id[:, None] * STRIDE0 + row_in[:, None] * (2 * D) + d[None, :]
        sm = head[:, None] & dmask[None, :]
        acc_kv = tl.where(head[:, None], tl.load(STATE + soff, mask=sm, other=0.0), acc_kv)
        acc_sc = tl.where(head[:, None], tl.load(STATE + soff + D, mask=sm, other=0.0), acc_sc)

    j = rows % R
    apev = tl.load(APE + j[:, None] * D + d[None, :],
                   mask=(in_chunk & ok)[:, None] & dmask[None, :], other=0.0)
    acc_sc = tl.where((real & in_chunk)[:, None], acc_sc + apev, acc_sc)

    good = (in_chunk & ok)[:, None] & dmask[None, :]
    s_masked = tl.where(good, acc_sc, MIN_SCORE)
    mx = tl.max(s_masked, axis=0)
    ex = tl.exp(s_masked - mx[None, :])
    pj = ex / tl.sum(ex, axis=0)[None, :]
    o = tl.sum(pj * acc_kv, axis=0)

    o2 = tl.where(dmask, o, 0.0)
    inv = 1.0 / tl.sqrt(tl.sum(o2 * o2, axis=0) * (1.0 / D) + EPS)
    y = o2 * inv * tl.load(NORMW + d, mask=dmask, other=0.0)

    orow = obase + sc
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    # ---- INTERLEAVE rope：逐列标量 store（索引 = 标量+常量，绝对单位步长）----
    ROFF: tl.constexpr = D - RD
    # 不用 tl.reshape / tl.join / 2-D reduce —— 把「索引正确性」与「lowering 正确性」解耦。
    # 取列用 1-D `tl.sum(tl.where(d == k, y, 0.0))`（与 HALF 同款、已验证可用）。
    # out[i] = cos[i]*y[i] + sin[i]*(i 偶 ? -y[i+1] : +y[i-1])；偶/奇分两循环，无越界。
    for i in range(0, RD, 2):                       # 偶列，partner = i+1 ≤ RD-1
        vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
        vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
        ci = tl.load(RCOS + orow * RD + i)
        si = tl.load(RSIN + orow * RD + i)
        tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty))
    for i in range(1, RD, 2):                       # 奇列，partner = i-1 ≥ 0
        vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
        vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
        ci = tl.load(RCOS + orow * RD + i)
        si = tl.load(RSIN + orow * RD + i)
        tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty))
