"""kv_compress_epilog —— AscendC → Triton（quant_mode=1 GROUP_FP8 + layout=1）

来源：_repos/vllm-ascend/csrc/attention/kv_compress_epilog/
接口（op_host/kv_compress_epilog_def.cpp L24-50）：
    Input : kv_compress_cache  fp8_e5m2 / fp8_e4m3fn（原地写回）
            x                  bf16  [rows, d]
            slot_mapping       int32 或 int64（def 里有 4 种 dtype 组合）
    Attr  : quant_group_size(128), quant_mode(1), round_scale(1), layout(1), block_stride(0)
    Output: kv_compress_cache

几何与公式出处：
  op_kernel/kv_compress_epilog.h
    L104-124  **slot_mapping==-1 的行被跳过并"压实"**：有效行紧凑写进 xLocal(validIdx 机制)，
              落盘时各自用自己的 slot —— 不是"按行号对齐"
    L130-156  quantMode 分派：1 → VFProcessDynamicBlockQuant；2 → MXFP8 变体
    L184-196  layout=1 落盘：rowStride = block_stride>0 ? block_stride : kvCacheCol
                              gmOffset  = slot * rowStride
  op_kernel/kv_compress_epilog_common.h L127-266 VFProcessDynamicBlockQuant：
    quantColNum = d - 64                      # ⚠️ x 的一行 = [量化的 d-64 列 | 原样透传的 64 列 rope]
    scaleColNum = ceil(quantColNum / 128)     # 每 128 元素一个 fp32 scale
    scale = max(|x_sub|) * coeff              # coeff = (float)(1/448)，**乘法不是除法**
    y     = x / scale  → fp8（CAST_RINT）；max==0 → 0
    一行输出（kvCacheCol 字节，见 L189/211/249-260 的指针推进）：
        [0, quantColNum)                    fp8 量化值
        [quantColNum, +128)                 x 的最后 64 列（BF16 原样透传 = 128 字节）
        [+128, +scaleColNum*4)              各子块 fp32 scale（小端）
        [concatCol, kvCacheCol)             pad 0
  op_host/kv_compress_epilog_tiling_arch35.cpp L211-263：
    concatCol = (d-64) + 128 + scaleCol*4 ; kvCacheCol = RoundUp(concatCol, 128) ; padCol = kvCacheCol-concatCol

**未实现（显式 NotImplementedError）**：
    - quant_mode != 1（=2 是 MXFP8，E8M0 幂次 scale，common.h L269-724 另两套）
    - layout != 1（layout=2 的 value/scale 分区见 tiling L220-248）
    - fp8_e5m2（只实现 e4m3fn；e5m2 需另一套编码，coeff=1/57344）
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["kv_compress_epilog"]

_ROPE = 64                      # SLICE_SIZE：原样透传的 rope 列数
_PER_BLOCK = 128                # 每 128 元素一个 scale
_FP8_MAX = 448.0                # FP8_E4M3FN_MAX_VALUE


@triton.jit
def _to_e4m3_bits(y):
    """fp32 → e4m3fn 8 位编码（RNE/就近取偶，与 kernel 的 CAST_RINT 一致）。

    与 indexer_compress_epilog_v2 中同一实现（已逐字节对拍 torch float8_e4m3fn 验证）。

    ⚠️ **平局（round-half-to-even）必须"朝偶数靠"**（2026-10-12 修）：
    `fl = floor(q)` 本身是整数，两个邻居必为"奇/偶"一对，所以平局时
    取偶数邻居即可：`fl` 为**奇数** → 进位到 `fl+1`；`fl` 为**偶数** → 保持 `fl`。
    这是 IEEE-754 RNE 的定义，也是 AscendC `CAST_RINT` 的语义。
    旧写法 `r = fl + tl.where(odd, 1.0, 0.0)` 字面上是同一件事，但**实测差 1**
    （见下方证据），故改成显式三分支写法，避免依赖那条表达式的后端行为。

    后果与证据：只有 `|x|/scale` 恰好落在两个 fp8 编码正中间（`diff == 0.5`）
    时才差 1。`kv_compress_epilog` 的 `d=576 group=64 rscale=0 rows=9 含-1` 用例
    恰好 5/15360 字节 `got = ref - 1`，offset 1690/1725/13225/13517/13531，
    对应 q = 11.5 / 9.5 / 10.5 / … —— **全部是精确平局**（`diff - 0.5 == 0`）。
    `rscale=1` 时 scale 取整成 2 的幂，商几乎落不到平局点，所以只有 rscale=0 暴露。
    """
    b = y.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    x = (b & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    x = tl.minimum(x, 448.0)
    ex = ((b >> 23) & 0xFF) - 127
    sc_bits = ((3 - ex) + 127) << 23
    q = x * sc_bits.to(tl.float32, bitcast=True)
    fl = tl.floor(q)
    diff = q - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0
    r = tl.where(diff > 0.5, fl + 1.0,
                 tl.where(diff < 0.5, fl, fl + tl.where(odd, 1.0, 0.0)))
    m = r - 8.0
    carry = m >= 8.0
    e_final = tl.where(carry, ex + 8, ex + 7)
    m_final = tl.where(carry, 0.0, m)
    code_norm = (e_final << 3) + m_final.to(tl.int32)
    qs = x * 512.0
    fls = tl.floor(qs)
    difs = qs - fls
    odds = (fls - tl.floor(fls * 0.5) * 2.0) != 0.0
    code_sub = tl.where(difs > 0.5, fls + 1.0,
                        tl.where(difs < 0.5, fls, fls + tl.where(odds, 1.0, 0.0))).to(tl.int32)
    code = tl.where(x < 0.015625, code_sub, code_norm)
    code = tl.where(x == 0.0, 0, code)
    return (sign | code) & 0xFF


@triton.jit
def _kce_kernel(
    X, SLOT, CACHE,
    ROWS, D, QUANT_COL, SCALE_COL, CONCAT_COL, KV_COL, ROW_STRIDE,
    BLK_Q: tl.constexpr,
    BLK_PAD: tl.constexpr,
):
    r = tl.program_id(0)                 # 一个 program 一行
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            base = slot * ROW_STRIDE
            # ---- 量化段：每 128 元素一个 scale ----
            for s in range(0, SCALE_COL):
                cols = s * 128 + tl.arange(0, BLK_Q)
                m = cols < QUANT_COL
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                scale = tl.where(mx > 0.0, mx * (1.0 / 448.0), 0.0)   # Mul，不是 Div
                y = tl.where(mx > 0.0, v / scale, 0.0)
                tl.store(CACHE + base + cols, _to_e4m3_bits(y).to(tl.uint8), mask=m)
                si = scale.to(tl.int32, bitcast=True)
                tl.store(CACHE + base + QUANT_COL + 128 + s * 4 + tl.arange(0, 4),
                         ((si >> (tl.arange(0, 4) * 8)) & 0xFF).to(tl.uint8))
            # ---- rope 透传段：最后 64 列 BF16 原样（64*2=128 字节，小端）----
            off = tl.arange(0, 64)
            rv = tl.load(X + r * D + QUANT_COL + off)                 # bf16
            u = rv.to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(CACHE + base + QUANT_COL + off * 2, (u & 0xFF).to(tl.uint8))
            tl.store(CACHE + base + QUANT_COL + off * 2 + 1, ((u >> 8) & 0xFF).to(tl.uint8))
            # ---- pad 0 ----
            pc = tl.arange(0, BLK_PAD)
            tl.store(CACHE + base + CONCAT_COL + pc,
                     tl.zeros([BLK_PAD], dtype=tl.uint8), mask=CONCAT_COL + pc < KV_COL)


@triton.jit
def _kce_mxfp8_kernel(
    X, SLOT, CACHE,
    ROWS, D, QUANT_COL, SCALE_COL, CONCAT_COL, KV_COL, ROW_STRIDE,
    ROUND_SCALE: tl.constexpr,
    BLK_G: tl.constexpr,
    BLK_PAD: tl.constexpr,
):
    """quant_mode=2 (MXFP8)：**行内布局与 group_fp8 相反** —— rope 在前 128 字节。

    出处 kv_compress_epilog_common.h L269-433（VFProcessDynamicMxFP8Quant）：
        L281 quantColNum = curColNum - 64 ; L282 scaleColNum = CeilDiv(quantColNum, perGroupSize)
        L338 rope 写到 `ropeYLocalAddr + i*dstCol/2`（**字节偏移 0**）
        L380/386 量化值写到 `j*VL_FP32 + i*dstCol + **ropeNum(128)**`
        L391 scale 写到 `quantColNum + ropeNum + scaleIdx`（**每 group 1 字节**）
        L361-362 sf = max(groupMax, 1e-4) * coeff
        L363-372 roundScale=True：e=(bits>>23)&0xFF；man=bits&0x7FFFFF；
                   eb = (e - 127) + (man!=0) + 127  ⇒ **scale = 2^ceil(log2(sf))**（进位）
        L376-379 y = clamp(x/scale, fp8Min, fp8Max)
        L389-391 scale 字节 = (scale_bits >> 23) & 0xFF（E8M0）
    """
    r = tl.program_id(0)
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            base = slot * ROW_STRIDE
            # ---- rope 透传段：**前 128 字节**（64 个 bf16，小端）----
            off = tl.arange(0, 64)
            rv = tl.load(X + r * D + QUANT_COL + off)
            u = rv.to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(CACHE + base + off * 2, (u & 0xFF).to(tl.uint8))
            tl.store(CACHE + base + off * 2 + 1, ((u >> 8) & 0xFF).to(tl.uint8))
            # ---- 量化段：从字节 128 开始，每 BLK_G 个元素共享一个 E8M0 scale ----
            for s in range(0, SCALE_COL):
                cols = s * BLK_G + tl.arange(0, BLK_G)
                m = cols < QUANT_COL
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                sf = tl.maximum(mx, 1.0e-4) * (1.0 / 448.0)      # Mul，不是 Div
                bits = sf.to(tl.int32, bitcast=True)
                e0 = (bits >> 23) & 0xFF
                man = bits & 0x7FFFFF
                if ROUND_SCALE:
                    eb = e0 + tl.where(man != 0, 1, 0)           # ceil(log2(sf))
                    scale = (eb << 23).to(tl.float32, bitcast=True)
                else:
                    eb = e0
                    scale = sf
                y = tl.minimum(tl.maximum(v / scale, -448.0), 448.0)
                tl.store(CACHE + base + 128 + cols, _to_e4m3_bits(y).to(tl.uint8), mask=m)
                tl.store(CACHE + base + 128 + QUANT_COL + s, (eb & 0xFF).to(tl.uint8))
            # ---- pad 0 ----
            pc = tl.arange(0, BLK_PAD)
            tl.store(CACHE + base + CONCAT_COL + pc,
                     tl.zeros([BLK_PAD], dtype=tl.uint8), mask=CONCAT_COL + pc < KV_COL)


def kv_compress_epilog(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_compress_cache: torch.Tensor,
    quant_mode: int = 1,
    layout: int = 1,
    quant_group_size: int = 128,
    round_scale: int = 1,
    block_stride: int = 0,
) -> torch.Tensor:
    """等价于 AscendC 的 KvCompressEpilog（原地写回）。

    **生产配置**（`dsa_attn_kv_plan.py` L195-203 / vllm-fl `device_op.py` L1542-1550）：
        `quant_mode=2, quant_group_size=64, round_scale_flag=True, layout=1`
    ⇒ **quant_mode=2 才是生产路径**；quant_mode=1 为兼容保留。

    Args:
        x: bf16 [rows, d]；一行 = [被量化的 d-64 列 | 原样透传的 64 列 rope]
        slot_mapping: [rows] int32/int64；-1 跳过
        kv_compress_cache: fp8_e4m3fn（或 uint8 视图）平铺缓冲
        quant_mode: 1 = GROUP_FP8（fp32 scale，每 128 元素一个）；
                    2 = MXFP8（**E8M0 单字节 scale**，每组 `quant_group_size` 个）
        layout: 1（生产）或 2
        quant_group_size: 1 固定 128（内核硬编码）；**2 支持 64 / 128**（生产用 64）
        round_scale: 1 = E8M0 取 `2^ceil(log2(sf))`（进位）；0 = 直接用 sf（仅 mode 2 有意义）
        block_stride: 0 → kvCacheCol；非 0 必须 >= kvCacheCol

    **一行的字节布局（两种模式不同，务必注意）**：
        mode 1: `[fp8 quantCol][rope 128][fp32 scale ×4B][pad]`
        mode 2: `[rope 128][fp8 quantCol][E8M0 scale ×1B][pad]`   ← rope 在前
    """
    if int(quant_mode) not in (1, 2):
        raise NotImplementedError(f"quant_mode={quant_mode} 未实现（只支持 1=GROUP_FP8 / 2=MXFP8）")
    if int(layout) not in (1, 2):
        raise NotImplementedError(
            f"layout={layout} 未实现（只支持 1 或 2）")
    group = int(quant_group_size)
    if int(quant_mode) == 1:
        if group != _PER_BLOCK:
            raise NotImplementedError(
                f"quant_mode=1 的 quant_group_size 固定 128（内核 L146 硬编码），实际 {group}")
    elif group not in (64, 128):
        raise NotImplementedError(
            f"quant_mode=2 的 quant_group_size 只支持 64/128（生产用 64），实际 {group}")
    if x.dim() != 2:
        raise ValueError(f"x 必须是 2D [rows, d]，实际 {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x 必须是 bf16（def.cpp L32），实际 {x.dtype}")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"slot_mapping 必须是 int32 或 int64，实际 {slot_mapping.dtype}")
    if kv_compress_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise NotImplementedError(
            f"kv_compress_cache dtype={kv_compress_cache.dtype} 未实现"
            f"（只实现 fp8_e4m3fn；e5m2 需另一套编码）")

    rows, d = x.shape
    if slot_mapping.numel() != rows:
        raise ValueError(f"slot_mapping 元素数 {slot_mapping.numel()} != rows={rows}")
    if d <= _ROPE:
        raise ValueError(f"d={d} 必须 > 64（一行 = 量化段 d-64 + rope 段 64）")
    quant_col = d - _ROPE
    # 几何随模式不同（common.h L145-146 vs L281-282；scale 宽度 4B(fp32) vs 1B(E8M0)）
    if int(quant_mode) == 1:
        scale_col = ceil_div(quant_col, _PER_BLOCK)
        concat_col = quant_col + 128 + scale_col * 4
    else:
        scale_col = ceil_div(quant_col, group)
        concat_col = quant_col + 128 + scale_col
    kv_col = ceil_div(concat_col, 128) * 128
    # 落盘行距（kv_compress_epilog.h L184-196）：
    #   layout==1 → blockStride>0 ? blockStride : kvCacheCol
    #   layout!=1（quant_mode=1 时）→ 直接用 kvCacheCol（不走 blockStride 分支）
    if int(layout) == 1:
        row_stride = int(block_stride) if int(block_stride) != 0 else kv_col
    else:
        row_stride = kv_col
    if row_stride < kv_col:
        raise ValueError(f"block_stride={row_stride} 必须 >= kvCacheCol={kv_col}")

    slot = slot_mapping.contiguous()
    if slot.dtype == torch.int64:
        smax = int(slot.detach().abs().max().item()) if slot.numel() else -1
        if smax >= (1 << 31):
            raise ValueError("slot_mapping(int64) 超 int32 范围，未实现")
        slot = slot.to(torch.int32)

    cache_u8 = (kv_compress_cache if kv_compress_cache.dtype == torch.uint8
                else kv_compress_cache.view(torch.uint8))
    cap = cache_u8.numel()
    smax = int(slot.detach().max().item()) if slot.numel() else -1
    if smax >= 0:
        need = smax * row_stride + kv_col
        if cap < need:
            raise ValueError(f"cache 太小：需要 {need} 字节（slot_max={smax}, rowStride={row_stride}），"
                             f"实际 {cap}")

    if int(quant_mode) == 1:
        _kce_kernel[(rows,)](
            x.contiguous(), slot, cache_u8,
            rows, d, quant_col, scale_col, concat_col, kv_col, row_stride,
            BLK_Q=128, BLK_PAD=128,
        )
    else:
        _kce_mxfp8_kernel[(rows,)](
            x.contiguous(), slot, cache_u8,
            rows, d, quant_col, scale_col, concat_col, kv_col, row_stride,
            ROUND_SCALE=bool(int(round_scale)),
            BLK_G=group, BLK_PAD=128,
        )
    return kv_compress_cache
