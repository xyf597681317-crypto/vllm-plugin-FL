"""indexer_compress_epilog_v2 —— AscendC → Triton

来源：_repos/vllm-ascend/csrc/attention/indexer_compress_epilog_v2/
接口（op_host/indexer_compress_epilog_v2_def.cpp L22-46）：
    Input : indexer_compress_cache  uint8  4D [numBlocks, cacheBs, 1, cacheD]（原地写回）
            x                       fp16/bf16  [rows, d]
            slot_mapping            int32      [rows]（-1 = 跳过）
    Attr  : layout(默认2), block_stride(默认0)
    Output: indexer_compress_cache（原地）

shape/几何规则（op_host/indexer_compress_epilog_v2_tiling.cpp L94-140）：
    cacheBs   = cache.shape[1]
    cache.shape[2] 必须为 1；cacheD = cache.shape[3]
    d         = x.shape[-1]
    scaleCol  = ceil(d / PER_BLOCK_FP16)          # 128 元素一个 scale
    blockStride 缺省(0) → cacheBs*cacheD；非 0 时必须 >= cacheBs*cacheD
    layout 只支持 2

公式出处（op_kernel/indexer_compress_epilog_v2_base.h L133-243 VFProcessDynamicBlockQuant
            与 op_kernel/indexer_compress_epilog_v2_single_row.h L49-98 Process）：
    单行（slot != -1）：
        # 每 128 元素子块独立求 scale（fold 循环把 2 个 VL_FP32=64 的块并成 128，写 1 个 scale）
        scale = max(|x|) * (1/448)      # INV_FP8_E4M3_MAX_VALUE；max==0 → scale=0
        y     = x / scale               # scale==0 时 Div 得 NaN → Select 保留原 x(=0)
        store y 为 fp8_e4m3（RoundMode::CAST_RINT，即就近取偶；NO_SAT）
    落盘寻址（single_row.h L83-93，单位 **字节**）：
        blkNumIdx = slot // cacheBs ; blkSizeIdx = slot % cacheBs
        valueOffset = blkNumIdx*blockStride + blkSizeIdx*d
        scaleOffset = blkNumIdx*blockStride + cacheBs*d + blkSizeIdx*scaleCol*4
      ⇒ 每个 block 排布 = [cacheBs*d 个 fp8 字节][cacheBs*scaleCol 个 fp32 scale]

实现说明：
    - cache 当 **uint8 平铺缓冲** 处理；fp8 值按字节写、scale 按 fp32 的 4 字节小端写
    - 一个 program 处理一行（1-D，避免标量经 [:, None] 广播成 2-D 基址）
    - 未用 tl.reshape 造偏移（避开后端"广播偏移 reshape 静默错位"坑）

**语义假设（未在 AscendC 侧验证）**：`blockStride` 的缺省值 cacheBs*cacheD 会让 scale 区
与下一个 block 的 value 区**重叠**（tiling 只校验 >= cacheBs*cacheD）。本实现严格照 kernel
公式算字节偏移；调用方若要 value/scale 不重叠，应显式传
block_stride = cacheBs*cacheD + cacheBs*scaleCol*4。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div

__all__ = ["indexer_compress_epilog_v2"]

_PER_BLOCK_FP16 = 128          # tiling: PER_BLOCK_FP16
_INV_FP8_E4M3_MAX = 1.0 / 448.0


@triton.jit
def _to_e4m3_bits(y):
    """fp32 → e4m3fn 的 8 位编码（RNE/就近取偶，与 kernel 的 CAST_RINT 一致）。

    e4m3fn：sign1/exp4/man3、bias=7、**无 inf**、E=15&m=7 为 NaN、最大有限值 448(=0x7E)。
    该后端不支持 fp8 cast（'unrecognized float type f8E4M3FN'），因此手写编码：
      - |y| >= 2^-6：正规化到 [8,16)，m = round(x*2^(3-e)) - 8，m==8 → 进位
      - |y| <  2^-6：次正规，码 = round(x / 2^-9)
      - 0 → 0；超过 448 → 饱和到 448（scale=max/448 保证不会发生，仅作保护）
    """
    b = y.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    x = (b & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    x = tl.minimum(x, 448.0)
    ex = ((b >> 23) & 0xFF) - 127

    # 正规路径：q = x * 2^(3-ex) ∈ [8,16)
    sc_bits = ((3 - ex) + 127) << 23
    q = x * sc_bits.to(tl.float32, bitcast=True)
    fl = tl.floor(q)
    diff = q - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0     # fl 为奇数
    r = tl.where(diff > 0.5, fl + 1.0,
                 tl.where(diff < 0.5, fl, fl + tl.where(odd, 1.0, 0.0)))
    m = r - 8.0
    carry = m >= 8.0
    e_final = tl.where(carry, ex + 8, ex + 7)
    m_final = tl.where(carry, 0.0, m)
    code_norm = (e_final << 3) + m_final.to(tl.int32)

    # 次正规路径：qs = x / 2^-9 = x*512
    qs = x * 512.0
    fls = tl.floor(qs)
    difs = qs - fls
    odds = (fls - tl.floor(fls * 0.5) * 2.0) != 0.0
    code_sub = tl.where(difs > 0.5, fls + 1.0,
                        tl.where(difs < 0.5, fls, fls + tl.where(odds, 1.0, 0.0))).to(tl.int32)

    code = tl.where(x < 0.015625, code_sub, code_norm)      # 2^-6 分界
    code = tl.where(x == 0.0, 0, code)
    return (sign | code) & 0xFF


@triton.jit
def _ice_kernel(
    X, SLOT, CACHE,
    ROWS, D, CACHE_BS, SCALE_COL, STRIDE,
    BLK_Q: tl.constexpr,
):
    r = tl.program_id(0)          # 一个 program 一行
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            blk_num = slot // CACHE_BS
            blk_off = slot % CACHE_BS
            v_base = blk_num * STRIDE + blk_off * D
            s_base = blk_num * STRIDE + CACHE_BS * D + blk_off * SCALE_COL * 4
            for s in range(0, SCALE_COL):
                cols = s * 128 + tl.arange(0, BLK_Q)
                m = cols < D
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                scale = tl.where(mx > 0.0, mx * (1.0 / 448.0), 0.0)
                # scale==0（全 0 子块）→ 输出 0（对应 AscendC 里 Select 保留原值 0）
                y = tl.where(scale > 0.0, v / scale, 0.0)
                tl.store(CACHE + v_base + cols, _to_e4m3_bits(y).to(tl.uint8), mask=m)
                # scale 按小端 4 字节**一次性向量写**；第 s 个子块的 scale 落在 s_base + s*4
                si = scale.to(tl.int32, bitcast=True)
                sh = tl.arange(0, 4) * 8
                b4 = ((si >> sh) & 0xFF).to(tl.uint8)
                tl.store(CACHE + s_base + s * 4 + tl.arange(0, 4), b4)


def indexer_compress_epilog_v2(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    indexer_compress_cache: torch.Tensor,
    layout: int = 2,
    block_stride: int = 0,
) -> torch.Tensor:
    """等价于 AscendC 的 IndexerCompressEpilogV2（原地写回 cache 并返回它）。

    Args:
        x: [rows, d] fp16/bf16（rows = slot_mapping 的元素数）
        slot_mapping: [rows] int32；-1 跳过
        indexer_compress_cache: uint8 4D [numBlocks, cacheBs, 1, cacheD]（可 view 成足够大的平铺缓冲）
        layout: 只支持 2
        block_stride: 0 → cacheBs*cacheD；非 0 必须 >= cacheBs*cacheD（单位：字节）

    Returns:
        indexer_compress_cache（已更新）
    """
    if int(layout) != 2:
        raise NotImplementedError(f"layout={layout} 未实现（tiling 只接受 2）")
    if x.dim() != 2:
        raise ValueError(f"x 必须是 2D [rows, d]，实际 {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x 只支持 fp16/bf16（def.cpp L30），实际 {x.dtype}")
    if slot_mapping.dtype != torch.int32:
        raise ValueError(f"slot_mapping 必须是 int32，实际 {slot_mapping.dtype}")
    if indexer_compress_cache.dtype != torch.uint8:
        raise ValueError(f"indexer_compress_cache 必须是 uint8，实际 {indexer_compress_cache.dtype}")
    if indexer_compress_cache.dim() != 4:
        raise ValueError("indexer_compress_cache 必须是 4D [numBlocks, cacheBs, 1, cacheD]")
    if indexer_compress_cache.shape[2] != 1:
        raise ValueError("indexer_compress_cache 的第 3 维必须是 1")
    rows, d = x.shape
    if slot_mapping.numel() != rows:
        raise ValueError(f"slot_mapping 元素数 {slot_mapping.numel()} != rows={rows}")
    cache_bs = int(indexer_compress_cache.shape[1])
    cache_d = int(indexer_compress_cache.shape[3])
    if cache_d != d:
        raise ValueError(f"cacheD={cache_d} 必须等于 x 的最后一维 d={d}")
    scale_col = ceil_div(d, _PER_BLOCK_FP16)
    # 调用点（csrc/torch_binding.cpp L1902-1904）传的是 indexer_compress_cache.stride(0) 作为
    # block_stride；attr 缺省 0 才回落到 cacheBs*cacheD（tiling L181-183）
    stride0 = int(indexer_compress_cache.stride(0))
    stride = int(block_stride) if int(block_stride) != 0 else max(stride0, cache_bs * cache_d)
    if stride < cache_bs * cache_d:
        raise ValueError(f"block_stride={stride} 必须 >= cacheBs*cacheD={cache_bs * cache_d}")

    num_blocks = int(indexer_compress_cache.shape[0])
    need = (num_blocks - 1) * stride + cache_bs * d + cache_bs * scale_col * 4
    # cache 常是更大缓冲的 view（indexer_full_cache.view(torch.uint8)），按底层 storage 算容量
    capacity = (indexer_compress_cache.untyped_storage().nbytes()
                - indexer_compress_cache.storage_offset())
    if capacity < need:
        raise ValueError(
            f"cache 底层缓冲太小：需要 {need} 字节（blockStride={stride}, cacheBs={cache_bs}, "
            f"cacheD={cache_d}, scaleCol={scale_col}），实际底层 {capacity} 字节")

    blk_q = 128
    _ice_kernel[(rows,)](
        x.contiguous(), slot_mapping.contiguous(), indexer_compress_cache,
        rows, d, cache_bs, scale_col, stride,
        BLK_Q=blk_q,
    )
    return indexer_compress_cache
