"""store_kv_block —— AscendC → Triton

来源：_repos/vllm-ascend/csrc/attention/store_kv_block/
接口（op_host/store_kv_block_def.cpp L27-57，**位置参数**）：
    Input : keyIn            int8/fp16/bf16  [numTokens, numHeads, headSize...]（2..7 维）
            keyCacheIn       int8/fp16/bf16  [numCache, ...]（numCache = dim0*dim1）
            groupLen         int32  [numGroups]
            groupKeyIdx      int32  [numGroups]
            groupKeyCacheIdx int32  [numGroups]
    Attr  : blockSize
    Output: keyCacheIn（原地写回）

真实调用点（vllm_ascend/attention/indexer.py L261-268）：
    torch.ops._C_ascend.store_kv_block(
        k_li,                    # keyIn
        indexer_k_cache,         # keyCacheIn
        group_len,               # groupLen
        group_key_idx,           # groupKeyIdx
        group_key_cache_idx,     # groupKeyCacheIdx
        block_size,              # blockSize
    )

公式出处（op_kernel/store_kv_block.h）：
    L71-74   tokenSize = 每行元素数（numHeads * headSize...），tokenByteSize = tokenSize*typeByte
    L112-126 Process 主循环：
        for i in corePerNum:
            idx = coreId + i*blockNum
            if groupLen[idx] <= 0 || groupKeyIdx[idx] < 0 || groupKeyCacheIdx[idx] < 0: continue
            blockLen = groupLen[idx] * tokenByteSize
            DataCopyPad(tokenLocal, keyInputGt[groupKeyIdx[idx]*tokenSize], blockLen)      # 拷出
            DataCopyPad(keyCacheInputGt[groupKeyCacheIdx[idx]*tokenSize], tokenLocal, blockLen)  # 拷入
    ⇒ 语义：逐 group 的**连续行块拷贝**
        keyCache[dst : dst+len] = keyIn[src : src+len]
        其中 len = groupLen[g]、src = groupKeyIdx[g]、dst = groupKeyCacheIdx[g]
        len<=0 或 src<0 或 dst<0 → 整组跳过（正是 indexer.py 用 "分组连续 run" 代替
        逐 token scatter 的 C8 reshape 优化：run 内 src/dst 都连续，因此一次块拷贝即可）

实现说明：
    把块拷贝摊平成**一维**（src*TOKEN + [0, len*TOKEN) → dst*TOKEN + 同区间），
    一个 program 处理一个 group，避免标量经 [:, None] 广播成 2-D 指针基址（Ascend 后端的已知坑）。
    纯拷贝算子：输出与输入**逐位相等**（int8/fp16/bf16 都按位相同）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div

__all__ = ["store_kv_block"]


@triton.jit
def _store_kv_block_kernel(
    KEYIN, KEYCACHE, GRLEN, GRKEY, GRCACHE,
    TOKEN,                       # 每行元素数
    BLK: tl.constexpr,
):
    g = tl.program_id(0)         # 一个 program 处理一个 group
    ln = tl.load(GRLEN + g)
    src = tl.load(GRKEY + g)
    dst = tl.load(GRCACHE + g)
    # 无效组（AscendC 里是 continue）：把长度/基址都归零，靠掩码整体跳过
    ok = (ln > 0) & (src >= 0) & (dst >= 0)
    ln = tl.where(ok, ln, 0)
    src = tl.where(ok, src, 0)
    dst = tl.where(ok, dst, 0)
    total = ln * TOKEN
    s_base = src * TOKEN         # 标量基址（只与一维 arange 相加，不做 [:, None] 广播）
    d_base = dst * TOKEN
    for c in range(0, tl.cdiv(total, BLK)):
        i = c * BLK + tl.arange(0, BLK)
        m = i < total
        v = tl.load(KEYIN + s_base + i, mask=m, other=0)
        tl.store(KEYCACHE + d_base + i, v, mask=m)


def store_kv_block(
    key_in: torch.Tensor,
    key_cache_in: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """等价于 AscendC 的 StoreKVBlock（原地写回 key_cache_in 并返回它）。

    Args:
        key_in:   [numTokens, numHeads, headSize...]（元素连续，行 = token）
        key_cache_in: [numCache, ...] 与 key_in 同行宽；原地写入
        group_len: [numGroups] int32，每个 group 要拷的行数
        group_key_idx: [numGroups] int32，源起始行
        group_key_cache_idx: [numGroups] int32，目标起始行
        block_size: 属性（tiling 里用于 UB 容量校验）

    Returns:
        key_cache_in（已更新）
    """
    for nm, t in (("group_len", group_len), ("group_key_idx", group_key_idx),
                  ("group_key_cache_idx", group_key_cache_idx)):
        if t.dtype != torch.int32:
            raise ValueError(f"{nm} 必须是 int32，实际 {t.dtype}")
    if not (group_len.numel() == group_key_idx.numel() == group_key_cache_idx.numel()):
        raise ValueError("group_len / group_key_idx / group_key_cache_idx 长度必须一致")
    if key_in.dtype not in (torch.int8, torch.float16, torch.bfloat16):
        raise ValueError(f"keyIn 只支持 int8/fp16/bf16，实际 {key_in.dtype}")
    if key_cache_in.dtype != key_in.dtype:
        raise ValueError(f"keyCacheIn dtype {key_cache_in.dtype} 与 keyIn {key_in.dtype} 不一致")
    # 行宽（一个 token 的元素数）由 **keyIn** 决定：keyIn 按 dim0 分行，每行 tokenSize 个元素
    # （AscendC kernel 里 tokenSize = product(shape[1:])，cache 只是"按该行宽铺开的一维缓冲"，
    #   kernel 用 keyCacheInputGt[groupKeyCacheIdx*tokenSize] 定位，并不看 cache 自己的形状）
    if key_in.dim() < 2 or key_cache_in.dim() < 2:
        raise ValueError("keyIn / keyCacheIn 至少 2 维")
    token = key_in.numel() // int(key_in.shape[0])
    if token * int(key_in.shape[0]) != key_in.numel():
        raise ValueError("keyIn 元素数不是 dim0 的整数倍")
    if key_cache_in.numel() % token != 0:
        raise ValueError(f"keyCacheIn 元素数 {key_cache_in.numel()} 不是行宽 {token} 的整数倍")
    ng = group_len.numel()
    if ng == 0:
        return key_cache_in
    blk = 1024
    grid = (ng,)
    _store_kv_block_kernel[grid](
        key_in.reshape(-1), key_cache_in.reshape(-1),
        group_len.contiguous(), group_key_idx.contiguous(), group_key_cache_idx.contiguous(),
        token, BLK=blk,
    )
    return key_cache_in


def _token_size(t: torch.Tensor) -> int:
    """一行（一个 token）的元素数 = shape[1:] 的乘积。"""
    n = 1
    for s in t.shape[1:]:
        n *= int(s)
    return n
