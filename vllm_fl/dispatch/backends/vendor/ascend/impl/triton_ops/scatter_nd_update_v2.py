"""scatter_nd_update_v2 —— AscendC → Triton

来源：csrc/ascend/moe/scatter_nd_update_v2/
接口（op_host/scatter_nd_update_v2_def.cpp L27-72 + infershape L30-45 + tiling L302-363）：
    Input : var      [d0, ..., d_{K-1}, *tail]   （原地更新，输出与 var 同地址）
            indices  [*lead, K]   int32/int64
            updates  [*lead, *tail]
    Output: var      [d0, ..., d_{K-1}, *tail]   dtype 同输入 var
    Attr  : strides (REQUIRED, ListInt, 长度 = K)

tiling 推导（scatter_nd_update_v2_tiling.cpp L306-352）：
    K          = indices.shape[-1]
    indexDim_  = K
    indicesMask_[k] = strides[k]                      # L126-129：strides 直接当线性化系数
    scatterLength_  = prod(var.shape[K:])             # L325-329（每个 index 元组更新的连续长度）
    indexRow        = prod(indices.shape[:-1])        # L330-333
    totalLength     = prod(var.shape[:K])             # L316-319

公式（op_kernel/scatter_nd_update_linear_index.h L191-216 Compute4LinearIndex）：
    linear_index[n] = sum_{k<K} indices[n, k] * strides[k]          # L208 Muls + L210 Add
（op_kernel/scatter_nd_update_no_sort.h L96-121 ProcessOneIndex）：
    var_flat[linear_index[n] + s] = updates_flat[n*S + s],  s ∈ [0, S)   # L119-121
（op_kernel/scatter_nd_update_v2.h L157-215 Compute/CopyOut，sorted 路径）：
    先按线性索引降序排序（scatter_nd_update_linear_index.h L218-253 ComputeForSort），
    再逆序遍历并 `if (curIdxValue == lastProcessedIdx_) continue;`（L205-208）跳过重复 ——
    即**同一 linear_index 只有一条 update 生效**。

语义假设（重要）：两条路径（sort / no_sort）在设计上等价于
    「同一 linear_index 上，**扁平序号 n 最大的那条 update 生效**（last write wins）」。
    no_sort 路径按 n 递增顺序写入（L96-104）天然是 last-wins；sort 路径保留重复组
    「排序数组中位置最靠后的那一条」（稳定排序下即原始 n 最大者）。
    注：AscendC 的 Sort 是不稳定排序、且排序只在 blockLength 分块内进行，跨块重复索引
    时原实现本身结果不确定；这里实现确定性的 last-wins。

实现：两段式（均为纯 Triton）
    1) linear index kernel：li[n] = Σ indices[n,k]*strides[k]，同时用 tl.atomic_max
       在 LAST[li] 上记录该地址的最大 n（即最后写入者）；
    2) scatter kernel：仅当 LAST[li[n]] == n 时才写入 var_flat[li[n] + s]。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["scatter_nd_update_v2"]

#: `LAST` 去重表缓存：(phys_range, device) -> int32 全 -1 缓冲。
#: 生产上这个算子在**每一层、每一步**都会调用（DeepSeek-V4 43 层 × 2 处），
#: 每次 `torch.full((numel(var),), ...)` 都会新申请一块（可达数十 MB）再丢弃，
#: 在 60GB 显存已被权重占满的情况下很容易触发设备侧分配失败
#: （表现为 `wait for compute device to finish failed, runtime result = 507035`）。
#: 缓存后只做一次 `fill_(-1)`，把重复分配消除掉。
_LAST_CACHE: dict = {}


def _alloc_last(phys_range: int, dev) -> torch.Tensor:
    key = (int(phys_range), str(dev))
    buf = _LAST_CACHE.get(key)
    if buf is None:
        buf = torch.full((phys_range,), -1, dtype=torch.int32, device=dev)
        _LAST_CACHE[key] = buf
    else:
        buf.fill_(-1)
    return buf


@triton.jit
def _snd_linear_index_kernel(
    IDX, STRIDES, LI, LAST, N, K, LAST_N,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_N + tl.arange(0, BLK_N)
    rmask = rows < N

    kk = tl.arange(0, BLK_K)
    kmask = kk < K

    st = tl.load(STRIDES + kk, mask=kmask, other=0).to(tl.int32)
    idx = tl.load(
        IDX + rows[:, None] * K + kk[None, :],
        mask=rmask[:, None] & kmask[None, :], other=0,
    ).to(tl.int32)

    li = tl.sum(idx * st[None, :], axis=1).to(tl.int32)
    # ⚠️ 地址必须夹住：LAST 只有 LAST_N 个元素，越界 atomic 会踩坏显存
    # （生产上表现为 8 卡同时 `PTA call acl api failed` / runtime 507035）。
    ok = rmask & (li >= 0) & (li < LAST_N)
    li_c = tl.minimum(tl.maximum(li, 0), LAST_N - 1)
    # LI 保存**原始**线性索引（可能越界），由 scatter kernel 自己再判断一次
    tl.store(LI + rows, li, mask=rmask)
    # LAST[li] = max(LAST[li], n)  → 该地址最后一条 update 的扁平序号
    tl.atomic_max(LAST + li_c, rows, mask=ok)


@triton.jit
def _snd_scatter_kernel(
    LI, LAST, UPD, OUT, N, S, LAST_N, OUT_N,
    BLK_N: tl.constexpr,
    BLK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    rows = pid_n * BLK_N + tl.arange(0, BLK_N)
    rmask = rows < N

    li = tl.load(LI + rows, mask=rmask, other=-1).to(tl.int32)
    li_ok = rmask & (li >= 0) & (li < LAST_N)
    # 地址一律用夹住后的值：li 越界时（LI 里可能是未初始化的垃圾）
    # `OUT + li + s` 会算出野地址，即使该 lane 被 mask 掉，Ascend 侧仍可能
    # 在地址计算阶段触发 "vector core exception"（507035）。
    li_c = tl.minimum(tl.maximum(li, 0), LAST_N - 1)
    last = tl.load(LAST + li_c, mask=li_ok, other=-1)
    win = li_ok & (last == rows)

    s = pid_s * BLK_S + tl.arange(0, BLK_S)
    smask = s < S
    off = li_c[:, None] + s[None, :]
    m = win[:, None] & smask[None, :] & (off < OUT_N)

    u = tl.load(UPD + rows[:, None] * S + s[None, :], mask=m, other=0)
    tl.store(OUT + off, u, mask=m)


def scatter_nd_update_v2(
    var: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    strides,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """等价于 AscendC 的 ScatterNdUpdateV2（原地更新 var）。

    Args:
        var:     [d0..d_{K-1}, *tail]
        indices: [*lead, K]，int32 或 int64
        updates: [*lead, *tail]
        strides: 长度 K 的整数序列（属性 strides）
        out:     可选，预先填好 var 内容的输出缓冲（默认 clone(var)）

    Returns:
        更新后的张量（dtype/shape 同 var）
    """
    assert indices.dim() >= 1, "indices 至少 1 维"
    K = int(indices.shape[-1])
    assert K >= 1, f"indices 最后一维 (K) 非法: {K}"
    assert len(strides) == K, f"strides 长度 {len(strides)} != K={K}"
    assert var.dim() >= K, f"var 维度 {var.dim()} < K={K}"

    N = 1
    for d in indices.shape[:-1]:
        N *= int(d)
    S = 1
    for d in var.shape[K:]:
        S *= int(d)

    tail = tuple(int(d) for d in var.shape[K:])
    want = tuple(int(d) for d in indices.shape[:-1]) + tail
    got = tuple(int(d) for d in updates.shape)
    if got != want:
        # 生产调用点（vllm_fl 的 indexer_quant_scatter_part1）会把 update 压平成
        # ``indices.shape[:-1] + tail`` 之外的形式，例如 indices[1,1,1] + var[...,128]
        # 时传 updates[1,128]；AscendC 版按原始缓冲区线性寻址，天然容忍。
        # 只要元素个数一致就 reshape 成规范形状，否则才报错。
        if int(updates.numel()) == (math.prod(want) if want else 1):
            updates = updates.reshape(want)
        else:
            raise ValueError(
                f"updates shape {got}（numel={updates.numel()}）既不是规范形状 {want}，"
                f"元素个数也不匹配 → 无法安全解释"
            )

    st = [int(s) for s in strides]
    # 线性索引可达的最大偏移 + 单条更新长度 → LAST 去重表大小
    max_off = 0
    for k in range(K):
        max_off += (int(var.shape[k]) - 1) * st[k]
    phys_range = max_off + S
    assert phys_range < 2 ** 31 - 1, f"线性索引范围 {phys_range} 超出 int32"

    y = var.contiguous().clone() if out is None else out
    y = y.contiguous() if not y.is_contiguous() else y
    if N == 0:
        return y

    dev = var.device
    idx = indices.contiguous().view(N, K)
    upd = updates.contiguous().view(N, S)
    y_flat = y.view(-1)

    # LI 用 -1 而不是 empty：越界行不会被写入，未初始化的垃圾值会让地址计算
    # 落到野地址（见 _snd_scatter_kernel 的注释）。
    li = torch.full((N,), -1, dtype=torch.int32, device=dev)
    last = _alloc_last(phys_range, dev)
    st_t = torch.tensor(st, dtype=torch.int32, device=dev)

    BLK_N = 64
    BLK_K = max(2, next_pow2(K))
    _snd_linear_index_kernel[(ceil_div(N, BLK_N),)](
        idx, st_t, li, last, N, K, phys_range,
        BLK_N=BLK_N, BLK_K=BLK_K,
    )

    BLK_N2 = 32
    BLK_S = min(next_pow2(S), 256)
    _snd_scatter_kernel[(ceil_div(N, BLK_N2), ceil_div(S, BLK_S))](
        li, last, upd, y_flat, N, S, phys_range, int(y_flat.numel()),
        BLK_N=BLK_N2, BLK_S=BLK_S,
    )
    return y
