"""moe_init_routing_custom —— AscendC → Triton

来源：csrc/ascend/moe/moe_init_routing_custom/

接口（op_host/moe_init_routing_custom_def.cpp L22-97）：
    Input : x          [N, cols]              int8/fp16/bf16/fp32   (REQUIRED)
            expert_idx [N, k]                 int32                 (REQUIRED)
            scale      [expert_num, cols] 或 [1] fp32               (OPTIONAL)
            offset     [1] fp32                                     (OPTIONAL)
    Output: expanded_x                     dropless: [xOutNum, cols]
            expanded_row_idx               [N*k]            int32
            expert_tokens_count_or_cumsum  见下             int64
            expanded_scale                 [xOutNum]        fp32
    Attr  : active_num(-1), expert_capacity(-1), expert_num(-1), drop_pad_mode(0),
            expert_tokens_num_type(0), expert_tokens_num_flag(false), quant_mode(-1),
            active_expert_range([], 空表示 [0, expert_num)), row_idx_type(0)

--------------------------------------------------------------------------
公式 / 语义出处
--------------------------------------------------------------------------
[A] 主流程（tiling_key = MOE_INIT_ROUTING_CUSTOM_PERFORMANCE）：
    op_kernel/moe_custom_full_load.h
    L79-89   CopyIn       —— 读 expert_idx（展平 N*k）
    L91-129  SortCompute  —— **按专家号升序稳定排序**
                             key = -expert_id（取负后交给降序 Sort ⇒ 专家号升序）
                             payload = 展平后的源下标 0..N*k-1
                             L121 Extract 拆出 (排序后的专家号, 排序后的源下标)
    L131-163 ExpertCountCompute —— KEY_VALUE 形式的 (expert_id, count) 对
    L165-227 CopyOutDynamicQuant —— 动态量化 + 写 expanded_row_idx
        L211-224: row_idx_type==1 ⇒ expanded_row_idx[dst] = src
                  row_idx_type==0 ⇒ expanded_row_idx[src] = dst（把源下标写成目标下标）
[B] 计数 / 前缀和（expert_tokens_count_or_cumsum 的三种语义）：
    op_kernel/moe_custom_expert_tokens_count.h
    L24-31     CUMSUM_MODE=0, COUNT_MODE=1, KEY_VALUE_MODE=2
    L237-278   Compute: COUNT → 每个专家一个计数；
               CUMSUM → `expertCountOutLocal[j] = loopTokenCount`（累加后写入）
                        ⇒ **含端点的前缀和** cumsum[e] = #{token : exp<=e}
               KEY_VALUE → L338-350：只保留 count!=0 的专家，按专家号升序
                           写成 (expert_id, count) 对，其余补 0
    L146-151   KEY_VALUE 输出长度 = min(actualExpertNum+1, expertNum) * 2
[C] 形状推导：op_host/moe_init_routing_custom_infershape.cpp
    L609-661   n = max(x.dim0, expert_idx.dim0); total = n*k;
               activeNum ∈ {0,-1} ⇒ total，否则 min(active_num, total)
               dropless: expanded_x = [xOutNum, cols]
               expanded_row_idx = [n*k]
               cumsum/count: KEY_VALUE ⇒ [expert_num, 2]；否则 [expertEnd-expertStart]
               expanded_scale: NON_QUANT/DYNAMIC_QUANT 且 dropless ⇒ [xOutNum]
    L686-697   量化模式下 expanded_x dtype = int8

--------------------------------------------------------------------------
实现范围说明（重要，已在汇报中声明）
--------------------------------------------------------------------------
* **支持 drop_pad_mode=0（dropless）+ quant_mode ∈ {-1 非量化, 1 动态量化}**。
  drop_pad_mode=1（droppad，output 为 [expert_num, expert_capacity, cols]）与
  quant_mode=0（静态量化）不在本 Triton 实现范围内，调用时显式抛错。
* 稳定排序假设：同专家号元素按**展平源下标升序**排列（AscendC 硬件 Sort 对相等
  键的稳定性未在文档中保证；本实现取确定性约定，与 fp64 参考一致）。
* 动态量化链路严格照抄 AscendC（full_load.h L188-199）：
      smooth = x_row * scale[expert_of_this_token, :]      # fp32
      s      = max(|smooth|) / 127
      int8   = CAST_RINT(CAST_ODD(smooth / s))             # 经 fp16 中转
  AscendC 的 CAST_ODD(fp32→fp16) 舍入模式与 Triton 默认 RN 存在 ±1 LSB 级差异，
  因此 int8 输出按 **±1 LSB** 判定（见测试文件说明）。
* expert_tokens_num_flag=False 时 AscendC 不写 expert_tokens_count_or_cumsum
  （infershape 也不设 shape）；本实现返回全 0 张量。
* **后端坑（本版已修，证据见 `_analysis/triton_ops/_probe_moe_store.py` /
  `_probe_moe_bisect.py`）**：`[BLK_D, BLK_C]` 2-D block + **2-D 掩码 store** 在
  BiShengIR/FlagTree 上会**静默丢掉部分 lane**（被丢的元素保持输出缓冲初值：
  非量化路径表现为整段 0，量化路径表现为逐行前缀缺失），且丢哪些行/列随 tile
  配置与编译产物变化。列宽非 2 幂时必现（cols=17/24/33/48 全部 FAIL），
  2 幂列宽（cols=32/64/128）不出现——旧测试恰好只覆盖了 2 幂列宽，所以一直被掩盖。
  修法：`_gather_kernel` / `_rowscale_kernel` / `_apply_quant_kernel` 全部改成
  **1-D block**（行号是标量、列是 1-D tile、行内 runtime 列循环），实测所有列宽
  逐元素一致；顺带也消除了 `[1, N]` 退化块触发 FlagTree
  `BlockPtrAnalysis.cpp:309 BlockData::divBlock` 断言的风险。
* active_num 落在 (0, N*k) 之间时 AscendC 会「先对 N*k 排序、再截断输出」，
  本实现为避免语义歧义对该区间显式抛错（-1/0/>=N*k 均支持）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["moe_init_routing_custom"]

# expert_tokens_num_type
CUMSUM = 0
COUNT = 1
KEY_VALUE = 2
# quant_mode
NON_QUANT = -1
STATIC_QUANT = 0
DYNAMIC_QUANT = 1
# row_idx_type
GATHER = 0
SCATTER = 1

_BLK = 64


@triton.jit
def _hist_kernel(E_FLAT, HIST, CNT, NB_E, E, n, BLK: tl.constexpr, USE_CNT: tl.constexpr):
    """统计全局直方图与「每块 × 每专家」计数（原子加，计数与顺序无关）。"""
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    m = o < n
    e = tl.load(E_FLAT + o, mask=m, other=0).to(tl.int32)
    valid = m & (e >= 0) & (e < E)
    tl.atomic_add(HIST + e, 1, mask=valid)
    if USE_CNT:
        tl.atomic_add(CNT + b * E + e, 1, mask=valid)


@triton.jit
def _excl_scan_kernel(HIST, OFF, N_TOTAL, E, BLK_E: tl.constexpr):
    """单 program：OFF = HIST 的排他前缀和，并把总数写回 N_TOTAL[0]。

    注意：块内必须先做排他前缀（run + 块内排他 cumsum）再存入，最后才累加 run；
    早期写成「先存 run 再累加」在 nchunk==1 时会存成全 0。
    """
    run = tl.zeros([BLK_E], dtype=tl.int32)
    nchunk = tl.cdiv(E, BLK_E)
    for c in range(nchunk):
        e_off = c * BLK_E + tl.arange(0, BLK_E)
        m = e_off < E
        h = tl.where(m, tl.load(HIST + e_off, mask=m, other=0), 0)
        excl = tl.cumsum(h, axis=0) - h
        tl.store(OFF + e_off, run + excl, mask=m)
        run = run + h
    tl.store(N_TOTAL + tl.arange(0, 1), tl.sum(run, axis=0))


@triton.jit
def _block_base_kernel(CNT, BASE, NB, E, BLK_E: tl.constexpr):
    """单 program：BASE[b, e] = sum_{b'<b} CNT[b', e]（按块串行扫描）。"""
    run = tl.zeros([BLK_E], dtype=tl.int32)
    nchunk = tl.cdiv(E, BLK_E)
    for b in range(NB):
        for c in range(nchunk):
            e_off = c * BLK_E + tl.arange(0, BLK_E)
            m = e_off < E
            cbb = tl.load(CNT + b * E + e_off, mask=m, other=0)
            tl.store(BASE + b * E + e_off, run, mask=m)
            run = run + cbb


@triton.jit
def _route_kernel(E_FLAT, OFF, BASE, ROWO, DE, E, K, n, BLK: tl.constexpr,
                  ROW_IDX_TYPE: tl.constexpr):
    """稳定计数排序：dst[i] = OFF[e_i] + base[block_i, e_i] + rank_local_i。

    rank_local_i = #{ j 同块 : j < i 且 e_j == e_i }（2-D 掩码归约，精确）。
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    i_off = tl.arange(0, BLK)
    m = o < n
    e = tl.load(E_FLAT + o, mask=m, other=0).to(tl.int32)

    same = (e[:, None] == e[None, :]) & (i_off[None, :] < i_off[:, None])
    rank = tl.sum(tl.where(same, 1, 0), axis=1)

    off = tl.load(OFF + e, mask=m, other=0)
    base = tl.load(BASE + b * E + e, mask=m, other=0)
    dst = off + base + rank

    # 目标位置 -> 源展平下标（后续 gather expanded_x 用）
    if ROW_IDX_TYPE == 1:      # SCATTER: ROWO[dst] = src
        tl.store(ROWO + dst, o, mask=m)
    else:                      # GATHER:  ROWO[src] = dst
        tl.store(ROWO + o, dst, mask=m)
    tl.store(DE + dst, e, mask=m)


@triton.jit
def _dst_to_src_kernel(ROWO, SRC, n, BLK: tl.constexpr, ROW_IDX_TYPE: tl.constexpr):
    """把 expanded_row_idx 还原成 dst->src 映射。"""
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < n
    v = tl.load(ROWO + o, mask=m, other=0)
    if ROW_IDX_TYPE == 1:      # SCATTER: ROWO[dst] = src
        tl.store(SRC + o, v, mask=m)
    else:                      # GATHER:  ROWO[src] = dst
        tl.store(SRC + v, o, mask=m)


@triton.jit
def _gather_kernel(X, SRC, OUTX, TOTAL, K, COLS, BLK_R: tl.constexpr, BLK_C: tl.constexpr):
    """非量化路径：按 dst->src 把 x 的整行搬到 expanded_x。

    ⚠️ **只用 1-D block**（行号是标量，列是 1-D tile）：``[BLK_D, BLK_C]`` 的 2-D
    block + 2-D 掩码 store 在本后端上有静默丢写（见文件头「实现范围说明」第 4 条），
    非 2 幂列宽（cols=17/24/33/48）必现；1-D 写法在所有实测列宽下逐元素一致。
    行块 BLK_R 行在一个 program 内串行处理，行内按 BLK_C 分块（runtime 循环）。
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                ob = d * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    v = tl.load(X + xb + c, mask=mc, other=0.0)
                    tl.store(OUTX + ob + c, v, mask=mc)


@triton.jit
def _rowscale_kernel(X, SRC, DE, SMOOTH, ESCALE, TOTAL, K, COLS,
                     BLK_R: tl.constexpr, BLK_C: tl.constexpr,
                     HAS_SMOOTH: tl.constexpr = True):
    """smooth = x_row * scale[expert_of_row]; s = max|smooth| / 127（每行一个 s）。

    ``HAS_SMOOTH=False`` 对应生产调用点的 ``quant_mode=1, scale=None``：
    AscendC 的 dynamic-quant 分支（``moe_v2_fullload_dynamic_quant.h`` L238-247）
    只做 ``s = ReduceMax(Abs(x_row)) / 127``，没有 smooth 因子。

    同样只用 1-D block（见 `_gather_kernel`）：行号是标量，列按 BLK_C 分块做在线 max，
    整行 max 仍在一个 program 内完成（不跨 program 归约）。
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                if HAS_SMOOTH:
                    sb = tl.load(DE + d) * COLS
                run = tl.zeros([BLK_C], dtype=tl.float32)
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    xv = tl.load(X + xb + c, mask=mc, other=0.0).to(tl.float32)
                    if HAS_SMOOTH:
                        sm = tl.load(SMOOTH + sb + c, mask=mc,
                                     other=0.0).to(tl.float32)
                        z = xv * sm
                    else:
                        z = xv
                    run = tl.maximum(run, tl.where(mc, tl.abs(z), 0.0))
                tl.store(ESCALE + d + tl.arange(0, 1),
                         tl.max(run, axis=0, keep_dims=True) / 127.0)


@triton.jit
def _apply_quant_kernel(X, SRC, DE, SMOOTH, ESCALE, OUTX, TOTAL, K, COLS,
                        BLK_R: tl.constexpr, BLK_C: tl.constexpr,
                        HAS_SMOOTH: tl.constexpr = True):
    """int8 = CAST_RINT(CAST_ODD(smooth / s))，fp16 中转（AscendC full_load.h L197-199）。

    1-D block（见 `_gather_kernel`）：2-D 掩码 store 在非 2 幂列宽下会丢写。
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                ob = d * COLS
                # 整零行的 s == 0：用 1.0 占位（该行元素全 0，量化结果仍是 0），
                # 与 rms_norm_dynamic_quant.py 同款处理；否则 0/0=NaN 会被
                # `tl.maximum(NaN,-128)` 变成 -128，而参考实现给 0。
                s = tl.maximum(tl.load(ESCALE + d + tl.arange(0, 1)), 1e-30)
                if HAS_SMOOTH:
                    sb = tl.load(DE + d) * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    xv = tl.load(X + xb + c, mask=mc, other=0.0).to(tl.float32)
                    if HAS_SMOOTH:
                        sm = tl.load(SMOOTH + sb + c, mask=mc,
                                     other=0.0).to(tl.float32)
                        num = xv * sm
                    else:
                        num = xv
                    q = num / s
                    f = q.to(tl.float16).to(tl.float32)
                    r = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
                    # AscendC 的 CAST_RINT 是**饱和**转换（参考实现同样 clamp 到 [-128,127]）：
                    # 不加这一步时，amax=0 的整零行会让 q=inf/nan，int8 结果与 AscendC 差到 239。
                    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
                    tl.store(OUTX + ob + c, r.to(tl.int8), mask=mc)


@triton.jit
def _counts_kernel(HIST, COUT, expert_start, expert_end, expert_num,
                   BLK_E: tl.constexpr, MODE: tl.constexpr):
    """单 program：把直方图整理成 COUNT / CUMSUM / KEY_VALUE 三种输出。"""
    run = tl.zeros([1], dtype=tl.int64)
    kv = tl.zeros([1], dtype=tl.int64)
    nchunk = tl.cdiv(expert_num, BLK_E)
    for c in range(nchunk):
        e_off = c * BLK_E + tl.arange(0, BLK_E)
        m = e_off < expert_num
        h = tl.load(HIST + e_off, mask=m, other=0).to(tl.int64)
        inrange = m & (e_off >= expert_start) & (e_off < expert_end)
        if MODE == 1:      # COUNT：每专家计数
            tl.store(COUT + (e_off - expert_start), h, mask=inrange)
        elif MODE == 0:    # CUMSUM：含端点前缀和 cumsum[e] = #{token : expert <= e}
            cum = tl.cumsum(tl.where(m, h, 0), axis=0) + run
            tl.store(COUT + (e_off - expert_start), cum, mask=inrange)
        else:              # KEY_VALUE(2)：按专家号升序写 (expert_id, count)，只保留 count != 0
            nz = inrange & (h != 0)
            pos = tl.cumsum(tl.where(nz, 1, 0), axis=0) - 1 + kv
            # 用 2-D [BLK_E,2] 一次写完 (expert_id, count)：两次「跨步 + 掩码」store
            # 会触发 FlagTree FLIR 的 InterleaveStatusWithMaskOptimization 断言
            # （InterleaveOptimization.cpp:635），详见 inplace_partial_rotary_mul.py。
            col = tl.arange(0, 2)
            v2 = tl.where(col[None, :] == 0, e_off.to(tl.int64)[:, None], h[:, None])
            tl.store(COUT + pos[:, None] * 2 + col[None, :], v2, mask=nz[:, None])
        run = run + tl.sum(tl.where(m, h, 0), axis=0)
        kv = kv + tl.sum(tl.where(inrange & (h != 0), 1, 0), axis=0)


def moe_init_routing_custom(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    scale: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    *,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = CUMSUM,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = NON_QUANT,
    active_expert_range: tuple[int, int] | list[int] = (),
    row_idx_type: int = GATHER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 MoeInitRoutingCustom（dropless；非量化 / 动态量化）。

    Returns:
        (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale)
    """
    assert drop_pad_mode == 0, "本 Triton 实现只覆盖 drop_pad_mode=0（dropless）"
    assert quant_mode in (NON_QUANT, DYNAMIC_QUANT), "只覆盖 quant_mode ∈ {-1, 1}"
    assert expert_num > 0, "必须给出 expert_num"
    assert x.dim() == 2 and expert_idx.dim() == 2, "x/expert_idx 需为 2 维"
    assert expert_idx.dtype == torch.int32, "expert_idx 需为 int32"

    N, cols = x.shape
    k = expert_idx.shape[1]
    n = max(N, expert_idx.shape[0])
    total = n * k
    assert active_num in (-1, 0) or active_num >= total, \
        "active_num 落在 (0, N*k) 的截断语义与 AscendC 不一致，本实现不支持"
    assert x.shape[0] == expert_idx.shape[0], "x 与 expert_idx 第 0 维需一致"

    if len(active_expert_range) == 2:
        expert_start, expert_end = int(active_expert_range[0]), int(active_expert_range[1])
    else:
        expert_start, expert_end = 0, expert_num
    assert 0 <= expert_start < expert_end <= expert_num

    dev = x.device
    e_flat = expert_idx.reshape(-1).contiguous()
    assert e_flat.numel() == total

    nb = ceil_div(total, _BLK)
    blk_e = max(next_pow2(min(expert_num, 1024)), 2)
    hist = torch.zeros(expert_num, dtype=torch.int32, device=dev)
    cnt = torch.zeros(nb * expert_num, dtype=torch.int32, device=dev)
    off = torch.zeros(expert_num, dtype=torch.int32, device=dev)
    base = torch.zeros(nb * expert_num, dtype=torch.int32, device=dev)
    n_total = torch.zeros(1, dtype=torch.int32, device=dev)
    rowo = torch.full((total,), -1, dtype=torch.int32, device=dev)
    de = torch.zeros(total, dtype=torch.int32, device=dev)

    _hist_kernel[(nb,)](e_flat, hist, cnt, nb, expert_num, total, BLK=_BLK, USE_CNT=True)
    _excl_scan_kernel[(1,)](hist, off, n_total, expert_num, BLK_E=blk_e)
    _block_base_kernel[(1,)](cnt, base, nb, expert_num, BLK_E=blk_e)
    _route_kernel[(nb,)](e_flat, off, base, rowo, de, expert_num, k, total,
                         BLK=_BLK, ROW_IDX_TYPE=int(row_idx_type))

    src = torch.zeros(total, dtype=torch.int32, device=dev)
    _dst_to_src_kernel[(ceil_div(total, _BLK),)](rowo, src, total, BLK=_BLK,
                                                 ROW_IDX_TYPE=int(row_idx_type))

    # 量化/搬运路径全部改为 1-D block（行号标量 + 列 1-D tile）：
    #   * BLK_R：每个 program 串行处理 BLK_R 行（>= 2，保留行级并行；
    #     1-D block 下已不存在 [1, N] 退化块，故不再需要靠 BLK_D>=2 规避
    #     FlagTree `BlockPtrAnalysis.cpp:309 BlockData::divBlock` 断言）
    #   * BLK_C：行内列分块（runtime 循环），UB 占用 ≈ 常数 × BLK_C × 32 bit
    blk_r_q = 4
    blk_c_q = max(16, min(next_pow2(cols), 1024))
    if quant_mode == DYNAMIC_QUANT:
        # scale is None → 走 AscendC 的「纯动态量化」分支：只按每行 amax 求 scale，
        # 不含 per-expert smooth 因子。生产（token_dispatcher.py L401 /
        # fused_moe_layer.py）就是这么调的，旧实现直接 assert 挂掉。
        has_smooth = scale is not None
        if has_smooth:
            smat = scale.contiguous()
            if smat.dim() == 1:
                smat = smat.expand(expert_num, cols).contiguous()
            assert smat.shape == (expert_num, cols), \
                f"scale 期望 [{expert_num},{cols}]，实际 {tuple(smat.shape)}"
        else:
            # 占位张量：HAS_SMOOTH=False 时内核不会读它
            smat = torch.zeros(1, dtype=torch.float32, device=dev)
        xq = torch.empty((total, cols), dtype=torch.int8, device=dev)
        escale = torch.zeros(total, dtype=torch.float32, device=dev)
        _rowscale_kernel[(ceil_div(total, blk_r_q),)](
            x, src, de, smat, escale, total, k, cols, BLK_R=blk_r_q, BLK_C=blk_c_q,
            HAS_SMOOTH=has_smooth)
        _apply_quant_kernel[(ceil_div(total, blk_r_q),)](
            x, src, de, smat, escale, xq, total, k, cols, BLK_R=blk_r_q, BLK_C=blk_c_q,
            HAS_SMOOTH=has_smooth)
        expanded_x = xq
    else:
        ex = torch.zeros((total, cols), dtype=x.dtype, device=dev)
        blk_r_g = 8
        blk_c_g = max(16, min(next_pow2(cols), 512))
        _gather_kernel[(ceil_div(total, blk_r_g),)](
            x, src, ex, total, k, cols, BLK_R=blk_r_g, BLK_C=blk_c_g)
        expanded_x = ex
        escale = torch.zeros(total, dtype=torch.float32, device=dev)

    # ---- expert_tokens_count_or_cumsum ----
    if expert_tokens_num_flag:
        if expert_tokens_num_type == KEY_VALUE:
            cout = torch.zeros((expert_num, 2), dtype=torch.int64, device=dev)
            _counts_kernel[(1,)](hist, cout.reshape(-1), expert_start, expert_end, expert_num,
                                 BLK_E=blk_e, MODE=KEY_VALUE)
        else:
            cout = torch.zeros(expert_end - expert_start, dtype=torch.int64, device=dev)
            _counts_kernel[(1,)](hist, cout, expert_start, expert_end, expert_num,
                                 BLK_E=blk_e, MODE=int(expert_tokens_num_type))
    else:
        cout = torch.zeros(expert_end - expert_start, dtype=torch.int64, device=dev)

    return expanded_x, rowo, cout, escale
