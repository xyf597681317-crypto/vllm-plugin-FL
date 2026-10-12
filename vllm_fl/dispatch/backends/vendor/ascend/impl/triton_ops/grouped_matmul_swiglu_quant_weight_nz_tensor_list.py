"""grouped_matmul_swiglu_quant_weight_nz_tensor_list —— AscendC → Triton

来源：`_repos/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/`

--------------------------------------------------------------------------
调用点与别名关系（重要，已核对源码）
--------------------------------------------------------------------------
`grouped_matmul_swiglu_quant_torch_adpt.h`
  L20-51  `grouped_matmul_swiglu_quant_weight_nz(x, weight, weight_scale, x_scale,
                                                 group_list, bias, offset, swiglu_limit)`
          → `aclnnGroupedMatmulSwigluQuantWeightNZ`（**单张量**版：weight 是 at::Tensor）
  L53-60  `grouped_matmul_swiglu_quant(...)` **直接 return 上面那个函数** ⇒ **纯别名**
  L62-99  `grouped_matmul_swiglu_quant_weight_nz_tensor_list(x, weight[], weight_scale[],
                                                             x_scale, group_list, bias, offset,
                                                             swiglu_limit)`
          → `aclnnGroupedMatmulSwigluQuantWeightNzTensorList`（**TensorList** 版）

并且 `_repos/vllm-ascend/csrc/gmm/` 下**只有 4 个算子目录**
（`gmm_dequant_situ_quant`、`grouped_matmul_swiglu_quant`、`grouped_matmul_swiglu_quant_v2`、
`grouped_matmul_swiglu_quant_weight_nz_tensor_list`），**没有 `*_weight_nz` 目录**，
唯一的 OpDef 是 `class GroupedMatmulSwigluQuantWeightNzTensorList`
（`op_host/..._def.cpp` L19 / `OP_ADD` L65）。

⇒ **结论**：`grouped_matmul_swiglu_quant` 与 `grouped_matmul_swiglu_quant_weight_nz`
在本仓库内**没有独立的 AscendC 内核源码**（走 CANN/OPP 提供的 aclnn），
只有 `_tensor_list` 版有本仓库的 AscendC 实现。因此本文件实现 `_tensor_list` 的数学，
并把 `grouped_matmul_swiglu_quant` / `..._weight_nz` 作为**同一数学的别名**导出
（`grouped_matmul_swiglu_quant = grouped_matmul_swiglu_quant_weight_nz = ..._tensor_list`
的薄包装，weight 传单张量时内部转成 list）。

--------------------------------------------------------------------------
接口（op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp L23-51）
--------------------------------------------------------------------------
    Input : x            [M, K]            int8
            weight       DYNAMIC TensorList[G], FORMAT_FRACTAL_NZ, int8
            weight_scale DYNAMIC TensorList[G], fp32 / bf16 / fp16
            x_scale      [M]               fp32
            group_list   [G]               int64
    Output: y        [M, N/2]  int8
            y_scale  [M]       fp32
            （adapter 还额外返回 output_offset：标量 fp32，L34/L79）
    Attr  : swiglu_limit (REQUIRED, float, 默认 0.0)

adapter 的形状推导（L72-79）：`n = weight[0].sizes()[1]`、`m = x.sizes()[0]`、
`k = x.sizes()[1]`；`output = [m, n/2] int8`、`output_scale = [m] fp32`。

--------------------------------------------------------------------------
数学（SwiGLU + per-token 动态量化；与本仓库 mc2 的 epilogue 同源）
--------------------------------------------------------------------------
对 group g 覆盖的行区间 [start_g, end_g)：
    C   = (x[start:end] · x_scale[:, None]) @ (W_g · weight_scale_g[None, :])   # [rows, N]
        （x/W 为 int8，累加在 fp32；x_scale 是 per-token，weight_scale 是 per-N 通道）
    gate = C[:, :N/2]; up = C[:, N/2:]
    sw  = (gate * sigmoid(gate)) * up          # swiglu_limit>0 时先 clamp(gate, max=limit)
    mx  = max|sw|  (per row)
    y   = rint(sw * 127 / mx)  → int8          # per-token 动态量化
    y_scale = mx / 127
公式出处：`_repos/vllm-fl/.../csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/
block_epilogue_pertoken_swiglu.hpp` L211-265（同族 SwiGLU+per-token 量化 epilogue）。

`group_list` 语义：本实现按 **累计结束偏移**（`group_list[g]` = 第 g 组最后一行的下一行，
即 cumsum 形式）解释 —— 与本仓库 MoE 的 `expert_tokens_count_or_cumsum`(CUMSUM) 约定一致；
起点由上一组末尾推出（第 0 组起点为 0）。已在文件与测试中显式标注为**语义假设**。

--------------------------------------------------------------------------
参数交代（不静默忽略）
--------------------------------------------------------------------------
x / weight / weight_scale / x_scale / group_list —— 使用
swiglu_limit —— 使用（>0 时 clamp gate）
bias / offset    —— adapter L41-42/L86-87 传给了 aclnn；**def.cpp 里没有 bias/offset 形参**，
                    本实现不支持，传入非空即抛 NotImplementedError（不静默忽略）。
--------------------------------------------------------------------------
实现说明（刻意规避已知 Ascend 后端静默错误）
--------------------------------------------------------------------------
* **无运行时 range 循环、无 2-D reduce 之外的复杂结构**：grid = (G, row_blocks, col_blocks)，
  每个 program 只做 load → tl.dot → 逐元素 → store。
* 已知坑规避：不使用「标量掩码 & 向量掩码」广播；不把 `1e-30` 这类字面量写进 kernel
  （`EPS` 走运行时标量参数）；不依赖循环携带张量。
* `weight` 逻辑形状按 [K, N] 处理（AscendC 侧是 FRACTAL_NZ 分形格式；Triton 实现使用
  其逻辑等价的 [K, N] 行主序布局，已在测试中说明）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2
# 共用 UB 预算模型（实测拟合式 + 常量只在 v2 文件里维护一份，避免两处漂移）
from .grouped_matmul_swiglu_quant_v2 import (
    BLK_K_CAP, BLK_N_CAP, UB_LIMIT_BITS, ub_budget_bits,
)

__all__ = [
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant",
]


@triton.jit
def _gmm_swiglu_quant_kernel(
    X, W, WS, XS, GSTART, GEND, Y, YS, YOFF,
    K, N, HALF_N, EPS,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """一个 program 处理 (group g, BLK_M 行)，**在 program 内沿 N 轴与 K 轴运行期分块**。

    ⚠️ SwiGLU 对齐要点：gate 取 W 的第 [0, HALF_N) 列，up 取第 [HALF_N, 2*HALF_N) 列，
    两者都用**同一个输出列索引 cn** 表达，天然对齐。
    错误写法 `gate = where(n<I, c, 0)` + `up = where(n>=I, c, 0)` 作用在**不相交**的列区间，
    乘积恒为 0 —— 这正是先前三个算子 out 精确等于 0 的根因（不是 tl.dot 的问题）。

    ⚠️ 为什么必须分块（910_93 + BiShengIR 实测，见 BACKEND_NOTES.md 坑 3）：
    生产 `I = moe_intermediate_size ∈ [512, 2304]`、`K = hidden ∈ {2048, 7168}`，
    而单块 `[BLK_M, BLK_N]` + 一次性载入整个 K 的写法在 `BLK_N ≥ 512` 或 `BLK_K ≥ 512`
    时必然 `ub overflow`（限额 1,572,864 bits：BLK_N=512 → 2,359,808；BLK_K=512 → 3,539,456）。
    **缩小 BLK_M 无效**（BLK_M=1、BLK_N=512 实测仍需 1,704,192 bits）。

    ⚠️ 代价：per-token 量化要整行的 max|sw|，分块后单块看不到整行 ⇒ 走
    「趟 1 求行内最大值 / 趟 2 重算并量化写出」，tl.dot 计算量 ×2。
    """
    g = tl.program_id(0)
    start = tl.load(GSTART + g)
    end = tl.load(GEND + g)
    rows = start + tl.program_id(1) * BLK_M + tl.arange(0, BLK_M)
    rm = rows < end
    k_off = tl.arange(0, BLK_K)
    n_off = tl.arange(0, BLK_N)          # 输出列索引（块内）
    xs = tl.load(XS + rows, mask=rm, other=1.0).to(tl.float32)
    nchunk = tl.cdiv(HALF_N, BLK_N)

    # ---- 趟 1：逐 N 块算 sw，累计 |sw| 行内最大值 ----
    m1 = tl.zeros([BLK_M], dtype=tl.float32)
    for nc in range(0, nchunk):
        cn = nc * BLK_N + n_off
        nm = cn < HALF_N
        cg = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        cu = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        for k0 in range(0, K, BLK_K):
            kk = k0 + k_off
            km = kk < K
            xi = tl.load(X + rows[:, None] * K + kk[None, :],
                         mask=rm[:, None] & km[None, :], other=0).to(tl.float16)
            wg = tl.load(W + g * K * N + kk[:, None] * N + cn[None, :],
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            wu = tl.load(W + g * K * N + kk[:, None] * N + (cn[None, :] + HALF_N),
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            cg += tl.dot(xi, wg, out_dtype=tl.float32)
            cu += tl.dot(xi, wu, out_dtype=tl.float32)
        sg = tl.load(WS + g * N + cn, mask=nm, other=0.0).to(tl.float32)
        su = tl.load(WS + g * N + cn + HALF_N, mask=nm, other=0.0).to(tl.float32)
        gate = cg * xs[:, None] * sg[None, :]
        up = cu * xs[:, None] * su[None, :]
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        sw = (gate / (1.0 + tl.exp(-gate))) * up
        sw = tl.where(nm[None, :], sw, 0.0)
        m1 = tl.maximum(m1, tl.max(tl.abs(sw), axis=1))

    mxs = tl.maximum(m1, EPS)
    s = mxs / 127.0

    # ---- 趟 2：重算 sw（与趟 1 逐字相同）并量化写出 ----
    for nc in range(0, nchunk):
        cn = nc * BLK_N + n_off
        nm = cn < HALF_N
        cg = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        cu = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        for k0 in range(0, K, BLK_K):
            kk = k0 + k_off
            km = kk < K
            xi = tl.load(X + rows[:, None] * K + kk[None, :],
                         mask=rm[:, None] & km[None, :], other=0).to(tl.float16)
            wg = tl.load(W + g * K * N + kk[:, None] * N + cn[None, :],
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            wu = tl.load(W + g * K * N + kk[:, None] * N + (cn[None, :] + HALF_N),
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            cg += tl.dot(xi, wg, out_dtype=tl.float32)
            cu += tl.dot(xi, wu, out_dtype=tl.float32)
        sg = tl.load(WS + g * N + cn, mask=nm, other=0.0).to(tl.float32)
        su = tl.load(WS + g * N + cn + HALF_N, mask=nm, other=0.0).to(tl.float32)
        gate = cg * xs[:, None] * sg[None, :]
        up = cu * xs[:, None] * su[None, :]
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        sw = (gate / (1.0 + tl.exp(-gate))) * up
        sw = tl.where(nm[None, :], sw, 0.0)

        q = sw * (127.0 / mxs)[:, None]
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.minimum(tl.maximum(
            tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)
        tl.store(Y + rows[:, None] * HALF_N + cn[None, :], qr.to(tl.int8),
                 mask=rm[:, None] & nm[None, :])

    tl.store(YS + rows, s, mask=rm)
    if HAS_BIAS:
        tl.store(YOFF + tl.arange(0, 1), tl.sum(tl.where(rm, s, 0.0), axis=0))


def _split_groups(group_list: torch.Tensor, n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把 group_list 转成 (start, end)。约定 group_list 是**累计结束偏移**(cumsum)。"""
    ends = group_list.contiguous().to(torch.int32)
    assert ends.numel() == n_groups, f"group_list 长度 {ends.numel()} != 组数 {n_groups}"
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    return starts, ends


def _impl(
    x, weight_list, weight_scale_list, x_scale, group_list, bias, offset, swiglu_limit,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.dtype == torch.int8, "x 需为 [M, K] int8"
    if bias is not None:
        raise NotImplementedError(
            "bias 非空：grouped_matmul_swiglu_quant_weight_nz_tensor_list 的 def.cpp"
            "（L23-51）没有 bias 形参，AscendC 原实现不使用 bias；本实现保持同样行为，"
            "不支持传入 bias。")
    if offset is not None:
        raise NotImplementedError(
            "offset 非空：def.cpp 没有 offset 形参，AscendC 原实现不使用；本实现不支持。")

    M, K = x.shape
    if isinstance(weight_list, (list, tuple)):
        G = len(weight_list)
        w = torch.stack([t.contiguous() for t in weight_list], dim=0)
    else:
        assert weight_list.dim() == 3, "单张量 weight 需为 [G, K, N]"
        G = weight_list.shape[0]
        w = weight_list.contiguous()
    assert w.dtype == torch.int8 and w.shape[1] == K, f"weight 期望 int8 [G,{K},N]"
    N = w.shape[2]
    assert N % 2 == 0, f"N={N} 必须为偶数（gate‖up）"
    I = N // 2
    if isinstance(weight_scale_list, (list, tuple)):
        assert len(weight_scale_list) == G
        ws = torch.stack([t.contiguous() for t in weight_scale_list], dim=0).to(torch.float32)
    else:
        ws = weight_scale_list.contiguous().to(torch.float32)
    assert tuple(ws.shape) == (G, N), f"weight_scale 期望 [{G},{N}]，实际 {tuple(ws.shape)}"

    dev = x.device
    x = x.contiguous()
    xs = x_scale.contiguous().to(torch.float32)
    assert xs.numel() == M, f"x_scale 长度 {xs.numel()} != M={M}"
    starts, ends = _split_groups(group_list, G)
    assert int(ends[-1].item()) <= M, "group_list 末项不能超过 M"

    y = torch.empty((M, I), dtype=torch.int8, device=dev)
    y_scale = torch.empty(M, dtype=torch.float32, device=dev)
    y_off = torch.zeros(1, dtype=torch.float32, device=dev)

    # ---- tile 选择：受 UB 限额约束（实测拟合式见 grouped_matmul_swiglu_quant_v2 文件头；
    #      两个算子 host tiling 同源，共用同一份常量以免各写一份漂移）----
    # 小形状下 blk_k / blk_n 覆盖整个 K / I ⇒ 只有 1 个块，与分块前的数值行为完全一致。
    if K <= 0 or I <= 0:
        raise NotImplementedError(f"K={K} / I={I} 非法（必须为正）")
    blk_m = 16
    blk_k = min(max(next_pow2(K), 16), BLK_K_CAP)
    blk_n = min(max(next_pow2(I), 16), BLK_N_CAP)   # 输出列只需覆盖 HALF_N = I
    need = ub_budget_bits(blk_m, blk_k, blk_n)
    if need > UB_LIMIT_BITS:
        raise NotImplementedError(
            f"grouped_matmul_swiglu_quant_weight_nz_tensor_list: tile "
            f"(BLK_M={blk_m}, BLK_K={blk_k}, BLK_N={blk_n}) 预估 UB 需求 {need} bits "
            f"> 限额 {UB_LIMIT_BITS} bits（K={K}, I={I}）。请调小 BLK_K_CAP/BLK_N_CAP。")
    grid = (G, ceil_div(M, blk_m))
    _gmm_swiglu_quant_kernel[grid](
        x, w, ws, xs, starts, ends, y, y_scale, y_off,
        K, N, I, 1e-30,
        BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
        HAS_BIAS=True,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return y, y_scale, y_off[0]


def grouped_matmul_swiglu_quant_weight_nz_tensor_list(
    x: torch.Tensor,
    weight,
    weight_scale,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TensorList 版：每个专家一张 int8 权重。返回 (y, y_scale, y_offset)。"""
    return _impl(x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)


def grouped_matmul_swiglu_quant_weight_nz(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单张量版（adapter L20-51）。**本仓库无独立 AscendC 内核**（见模块 docstring），
    数学与 TensorList 版一致，故复用同一实现；weight 需为 [G, K, N]（未堆叠时会被视为 [1,K,N]）。"""
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)
        weight_scale = weight_scale.reshape(1, -1)
    return _impl(x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """**纯别名**：adapter L53-60 直接 `return grouped_matmul_swiglu_quant_weight_nz(...)`。"""
    return grouped_matmul_swiglu_quant_weight_nz(
        x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)
