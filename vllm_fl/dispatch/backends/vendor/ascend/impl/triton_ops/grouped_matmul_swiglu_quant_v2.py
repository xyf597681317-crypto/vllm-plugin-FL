"""grouped_matmul_swiglu_quant_v2 —— AscendC → Triton

来源：`_repos/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2/`

--------------------------------------------------------------------------
调用点签名（逐字对齐 grouped_matmul_swiglu_quant_v2_torch_adpt.h L20-36）
--------------------------------------------------------------------------
    std::tuple<at::Tensor,at::Tensor> grouped_matmul_swiglu_quant_v2(
        const at::Tensor& x,                                   // [M, K] int8
        const at::TensorList& weight,                          // 每专家 [K, N] int8 (FRACTAL_NZ)
        const at::TensorList& weight_scale,                    // 每专家 [N]
        const at::Tensor& x_scale,                             // [M] fp32  per-token
        const at::Tensor& group_list,                          // [G] int64
        const c10::optional<at::Tensor>& smooth_scale,          // 可选, (E,N/2) 或 (E,)
        const c10::optional<at::TensorList>& weight_assist_matrix,
        const c10::optional<at::Tensor>& bias,
        c10::optional<int64_t> dequant_mode, dequant_dtype, quant_mode, quant_dtype,
        bool transpose_weight, int64_t group_list_type,
        at::IntArrayRef tuning_config, double swiglu_limit)

形状推导（adapter L39-45）：`n = weight_scale[0].sizes().back()`、`m = x.sizes()[0]`、
`k = x.sizes()[1]`；`output = [m, n/2] int8`、`output_scale = [m] fp32`。
（注意 v2 从 **weight_scale** 取 n，v1 从 weight 取 —— 这是两者的接口差异之一。）

--------------------------------------------------------------------------
与 v1（`grouped_matmul_swiglu_quant_weight_nz_tensor_list`）的差异
--------------------------------------------------------------------------
1. **`smooth_scale`（v2 新增，可选）**：在 **SwiGLU 之后、per-token 量化之前**
   乘到 SwiGLU 结果上。出处：op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_post.h
     L137-139 `if (smoothScaleDimNum != 0) ApplySmoothScale(...)`（为空则跳过）
     L214-217 `smoothScaleDimNum == 2` ⇒ 形状 (E, N/2)，取当前 group 那一行
     L221-223 `smoothScaleDimNum == 1` ⇒ 形状 (E,)，单标量广播到 N/2
     L350 注释「四步 vector 计算（perToken反量化、Swiglu、SmoothScale、Quant）」
2. **`bias`（v2 def 里有形参但内核不用）**：
   op_kernel/grouped_matmul_swiglu_quant_v2_utils.h L84 明确写着
   `// bias未被使用但高阶模板参数需要传入`。
   ⇒ 与 v1 一样，**AscendC 原实现不加 bias**；本实现传入非空 bias 时抛
   NotImplementedError（不静默忽略）。
3. `weight_assist_matrix`：W4A8/A4W4 路径的辅助量化矩阵（int4 打包）。
   本实现面向 **int8 权重（A8W8）**，传入非空 assist matrix 时抛 NotImplementedError。
4. `group_list_type`：0 ⇒ `group_list` 是**累计结束偏移**（cumsum，默认）；
   1 ⇒ 是**每组计数**（本实现内部做前缀和换算）。已显式标注为语义假设。
5. 其余数学与 v1 完全一致（同族 SwiGLU + per-token 动态量化）：
     C1 = (x·x_scale) @ (W_g·weight_scale_g)      # [rows, N]
     gate = C1[:, :N/2] ; up = C1[:, N/2:]         # ← up 必须左移 N/2 对齐
     sw = silu(gate) * up                           # swiglu_limit>0 时 clamp(gate)
     sw = sw * smooth_scale_g                       # v2 新增
     y = rint(sw*127/max|sw|) → int8 ; y_scale = max|sw|/127

--------------------------------------------------------------------------
实现要点（沿用本任务已验证 PASS 的写法）
--------------------------------------------------------------------------
* gate/up 用**同一个输出列索引**分别取 W 的两半（两次 tl.dot），天然对齐 ——
  绝不写成 `up = where(n>=I, c, 0)`（那是"支撑在不相交列区间"，乘积恒为 0）。
* 逐段可验证：本 kernel 只用 load → tl.dot → 逐元素 → store，无运行时循环携带张量。
* 不做「标量掩码 & 向量掩码」广播；`smooth_scale` 的两种形状分别处理。

后端形态说明（910_93 + BiShengIR/FlagTree；探针：`_probe_2dstore_gmmv2.py`）：
  L129 是「[BLK_M, BLK_N] 2-D 掩码 store + 长值链（dot→silu→逐行 max→除法→fp16 往返→rint）」，
  经 sentinel 双填充实测**没有**出现静默丢写：I ∈ {17,24,33,48,100,127}（含 smooth col/row、
  swiglu_limit、K=128/256）全部 dropped=0、wrong>1LSB=0、scale_err ≤ 2e-7。
  故本文件**不做改写**；但注意 I ≥ 257（BLK_N ≥ 512）在本后端的 UB 上会编译失败
  （ub overflow），如需支持需降 BLK_N 分块 —— 与本条无关，是 UB 容量问题。
  ⚠️ 不要用 `tl.reshape` 2-D→1-D 再 store：同后端会大面积丢 lane。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["grouped_matmul_swiglu_quant_v2"]


# ---------------------------------------------------------------------------
# UB 预算（910_93 + BiShengIR，**实测量出来的**，见 BACKEND_NOTES.md 坑 3）
# ---------------------------------------------------------------------------
# 限额 1,572,864 bits（原始报错 "ub overflow, requires N bits while 1572864 bits available"）。
# 实测（M=16/64，BLK_M=16，BLK_K=128，**旧「单块一次性」写法**）：
#     BLK_N=256 → OK ; 512 → 2,359,808 ; 1024 → 4,719,104 ; 2048 → 9,437,696 ; 4096 → 18,874,880
# 拟合式：ub ≈ BLK_N × (3243 + 86×BLK_M + 24×max(BLK_K-128, 0))
# 该式的实测校验（见 _probe_gmm_prod.py 段 C，**分块后新写法**）：
#     BM=16,BK=512,BN=256  → 预测 3,541,760  实测 3,540,992   （一致，<1%）
#     BM=16,BK=2048,BN=512 → 预测 25,957,888 实测 25,954,304  （一致，<1%）
#     BM=16,BK=128,BN=512  → 预测 2,364,928  实测 3,409,920   （**低预测 44%**）
# ⇒ 该拟合式只在「K 轴分块数很多」或「N 块很大」时准确；改成分块循环后 BLK_N=512 那一档
#   比旧写法更费 UB。所以**真正的保护是下面两个硬上限**，拟合式只作粗网：
UB_LIMIT_BITS = 1_572_864
BLK_N_CAP = 256
BLK_K_CAP = 128
# 选定 tile(BM=16, BK=128, BN=256) 的预算：1,182,464 bits ≤ 限额 1,572,864（余量 ~25%），
# 已在生产形状 K∈{2048,7168} × I∈{512,1024,2048,2304} 上实测「编译+执行+逐元素一致」通过。


def ub_budget_bits(blk_m: int, blk_k: int, blk_n: int) -> int:
    """实测拟合出的 UB 需求（bits），供 host 侧显式拦截。"""
    return blk_n * (3243 + 86 * blk_m + 24 * max(blk_k - 128, 0))


@triton.jit
def _gmm_v2_kernel(
    X, W, WS, XS, SMOOTH, GSTART, GEND, Y, YS,
    K, N, HALF_N,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_N: tl.constexpr,
    HAS_SMOOTH_COL: tl.constexpr,     # smooth_scale 形状 (E, N/2)
    HAS_SMOOTH_ROW: tl.constexpr,     # smooth_scale 形状 (E,)
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """一个 program 处理 (group g, BLK_M 行)，**在 program 内沿 N 轴与 K 轴运行期分块**。

    为什么分块（910_93 + BiShengIR 实测，BACKEND_NOTES.md 坑 3）：
      生产 `I = moe_intermediate_size ∈ [512, 2304]`、`K = hidden ∈ {2048, 7168}`。
      单块 `[BLK_M, BLK_N]` 且一次性载入整个 K 的写法，在 `BLK_N ≥ 512` 或 `BLK_K ≥ 512`
      时必然 `ub overflow`（限额 1,572,864 bits），且**缩小 BLK_M 无效**
      （BLK_M=1、BLK_N=512 实测仍需 1,704,192 bits）。故 N、K 都改为运行期循环，
      单块只占 BLK_N ≤ 256 / BLK_K ≤ 128。

    ⚠️ 代价（必须知情）：per-token 量化要**整行**的 max|sw|，分块后单块看不到整行，
      因此这里是「**趟 1 求行内最大值 / 趟 2 重算并量化写出**」，tl.dot 计算量 ×2。
      不用 scratch 暂存 sw 是因为它要 M×I×4 字节（生产 M 可达数万 token → 几百 MB）。

    ⚠️ 对齐要点（与旧版一致）：gate 取 W 的 [0, HALF_N) 列、up 取 [HALF_N, 2*HALF_N) 列，
      两者用**同一个输出列索引 cn** 表达，天然对齐。
    """
    g = tl.program_id(0)
    start = tl.load(GSTART + g)
    end = tl.load(GEND + g)
    rows = start + tl.program_id(1) * BLK_M + tl.arange(0, BLK_M)
    rm = rows < end
    k_off = tl.arange(0, BLK_K)
    n_off = tl.arange(0, BLK_N)
    xs = tl.load(XS + rows, mask=rm, other=1.0).to(tl.float32)
    nchunk = tl.cdiv(HALF_N, BLK_N)

    # ---- 趟 1：逐 N 块算 sw，累计 |sw| 的行内最大值（K 轴同样分块累加）----
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
        if HAS_SMOOTH_COL:
            sm = tl.load(SMOOTH + g * HALF_N + cn, mask=nm, other=1.0).to(tl.float32)
            sw = sw * sm[None, :]
        if HAS_SMOOTH_ROW:
            smr = tl.load(SMOOTH + g).to(tl.float32)
            sw = sw * smr
        sw = tl.where(nm[None, :], sw, 0.0)
        m1 = tl.maximum(m1, tl.max(tl.abs(sw), axis=1))

    mxs = tl.where(m1 > 0.0, m1, 1.0)
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
        if HAS_SMOOTH_COL:
            sm = tl.load(SMOOTH + g * HALF_N + cn, mask=nm, other=1.0).to(tl.float32)
            sw = sw * sm[None, :]
        if HAS_SMOOTH_ROW:
            smr = tl.load(SMOOTH + g).to(tl.float32)
            sw = sw * smr
        sw = tl.where(nm[None, :], sw, 0.0)

        q = sw * (127.0 / mxs)[:, None]
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.minimum(tl.maximum(
            tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)
        tl.store(Y + rows[:, None] * HALF_N + cn[None, :], qr.to(tl.int8),
                 mask=rm[:, None] & nm[None, :])

    tl.store(YS + rows, s, mask=rm)


def _ends_from_group_list(group_list, e, group_list_type):
    """group_list_type 0 ⇒ cumsum（累计结束偏移）；1 ⇒ 每组计数（内部前缀和）。"""
    gl = group_list.contiguous().to(torch.int32)
    assert gl.numel() == e, f"group_list 长度 {gl.numel()} != 组数 {e}"
    if int(group_list_type) == 1:
        ends = torch.cumsum(gl.to(torch.int64), dim=0).to(torch.int32)
    else:
        ends = gl
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    return starts, ends


def grouped_matmul_swiglu_quant_v2(
    x: torch.Tensor,
    weight,
    weight_scale,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
    weight_assist_matrix=None,
    bias: torch.Tensor | None = None,
    dequant_mode=None,
    dequant_dtype=None,
    quant_mode=None,
    quant_dtype=None,
    transpose_weight: bool = False,
    group_list_type: int = 0,
    tuning_config=(0,),
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 GroupedMatmulSwigluQuantV2（int8/A8W8 路径）。

    Returns:
        (y [M, N//2] int8, y_scale [M] fp32)
    """
    if bias is not None:
        raise NotImplementedError(
            "bias 非空：op_kernel/grouped_matmul_swiglu_quant_v2_utils.h L84 明确写着"
            "「bias未被使用但高阶模板参数需要传入」，AscendC 原实现不使用 bias；"
            "本实现保持同样行为，不支持传入 bias。")
    if weight_assist_matrix is not None and len(weight_assist_matrix) > 0:
        raise NotImplementedError(
            "weight_assist_matrix 非空：它是 W4A8/A4W4 的辅助量化矩阵，"
            "本实现只覆盖 int8(A8W8) 权重路径。")
    if transpose_weight:
        raise NotImplementedError("transpose_weight=True 未在本实现覆盖范围内。")

    assert x.dim() == 2 and x.dtype == torch.int8, "x 需为 [M, K] int8"
    M, K = x.shape
    E = len(weight) if isinstance(weight, (list, tuple)) else weight.shape[0]
    w = (torch.stack([t.contiguous() for t in weight], dim=0)
         if isinstance(weight, (list, tuple)) else weight.contiguous())
    assert w.dtype == torch.int8 and w.shape[1] == K, f"weight 期望 int8 [E,{K},N]"
    ws = (torch.stack([t.contiguous() for t in weight_scale], dim=0).to(torch.float32)
          if isinstance(weight_scale, (list, tuple)) else weight_scale.contiguous().to(torch.float32))
    N = ws.shape[-1]                       # ← 与 adapter 一致：n 取自 weight_scale
    assert N % 2 == 0, f"N={N} 必须为偶数（gate‖up）"
    I = N // 2
    assert tuple(ws.shape) == (E, N), f"weight_scale 期望 [{E},{N}]，实际 {tuple(ws.shape)}"
    assert tuple(w.shape) == (E, K, N), f"weight 期望 [{E},{K},{N}]，实际 {tuple(w.shape)}"

    dev = x.device
    x = x.contiguous()
    xs = x_scale.contiguous().to(torch.float32)
    assert xs.numel() == M
    starts, ends = _ends_from_group_list(group_list, E, group_list_type)

    has_col = has_row = False
    sm = x
    if smooth_scale is not None:
        smt = smooth_scale.contiguous().to(torch.float32)
        if smt.dim() == 2:
            assert tuple(smt.shape) == (E, I), f"smooth_scale 期望 [{E},{I}]，实际 {tuple(smt.shape)}"
            has_col = True
        elif smt.dim() == 1:
            assert smt.numel() == E, f"smooth_scale 期望 [{E}]，实际 {tuple(smt.shape)}"
            has_row = True
        else:
            raise NotImplementedError(f"smooth_scale 维度 {smt.dim()} 不支持（只支持 (E,N/2) 或 (E,)）")
        sm = smt

    y = torch.empty((M, I), dtype=torch.int8, device=dev)
    y_scale = torch.empty(M, dtype=torch.float32, device=dev)

    # ---- tile 选择：受 UB 限额约束（见文件头的实测拟合式）----
    # 小形状下 blk_k / blk_n 覆盖整个 K / I ⇒ 只有 1 个块，与分块前的数值行为完全一致；
    # 大形状才真正分块（BLK_N ≤ 256、BLK_K ≤ 128）。
    if K <= 0 or I <= 0:
        raise NotImplementedError(f"K={K} / I={I} 非法（必须为正）")
    blk_m = 16
    blk_k = min(max(next_pow2(K), 16), BLK_K_CAP)
    blk_n = min(max(next_pow2(I), 16), BLK_N_CAP)
    need = ub_budget_bits(blk_m, blk_k, blk_n)
    if need > UB_LIMIT_BITS:
        # 显式拒绝，而不是让 BiShengIR 在编译期抛 ub overflow（也绝不静默错值）
        raise NotImplementedError(
            f"grouped_matmul_swiglu_quant_v2: tile (BLK_M={blk_m}, BLK_K={blk_k}, BLK_N={blk_n}) "
            f"预估 UB 需求 {need} bits > 限额 {UB_LIMIT_BITS} bits"
            f"（K={K}, I={I}）。请调小 BLK_K_CAP/BLK_N_CAP。")
    _gmm_v2_kernel[(E, ceil_div(M, blk_m))](
        x, w, ws, xs, sm, starts, ends, y, y_scale,
        K, N, I,
        BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
        HAS_SMOOTH_COL=has_col, HAS_SMOOTH_ROW=has_row,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return y, y_scale
