"""dequant_swiglu_quant —— AscendC → Triton

来源：csrc/ascend/moe/dequant_swiglu_quant/
接口（op_host/dequant_swiglu_quant_def.cpp L30-101 + infershape L44-69）：
    Input : x                [M, 2H]  (int32 / bf16 / fp16)
            weight_scale     [2H] 或 [G, 2H]   fp32
            activation_scale [M] 或 [G, M]     fp32   （逐行缩放）
            bias             [2H] 或 [G, 2H]   fp32/fp16/bf16/int32
            quant_scale      [H]  或 [G, H]    fp32   （逐通道 smooth/quant scale）
            quant_offset     [H]  或 [G, H]    fp32
            group_index      [G] 或 [G,2]      int32/int64（MoE 每组 token 数）
    Output: y     [M, H]       int8（dst_type=2）
            scale [M]          fp32
    Attr  : activate_left=false, quant_mode="static", dst_type=2(int8),
            round_mode="rint", activate_dim=-1, swiglu_mode=0,
            clamp_limit=0.0, glu_alpha=1.0, glu_bias=0.0
  切分轴 = activate_dim（默认最后一维），Y 在该轴上长度减半（infershape L60-66）。

公式（op_kernel/dequant_swiglu_quant.h）：
  反量化（L400-427，仅当 x 为 int32）：
      xf = fp32(x)                                    # L401
      xf = weight_scale * xf                          # L405（逐通道广播；多组时按组取行，见下）
      xf = activation_scale * xf                      # L413（逐行广播，若提供）
      xf = xf + bias                                  # L424（浮点 bias；int32 bias 在 L386-390 先加）
  SwiGLU（L432-472，swiglu_mode=0 普通路径）：
      actOffset_ = actRight * H, gateOffset_ = H - actOffset_     # L182-183
      activate_left=true  -> act = xf[:, :H], gate = xf[:, H:]    # tiling_base L155：false<->激活右侧
      activate_left=false -> act = xf[:, H:], gate = xf[:, :H]
      y = gate * (act / (1 + exp(-act)))              # L459-472（glu_alpha=1 时即标准 SiLU）
  SwiGLU-OAI（L630-674，swiglu_mode=1，即 swigluoai_uninterleave）：
      if clamp_limit > 0:  gate = min(max(gate, -c), c)          # L650-656（gate 双侧 clamp）
      gate = gate + glu_bias                                     # L657（**无条件**加）
      if clamp_limit > 0:  act  = min(act, c)                    # L659-663（act **只**有上限）
      y = gate * (act / (1 + exp(-glu_alpha * act)))              # L664-672
  动态量化 DynamicQuant（L677-737，quant_mode="dynamic"）：
      if smooth: y = y * quant_scale[H]               # L681-694
      rowmax = max_d |y|                              # L696-711
      scale  = rowmax * (1/127)                       # L714（DYNAMIC_QUANT_FACTOR，L33）
      yq     = RINT(y / scale) -> int8                # L732-735 + L789-800（CAST_RINT=就近偶数）
  MoE 多组（op_kernel/dequant_swiglu_quant_cut_group.h L38-58 + .h L224-238）：
      group_index[g] = 「第 g 组的 token 数」（**不是**前缀和；调用点见
      vllm_fl/.../moe_mlp.py L248 cumsum_group_list(group_list, group_list_type, 1)，
      dst_list_type=1 即 per-group 计数，见 moe_mlp.py L54-83）
      组 g 覆盖输出行 [Σ_{k<g} count_k, Σ_{k≤g} count_k)，取 weight_scale[g] 那一行（L226-234）
      speGroupType=1（group_index 为 [G,2]）时：列 0 = 真实组号，列 1 = 该组 token 数（L44-47）

Triton 实现说明 / 语义假设：
  1. 覆盖：x=int32 + quant_mode="dynamic"(或 1) + dst_type=int8 + activate_dim=-1 +
     swiglu_mode∈{0,1} + group_index∈{None, [G] 计数, [G,2](组号,计数)} + activate_left∈{T,F}。
     未实现：static 量化、bf16/fp16 直通路径（未验证故不复现）、int4/fp8 输出、
     int32 bias 前置加法、[G,M] 形状的 activation_scale、activate_dim≠最后一维。
  2. 取整用「就近偶数」（CAST_RINT），与 torch.round 一致。
  3. rowmax==0 的行 scale 记 0（AscendC 该退化情形为 0/0，测试数据回避）。
  4. 组号映射用独立小 kernel 逐个累加得到（避免 tl.sum(tl.where(...)) 从 int32 向量取标量
     的 Ascend 后端坑）。

后端形态说明（910_93 + BiShengIR/FlagTree；探针：`_probe_2dstore_dsq.py`）：
  本文件两个 kernel 的 store 都是「**标量行号 + 1-D 列 tile + 1-D 掩码**」
  （`row = tl.program_id(0)`；`mask = hm = h < H`），形态上不属于
  「[BLK_D,BLK_C] 2-D block + 2-D 掩码 store」。sentinel 双填充实测
  H ∈ {17,24,33,48,100,127,129,256,1000,7168}（含 activate_left / swiglu_mode1）
  全部 dropped=0；端到端与 fp64 参考逐元素一致（>1LSB 数=0）、同输入跑两次输出 0 差异。
  ⇒ 本文件**不需要改写**。
"""

from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["dequant_swiglu_quant"]

DYNAMIC_QUANT_FACTOR = 1.0 / 127.0


@triton.jit
def _dsq_group_ids_kernel(
    GROUP_INDEX, GID,
    M, G,
    GROUP_STRIDE: tl.constexpr,
    BLK: tl.constexpr,
):
    """把「每组 token 数」展开成每行的组号（组号 = 组在 weight_scale 中的行号）。"""
    pid = tl.program_id(0)
    rows = pid * BLK + tl.arange(0, BLK)
    gid = tl.zeros([BLK], dtype=tl.int32)
    off = 0
    for g in range(G):
        if GROUP_STRIDE == 2:
            c = tl.load(GROUP_INDEX + g * 2 + 1).to(tl.int32)   # [G,2]: 列1 = token 数
            gv = tl.load(GROUP_INDEX + g * 2).to(tl.int32)      # 列0 = 真实组号
        else:
            c = tl.load(GROUP_INDEX + g).to(tl.int32)
            gv = tl.full([], g, tl.int32)
        gid = tl.where((rows >= off) & (rows < off + c), gv, gid)
        off += c
    tl.store(GID + rows, gid, mask=rows < M)


@triton.jit
def _dsq_swiglu_kernel(
    X, WSCALE, ASCALE, BIAS, QSCALE, GID, YBUF,
    M, H,
    CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS,
    IS_INT32: tl.constexpr,
    HAS_ASCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_QSCALE: tl.constexpr,
    HAS_GROUP: tl.constexpr,
    ACTIVATE_LEFT: tl.constexpr,
    SWIGLU_MODE1: tl.constexpr,
    BLK_H: tl.constexpr,
    BLK_M: tl.constexpr,
):
    # ⚠️ grid 第 0 维上限 65535（实测：M=65536 时
    #    `KernelLaunch failed ... value 65536 for parameter coreDim is invalid`，
    #    且 M=65537 又能过 —— 非单调，故不能靠"低于上限"推理豁免，必须按 BLK_M 分块）。
    #    这里一个 program 处理 BLK_M 行，grid = ceil_div(M, BLK_M)。
    pid = tl.program_id(0)
    dh = tl.arange(0, BLK_H)
    nloop = tl.cdiv(H, BLK_H)
    for mi in range(BLK_M):
        row = pid * BLK_M + mi
        if row < M:
            g = 0
            if HAS_GROUP:
                g = tl.load(GID + row)
            wbase = g * 2 * H
            for i in range(nloop):
                h = i * BLK_H + dh
                hm = h < H
                if ACTIVATE_LEFT:
                    a = tl.load(X + row * 2 * H + h, mask=hm, other=0.0).to(tl.float32)
                    gg = tl.load(X + row * 2 * H + H + h, mask=hm, other=0.0).to(tl.float32)
                else:
                    gg = tl.load(X + row * 2 * H + h, mask=hm, other=0.0).to(tl.float32)
                    a = tl.load(X + row * 2 * H + H + h, mask=hm, other=0.0).to(tl.float32)
                if IS_INT32:
                    if HAS_GROUP:
                        ws_a = tl.load(WSCALE + wbase + h, mask=hm, other=0.0).to(tl.float32)
                        ws_g = tl.load(WSCALE + wbase + H + h, mask=hm, other=0.0).to(tl.float32)
                    else:
                        if ACTIVATE_LEFT:
                            ws_a = tl.load(WSCALE + h, mask=hm, other=0.0).to(tl.float32)
                            ws_g = tl.load(WSCALE + H + h, mask=hm, other=0.0).to(tl.float32)
                        else:
                            ws_g = tl.load(WSCALE + h, mask=hm, other=0.0).to(tl.float32)
                            ws_a = tl.load(WSCALE + H + h, mask=hm, other=0.0).to(tl.float32)
                    a = a * ws_a
                    gg = gg * ws_g
                if HAS_ASCALE:
                    asc = tl.load(ASCALE + row).to(tl.float32)
                    a = a * asc
                    gg = gg * asc
                if HAS_BIAS:
                    if HAS_GROUP:
                        ba = tl.load(BIAS + wbase + h, mask=hm, other=0.0).to(tl.float32)
                        bg = tl.load(BIAS + wbase + H + h, mask=hm, other=0.0).to(tl.float32)
                    elif ACTIVATE_LEFT:
                        ba = tl.load(BIAS + h, mask=hm, other=0.0).to(tl.float32)
                        bg = tl.load(BIAS + H + h, mask=hm, other=0.0).to(tl.float32)
                    else:
                        bg = tl.load(BIAS + h, mask=hm, other=0.0).to(tl.float32)
                        ba = tl.load(BIAS + H + h, mask=hm, other=0.0).to(tl.float32)
                    a = a + ba
                    gg = gg + bg
                if SWIGLU_MODE1:
                    # L650-663：gate 双侧 clamp 后加 glu_bias；act 只做上限 clamp
                    gg = tl.where(CLAMP_LIMIT > 0.0, tl.minimum(gg, CLAMP_LIMIT), gg)
                    gg = tl.where(CLAMP_LIMIT > 0.0, tl.maximum(gg, -CLAMP_LIMIT), gg)
                    gg = gg + GLU_BIAS
                    a = tl.where(CLAMP_LIMIT > 0.0, tl.minimum(a, CLAMP_LIMIT), a)
                    y = gg * (a / (1.0 + tl.exp(0.0 - GLU_ALPHA * a)))       # L664-672
                else:
                    y = gg * (a / (1.0 + tl.exp(0.0 - a)))                   # L459-472
                if HAS_QSCALE:
                    qs = tl.load(QSCALE + h, mask=hm, other=1.0).to(tl.float32)
                    y = y * qs
                tl.store(YBUF + row * H + h, tl.where(hm, y, 0.0), mask=hm)


@triton.jit
def _dsq_quant_kernel(
    YBUF, YOUT, SCALE,
    M, H, QFACTOR,
    BLK_H: tl.constexpr,
    BLK_M: tl.constexpr,
):
    """动态量化：scale = rowmax/127，yq = RINT(y/scale)。

    与 `_dsq_swiglu_kernel` 同样按 BLK_M 分块（grid 第 0 维上限 65535）。
    """
    pid = tl.program_id(0)
    dh = tl.arange(0, BLK_H)
    nloop = tl.cdiv(H, BLK_H)
    for mi in range(BLK_M):
        row = pid * BLK_M + mi
        if row < M:
            mx = tl.zeros([1], dtype=tl.float32)
            for i in range(nloop):
                h = i * BLK_H + dh
                hm = h < H
                v = tl.load(YBUF + row * H + h, mask=hm, other=0.0)
                mx = tl.maximum(mx, tl.expand_dims(tl.max(tl.abs(v), axis=0), 0))
            sc = tl.sum(mx) * QFACTOR
            sc = tl.where(sc > 0.0, sc, 0.0)
            tl.store(SCALE + row, sc)
            for i in range(nloop):
                h = i * BLK_H + dh
                hm = h < H
                v = tl.load(YBUF + row * H + h, mask=hm, other=0.0)
                v = tl.where(sc > 0.0, v / sc, 0.0)
                f = tl.floor(v)
                fr = v - f
                od = (f * 0.5) != tl.floor(f * 0.5)          # RINT：就近偶数
                q = tl.where((fr > 0.5) | ((fr == 0.5) & od), f + 1.0, f)
                tl.store(YOUT + row * H + h, q.to(tl.int8), mask=hm)


def _is_dynamic(quant_mode) -> bool:
    if isinstance(quant_mode, str):
        return quant_mode.lower() == "dynamic"
    return int(quant_mode) == 1


def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_offset: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    activate_left: bool = False,
    quant_mode="dynamic",
    dst_type: torch.dtype = torch.int8,
    round_mode: str = "rint",
    activate_dim: int = -1,
    swiglu_mode: int = 0,
    clamp_limit: float | None = 0.0,
    glu_alpha: float | None = 1.0,
    glu_bias: float | None = 0.0,
):
    """等价于 AscendC 的 DequantSwigluQuant（int32 + dynamic + int8 主路径）。

    Args:
        x: [M, 2H] int32
        weight_scale: [2H]（单组）或 [G, 2H]（多组，与 group_index 长度一致）fp32
        activation_scale: [M] fp32 可选（逐行/pertoken 缩放）
        bias: [2H] 或 [G, 2H] 可选（浮点）
        quant_scale: [H] fp32 可选（逐通道 smooth scale，dynamic 模式）
        quant_offset: dynamic 模式不使用
        group_index: None / [G] int32|int64（每组 token 数）/ [G,2]（列0=组号, 列1=token 数）
        activate_left: True -> 激活作用在左半（生产调用点用 True）
        quant_mode: "dynamic"/1（"static"/0 未实现）
        dst_type: 仅 torch.int8
        swiglu_mode: 0=标准 SiLU SwiGLU；1=SwiGLU-OAI（clamp/glu_alpha/glu_bias 生效）
        clamp_limit / glu_alpha / glu_bias: None 视为默认 0.0 / 1.0 / 0.0

    Returns:
        (y [M, H] int8, scale [M] fp32)
    """
    assert _is_dynamic(quant_mode), "仅实现 quant_mode='dynamic'(1)"
    assert dst_type == torch.int8, "仅实现 dst_type=int8"
    assert swiglu_mode in (0, 1), "swiglu_mode 只支持 0/1"
    assert x.dim() == 2, f"x 应为 [M,2H]，实际 {tuple(x.shape)}"
    if activate_dim not in (-1, 1):
        raise NotImplementedError("仅实现 activate_dim=-1（最后一维对半切）")
    c_lim = 0.0 if clamp_limit is None else float(clamp_limit)
    g_alpha = 1.0 if glu_alpha is None else float(glu_alpha)
    g_bias = 0.0 if glu_bias is None else float(glu_bias)
    if swiglu_mode == 0:
        assert c_lim == 0.0 and g_alpha == 1.0 and g_bias == 0.0, \
            "clamp/glu_alpha/glu_bias 只在 swiglu_mode=1 生效（AscendC 普通路径不含它们）"

    M, two_h = x.shape
    assert two_h % 2 == 0, "最后一维必须为偶数"
    H = two_h // 2
    assert x.dtype == torch.int32, "本实现只覆盖 x=int32 的反量化主路径"
    assert bias is None or bias.dtype != torch.int32, "int32 bias 前置加法未实现"

    has_group = group_index is not None
    if has_group:
        assert group_index.dim() in (1, 2), "group_index 应为 [G] 或 [G,2]"
        n_groups = group_index.shape[0]
        group_stride = 2 if group_index.dim() == 2 else 1
        assert weight_scale.numel() == n_groups * two_h, \
            f"多组时 weight_scale 应为 [{n_groups},{two_h}]，实际 {tuple(weight_scale.shape)}"
    else:
        group_stride = 1
        assert weight_scale.numel() == two_h, f"weight_scale 应为 [{two_h}]"
    for nm, t, n in (("activation_scale", activation_scale, M),
                     ("quant_scale", quant_scale, H)):
        if t is not None:
            assert t.numel() == n, f"{nm} 长度 {t.numel()} != {n}"

    dev = x.device
    x = x.contiguous()
    # ⚠️ 用 zeros 而不是 empty：`_dsq_quant_kernel` 无条件读 YBUF，一旦上游
    # `_dsq_swiglu_kernel` 因任何原因没写满（launch 失败 / 越界），empty 会
    # **静默**把未初始化内存算进结果（实测：上游 launch 失败后下游整片变成 0/垃圾，
    # 不报错）。zeros 让这种情况表现为确定的 0，且成本仅 M×H×4 字节。
    ybuf = torch.zeros((M, H), dtype=torch.float32, device=dev)
    y = torch.empty((M, H), dtype=torch.int8, device=dev)
    scale = torch.empty((M,), dtype=torch.float32, device=dev)
    gid = torch.zeros((M,), dtype=torch.int32, device=dev)
    if has_group:
        blk = 256
        _dsq_group_ids_kernel[(ceil_div(M, blk),)](
            group_index.contiguous(), gid, M, n_groups,
            GROUP_STRIDE=group_stride, BLK=blk,
        )

    blk_h = 128 if H > 128 else next_pow2(H)
    # ⚠️ grid 第 0 维上限 65535（实测 M=65536 → `KernelLaunch failed ... coreDim is invalid`，
    #    而 M=65537 又能过：**非单调**，不能靠"低于上限"推理豁免）。一个 program 处理
    #    BLK_M 行 ⇒ grid = ceil_div(M, BLK_M)，M 上限提升到 65535*BLK_M。
    blk_m = max(1, int(_os.environ.get("DSQ_BLK_M", "8")))
    n_row_blocks = ceil_div(M, blk_m)
    assert n_row_blocks <= 65535, (
        f"M={M} 需要 grid={n_row_blocks} > 65535（coreDim 上限）；"
        f"请增大 DSQ_BLK_M（当前 {blk_m}）")
    _dsq_swiglu_kernel[(n_row_blocks,)](
        x,
        weight_scale.contiguous(),
        activation_scale.contiguous() if activation_scale is not None else x,
        bias.contiguous() if bias is not None else x,
        quant_scale.contiguous() if quant_scale is not None else x,
        gid,
        ybuf,
        M, H,
        c_lim, g_alpha, g_bias,
        IS_INT32=True,
        HAS_ASCALE=activation_scale is not None,
        HAS_BIAS=bias is not None,
        HAS_QSCALE=quant_scale is not None,
        HAS_GROUP=has_group,
        ACTIVATE_LEFT=bool(activate_left),
        SWIGLU_MODE1=(swiglu_mode == 1),
        BLK_H=blk_h,
        BLK_M=blk_m,
    )
    _dsq_quant_kernel[(n_row_blocks,)](ybuf, y, scale, M, H, DYNAMIC_QUANT_FACTOR,
                                       BLK_H=blk_h, BLK_M=blk_m)
    return y, scale
