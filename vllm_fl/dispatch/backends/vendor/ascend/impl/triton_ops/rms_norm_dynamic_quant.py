"""rms_norm_dynamic_quant —— AscendC → Triton

来源：csrc/ascend/attention/rms_norm_dynamic_quant/
接口（op_host/rms_norm_dynamic_quant_def.cpp L22-78）：
    Input : x            [numTokens, D]  fp16/bf16
            gamma        [D]             fp16/bf16
            smooth_scale1[D]             fp16/bf16  (OPTIONAL)
            smooth_scale2[D]             fp16/bf16  (OPTIONAL)
            beta         [D]             fp16/bf16  (OPTIONAL)
    Output: y1     [numTokens, D]  int8 / int4
            y2     [numTokens, D]  int8 / int4
            scale1 [numTokens]     fp32
            scale2 [numTokens]     fp32
    Attr  : epsilon = 1e-6, output_mask = ListBool, dst_type = DT_INT8

公式（op_kernel/rms_norm_dynamic_quant_normal_kernel.h）：
  ComputeRmsNorm（L129-174）：
      xf        = fp32(x)                                    # L134
      squareSum = sum_{d<D} xf^2                             # L143 ReduceSumHalfInterval
      rstd      = 1 / sqrt(squareSum * avgFactor + eps)      # L144，avgFactor = 1/D（tiling.cpp L309）
      t         = xf * rstd                                  # L148-150
      t         = t * fp32(gamma)                            # L154-161
      if beta: t = t + fp32(beta)                            # L163-172
  doQuant1withFlag（L187-229）/ doQuant2withFlag（L231-272）：
      z        = t * (smooth_k 若存在 否则 1.0)              # L197-215 / L242-258
      maxAbs   = max_{d<D} |z|                               # L301-308
      scaleTmp = quantMaxVal / maxAbs                        # L309（int8: 127.0, int4: 7.0，helper.h L28-29）
      outScale = 1 / scaleTmp  = maxAbs / quantMaxVal        # L310  ← 存的是反量化 scale
      zq       = RINT(z * scaleTmp)                          # L216-218（CAST_RINT = 就近偶数）
      y        = cast_int8(zq)                               # L220-226 经 int32→half→int8(TRUNC)，整数故无损
  使能关系（base.h L69-72 + tiling.cpp L216-222）：
      output_mask 为空 -> outQuant1Flag=outQuant2Flag=-1（isOld），y1 必算；
                          y2 仅当 smooth1 与 smooth2 都存在时计算（oldDouble）
      output_mask=[b1,b2] -> flag_i = 1(b_i)/0，flag_i==1 才计算并写 y_i

Triton 实现说明 / 语义假设：
  1. 只支持 dst=int8（quantMaxVal=127）；int4 分支未实现（无 int4 张量类型）。
  2. 全零行：AscendC 会得到 scaleTmp=inf -> NaN；这里若 maxAbs==0 则取 0（安全化处理）。
  3. 三个内核模板（normal / single_row / cut_d）只切分方式不同，数学一致
     （cut_d 见 rms_norm_dynamic_quant_cut_d_kernel.h L80/L93/L255/L364，公式相同）。
  4. 未使能的输出返回 None（AscendC 中对应 GM 不写，内容未定义）。
  5. Ascend UB 只有 192KB，逐行+分块（3 趟：平方和 / 行内最大值 / 写回）降低寄存器压力。

后端形态说明（910_93 + BiShengIR/FlagTree；探针：`_probe_2dstore_rmsdq.py`）：
  本 kernel 的 2-D 掩码 store（L148/L160）经 **sentinel 双填充**实测**没有**静默丢写：
  D ∈ {17,24,33,48,100,127,128,129,255,256,1000,1024,7168} 全部 dropped=0 / wrong=0，
  且「2-D 原形」与「1-D 改写形（标量行号 + 1-D 列 tile）」逐元素完全一致（scale_err ≤ 1e-8）；
  生产算子端到端与 fp64 参考 0 差异、同输入跑两次输出一致。故本文件**不做改写**。
  ⚠️ 但不要用 `tl.reshape` 把 2-D tile 拉平成 1-D 再 store —— 同后端实测
  cols=1024（无任何掩码）时仍丢 896/1024 个 lane（见 `_probe_2dstore_form.py`）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["rms_norm_dynamic_quant", "DYNAMIC_QUANT_DIVIDEND"]

DYNAMIC_QUANT_DIVIDEND = 127.0


@triton.jit
def _rms_norm_dynamic_quant_kernel(
    X, GAMMA, SMOOTH1, SMOOTH2, BETA,
    Y1, Y2, SCALE1, SCALE2,
    ROWS, D,
    EPS, INV_D, QMAX,
    BLK_R: tl.constexpr,
    BLK_D: tl.constexpr,
    HAS_SMOOTH1: tl.constexpr,
    HAS_SMOOTH2: tl.constexpr,
    HAS_BETA: tl.constexpr,
    DO_Q1: tl.constexpr,
    DO_Q2: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS
    dr = tl.arange(0, BLK_D)
    nloop = tl.cdiv(D, BLK_D)

    # 趟 1：行内平方和
    ss = tl.zeros([BLK_R], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_D + dr
        m2 = rmask[:, None] & (d < D)[None, :]
        x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    rstd = 1.0 / tl.sqrt(ss * INV_D + EPS)

    # 趟 2：|z_k| 的行内最大值
    m1 = tl.zeros([BLK_R], dtype=tl.float32)
    mv2 = tl.zeros([BLK_R], dtype=tl.float32)
    if DO_Q1:
        for i in range(nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
            t = x * rstd[:, None] * g[None, :]
            if HAS_BETA:
                bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
                t = t + bt[None, :]
            if HAS_SMOOTH1:
                s1 = tl.load(SMOOTH1 + d, mask=d < D, other=1.0).to(tl.float32)
                z1 = t * s1[None, :]
            else:
                z1 = t
            m1 = tl.maximum(m1, tl.max(tl.where(m2, tl.abs(z1), 0.0), axis=1))
    if DO_Q2:
        for i in range(nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
            t = x * rstd[:, None] * g[None, :]
            if HAS_BETA:
                bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
                t = t + bt[None, :]
            if HAS_SMOOTH2:
                s2 = tl.load(SMOOTH2 + d, mask=d < D, other=1.0).to(tl.float32)
                z2 = t * s2[None, :]
            else:
                z2 = t
            mv2 = tl.maximum(mv2, tl.max(tl.where(m2, tl.abs(z2), 0.0), axis=1))

    # maxAbs==0 时用 1.0 占位（z 全为 0，量化结果仍是 0），避免 inf/NaN
    st1 = QMAX / tl.where(m1 > 0.0, m1, 1.0)
    st2 = QMAX / tl.where(mv2 > 0.0, mv2, 1.0)

    # 趟 3：量化并写回
    for i in range(nloop):
        d = i * BLK_D + dr
        m2 = rmask[:, None] & (d < D)[None, :]
        x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
        t = x * rstd[:, None] * g[None, :]
        if HAS_BETA:
            bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
            t = t + bt[None, :]
        if DO_Q1:
            if HAS_SMOOTH1:
                s1 = tl.load(SMOOTH1 + d, mask=d < D, other=1.0).to(tl.float32)
                z1 = t * s1[None, :]
            else:
                z1 = t
            v1 = z1 * st1[:, None]
            f1 = tl.floor(v1)
            fr1 = v1 - f1
            od1 = (f1 * 0.5) != tl.floor(f1 * 0.5)          # 就近偶数（CAST_RINT）
            q1 = tl.where((fr1 > 0.5) | ((fr1 == 0.5) & od1), f1 + 1.0, f1)
            tl.store(Y1 + rows[:, None] * D + d[None, :], q1.to(tl.int8), mask=m2)
        if DO_Q2:
            if HAS_SMOOTH2:
                s2 = tl.load(SMOOTH2 + d, mask=d < D, other=1.0).to(tl.float32)
                z2 = t * s2[None, :]
            else:
                z2 = t
            v2 = z2 * st2[:, None]
            f2 = tl.floor(v2)
            fr2 = v2 - f2
            od2 = (f2 * 0.5) != tl.floor(f2 * 0.5)
            q2 = tl.where((fr2 > 0.5) | ((fr2 == 0.5) & od2), f2 + 1.0, f2)
            tl.store(Y2 + rows[:, None] * D + d[None, :], q2.to(tl.int8), mask=m2)

    if DO_Q1:
        tl.store(SCALE1 + rows, tl.where(m1 > 0.0, m1 / QMAX, 0.0), mask=rmask)
    if DO_Q2:
        tl.store(SCALE2 + rows, tl.where(mv2 > 0.0, mv2 / QMAX, 0.0), mask=rmask)


def rms_norm_dynamic_quant(
    x: torch.Tensor,
    gamma: torch.Tensor,
    smooth_scale1: torch.Tensor | None = None,
    smooth_scale2: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    output_mask: list[bool] | None = None,
    dst_type: torch.dtype = torch.int8,
):
    """等价于 AscendC 的 RmsNormDynamicQuant（dst=int8）。

    Args:
        x: [numTokens, D] fp16/bf16
        gamma: [D] 同 x dtype
        smooth_scale1/smooth_scale2/beta: [D] 可选
        epsilon: 属性，默认 1e-6
        output_mask: 属性 ListBool；None 表示 AscendC 的「旧模式」（空 mask）
        dst_type: 目前仅支持 torch.int8

    Returns:
        (y1, y2, scale1, scale2)，未使能的输出为 None
    """
    assert x.dim() == 2, f"x 应为 [numTokens, D]，实际 {tuple(x.shape)}"
    assert dst_type == torch.int8, "仅实现 dst_type=int8（quantMaxVal=127）"
    rows, D = x.shape
    assert gamma.numel() == D, f"gamma 长度 {gamma.numel()} != D={D}"
    for nm, t in (("smooth_scale1", smooth_scale1), ("smooth_scale2", smooth_scale2), ("beta", beta)):
        if t is not None:
            assert t.numel() == D, f"{nm} 长度 {t.numel()} != D={D}"

    has1 = smooth_scale1 is not None
    has2 = smooth_scale2 is not None
    if output_mask is None:
        do1 = True
        do2 = has1 and has2          # oldDouble（base.h L70）
    else:
        do1 = bool(output_mask[0])
        do2 = bool(output_mask[1])

    dev = x.device
    x = x.contiguous()
    gamma = gamma.contiguous()
    y1 = torch.empty((rows, D), dtype=torch.int8, device=dev) if do1 else None
    y2 = torch.empty((rows, D), dtype=torch.int8, device=dev) if do2 else None
    s1 = torch.empty((rows,), dtype=torch.float32, device=dev) if do1 else None
    s2 = torch.empty((rows,), dtype=torch.float32, device=dev) if do2 else None
    if not do1 and not do2:
        return y1, y2, s1, s2

    # Ascend UB 仅 192KB：BLK_D 取小值（128）以保证编译通过；D 更大时按 chunk 循环
    blk_d = 128 if D > 128 else next_pow2(D)
    blk_r = 1
    _rms_norm_dynamic_quant_kernel[(ceil_div(rows, blk_r),)](
        x, gamma,
        smooth_scale1.contiguous() if has1 else x,
        smooth_scale2.contiguous() if has2 else x,
        beta.contiguous() if beta is not None else x,
        y1 if do1 else x, y2 if do2 else x,
        s1 if do1 else gamma, s2 if do2 else gamma,
        rows, D,
        float(epsilon), 1.0 / float(D), DYNAMIC_QUANT_DIVIDEND,
        BLK_R=blk_r, BLK_D=blk_d,
        HAS_SMOOTH1=has1, HAS_SMOOTH2=has2, HAS_BETA=beta is not None,
        DO_Q1=do1, DO_Q2=do2,
    )
    return y1, y2, s1, s2
