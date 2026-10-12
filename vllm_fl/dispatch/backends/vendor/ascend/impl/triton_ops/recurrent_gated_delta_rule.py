"""recurrent_gated_delta_rule —— AscendC → Triton

来源：csrc/ascend/attention/recurrent_gated_delta_rule/
接口（op_host/recurrent_gated_delta_rule_def.cpp L21-81 + _infershape.cpp L38-81）：
    Input : query              [T, NK, DK]  bf16
            key                [T, NK, DK]  bf16
            value              [T, NV, DV]  bf16
            beta               [T, NV]      bf16
            state              [S, NV, DV, DK]  bf16 / fp32     （S = 状态槽数）
            actual_seq_lengths [B+1]        int32（累加前缀和 cu_seqlens）
            ssm_state_indices  [T]          int32（每个 token 对应的 state 槽号）
            g                  [T, NV]      fp32（OPTIONAL，log 空间）
            gk                 [T, NV, DK]  fp32（OPTIONAL，log 空间，k 方向门控）
            num_accepted_tokens[B]          int32（OPTIONAL）
    Output: out                [T, NV, DV]  同 value dtype
            state              [S, NV, DV, DK]  同输入 state dtype
    Attr  : scale_value = 1.0

公式（op_kernel/recurrent_gated_delta_rule.h）：
    每 (batch, v-head, v 分块) 独立递推，state 记为 S[DV, DK]：
      gama  = exp(g[t, h])                    # L437 Exp(gamaInUb)；L477 gama_
      gammaK= exp(gk[t, h, :])                # L226 Exp(gamaKInUb)
      S    *= gama                            # L366-368 Muls(stateInUb, stateInUb, gama_)
      S    *= gammaK                          # L369-371 MatVecMul(stateInUb, gamaKInUb, ..., isAdd=false)
      d     = v[t,h,:] - S @ k[t, hk, :]      # L375  MatVecMul + L377 ReduceSumDispatch + L379 Sub
      d    *= beta[t, h]                      # L381 Muls(delta, delta, beta_)
      S    += outer(d, k)                     # L383 Broadcast + L385 MatVecMul(..., isAdd=true)
      o     = S @ (q[t, hk, :] * scale)       # L387 + L389 + L242 Muls(q, q, scale_)
      out[t, h, :] = o                        # L398/L402-408 Cast(CAST_RINT) + DataCopyPad
      final_state[ssm_state_indices[t], h] = S     # L476 + L410-417，每个 token 都写

    - hk = h / (NV / NK)                        # L445 qkOffset 用 head_i/(NV_/NK_)
    - 初始状态槽：stateTokenIdx = seq0；若给 num_accepted_tokens 则 = seq0 + n - 1（L181-188）
    - g/gk 载入后立刻 Exp（L437 / L226），即输入是 log 空间
    - 所有中间计算都在 fp32（Cast 到 fp32 后运算），最后 CAST_RINT 回输出 dtype

实现：每个 (batch, v-head, v 分块) 一个 program，program 内按 token 顺序串行递推。
（AscendC 用 IsCurrentBlock 把 (batch,head) 分给不同核，但每块只由一个核计算，结果等价。）

已知限制（Ascend bishengir 编译器缺陷，非算法问题）：
    当 state 为 fp32（→ final 也是 fp32）时，bishengir 在 ConvertLinalgRToBinary 阶段 LLVM crash
    （"PLEASE submit a bug report to llvm-project ... crash backtrace"），0.5s 内失败、5/5 可复现，
    独立最小进程亦可复现，与负载/OOM 无关（MemAvailable 1.8TB）。
    已定位触发点 = **动态循环体内对循环携带的 2-D state 做 2-D tl.store**
    （删掉/移出这条 store，其余代码不动即可编过；见 _analysis/triton_ops/后端崩溃最小复现.md）
    已试过但**无效**的规避：去掉 no-op cast、FINAL 单独 constexpr 分支、统一 .to(dtype)、
    2-D 地址 pointer hoist、把 store 改成 1-D 展平（值 tl.reshape + 算术偏移）、
    减小 tile（BLK_V=8/16/32）、num_warps/num_stages 各值、重试。
    state=bf16/fp16（op_host def 的主用组合）一切正常，8/8 + 调用点风格 1/1 全 PASS。
    同一个 1-D store 改法**在 chunk_gated_delta_rule_fwd_h 上已生效**（该算子 fp32 路径已可用），
    说明 recurrent 还存在第二个未定位的触发点。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["recurrent_gated_delta_rule"]


@triton.jit
def _rgdr_kernel(
    Q, KEY, VAL, BETA, GAMA, GAMAK, STATE, CU, SSM, OUT, FINAL, NACC,
    T, NK, NV, DK, DV, SCALE,
    HAS_G: tl.constexpr,
    HAS_GK: tl.constexpr,
    HAS_ACC: tl.constexpr,
    FINAL_F32: tl.constexpr,
    STATE_F32: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_V: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq0 = tl.load(CU + pid_b).to(tl.int32)
    seq1 = tl.load(CU + pid_b + 1).to(tl.int32)
    if seq1 <= seq0:
        return

    if HAS_ACC:
        nacc = tl.load(NACC + pid_b).to(tl.int32)
        state_tok = seq0 + nacc - 1
    else:
        state_tok = seq0
    slot = tl.load(SSM + state_tok).to(tl.int32)

    k_off = tl.arange(0, BLK_K)
    kmask = k_off < DK
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < DV

    hk = pid_h // (NV // NK)

    sptr = STATE + ((slot * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :]
    # 统一走 .to(tl.float32)（fp32 时为 no-op，但保留 convert op —— 去掉这个 constexpr 分支后
    # 与已修好的 chunk 内核 IR 形态一致，bishengir 不再崩）
    sm = tl.load(sptr, mask=vmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)

    # workaround（Ascend bishengir bug）：循环内「2-D tl.store」会让 bishengir 在
    # ConvertLinalgRToBinary 阶段 LLVM crash（fp32 值无 dtype 转换时 5/5 必崩，bf16 不崩）。
    # 把 state 落盘改成**等价的 1-D store**（值用 tl.reshape 展平，偏移用算术式重算；
    # 注意不能 reshape 由广播算出来的偏移张量，那样会错位）。详见 后端崩溃最小复现.md
    i1 = tl.arange(0, BLK_V * BLK_K)
    row_i = i1 // BLK_K                     # tile 内行号（= v 方向）
    col_i = i1 % BLK_K                      # tile 内列号（= k 方向）
    vglob = pid_v * BLK_V + row_i
    off1 = vglob * DK + col_i
    fm1 = (vglob < DV) & (col_i < DK)

    t = seq0
    while t < seq1:
        q = tl.load(Q + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(KEY + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32)
        vv = tl.load(VAL + (t * NV + pid_h) * DV + v_off, mask=vmask, other=0.0).to(tl.float32)
        bb = tl.load(BETA + t * NV + pid_h).to(tl.float32)

        if HAS_G:
            gm = tl.exp(tl.load(GAMA + t * NV + pid_h).to(tl.float32))
            sm = sm * gm
        if HAS_GK:
            gk = tl.exp(tl.load(GAMAK + (t * NV + pid_h) * DK + k_off, mask=kmask, other=0.0).to(tl.float32))
            sm = sm * gk[None, :]

        d = vv - tl.sum(sm * kk[None, :], axis=1)
        d = d * bb
        sm = sm + d[:, None] * kk[None, :]

        o = tl.sum(sm * q[None, :], axis=1)
        tl.store(OUT + (t * NV + pid_h) * DV + v_off, o.to(OUT.dtype.element_ty), mask=vmask)

        slot_t = tl.load(SSM + t).to(tl.int32)
        fbase = (slot_t * NV + pid_h) * DV * DK
        tl.store(FINAL + fbase + off1,
                 tl.reshape(sm.to(FINAL.dtype.element_ty), [BLK_V * BLK_K]), mask=fm1)
        t += 1


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    scale_value: float = 1.0,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """等价于 AscendC 的 RecurrentGatedDeltaRule（decoding 阶段的 gated delta rule）。

    参数顺序与 op_host/recurrent_gated_delta_rule_def.cpp 一致
    （query, key, value, beta, state, actual_seq_lengths, ssm_state_indices, g, gk, num_accepted_tokens）。
    另接受 `scale=` 作 `scale_value` 的别名：调用点 vllm_ascend/ops/gdn.py L556-567 / L581-593
    用 `scale=key.shape[-1]**-0.5` 传参，而该值对应 op_host 的 attr `scale_value`（L81）。

    Args:
        query/key: [T, NK, DK] bf16
        value:     [T, NV, DV] bf16
        beta:      [T, NV] bf16
        state:     [S, NV, DV, DK] —— **仅支持 bf16/fp16**；fp32 会显式抛 NotImplementedError
                   （原因见文末"已知限制"）
        actual_seq_lengths: [B+1] int32 累加序列长度
        ssm_state_indices:  [T] int32
        g:   [T, NV] fp32（log 空间），None 表示不做衰减
        gk:  [T, NV, DK] fp32（log 空间），None 表示不做 k 方向门控
        num_accepted_tokens: [B] int32，给定则初始状态取 seq0+n-1 槽
        scale_value: 标量属性（也可用关键字 scale 传入）

    Returns:
        (out [T, NV, DV] = value.dtype, final_state [S, NV, DV, DK] = state.dtype)

    state dtype 支持矩阵（显式声明，不静默降级）：
        bf16 / fp16 → ✅ 可用（8/8 精度用例 + 1/1 调用点风格用例 PASS）
        fp32        → ❌ 抛 NotImplementedError（Ascend 后端编译器缺陷，非算法问题；
                        最小复现与定位见 _analysis/triton_ops/后端崩溃最小复现.md）
    """
    if scale is not None:
        scale_value = scale

    # ---- 已声明不支持的特化：state=fp32（→ final 也随之为 fp32）。显式报错，绝不静默算错 ----
    if state.dtype == torch.float32:
        raise NotImplementedError(
            "state=fp32（final 也随之为 fp32）**不支持**：受 Ascend 后端编译器缺陷阻塞——"
            "bishengir 在 ConvertLinalgRToBinary 阶段对 all-fp32 特化发生 LLVM crash"
            "（0.5s 内失败、5/5 可复现，与负载/OOM 无关）。最小复现、二分定位与 workaround 见 "
            "_analysis/triton_ops/后端崩溃最小复现.md；同一 workaround 已在 "
            "chunk_gated_delta_rule_fwd_h 上生效，但本算子仍崩（存在第二个未定位触发点）。"
            "请改用 bf16/fp16 的 state：op_host def 同时支持 DT_BF16/DT_FLOAT，"
            "bf16 路径 8/8 精度用例 + 1/1 调用点风格用例全部 PASS。")
    if state.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"state dtype {state.dtype} 不支持（仅 bf16/fp16）")
    assert query.shape == key.shape, f"query/key 形状不一致 {tuple(query.shape)} vs {tuple(key.shape)}"
    T, NK, DK = (int(s) for s in query.shape)
    NV = int(value.shape[1])
    DV = int(value.shape[2])
    assert NV % NK == 0, f"NV={NV} 必须是 NK={NK} 的整数倍"
    S = int(state.shape[0])
    assert tuple(state.shape) == (S, NV, DV, DK), f"state shape {tuple(state.shape)} 与 (S,NV,DV,DK) 不符"
    assert tuple(beta.shape) == (T, NV), f"beta shape {tuple(beta.shape)} != {(T, NV)}"

    cu = actual_seq_lengths.contiguous().to(torch.int32)
    B = int(cu.numel()) - 1
    ssm = ssm_state_indices.contiguous().to(torch.int32)

    out = torch.empty((T, NV, DV), dtype=value.dtype, device=value.device)
    final = state.contiguous().clone()

    BLK_K = next_pow2(DK)
    BLK_V = min(next_pow2(DV), 64)
    grid = (NV, ceil_div(DV, BLK_V), max(B, 1))
    _rgdr_kernel[grid](
        query.contiguous(), key.contiguous(), value.contiguous(), beta.contiguous(),
        (g if g is not None else query).contiguous(),
        (gk if gk is not None else query).contiguous(),
        state.contiguous(), cu, ssm, out, final,
        (num_accepted_tokens if num_accepted_tokens is not None else cu).contiguous(),
        T, NK, NV, DK, DV, float(scale_value),
        HAS_G=bool(g is not None),
        HAS_GK=bool(gk is not None),
        HAS_ACC=bool(num_accepted_tokens is not None),
        FINAL_F32=bool(final.dtype == torch.float32),
        STATE_F32=bool(state.dtype == torch.float32),
        BLK_K=BLK_K,
        BLK_V=BLK_V,
    )
    return out, final
