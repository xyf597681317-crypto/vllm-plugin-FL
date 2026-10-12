"""causal_conv1d —— AscendC → Triton

来源：csrc/ascend/moe/causal_conv1d/
接口（op_host/causal_conv1d_def.cpp L24-73 + tiling_utils.h L20-31）：
    Input : x              [T, dim]（varlen/prefill）或 [batch, seqlen, dim]（3D decode）
                          或 [batch, dim]（2D decode，seqlen=1）      fp16/bf16
            weight         [width, dim]            fp16/bf16
            bias           [dim]                   (OPTIONAL)
            convStates     [numCache, stateLen, dim]  fp16/bf16（in/out）
            queryStartLoc  [batch+1]               int32/int64 (OPTIONAL)
            cacheIndices   [batch]                 int32/int64 (OPTIONAL)
            initialStateMode [batch]               bool/int32/int64 (OPTIONAL)
            numAcceptedTokens [batch]              int32/int64 (OPTIONAL，投机解码)
    Output: y              [T, dim]（形状同 x）    同 x dtype
    Attr  : activationMode(0=none/1=silu), padSlotId=-1, runMode(0=fn/1=update)

**调用点参数名对齐**（`csrc/ascend/torch_binding.cpp` L1165-1169 的 schema，
调用点 `vllm_fl/dispatch/backends/vendor/ascend/impl/gdn.py` L199-303）：
    npu_causal_conv1d_custom(output, x, weight, conv_state, bias_opt, query_start_loc_opt,
                             cache_indices_opt, initial_state_mode_opt, num_accepted_tokens_opt,
                             activation_mode:int, pad_slot_id:int, run_mode:int) -> Tensor
  本实现同时接受调用点名（`*_opt` / `conv_state` / `activation_mode`）与 snake_case 名。

公式（op_kernel/causal_conv1d.h）：
  InitRing（L351-393）：ring 前 (width-1) 槽 = convStates[cacheIdx, stateTokenOffset+i, :]
      （hasInit 假则置 0，L376-381）；hasInit = initialStateMode[seq]（L945-947）
  RunSeq（L415-470）：对 t∈[0,len)：
      acc = bias                                                        # L429-433
      for j∈[MAX_WIDTH-width, MAX_WIDTH): tap = MAX_WIDTH-1-j           # L435-446
          slot = (tap==0) ? ring[当前 token] : ring[token(t-tap)]       # L437
          acc += fp32(ring[slot]) * weight[j]
      ⇒ y[t] = bias + Σ_{k=0..width-1} weight[k] * ext[t+k]
         ext = [旧状态(width-1 个历史 token), x[start..start+len)]
      activationMode!=0 -> Silu(acc)                                    # L450-452
  WriteBackState（L717-737）：state[pos] = ext[len+pos]，pos∈[0,width-2)
      （WriteBackStateSpec L740-830 是 width==4 且 stateLen 足够时的等价快速路径）
      cacheIdx == padSlotId 时不处理该槽（L933-936）
  任务窗口（L52-94 + tiling_validation.h L89-121）：
      inputMode=0 varlen：start=qsl[b], len=qsl[b+1]-qsl[b]（x 为 [cuSeqlen, dim]）
      inputMode=1 3D    ：start=b*seqLen, len=seqLen（x 为 [batch, seqlen, dim]）
      inputMode=2 2D    ：start=b, len=1（x 为 [batch, dim]，decode）
  投机解码（L1009-1018）：numAcceptedTokens[b] -> stateTokenOffset = clamp(nat-1, 0, stateLen-(width-1))
      即状态读取从该偏移开始（滑动窗口），写回仍是 ext[len+pos]

Triton 实现说明 / 语义假设：
  1. runMode=0/1 都支持；窗口模式由 x 的维度与 queryStartLoc 推出（与 tiling_validation 一致）。
  2. 累加 fp32（与内核 Cast 到 fp32 后乘加一致），最后按 T 就地 round-to-nearest。
  3. numAcceptedTokens 仅 width==4 时 AscendC 才允许（validation L362），本实现同样只在该条件下
     视为投机解码；否则忽略。
  4. 一个 program 负责一个 (batch, dim 分块)；沿 t 串行推进（因果卷积的固有依赖）。
  5. **约束**：同一次调用内各 batch 的 cacheIndices 必须指向互不相同的状态槽（生产路径满足；
     被 pad_slot_id 标记的槽会被跳过）。AscendC 逐 batch 串行，同一槽被两个活跃 batch 复用时
     后者的初态会读到前者写回的状态；Triton 侧 batch 并行，无法复现该串行副作用（会形成数据竞争）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]

PAD_SLOT_ID = -1


@triton.jit
def _causal_conv1d_kernel(
    X, W, BIAS, STATES, QSL, CACHE_IDX, INIT_MODE, NAT, Y,
    BATCH, DIM, T, SEQ_LEN,
    PAD_SLOT_ID_V,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WINDOW_MODE: tl.constexpr,      # 0=varlen, 1=batch3d, 2=decode2d
    HAS_BIAS: tl.constexpr,
    HAS_CACHE_IDX: tl.constexpr,
    HAS_INIT_MODE: tl.constexpr,
    HAS_NAT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    nchunk = tl.cdiv(DIM, BLK_D)
    b = pid // nchunk
    c = pid % nchunk
    if b >= BATCH:
        return
    d = c * BLK_D + tl.arange(0, BLK_D)
    dmask = d < DIM

    if WINDOW_MODE == 0:
        start = tl.load(QSL + b)
        end = tl.load(QSL + b + 1)
        length = end - start
    elif WINDOW_MODE == 1:
        start = b * SEQ_LEN
        length = SEQ_LEN
    else:
        start = b
        length = 1

    if HAS_CACHE_IDX:
        cache_idx = tl.load(CACHE_IDX + b)
    else:
        cache_idx = b
    if HAS_INIT_MODE:
        has_init = tl.load(INIT_MODE + b) != 0
    else:
        has_init = False
    sto = 0
    if HAS_NAT:
        sto = tl.minimum(tl.maximum(tl.load(NAT + b) - 1, 0), STATE_LEN - (WIDTH - 1))

    bias = tl.zeros([BLK_D], dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + d, mask=dmask, other=0.0).to(tl.float32)

    skip = (cache_idx == PAD_SLOT_ID_V) & HAS_CACHE_IDX
    state_base = cache_idx * STATE_LEN * DIM + d
    write_state = not skip

    for t in range(length):
        acc = bias
        for k in tl.static_range(WIDTH):
            i = t + k                      # ext 坐标
            src = i - (WIDTH - 1)          # 对应 x 行偏移（<0 表示取旧状态）
            vx = tl.load(X + (start + src) * DIM + d,
                         mask=(src >= 0) & dmask & (start + src < T) & (start + src >= 0), other=0.0).to(tl.float32)
            vs = tl.load(STATES + state_base + (sto + i) * DIM,
                         mask=(src < 0) & (i < WIDTH - 1) & dmask, other=0.0).to(tl.float32)
            vs = tl.where(has_init & (src < 0), vs, 0.0)
            w = tl.load(W + k * DIM + d, mask=dmask, other=0.0).to(tl.float32)
            acc += (vx + vs) * w
        if ACTIVATION:
            acc = acc / (1.0 + tl.exp(0.0 - acc))       # Silu
        tl.store(Y + (start + t) * DIM + d, acc.to(Y.dtype.element_ty), mask=dmask)

    # 状态回写：new_state[pos] = ext[length+pos]（等价于 WriteBackState / WriteBackStateSpec）
    for pos in tl.static_range(WIDTH - 1):
        i = length + pos
        src = i - (WIDTH - 1)
        vx = tl.load(X + (start + src) * DIM + d,
                     mask=(src >= 0) & dmask & (start + src < T) & (start + src >= 0), other=0.0).to(tl.float32)
        vs = tl.load(STATES + state_base + (sto + i) * DIM,
                     mask=(src < 0) & (i < WIDTH - 1) & dmask, other=0.0).to(tl.float32)
        vs = tl.where(has_init & (src < 0), vs, 0.0)
        tl.store(STATES + state_base + pos * DIM, (vx + vs).to(STATES.dtype.element_ty),
                 mask=dmask & write_state & (length > 0))


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    initial_state_mode: torch.Tensor | None = None,
    activation: int | str | None = 0,
    pad_slot_id: int = PAD_SLOT_ID,
    run_mode: int = 0,
    num_accepted_tokens: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    bias_opt: torch.Tensor | None = None,
    query_start_loc_opt: torch.Tensor | None = None,
    cache_indices_opt: torch.Tensor | None = None,
    initial_state_mode_opt: torch.Tensor | None = None,
    num_accepted_tokens_opt: torch.Tensor | None = None,
    activation_mode: int | None = None,
    has_initial_state: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """等价于 AscendC 的 CausalConv1d（npu_causal_conv1d_custom）。

    参数同时接受调用点名（`conv_state` / `bias_opt` / `query_start_loc_opt` /
    `cache_indices_opt` / `initial_state_mode_opt` / `num_accepted_tokens_opt` /
    `activation_mode`）与 snake_case 名；两者都给时以 `*_opt` 为准。

    Args:
        x: [T, dim]（varlen）/ [batch, seqlen, dim]（3D decode）/ [batch, dim]（2D decode）
        weight: [width, dim]
        conv_states / conv_state: [numCache, stateLen, dim]，**原地更新**
        query_start_loc(_opt): [batch+1]；varlen/3D 模式必需
        cache_indices(_opt): [batch]，默认用 batch 序号
        initial_state_mode(_opt) / has_initial_state: [batch]，真表示沿用 conv_states 里的初始状态
        activation / activation_mode: 0/None 无，1/"silu" 用 SiLU
        pad_slot_id: 命中该 cache 槽时不处理（不回写状态）
        run_mode: 0=fn(prefill)，1=update/decode
        num_accepted_tokens(_opt): [batch]，投机解码时状态读取的滑动偏移
        out: 可选输出张量（同 x 形状），未给出则内部新建

    Returns:
        y，形状同 x
    """
    bias = bias_opt if bias_opt is not None else bias
    conv_states = conv_state if conv_state is not None else conv_states
    query_start_loc = query_start_loc_opt if query_start_loc_opt is not None else query_start_loc
    cache_indices = cache_indices_opt if cache_indices_opt is not None else cache_indices
    if initial_state_mode_opt is not None:
        initial_state_mode = initial_state_mode_opt
    elif has_initial_state is not None:
        initial_state_mode = has_initial_state
    nat = num_accepted_tokens_opt if num_accepted_tokens_opt is not None else num_accepted_tokens
    if activation_mode is not None:
        activation = activation_mode
    act_on = activation not in (None, 0, False, "none")
    if isinstance(activation, str):
        assert activation in ("silu", "swish"), "activation 只支持 None/'silu'/'swish'"

    assert conv_states is not None, "conv_states/conv_state 必须提供"
    assert x.dim() in (2, 3), f"x 应为 2D/3D，实际 {tuple(x.shape)}"
    dim = x.shape[-1]
    width = weight.shape[0]
    assert weight.shape == (width, dim), f"weight 应为 [{width},{dim}]，实际 {tuple(weight.shape)}"
    assert 2 <= width <= 4, f"width={width} 超出支持范围（AscendC MAX_WIDTH=4）"
    state_len = conv_states.shape[1]
    assert state_len >= width - 1, f"stateLen={state_len} < width-1={width - 1}"
    assert conv_states.shape[2] == dim

    if x.dim() == 3:
        batch, seq_len = x.shape[0], x.shape[1]
        T = batch * seq_len
        window_mode = 1
    elif query_start_loc is not None:
        batch = query_start_loc.numel() - 1
        seq_len = 1
        T = x.shape[0]
        window_mode = 0
    else:
        batch = x.shape[0]
        seq_len = 1
        T = batch
        window_mode = 2
    if run_mode not in (0, 1):
        raise ValueError("run_mode 只支持 0(fn)/1(update)")

    has_nat = nat is not None and width == 4
    if nat is not None and not has_nat:
        raise NotImplementedError("num_accepted_tokens 仅在 width=4 时被 AscendC 支持")
    assert x.dtype == weight.dtype == conv_states.dtype
    if bias is not None:
        assert bias.numel() == dim and bias.dtype == x.dtype

    dev = x.device
    x = x.contiguous()
    weight = weight.contiguous()
    y = out if out is not None else torch.empty_like(x)
    blk_d = min(next_pow2(dim), 512)

    _causal_conv1d_kernel[(batch * ceil_div(dim, blk_d),)](
        x, weight, bias if bias is not None else x, conv_states,
        query_start_loc.to(torch.int32).contiguous() if query_start_loc is not None else x,
        cache_indices.to(torch.int32).contiguous() if cache_indices is not None else x,
        initial_state_mode.contiguous() if initial_state_mode is not None else x,
        nat.to(torch.int32).contiguous() if nat is not None else x,
        y,
        batch, dim, T, seq_len,
        int(pad_slot_id),
        WIDTH=width,
        STATE_LEN=state_len,
        WINDOW_MODE=window_mode,
        HAS_BIAS=bias is not None,
        HAS_CACHE_IDX=cache_indices is not None,
        HAS_INIT_MODE=initial_state_mode is not None,
        HAS_NAT=has_nat,
        ACTIVATION=bool(act_on),
        BLK_D=blk_d,
    )
    return y


def causal_conv1d_update(*args, **kwargs) -> torch.Tensor:
    """调用点名兼容入口（等价 causal_conv1d_fn(run_mode=1)）。

    调用点形如：causal_conv1d_update(x=..., conv_state=..., weight=..., bias=...,
    activation=..., conv_state_indices=..., num_accepted_tokens=..., query_start_loc=...,
    max_query_len=..., pad_slot_id=...)
    """
    kwargs = dict(kwargs)
    if "conv_state_indices" in kwargs and "cache_indices" not in kwargs:
        kwargs["cache_indices"] = kwargs.pop("conv_state_indices")
    kwargs.pop("max_query_len", None)
    kwargs.pop("block_idx_last_scheduled_token", None)
    kwargs.pop("initial_state_idx", None)
    kwargs.pop("validate_data", None)
    kwargs["run_mode"] = 1
    return causal_conv1d_fn(*args, **kwargs)
