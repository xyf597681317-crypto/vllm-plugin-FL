# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""causal_conv1d -- AscendC to Triton port

Source: csrc/ascend/moe/causal_conv1d/
Interface (op_host/causal_conv1d_def.cpp L24-73 + tiling_utils.h L20-31):
    Input : x              [T, dim] (varlen/prefill) or [batch, seqlen, dim] (3D decode)
                           or [batch, dim] (2D decode, seqlen=1)      fp16/bf16
            weight         [width, dim]            fp16/bf16
            bias           [dim]                   (OPTIONAL)
            convStates     [numCache, stateLen, dim]  fp16/bf16 (in/out)
            queryStartLoc  [batch+1]               int32/int64 (OPTIONAL)
            cacheIndices   [batch]                 int32/int64 (OPTIONAL)
            initialStateMode [batch]               bool/int32/int64 (OPTIONAL)
            numAcceptedTokens [batch]              int32/int64 (OPTIONAL, speculative decoding)
    Output: y              [T, dim] (same shape as x)    same dtype as x
    Attr: activationMode(0=none/1=silu), padSlotId=-1, runMode(0=fn/1=update)

**Accepted calling forms** (schema of `csrc/ascend/torch_binding.cpp` L1165-1169, call site
`vllm_fl/dispatch/backends/vendor/ascend/impl/gdn.py` L199-303):
    npu_causal_conv1d_custom(output, x, weight, conv_state, bias_opt, query_start_loc_opt,
                             cache_indices_opt, initial_state_mode_opt, num_accepted_tokens_opt,
                             activation_mode:int, pad_slot_id:int, run_mode:int) -> Tensor

  Form A -- AscendC contract, **output first and written in place** (the schema order, and the
            order the migrated call sites use):
        causal_conv1d_fn(output, x, weight, conv_state=..., bias_opt=...,
                         query_start_loc_opt=..., cache_indices_opt=...,
                         initial_state_mode_opt=..., num_accepted_tokens_opt=...,
                         activation_mode=..., pad_slot_id=..., run_mode=...)
    `output` is filled in place (a `copy_` is used when it is a strided view, so a caller that
    keeps its own view of the buffer still observes the result) and is also returned.
  Form B -- position-sorted entry used by the `_C_ascend` bridge/adapter, output passed by keyword:
        causal_conv1d_fn(x, weight, conv_state=..., bias_opt=..., ..., run_mode=..., out=output)

  Form A is selected unambiguously by the third positional argument: the AscendC contract puts
  `weight` there, i.e. a 2-D `[width, dim]` tensor with `2 <= width <= 4` (`MAX_WIDTH`), while
  Form B only ever passes the 1-D `[dim]` bias (or `None`) in that slot. `output=` may also be
  given explicitly in either form. If the first two positional arguments do not then have the
  same shape, the call is rejected instead of silently computing the wrong thing.

This implementation accepts both the call-site names (`*_opt` / `conv_state` / `activation_mode`)
and the snake_case names.

Formulas (op_kernel/causal_conv1d.h):
  InitRing (L351-393): the first (width-1) ring slots = convStates[cacheIdx, stateTokenOffset+i, :]
      (zeroed when hasInit is false, L376-381); hasInit = initialStateMode[seq] (L945-947)
  RunSeq (L415-470): for t in [0,len):
      acc = bias                                                        # L429-433
      for j in [MAX_WIDTH-width, MAX_WIDTH): tap = MAX_WIDTH-1-j        # L435-446
          slot = (tap==0) ? ring[current token] : ring[token(t-tap)]    # L437
          acc += fp32(ring[slot]) * weight[j]
      => y[t] = bias + sum_{k=0..width-1} weight[k] * ext[t+k]
         ext = [old state (width-1 historical tokens), x[start..start+len)]
      activationMode!=0 -> Silu(acc)                                    # L450-452
  WriteBackState (L717-737): state[pos] = ext[len+pos], pos in [0,width-2)
      (WriteBackStateSpec L740-830 is the equivalent fast path for width==4 with a large enough
      stateLen)
      a slot with cacheIdx == padSlotId is not processed (L933-936)
  Task window (L52-94 + tiling_validation.h L89-121):
      inputMode=0 varlen: start=qsl[b], len=qsl[b+1]-qsl[b] (x is [cuSeqlen, dim])
      inputMode=1 3D: start=b*seqLen, len=seqLen (x is [batch, seqlen, dim])
      inputMode=2 2D: start=b, len=1 (x is [batch, dim], decode)
  Speculative decoding (L1009-1018): numAcceptedTokens[b] -> stateTokenOffset = clamp(nat-1, 0, stateLen-(width-1))
      i.e. the state is read starting from that offset (sliding window); the write-back is still ext[len+pos]

Triton implementation notes / semantic assumptions:
  1. runMode=0/1 are both supported; the window mode is derived from the shape of x and
     queryStartLoc (consistent with tiling_validation).
  2. Accumulation is fp32 (matching the kernel, which casts to fp32 before the multiply-add) and the
     final result is rounded to nearest in place.
  3. AscendC only allows numAcceptedTokens when width==4 (validation L362); this implementation
     likewise only treats it as speculative decoding under that condition and ignores it otherwise.
  4. One program handles one (batch, dim tile) and advances serially along t (the inherent dependency
     of a causal convolution).
  5. **Constraint**: within a single call, the cacheIndices of the batches must point to pairwise
     distinct state slots (the production path satisfies this; slots marked by pad_slot_id are
     skipped). AscendC processes batches serially, so when two live batches reuse a slot the later
     one reads the state written back by the earlier one; the Triton side is batch-parallel and
     cannot reproduce that serial side effect (it would be a data race).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1

#: `MAX_WIDTH` of the AscendC operator (tiling_utils.h): a `[width, dim]` weight with a first
#: dimension in this range is what makes the AscendC positional form (Form A) recognisable.
MAX_WIDTH = 4

#: sentinel for "keyword not passed", which matters because several AscendC keywords are
#: meaningful when passed as an explicit `None` (e.g. `initial_state_mode_opt=None` means
#: "reuse the state stored in conv_state", it is not the same as "not passed").
_UNSET = object()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _is_ascendc_weight(candidate, dim: int) -> bool:
    """True when the *third* positional argument is an AscendC `weight` `[width, dim]`.

    The AscendC contract is `(output, x, weight, conv_state, ...)`, so in the
    `(x, weight, ...)` form the same slot only ever holds a 1-D `[dim]` bias or `None`.
    """
    if not torch.is_tensor(candidate) or candidate.dim() != 2:
        return False
    return 2 <= candidate.shape[0] <= MAX_WIDTH and candidate.shape[1] == dim


def _resolve_out(out, x: torch.Tensor):
    """Return `(buffer, copy_back_target)` for the `[T, dim]` kernel result.

    `out` is used directly when it is a contiguous buffer of the very same shape; otherwise a
    fresh contiguous buffer is returned together with the tensor that has to receive a
    `copy_` afterwards. That keeps a strided view (or a differently shaped view of the same
    storage) both correct and observably written in place.
    """
    if out is None:
        return torch.empty_like(x), None
    if not torch.is_tensor(out):
        raise TypeError(f"output/out must be a Tensor, got {type(out).__name__}")
    if out.dtype != x.dtype:
        raise TypeError(f"output/out dtype {out.dtype} must match x dtype {x.dtype}")
    if out.numel() != x.numel():
        raise ValueError(
            f"output/out has {out.numel()} elements but x has {x.numel()}")
    if out.is_contiguous() and tuple(out.shape) == tuple(x.shape):
        return out, None
    return torch.empty_like(x), out


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
        # note: **when initial_state_mode is not passed, the native behavior is to reuse
        #   conv_state**, not to assume "no initial state". Measured:
        #     `initial_state_mode_opt=None`: native y_err = 1.953e-03 (bf16 floor) -- it read conv_state;
        #                                    ours y_err = 1.378e-01 -- the old code zeroed the whole
        #                                    history here.
        #   The parameter exists to **turn the initial state off** (the first block of a prefill has no
        #   history), not to turn it on.
        #   The call site `impl/gdn.py` L207 (the spec path) passes None, so the old implementation
        #   necessarily produced silently wrong values in the model.
        has_init = True
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
            i = t + k                      # coordinate in ext
            src = i - (WIDTH - 1)          # corresponding x row offset (<0 means take the old state)
            vx = tl.load(X + (start + src) * DIM + d,
                         mask=(src >= 0) & dmask & (start + src < T) & (start + src >= 0), other=0.0).to(tl.float32)
            vs = tl.load(STATES + state_base + (sto + i) * DIM,
                         mask=(src < 0) & (i < WIDTH - 1) & dmask & write_state,
                         other=0.0).to(tl.float32)
            # note (second site): **a pad slot has to be skipped on reads as well**.
            #   The old code only skipped the write (`write_state`), but the `cache_idx` of a pad slot
            #   is -1 (the call site passes `cache_idx=[0,-1,1,-1], pad_slot=-1`), so `state_base`
            #   became a **negative offset** => an **out-of-bounds read** => garbage/NaN (that case
            #   measured `y_err=nan`). The AscendC semantics are "when cacheIdx == padSlotId the slot
            #   is not processed" (neither written, nor read for history).
            vs = tl.where(has_init & (src < 0) & write_state, vs, 0.0)
            w = tl.load(W + k * DIM + d, mask=dmask, other=0.0).to(tl.float32)
            acc += (vx + vs) * w
        if ACTIVATION:
            acc = acc / (1.0 + tl.exp(0.0 - acc))       # Silu
        tl.store(Y + (start + t) * DIM + d, acc.to(Y.dtype.element_ty), mask=dmask)

    # state write-back: new_state[pos] = ext[length+pos] (equivalent to WriteBackState / WriteBackStateSpec)
    for pos in tl.static_range(WIDTH - 1):
        i = length + pos
        src = i - (WIDTH - 1)
        vx = tl.load(X + (start + src) * DIM + d,
                     mask=(src >= 0) & dmask & (start + src < T) & (start + src >= 0), other=0.0).to(tl.float32)
        vs = tl.load(STATES + state_base + (sto + i) * DIM,
                     mask=(src < 0) & (i < WIDTH - 1) & dmask & write_state,
                     other=0.0).to(tl.float32)
        vs = tl.where(has_init & (src < 0) & write_state, vs, 0.0)
        tl.store(STATES + state_base + pos * DIM, (vx + vs).to(STATES.dtype.element_ty),
                 mask=dmask & write_state & (length > 0))


def _causal_conv1d_ascendc(
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
    initial_state_mode_opt=_UNSET,
    num_accepted_tokens_opt=_UNSET,
    activation_mode=_UNSET,
    has_initial_state: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equivalent of the AscendC CausalConv1d (npu_causal_conv1d_custom).

    Two positional layouts are accepted (see the module docstring):

      * AscendC / schema order -- ``(output, x, weight, ...)`` with the output written in
        place. Recognised by the third positional being a 2-D ``[width, dim]`` weight, or by
        passing ``output=`` explicitly. The schema is
        ``npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, Tensor conv_state,
        Tensor? bias_opt, ..., int activation_mode, int pad_slot_id, int run_mode) -> Tensor``
        and the migrated call sites keep exactly that order.
      * implementation order -- ``(x, weight, ...)`` with the output passed as ``out=`` (or
        allocated internally).

    Arguments are accepted under both the call-site names (`conv_state` / `bias_opt` /
    `query_start_loc_opt` / `cache_indices_opt` / `initial_state_mode_opt` /
    `num_accepted_tokens_opt` / `activation_mode`) and the snake_case names; when both are given,
    the `*_opt` form wins.

    Args:
        x: [T, dim] (varlen) / [batch, seqlen, dim] (3D decode) / [batch, dim] (2D decode)
        weight: [width, dim]
        conv_states / conv_state: [numCache, stateLen, dim], **updated in place**
        query_start_loc(_opt): [batch+1]; required in varlen/3D mode
        cache_indices(_opt): [batch], defaults to the batch index
        initial_state_mode(_opt) / has_initial_state: [batch]; true means reuse the initial state
            stored in conv_states
        activation / activation_mode: 0/None for none, 1/"silu" for SiLU
        pad_slot_id: a cache slot matching this id is not processed (no state write-back)
        run_mode: 0=fn(prefill), 1=update/decode
        num_accepted_tokens(_opt): [batch], sliding offset of the state read under speculative decoding
        out / output: output tensor, same shape/dtype as x; **filled in place**. Allocated
            internally when neither is given. `output` is the AscendC name, `out` the
            implementation name; passing either selects the AscendC semantics for that tensor.

    Returns:
        the output tensor (`output`/`out` when one was given, the fresh tensor otherwise)
    """
    # ---- AscendC positional form: (output, x, weight, ...) --------------------------------
    output = out if output is None else output
    if output is None and _is_ascendc_weight(bias, x.shape[-1]):
        # `x`/`weight`/`bias` hold the AscendC `output`/`x`/`weight` here.
        output, x, weight, bias = x, weight, bias, None
    if output is not None:
        if not torch.is_tensor(output) or not torch.is_tensor(x) or not torch.is_tensor(weight):
            raise TypeError(
                "causal_conv1d_fn(output, x, weight, ...) requires three tensors; got "
                f"{type(output).__name__}, {type(x).__name__}, {type(weight).__name__}")
        if tuple(output.shape) != tuple(x.shape):
            raise ValueError(
                "AscendC contract violated: the in-place output "
                f"{tuple(output.shape)} must have the same shape as x {tuple(x.shape)}")
        if output.numel() == 0:
            return output
        _causal_conv1d_impl(
            x, weight, bias=bias, conv_states=conv_states,
            query_start_loc=query_start_loc, cache_indices=cache_indices,
            initial_state_mode=initial_state_mode, activation=activation,
            pad_slot_id=pad_slot_id, run_mode=run_mode,
            num_accepted_tokens=num_accepted_tokens, conv_state=conv_state,
            bias_opt=bias_opt, query_start_loc_opt=query_start_loc_opt,
            cache_indices_opt=cache_indices_opt,
            initial_state_mode_opt=initial_state_mode_opt,
            num_accepted_tokens_opt=num_accepted_tokens_opt,
            activation_mode=activation_mode, has_initial_state=has_initial_state,
            out=output)
        return output
    return _causal_conv1d_impl(
        x, weight, bias=bias, conv_states=conv_states, query_start_loc=query_start_loc,
        cache_indices=cache_indices, initial_state_mode=initial_state_mode,
        activation=activation, pad_slot_id=pad_slot_id, run_mode=run_mode,
        num_accepted_tokens=num_accepted_tokens, conv_state=conv_state, bias_opt=bias_opt,
        query_start_loc_opt=query_start_loc_opt, cache_indices_opt=cache_indices_opt,
        initial_state_mode_opt=initial_state_mode_opt,
        num_accepted_tokens_opt=num_accepted_tokens_opt, activation_mode=activation_mode,
        has_initial_state=has_initial_state, out=out)


def _causal_conv1d_impl(
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
    initial_state_mode_opt=_UNSET,
    num_accepted_tokens_opt=_UNSET,
    activation_mode=_UNSET,
    has_initial_state: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Kernel entry point (form B); `causal_conv1d_fn` resolves the positional layout first."""
    bias = bias_opt if bias_opt is not None else bias
    conv_states = conv_state if conv_state is not None else conv_states
    query_start_loc = query_start_loc_opt if query_start_loc_opt is not None else query_start_loc
    cache_indices = cache_indices_opt if cache_indices_opt is not None else cache_indices
    if initial_state_mode_opt is not _UNSET:
        initial_state_mode = initial_state_mode_opt
    elif has_initial_state is not None:
        initial_state_mode = has_initial_state
    nat = num_accepted_tokens_opt if num_accepted_tokens_opt is not _UNSET else num_accepted_tokens
    if activation_mode is not _UNSET:
        activation = activation_mode
    act_on = activation not in (None, 0, False, "none")
    if isinstance(activation, str):
        assert activation in ("silu", "swish"), "activation only supports None/'silu'/'swish'"

    assert conv_states is not None, "conv_states/conv_state must be provided"
    assert x.dim() in (2, 3), f"x must be 2D/3D, got {tuple(x.shape)}"
    dim = x.shape[-1]
    width = weight.shape[0]
    assert weight.shape == (width, dim), f"weight must be [{width},{dim}], got {tuple(weight.shape)}"
    assert 2 <= width <= 4, f"width={width} is out of range (AscendC MAX_WIDTH=4)"
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
        raise ValueError("run_mode only supports 0(fn)/1(update)")

    has_nat = nat is not None and width == 4
    if nat is not None and not has_nat:
        raise NotImplementedError("AscendC only supports num_accepted_tokens when width=4")
    assert x.dtype == weight.dtype == conv_states.dtype
    if bias is not None:
        assert bias.numel() == dim and bias.dtype == x.dtype

    dev = x.device
    x = x.contiguous()
    weight = weight.contiguous()
    y, copy_back = _resolve_out(out, x)
    blk_d = min(_next_pow2(dim), 512)

    _causal_conv1d_kernel[(batch * _ceil_div(dim, blk_d),)](
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
    if copy_back is not None:
        copy_back.copy_(y.reshape(copy_back.shape))
        return copy_back
    return y


def _causal_conv1d_update_ascendc(*args, **kwargs) -> torch.Tensor:
    """Call-site-name compatible entry point (equivalent to causal_conv1d_fn(run_mode=1)).

    The call site looks like: causal_conv1d_update(x=..., conv_state=..., weight=..., bias=...,
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

# ==========================================================================================
# Legacy vLLM-compatible surface (kept verbatim)
# ==========================================================================================
# `vllm_fl/dispatch/backends/vendor/ascend/patch.py::patch_causal_conv1d` installs
# `causal_conv1d_fn` / `causal_conv1d_update_npu` from this module into vLLM's
# `model_executor.layers.mamba.ops.causal_conv1d`, whose callers use vLLM's own layout
# (`x` is `[dim, cu_seqlen]`, the state is `(..., dim, width-1)`, `activation="silu"`,
# `metadata=`). That contract is unrelated to the AscendC one and has to keep working
# unchanged, so the original implementation is retained here and the public
# `causal_conv1d_fn` below only routes to it when the AscendC layout is not in use.
from typing import Any, Optional

import torch.nn.functional as F


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def _causal_conv1d_fn_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_states: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    metadata: Optional[Any] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)]
                if has_initial_state[i]
                else None,
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


@triton.jit
def _causal_conv1d_update_kernel_npu_tiled(
    # Pointers
    x_ptr,  # (batch, dim, seqlen) OR (num_tokens, dim) for varlen
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # same shape as x_ptr
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,  # max seqlen for varlen, or exact seqlen
    state_len: tl.constexpr,  # effective state_len computed in wrapper
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # <= 6
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # tiling
    BLOCK_N: tl.constexpr,  # channel tile (C_TILE)
    B_TILE: tl.constexpr,  # batch tile
    T_CHUNK: tl.constexpr,  # token chunk for state update
):
    # program ids
    pid_b = tl.program_id(0)  # batch-tile id
    pid_c = tl.program_id(1)  # channel-tile id

    # channel indices for this program
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_w = idx_feats < dim

    # preload weights once per program (shared by B_TILE sequences)
    w_base = w_ptr + idx_feats * stride_w_dim
    # define to avoid "undefined" in branches
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # token index vector for chunked copy
    tok_vec = tl.arange(0, T_CHUNK)  # [T_CHUNK]

    # process B_TILE sequences inside the same program instance
    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi  # scalar tl.int32
        lane_active = b < batch  # scalar predicate

        # -------------------------
        # APC mapping (optional)
        # -------------------------
        if IS_APC_ENABLED:
            conv_state_init = tl.load(initial_state_idx + b, mask=lane_active, other=0).to(tl.int32)
            current_last_index = tl.load(
                block_idx_last_scheduled_token + b, mask=lane_active, other=0
            ).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
            seqlen_run = (qe - qs).to(tl.int32)
            # revise effective state_len for shorter sequences (same formula as original)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        # empty sequence -> skip (avoid early return because other lanes in tile)
        lane_active = lane_active & (seqlen_run > 0)

        # -------------------------
        # spec decoding offset (optional)
        # -------------------------
        if IS_SPEC_DECODING:
            conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64) - 1
            )
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(
                prior_tokens + 0 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0
            ).to(tl.float16)
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(
                prior_tokens + 1 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0
            ).to(tl.float16)
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(
                prior_tokens + 2 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0
            ).to(tl.float16)
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(
                prior_tokens + 3 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0
            ).to(tl.float16)
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(
                prior_tokens + 4 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0
            ).to(tl.float16)

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(tl.int32)

        # base pointers
        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = (
            conv_state_ptr
            + conv_states_offset * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            src_tok = (dst_tok + shift).to(tl.int32)  # [T_CHUNK]
            m_tok = (
                use_shift
                & (dst_tok < keep_shift)
                & (src_tok < state_len_run)
                & (dst_tok < state_len_run)
            )
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )

            src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            dst_tok = (keep_shift + x_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            x_tok = (tail_start + dst_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_offset < num_cache_lines)
            )

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # -------------------------
        # STEP 3/4/5: causal conv1d (+ optional SiLU) and store output
        # This is original STEP3~5, but per-lane and without debug_barrier.
        # -------------------------
        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        # accumulator preload (bias)
        acc_preload = acc_bias

        # compute each token; keep tl.range so varlen can use seqlen_run as runtime trip count (like original)
        for idx_token in tl.range(seqlen_run):
            acc = acc_preload

            # same selection logic as original (unrolled by KERNEL_WIDTH)
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    # only x[t] * w0
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                        tl.float16
                    )
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                            tl.float16
                        )
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                            tl.float16
                        )
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                            tl.float16
                        )
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                            tl.float16
                        )
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(
                            tl.float16
                        )

                acc += matrix_x.to(tl.float32) * matrix_w  # [BLOCK_N]

            # roll history window
            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1.0 + tl.exp(-acc))

            # store output
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if conv_state_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)

    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    _, width = weight.shape
    num_cache_lines, _, state_len_total = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1
        assert state_len_total >= width - 1
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    # overwrite-on-x strategy same as original
    out = x

    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    # -------- tiling heuristic--------
    # keep program count around ~[80..160]
    # vector core 40
    # TODO: use driver to get the vector core num
    CORE_HINT = 40
    # channel tile: 512 when dim large (reduce tasks), else 256
    block_n = 512 if dim >= 512 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * CORE_HINT  # ~80
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    # clamp to small set
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    # token chunk based on block_n (32KB UB idea); conservative
    t_chunk = 20 if block_n == 512 else 48

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)

# ==========================================================================================
# Public entry points
# ==========================================================================================
def causal_conv1d_fn(x, weight, bias=None, activation="silu", conv_states=None,
                     has_initial_state=None, cache_indices=None, query_start_loc=None,
                     metadata=None, pad_slot_id=PAD_SLOT_ID, **kwargs):
    """CausalConv1d under both the AscendC contract and the vLLM contract.

    AscendC (`npu_causal_conv1d_custom`, schema of `csrc/ascend/torch_binding.cpp`
    L1165-1169, used by the migrated call sites in `impl/gdn.py`)::

        causal_conv1d_fn(output, x, weight, conv_state=..., bias_opt=...,
                         query_start_loc_opt=..., cache_indices_opt=...,
                         initial_state_mode_opt=..., num_accepted_tokens_opt=...,
                         activation_mode=..., pad_slot_id=..., run_mode=...)

    `output` is filled **in place** (via `copy_` when it is a strided view, so a caller
    that keeps its own view of the buffer sees the result) and returned. This form is
    selected when `output=`/`out=` is given, when one of the AscendC-only keyword names is
    used, or when the third positional argument is a 2-D `[width, dim]` weight -- which is
    unambiguous, because the vLLM contract only ever puts a 1-D `[dim]` bias (or `None`)
    in that slot. If the output and the input then do not have the same shape the call is
    rejected instead of silently computing the wrong tensor.

    vLLM (`patch_causal_conv1d`)::

        causal_conv1d_fn(x, weight, bias, activation, conv_states, has_initial_state,
                         cache_indices, query_start_loc, metadata, pad_slot_id)

    with `x` of shape `[dim, cu_seqlen]`; unchanged, served by `_causal_conv1d_fn_vllm`.
    """
    if ("output" in kwargs or "out" in kwargs or "conv_state" in kwargs
            or "bias_opt" in kwargs or "run_mode" in kwargs
            or "query_start_loc_opt" in kwargs or "cache_indices_opt" in kwargs
            or "initial_state_mode_opt" in kwargs or "num_accepted_tokens_opt" in kwargs
            or "activation_mode" in kwargs
            or _is_ascendc_weight(bias, x.shape[-1] if torch.is_tensor(x) else -1)):
        out = kwargs.pop("output", None)
        out = kwargs.pop("out", None) if out is None else out
        call = dict(kwargs)
        call.setdefault("conv_states", conv_states)
        call.setdefault("query_start_loc", query_start_loc)
        call.setdefault("cache_indices", cache_indices)
        call.setdefault("initial_state_mode", None)
        call.setdefault("has_initial_state", has_initial_state)
        call.setdefault("pad_slot_id", pad_slot_id)
        if activation != "silu":
            call.setdefault("activation", activation)
        return _causal_conv1d_ascendc(x, weight, bias, out=out, **call)
    return _causal_conv1d_fn_vllm(
        x, weight, bias=bias, activation=activation, conv_states=conv_states,
        has_initial_state=has_initial_state, cache_indices=cache_indices,
        query_start_loc=query_start_loc, metadata=metadata, pad_slot_id=pad_slot_id)


def causal_conv1d_update(*args, **kwargs):
    """AscendC update/decode entry point (`run_mode=1`); see `causal_conv1d_fn`.

    `causal_conv1d_update_npu` above is a different, older entry point (vLLM layout) and
    is intentionally left untouched.
    """
    return _causal_conv1d_update_ascendc(*args, **kwargs)


__all__ = ["causal_conv1d_fn", "causal_conv1d_update", "causal_conv1d_update_npu",
           "causal_conv1d_ref", "PAD_SLOT_ID"]
