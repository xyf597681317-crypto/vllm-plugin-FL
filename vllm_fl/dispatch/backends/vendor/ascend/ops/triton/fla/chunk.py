# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors
import warnings

import torch
from einops import rearrange
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.utils import SUPPRESS_LEVEL

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: F401
from .chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from .chunk_o import chunk_fwd_o  # noqa: F401
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import input_guard, prepare_final_chunk_indices
from .wy_fast import recompute_w_u_fwd


def _require_device_tensor(value, name):
    # Shared guard for every path that hands a tensor to a Triton kernel.
    #
    # WHY THIS IS A SEPARATE FUNCTION (T-059): the first fix put the check inside
    # one branch only, so an already-materialised host tensor slipped through the
    # "return value unchanged" path and reached the kernel.  Centralising it means
    # all callers get the same guarantee, and it is testable in isolation.
    #
    # The runtime requires the pointer to be DEVICE-resident; a host tensor dies in
    # triton-ascend .../third_party/ascend/backend/driver.py:565 after
    # aclrtPointerGetAttributes() reports a non-DEVICE location, with the message
    #     ValueError: Pointer argument (at N) cannot be accessed from Triton
    #                (cpu tensor?)
    # Failing here instead turns "some pointer at N is on cpu" into a message that
    # names the argument and its actual device.
    dev = getattr(value, "device", None)
    if dev is not None and dev.type == "cpu":
        raise ValueError(
            "%s is a CPU tensor (device=%s) but is passed to a Triton kernel, which "
            "requires a device-resident pointer argument "
            "(runtime error: 'Pointer argument (at N) cannot be accessed from Triton "
            "(cpu tensor?)'). Move it to the compute device before the launch."
            % (name, dev)
        )
    return value


def _as_tensor_arg(value, name, device):
    # Contract adapter for arguments that are handed to a Triton kernel.
    #
    # The AscendC metadata builder stores `cu_seqlens_kern` as a plain
    # `tuple[int, ...]` (gdn_attn_builder.py:93, "cu_seqlens_kern: tuple[int, ...]
    # | None"), because the AscendC operator consumed it host-side.  The Python
    # implementation this call site now targets declares
    #     cu_seqlens: torch.LongTensor | None = None
    # (chunk_delta_h.py:188) and launches a Triton kernel with it.  Triton's
    # binder calls `mangle_type(arg)` on every non-constexpr argument, which does
    # `arg.dtype` (triton/runtime/jit.py:316) -- so a tuple blows up with
    #     AttributeError: 'tuple' object has no attribute 'dtype'
    # even when the kernel would not read the value (IS_VARLEN is derived from
    # `cu_seqlens is not None`, but the dtype mangling happens regardless).
    #
    # DEVICE IS NOT OPTIONAL HERE.  `torch.tensor(...)` defaults to CPU, and every
    # Triton pointer argument must be device-accessible: the runtime rejects a host
    # tensor with
    #     ValueError: Pointer argument (at N) cannot be accessed from Triton
    #                (cpu tensor?)
    # (triton-ascend .../third_party/ascend/backend/driver.py:565, which reads
    # `aclrtPointerGetAttributes(ptr).location.type` and requires DEVICE).
    # So the caller MUST say which device the kernel works on, and we convert
    # straight onto it.  We deliberately do NOT default to `.cpu()` nor call
    # `.npu()`: the first is what broke qwen27 v4, and the second hard-codes a
    # backend into a file that is also imported on other platforms.
    #
    # Convert tuple/list to a tensor (numerically identical values); pass every
    # other type through unchanged so behaviour is not altered.  Explicit `raise`
    # rather than `assert` (-O would strip an assert), and the message carries the
    # actual type/shape so the next occurrence is diagnosable at a glance.
    #
    # EVERY RETURN PATH IS GUARDED (T-059 follow-up).  An earlier revision checked
    # only the tuple branch, so an already-materialised CPU tensor was returned
    # untouched and reached Triton anyway -- the guard knew the risk but did not
    # cover the whole function.  A guard is only as good as the paths it covers.
    if isinstance(value, torch.Tensor):
        _require_device_tensor(value, name)
        return value
    if isinstance(value, (tuple, list)):
        if device is None:
            raise ValueError(
                "%s must be converted onto a device, but device=None was passed. "
                "A Triton kernel pointer argument may not be a host tensor "
                "(runtime error: 'Pointer argument (at N) cannot be accessed from "
                "Triton (cpu tensor?)')." % (name,)
            )
        return torch.tensor(value, dtype=torch.int64, device=device)
    if value is None:
        return None
    raise TypeError(
        "%s must be a torch.Tensor, tuple/list, or None, but got %s (%r). "
        "A Triton kernel argument is type-mangled via `.dtype`, so a non-tensor "
        "raises AttributeError: 'tuple' object has no attribute 'dtype'."
        % (name, type(value).__name__, value)
    )


def _guard_reusable_chunk_offsets(chunk_offsets, cu_seqlens_kern, chunk_size, keep_meta):
    # Guard for the reuse below.  The metadata table was built from the original
    # `cu_seqlens`, so it is only interchangeable with the compacted kernel view
    # when nothing was dropped.  If that invariant breaks we must fail loudly:
    # silently passing a stale table makes the kernel read the wrong chunk base.
    #
    # This deliberately uses explicit `raise` and never `assert`, because `assert`
    # is stripped under `python -O` and the failure mode here is a silent wrong
    # answer rather than a crash.  The check must therefore always run.
    #
    # ---------------------------------------------------------------------------
    # T-073 (V2): this check no longer READS THE DEVICE TABLE.
    #
    # `chunk_size` remains in the signature for call-site stability, but the check
    # no longer recomputes the cdiv recurrence, so it is intentionally unused here.
    #
    # It used to do `chunk_offsets.tolist()` and compare every entry against the
    # cumsum recurrence.  That is a device->host copy, i.e. a synchronising device
    # access on the generation path -- exactly what 507015 / "MTE DDR out of range"
    # was reported inside (copy_between_host_and_device_opapi).  It was also the
    # FIRST device data access of the step: this guard is called at L257, before the
    # cu_seqlens H2D argument at L278.
    #
    # The check is now pure host-side integer arithmetic on values we already hold.
    # WHY THAT IS STILL A REAL CHECK (not a weakening):
    #   _compact_empty_segments (gdn_attn_builder.py:158-170) returns keep_meta=None
    #   if and only if NO empty segment was dropped:
    #       if bool(keep.all()):  return cu_seqlens_host, initial_state, None
    #   so in this branch (we already returned when keep_meta is not None) the
    #   kernel's `cu_seqlens_kern` IS the very list the metadata table was built
    #   from.  The lengths must therefore agree, and a disagreement means the
    #   builder and this consumer have diverged -- which is what we must catch.
    #
    # WHAT IS NO LONGER CHECKED (stated so nobody assumes otherwise):
    #   the table's CONTENT.  Comparing values cannot be done without the device
    #   copy we are removing.  The remaining exposure is "the builder produced a
    #   table whose length is right but whose values are wrong"; that is the
    #   builder's invariant to hold, and V4 (builder self-validates on the host and
    #   passes the verdict down with the metadata) is the clean way to restore it.
    # ---------------------------------------------------------------------------
    if chunk_offsets is None or keep_meta is not None:
        return
    if cu_seqlens_kern is None or len(cu_seqlens_kern) < 2:
        raise ValueError(
            "D-1 chunk_offsets reuse is not applicable: expected at least one sequence in "
            "cu_seqlens_kern but got %r.  Note: the keep_meta is not None branch is not covered by D-1."
            % (cu_seqlens_kern,)
        )
    # prepare_chunk_offsets returns cat([0], cdiv(lens, CS)).cumsum(-1), i.e. one
    # entry per segment plus a trailing sentinel, so len(table) == len(cu_seqlens).
    #
    # `chunk_offsets.numel()` is host-visible metadata (it needs no device read),
    # which is what keeps this guard free of any H2D/D2H copy.
    n_segments = len(cu_seqlens_kern) - 1
    if chunk_offsets.numel() != n_segments + 1:
        raise ValueError(
            "D-1 chunk_offsets reuse would be wrong: metadata table has %d entries but "
            "cu_seqlens_kern decodes to %d sequences, so %d entries are required when "
            "keep_meta is None.  Note: the keep_meta is not None branch is not covered by D-1."
            % (chunk_offsets.numel(), n_segments, n_segments + 1)
        )
    # Host-side sanity on the very list the table was built from: a varlen
    # cu_seqlens is non-decreasing, and a negative or out-of-order entry would make
    # the cdiv recurrence meaningless.  This costs no device access.
    prev = int(cu_seqlens_kern[0])
    if prev < 0:
        raise ValueError(
            "D-1 chunk_offsets reuse is not applicable: cu_seqlens_kern[0] is %d, but varlen "
            "sequence starts must be >= 0." % prev
        )
    for i in range(len(cu_seqlens_kern) - 1):
        cur = int(cu_seqlens_kern[i + 1])
        if cur < prev:
            raise ValueError(
                "D-1 chunk_offsets reuse is not applicable: cu_seqlens_kern is not non-decreasing "
                "at index %d (%d then %d), so the chunk-offset recurrence is undefined."
                % (i + 1, prev, cur)
            )
        prev = cur



def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
):
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes
    chunk_size = 64
    block_indices_cumsum = None if prebuilt_meta is None else prebuilt_meta.block_indices_cumsum
    cu_seqlens_host = None if prebuilt_meta is None else prebuilt_meta.cu_seqlens_host
    chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64
    chunk_indices_chunk64_host = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64_host
    chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_offsets_chunk64
    update_chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.update_chunk_offsets_chunk64
    final_chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.final_chunk_indices_chunk64
    chunk_indices_large_block = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_large_block
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_indices=block_indices_cumsum,
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_chunk64,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_large_block=chunk_indices_large_block,
        chunk_indices_bt=chunk_indices_chunk64,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_chunk64,
    )

    k_ascendc = k.to(torch.bfloat16).contiguous()
    w_ascendc = w.to(torch.bfloat16).contiguous()
    u_ascendc = u.to(torch.bfloat16).contiguous()
    g_ascendc = g.transpose(1, 2).contiguous()
    q_ascendc = q.to(torch.bfloat16).contiguous()

    cu_seqlens = None if cu_seqlens is None else cu_seqlens.to(torch.int64)
    chunk_indices = None if chunk_indices_chunk64 is None else chunk_indices_chunk64.to(torch.int64)
    if cu_seqlens_host is None and cu_seqlens is not None:
        cu_seqlens_host = tuple(cu_seqlens.tolist())
    if chunk_indices_chunk64_host is None and chunk_indices is not None:
        chunk_indices_chunk64_host = tuple(chunk_indices.flatten().tolist())
    # Compact zero-length segments for the AscendC kernels (see
    # _compact_empty_segments).  chunk_indices_chunk64 is already compact-
    # ranked and is reused as-is; only cu_seqlens / initial_state need
    # compacting.
    if prebuilt_meta is not None and hasattr(prebuilt_meta, "keep_meta"):
        cu_seqlens_kern = cu_seqlens_host if prebuilt_meta.cu_seqlens_kern is None else prebuilt_meta.cu_seqlens_kern
        keep_meta = prebuilt_meta.keep_meta
        initial_state_kern = (
            initial_state[keep_meta] if initial_state is not None and keep_meta is not None else initial_state
        )
    else:
        cu_seqlens_kern, initial_state_kern = cu_seqlens_host, initial_state
        keep_meta = None
    # D-1: reuse the chunk offset table that the metadata builder already
    # computed (gdn_attn_builder.py:189) instead of letting the callee redo it.
    # The callee would otherwise take its `if chunk_offsets is None` branch and
    # call prepare_chunk_offsets(), which evaluates `triton.cdiv` over an NPU
    # tensor on the first request that reaches this Python implementation.
    # Reuse is only valid when nothing was compacted away: with keep_meta set,
    # cu_seqlens_kern is the shorter, zero-length-segment-free view while the
    # table was built from the original cu_seqlens, so the two disagree and the
    # callee must be left to recompute (that branch is NOT covered by D-1).
    # Pass None when there is no prebuilt metadata at all; that keeps the
    # legacy recompute path byte-for-byte unchanged.
    chunk_offsets_kern = chunk_offsets_chunk64 if keep_meta is None else None
    _guard_reusable_chunk_offsets(chunk_offsets_kern, cu_seqlens_kern, chunk_size=64, keep_meta=keep_meta)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k_ascendc,
        w_ascendc,
        u_ascendc,
        g=g_ascendc,
        gk=None,
        initial_state=initial_state_kern,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        # The kernel indexes `k` with these values (`k[k_seqlen_offs...]`), so the
        # lengths must live on the device `k` is on -- a host tensor is rejected by
        # the Triton runtime ("Pointer argument (at 8) cannot be accessed from
        # Triton (cpu tensor?)").  `k_ascendc` is a device tensor by construction
        # (L187), so its device is the right target and it is always available here.
        #
        # T-059: assert that the device SOURCE is itself on the device.  If `k` ever
        # arrives on CPU then every tensor derived from it is on CPU too, and the
        # failure would only surface deep inside Triton as "Pointer argument (at N)".
        # Checking here names the real culprit at the call that caused it.
        cu_seqlens=_as_tensor_arg(
            cu_seqlens_kern,
            "cu_seqlens_kern",
            _require_device_tensor(k_ascendc, "k_ascendc").device,
        ),
        chunk_indices=chunk_indices_chunk64,
        chunk_offsets=chunk_offsets_kern,
        use_exp2=False,
        transpose_state_layout=False,
    )
    if keep_meta is not None:
        # Scatter the compacted final_state back to the original [N, H, K, V]
        # layout the PCP state recursion expects; empty segments keep their
        # initial state.
        _fs_full = initial_state.clone()
        _fs_full[keep_meta] = final_state
        final_state = _fs_full

    if get_pcp_group().world_size > 1:
        # When integrating mtp, since `mix_qkv` has been split, `num_decode`
        # cannot be directly obtained from the metadata and needs to be recalculated.
        actual_num_decodes = getattr(prebuilt_meta, "num_decodes", None)
        if actual_num_decodes is None:
            actual_num_decodes = num_decodes
        h_update = chunk_gated_delta_rule_fwd_hupdate(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
            chunk_offsets=chunk_offsets_chunk64,
            update_chunk_offsets=update_chunk_offsets_chunk64,
            num_decodes=actual_num_decodes,
        )
        all_final_state = get_pcp_group().all_gather(final_state.unsqueeze(0), 0)
        final_chunk_indices = final_chunk_indices_chunk64
        if final_chunk_indices is None:
            final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, chunk_size)
        final_h_update = h_update[:, final_chunk_indices, :, :, :]
        all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)

        updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
        updated_state[0, ...] = all_final_state[0]
        for i in range(1, get_pcp_group().world_size):
            # correct_i = all_final_state[i] + Φ_i · (correct_{i-1} - s0) = Φ_i · correct_{i-1} + p_i
            updated_final_state = all_final_state[i] + torch.matmul(
                all_final_h_update[i, ...], updated_state[i - 1, ...] - initial_state
            )
            updated_state[i, ...] = updated_final_state

        final_state = updated_state[-1, ...]

        if get_pcp_group().rank_in_group == 0:
            updated_h_state = torch.zeros_like(final_state)
        else:
            updated_h_state = updated_state[get_pcp_group().rank_in_group - 1, ...]

        if get_pcp_group().rank_in_group > 0:
            rerun_initial_state = initial_state.clone()
            prefill_seq_offset = actual_num_decodes
            prefill_slice = slice(prefill_seq_offset, final_state.shape[0])
            rerun_initial_state[prefill_slice] = updated_h_state[prefill_slice]
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=g,
                initial_state=rerun_initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices_chunk64,
                chunk_offsets=chunk_offsets_chunk64,
            )
            h = h.contiguous()
            v_new = v_new.contiguous()

    # T-060: this call site was written against a DIFFERENT signature than the
    # implementation that is actually loaded (impl/fla/chunk_o.py:112):
    #     def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None,
    #                     chunk_size=64, chunk_offsets=None)
    # The old form passed `scale` POSITIONALLY as the 5th argument, which lands in
    # `g`, while also passing g=... -> TypeError: got multiple values for 'g'.
    # It additionally passed g_gamma= / chunk_indices= / transpose_state_layout=,
    # none of which exist in this signature.
    #
    # Everything is now passed BY KEYWORD so that a signature change can never
    # silently re-map an argument again; positional passing is exactly how this
    # broke.
    #
    #   g            <- g_ascendc           (the gate; chunk_o.py takes it ALREADY HEAD-FIRST: its kernel
#                                        indexes `g + bos + i_h * T_max`, i.e. [H,T], and it must NOT
#                                        transpose again.  Contract: the island owns g's transpose; the
#                                        callee owns nothing.  For k/w/u/q the island must NOT transpose,
#                                        because their kernels index them token-major.
#
# CONTRACT (final; matches the upstream vllm_ascend tree):
#   * the island transposes ONLY `g`;  q, k, w, u, v and the outputs stay TOKEN-MAJOR;
#   * every callee derives its dimensions from that token-major layout and does NOT transpose `g`
#     itself (chunk_delta_h / chunk_o / chunk_delta_hupdate all index `g + bos + i_h * T_max`, i.e. [H,T],
#     which is exactly what the island's single transpose produces).
#   * callees without the island's transposes (wy_fast, chunk_scaled_dot_kkt) receive raw `g` and
#     normalise it internally -- same rule, one owner per boundary.
#                                        because their kernels index them token-major.)
    #   scale        <- scale               (the caller's real scale, not the
    #                                        k.shape[-1]**-0.5 fallback at :128)
    #   cu_seqlens   <- cu_seqlens_kern     (must be a DEVICE TENSOR: the callee
    #                                        does `len(cu_seqlens)` at :134 and
    #                                        `prepare_chunk_offsets(cu_seqlens, BT)`
    #                                        at :136, and that does
    #                                        `cu_seqlens.new_tensor(...)` -- a tuple
    #                                        would blow up there; a CPU tensor would
    #                                        hit the T-059 pointer check)
    #   chunk_offsets<- chunk_offsets_kern  (the table the builder already made)
    #
    # Why chunk_offsets replaces the old chunk_indices=: the callee's only index
    # input is chunk_offsets (:136 computes it, :150 forwards it) and its kernel
    # takes chunk_offsets, not chunk_indices -- chunk location is derived from
    # cu_seqlens + chunk_offsets.  Passing the prebuilt table also avoids
    # `prepare_chunk_offsets()` running triton.cdiv over an NPU tensor at capture
    # time, which is the very thing D-1 exists to avoid.
    #
    # g_gamma= is dropped: nothing in the deployed ascend tree declares that
    # parameter -- the only occurrence of the name is this former call -- and the
    # callee only needs g to be non-None.
    # transpose_state_layout= is dropped: it belongs to the chunk_delta_h state
    # layout switch, which chunk_fwd_o neither accepts nor uses.
    #
    # chunk_offsets_kern is already "the table to use, or None when it does not
    # match" (L256), so it is passed as-is rather than re-deriving the keep_meta
    # condition here -- duplicating that test is how the two copies drift apart.
    o_ascendc = chunk_fwd_o(
        q=_require_device_tensor(q_ascendc, "q_ascendc"),
        k=_require_device_tensor(k_ascendc, "k_ascendc"),
        v=_require_device_tensor(v_new, "v_new"),
        h=_require_device_tensor(h, "h"),
        g=_require_device_tensor(g_ascendc, "g_ascendc"),
        scale=scale,
        cu_seqlens=_as_tensor_arg(
            cu_seqlens_kern,
            "cu_seqlens_kern",
            _require_device_tensor(q_ascendc, "q_ascendc").device,
        ),
        chunk_size=64,
        chunk_offsets=chunk_offsets_kern,
    )

    o = o_ascendc.to(torch.bfloat16).contiguous()
    v_new = v_new.to(torch.bfloat16).contiguous()
    h = h.to(torch.bfloat16).contiguous()

    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        prebuilt_meta=None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "chunk_gated_delta_rule: head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2,
        )
        q, k, v, beta, g = map(lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"chunk_gated_delta_rule: Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        prebuilt_meta,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state
