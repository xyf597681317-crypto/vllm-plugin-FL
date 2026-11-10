# Copyright (c) 2026 BAAI. All rights reserved.

"""
Triton implementation of the CANN `aclnnGroupedMatmulV5` (GroupedMatmul) operator.

Contract, measured on the 105 host from `torch_npu.profiler` `kernel_details.csv`
(Qwen3-VL-30B-A3B, TP2, decode 64x1024, 29 engine steps) and from the single
vendor call site that the bf16 model actually executes,
`impl/moe/moe_mlp.py::unquant_apply_mlp`:

    gmm1 (gate_up):  x=[512, 2048] bf16, w=[128, 2048, 768] bf16 (NCL view of a
                     [128, 768, 2048] contiguous parameter), out=[512, 768]
    gmm2 (down):     x=[512,  384] bf16, w=[128,  384, 2048] bf16, out=[512, 2048]
    group_list:      [128] int64, `group_list_type=1` (per-expert counts)
    no bias / scale / offset; ND output; every call takes split_item=2.

The two calls are the largest single line item of the graph-mode decode step
(16.6 ms/step of 45.4 ms, 36.6%) and they are **HBM-bandwidth bound**: each
call streams the whole expert weight tensor (402 MB / 201 MB) once, because
every one of the 128 experts owns at least one of the 512 routed rows, so the
29 GB of per-rank MoE weights must cross the bus on every token.  The native
kernel measured 216 us / 130 us, i.e. 1.74 TB/s combined, which is at the
roofline of this die; a kernel can only match it, not beat it.  This module
therefore aims at "same order as native" and keeps the native operator as the
fallback for every layout it does not support.

Supported subset (everything else returns `None` and the caller keeps the
native operator):

    * x          `[M, K]` bf16/fp16, contiguous
    * weight     `[E, K, N]` with the K dimension contiguous (the NCL view the
                 vendor call site passes) or `[E, N, K]` contiguous (plain
                 row-major, supported because `tl.dot` only needs strides)
    * group_list `[E]` int32/int64, count semantics (`group_list_type=1`)
    * no bias, no quantization, output dtype == x dtype
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Row tile.  A decode step routes `topk * batch` rows through `E` experts, so
# the per-expert row count is tiny (512/128 = 4 in the measured model); 16 is
# the smallest tile `tl.dot` accepts on this backend and the padding rows are
# free because the kernel is bandwidth bound, not compute bound.
_BM = 16
# Column tile / K tile fallbacks.  A weight row is `N` wide and every element is
# read exactly once, so the widest burst that still divides the extent wins;
# measured on 105 with the model's real `[E, K, N]` layout (402 MB weight,
# M=512, native = 0.350 ms = 1150 GB/s): BN=256/BK=256 -> 0.452 ms (891 GB/s),
# BN=256/BK=128 -> 0.476, BN=256/BK=64 -> 0.534, BN=128/BK=128 -> 0.693,
# BN=64/BK=256 -> 1.078.  K=384 (gmm2) cannot use BK=256 and lands on 128.
_BN = 256
_BK = 256

# The Ascend backend rejects a coreDim of exactly 65536.
_MAX_GRID = 65535
# Below this the masking overhead dominates; those shapes keep the native op.
_MIN_K = 64


def _pick_block(n: int, candidates: tuple) -> Optional[int]:
    """Largest candidate tile width that divides `n` (None when none does)."""
    for c in candidates:
        if n % c == 0:
            return c
    return None


if HAS_TRITON:

    @triton.jit
    def _grouped_matmul_kernel(
        x_ptr,
        w_ptr,
        gl_ptr,
        out_ptr,
        M,
        N,
        E,
        stride_we,
        stride_wk,
        stride_wn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        K: tl.constexpr,
        TRANS_B: tl.constexpr,
        GL_TYPE: tl.constexpr,
    ):
        """
        `out[start_e:start_e+rows_e] = x[...] @ w[e]` for one (expert, N tile).

        The expert range comes from an in-kernel prefix sum over `group_list`.
        The sum is a short scalar loop (E <= 128 entries, all L1 resident):
        a vector reduction over the 128-lane int64 group list was measured to
        miscompile on this backend (garbage `start` -> out-of-bounds reads ->
        NaN), and the scalar walk is what the validated 107 prototype used.
        `GL_TYPE=1` is the count form (one entry per expert), `GL_TYPE=0` is the
        cumulative-end form.  The row loop is dynamic so an expert that owns
        more than `BM` rows still gets all of them; empty experts exit at once.
        """
        e = tl.program_id(0)
        pid_n = tl.program_id(1)

        if GL_TYPE == 1:
            start = tl.zeros((), dtype=tl.int64)
            j = 0
            while j < e:
                start += tl.load(gl_ptr + j)
                j += 1
            rows = tl.load(gl_ptr + e)
        else:
            start = tl.where(e > 0, tl.load(gl_ptr + tl.maximum(e - 1, 0)), 0)
            rows = tl.load(gl_ptr + e) - start

        offs_n = pid_n * BN + tl.arange(0, BN)
        n_mask = offs_n < N

        m_tiles = tl.cdiv(rows, BM)
        for t in range(0, m_tiles):
            offs_m = start + t * BM + tl.arange(0, BM)
            m_mask = offs_m < start + rows
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                a = tl.load(
                    x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                )
                if TRANS_B:
                    # Weight rows are K-contiguous ([E, N, K] storage): read the
                    # tile in its natural [BN, BK] order so every DMA burst is
                    # contiguous, then feed the cube the transposed operand.
                    bt = tl.load(
                        w_ptr + e * stride_we + offs_n[:, None] * stride_wn
                        + offs_k[None, :] * stride_wk,
                        mask=n_mask[:, None],
                        other=0.0,
                    )
                    b = tl.trans(bt)
                else:
                    b = tl.load(
                        w_ptr + e * stride_we + offs_n[None, :] * stride_wn
                        + offs_k[:, None] * stride_wk,
                        mask=n_mask[None, :],
                        other=0.0,
                    )
                acc = tl.dot(a, b, acc)
            tl.store(
                out_ptr + offs_m[:, None] * N + offs_n[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )


def _triton_gmm_enabled() -> bool:
    return os.environ.get("VLLM_FL_TRITON_GMM", "1") != "0"


_TRACE = os.environ.get("VLLM_FL_TRITON_GMM_TRACE", "0") == "1"
_traced = {"entry": False, "reject": False}


def _reject(why: str):
    """Return `None` (native fallback) and, when tracing, say why exactly once."""
    if _TRACE and not _traced["reject"]:
        _traced["reject"] = True
        print("[triton_gmm] FALLBACK reason=%s" % why, flush=True)
    return None


def triton_grouped_matmul(
    x: list[torch.Tensor],
    weight: list[torch.Tensor],
    group_list: torch.Tensor,
    *,
    group_list_type: int = 1,
    split_item: int = 2,
    group_type: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    bias: Optional[list[torch.Tensor]] = None,
    scale: Optional[list[torch.Tensor]] = None,
    per_token_scale: Optional[list[torch.Tensor]] = None,
    antiquant_scale: Optional[list[torch.Tensor]] = None,
    antiquant_offset: Optional[list[torch.Tensor]] = None,
) -> Optional[list[torch.Tensor]]:
    """
    Drop-in Triton replacement for the unquantized `torch_npu.npu_grouped_matmul`.

    Returns a one-element list (the native operator's `split_item=2` shape) or
    `None` when this layout is outside the supported subset.
    """
    if _TRACE and not _traced["entry"]:
        _traced["entry"] = True
        try:
            xt0 = x[0] if isinstance(x, (list, tuple)) else x
            w0 = weight[0] if isinstance(weight, (list, tuple)) else weight
            print("[triton_gmm] entry x=%r w=%r gl=%r(%s) glt=%s split=%s grp=%s bias=%r"
                  % (getattr(xt0, "shape", None), getattr(w0, "shape", None),
                     getattr(group_list, "shape", None),
                     getattr(group_list, "dtype", None), group_list_type, split_item,
                     group_type, None if bias is None else len(bias)), flush=True)
            if isinstance(w0, torch.Tensor):
                print("[triton_gmm]   x_contig=%r w_strides=%r w_contig=%r dtypes=%r/%r"
                      % (getattr(xt0, "is_contiguous", lambda: None)(), w0.stride(),
                         w0.is_contiguous(), getattr(xt0, "dtype", None), w0.dtype), flush=True)
        except Exception as exc:  # noqa: BLE001
            print("[triton_gmm] trace failed %r" % (exc,), flush=True)
    if not HAS_TRITON or not _triton_gmm_enabled():
        return _reject("disabled")
    if bias is not None or scale is not None or per_token_scale is not None:
        return _reject("bias/scale")
    if antiquant_scale is not None or antiquant_offset is not None:
        return _reject("antiquant")
    if group_list_type not in (0, 1) or split_item != 2 or group_type != 0:
        return _reject("glt/split/group")
    if not isinstance(x, (list, tuple)) or len(x) != 1:
        return _reject("x-not-single")
    if not isinstance(weight, (list, tuple)) or len(weight) != 1:
        return _reject("w-not-single-list")
    xt, w = x[0], weight[0]
    if not isinstance(xt, torch.Tensor) or not isinstance(w, torch.Tensor):
        return _reject("x/w-not-tensor")
    if xt.dim() != 2 or w.dim() != 3:
        return _reject("dim")
    if xt.dtype not in (torch.bfloat16, torch.float16) or w.dtype != xt.dtype:
        return _reject("dtype")
    if output_dtype is not None and output_dtype != xt.dtype:
        return _reject("output_dtype")
    if xt.numel() == 0:
        return _reject("empty")
    if not xt.is_contiguous():
        # A `[M, K]` bf16 activation is a few MB; one copy is cheaper than the
        # native operator it replaces.
        xt = xt.contiguous()
    if not isinstance(group_list, torch.Tensor) or group_list.dim() != 1:
        return _reject("gl")
    if group_list.dtype not in (torch.int32, torch.int64):
        return _reject("gl-dtype")

    M, K = xt.shape
    E = group_list.numel()
    if w.shape[0] != E:
        return _reject("E-mismatch")
    # The logical weight is `[E, K, N]`; the vendor call site of the bf16 model
    # passes a *contiguous* `[E, K, N]` parameter (measured: shape
    # [128, 2048, 768], strides (1572864, 768, 1)), while other call sites pass
    # a transposed `[E, N, K]` storage viewed as `[E, K, N]`.  Both are
    # supported, and the load order follows whichever axis is unit stride.
    if w.shape[1] == K and w.shape[2] > 0:
        N = w.shape[2]
        stride_we, stride_wk, stride_wn = w.stride(0), w.stride(1), w.stride(2)
    elif w.shape[2] == K and w.shape[1] > 0:
        N = w.shape[1]
        stride_we, stride_wk, stride_wn = w.stride(0), w.stride(2), w.stride(1)
    else:
        return _reject("w-shape")
    if stride_wn != 1 and stride_wk != 1:
        return _reject("w-stride")
    # `[E, K, N]` row-major already gives a contiguous `[BK, BN]` tile; only the
    # transposed storage needs the `[BN, BK]` read plus a register transpose.
    trans_b = stride_wn != 1
    bn = _pick_block(N, (256, 128, 64))
    bk = _pick_block(K, (256, 128, 64))
    if bn is None or bk is None or K < _MIN_K:
        return _reject("tile/K")
    if E > _MAX_GRID or M > _MAX_GRID * _BM:
        return _reject("grid")
    # Decode routes ~M/E rows per expert (4 in the measured model) so the 16-row
    # tile already covers them; a prefill chunk routes far more and the wider
    # tile amortises the activation reads (measured 1.40x -> 1.12x at M=4096).
    bm = 32 if M > E * 16 else _BM

    out = torch.empty((M, N), dtype=xt.dtype, device=xt.device)
    grid = (E, triton.cdiv(N, bn))
    _grouped_matmul_kernel[grid](
        xt,
        w,
        group_list,
        out,
        M,
        N,
        E,
        stride_we,
        stride_wk,
        stride_wn,
        BM=bm,
        BN=bn,
        BK=bk,
        K=K,
        TRANS_B=trans_b,
        GL_TYPE=group_list_type,
    )
    if _TRACE and _traced["reject"] is False:
        _traced["reject"] = True
        print("[triton_gmm] ACTIVE M=%d K=%d N=%d E=%d bm=%d bn=%d bk=%d trans_b=%r"
              % (M, K, N, E, bm, bn, bk, trans_b), flush=True)
    return [out]
