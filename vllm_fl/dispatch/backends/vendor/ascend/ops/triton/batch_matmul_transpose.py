# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""batch_matmul_transpose -- AscendC to Triton.

Source: vllm-ascend/csrc/batch_matmul_transpose/ -- the AscendC/CANN operator behind
``torch.ops._C_ascend.batch_matmul_transpose``.

    csrc/torch_binding.cpp                       L2212-2213  registered schema + impl
    csrc/ops.h                                   L126-133    launch entry signature
    batch_matmul_transpose_torch_adpt.h          L22-51      host adapter -> kernel call
    op_host/batch_matmul_transpose.h             L43-120     dtype/shape checks + tiling
    op_host/tiling/tiling_data.h                 L42-50      OpShape {batchSize, m, k, n}
    op_kernel/batch_matmul_transpose_kernel.cpp  L122-629    PpMatmulEinSum::Process

Interface (csrc/torch_binding.cpp L2212, **the schema returns nothing**):

    batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c,
                           str? format_mode=None, str? quant_mode=None) -> ()

    Input : tensor_a   fp16 / bf16   [m, batch, k]    (ND, op_host L80-83)
            tensor_b   fp16 / bf16   [batch, k, n]    (ND, format_mode="ND")
            tensor_c   fp16 / bf16   [m, batch, n]    (**out-parameter**, in place)
    Attr  : format_mode  "ND" (default) / "NZ"
            quant_mode   "per_channel_symm" (default) / "per_channel_asymm" / "per_token_symm"

    dtype and shape contract (op_host/batch_matmul_transpose.h):
        L60-62  tensor_a/b/c share one dtype, and it must be fp16 or bf16
        L68     tensor_a is 3-D
        L69-72  ND: tensor_b is 3-D, tensor_a.shape[2] == tensor_b.shape[1] (= k) and
                n = tensor_b.shape[2]
        L73-76  NZ: tensor_b is 4-D, tensor_a.shape[2] == tensor_b.shape[2] and
                n = tensor_b.shape[1] * tensor_b.shape[3]
        L78     tensor_a.shape[1] == tensor_b.shape[0] (= batch)

Call site (vllm_fl/attention/ascend/sfa_v1.py L1192-1202, ``_v_up_proj``):

    x = x.view(-1, self.local_num_heads, self.kv_lora_rank)                 # [m, batch, k]
    res = torch.empty((num_input_tokens, self.local_num_heads,
                       self.v_head_dim), dtype=x.dtype, device=x.device)     # [m, batch, n]
    torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)           # W_UV [batch,k,n]
    x = res.reshape(-1, self.local_num_heads * self.v_head_dim)

    with (m, batch, k, n) = (num_input_tokens, local_num_heads, kv_lora_rank, v_head_dim).

Semantics -- ``tensor_c[m, b, n] = sum_k tensor_a[m, b, k] * tensor_b[b, k, n]``
(op_kernel/batch_matmul_transpose_kernel.cpp, TA = TB = false):

    L137        offset_c = pid_m*m0*batch*n + b*n + pid_n*n0
                -> tensor_c is [m, batch, n] with a row stride of n*batch; the same
                   stride is handed to the L0C->GM copy at L612-617
    L145-149    offset_a = pid_m*m0*batch*k + b*k + shuffle_k*k0
                -> tensor_a is [m, batch, k] with a row stride of k*batch
    L151-165    offset_b = b*k*n + shuffle_k*k0*n + pid_n*n0
                -> tensor_b is [batch, k, n] with a row stride of n (FormatB = ND)
    L534-541    the L1->L0A copy reads the A tile with mSrcStride=1, kSrcStride=m_round,
                i.e. row-major m x k, so A is *not* transposed on the cube
    L592-598    Mad(l0c_buf, l0a_buf, l0b_buf, m_actual, n_actual, k0_actual, init_c)
                with an m x k A tile and a k x n B tile
    =>  C[m, b, n] = sum_k A[m, b, k] * B[b, k, n]     (einsum "mbk,bkn->mbn")

    The "transpose" in the operator name is about tensor_a: the cube/ATB pp_matmul
    lineage wants the batch axis first, whereas tensor_a carries the batch axis in the
    middle, hence the k*batch row stride used for every A tile (L145-149). All 16
    instantiations pass TA = false (L662-709) and MatMulInfo never sets transA/transB
    (op_host/tiling/tiling_data.h L33-34 keep the ``false``/0 defaults) nor biasFlag
    (L35), so the ND path always takes the TB = false branch above and the epilogue is
    a plain accumulator store. The ``_t_`` (transB = true) instantiations are dead code
    for this operator.

    Cross-check against the fallbacks the call site itself uses when this operator is
    unavailable (sfa_v1.py L1203-1216):
        torch_npu.npu_transpose_batchmatmul(x, W_UV, perm_x1=(1,0,2), perm_y=(1,0,2))
        torch.bmm(x.view(-1,N,L).transpose(0,1), W_UV).transpose(0,1)
    -- both are the same einsum.

Parameter coverage:
    format_mode : None / "ND" are implemented. "NZ" raises NotImplementedError, which is
        exactly the restriction recorded at the call site (sfa_v1.py L689: "Currently,
        torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz").
    quant_mode  : validated against the three strings of op_host L13-17 and otherwise
        ignored. It only reaches the tiling key through IsI8Bf16Kernel
        (op_host/tiling/tiling_data.cpp L24-30), which requires isInt8; this operator
        only accepts fp16/bf16 (op_host L61-62), so quant_mode cannot change the result.
        An unknown string raises ValueError, mirroring the TORCH_CHECK in GetModeVal
        (op_host L39).
    m <= 1024   : the guard at the call site (BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024,
        sfa_v1.py L67) comes from the host-side tiling cache MAX_CAPTURE_NUM = 1024
        (op_host/batch_matmul_transpose.h L29; the cached block is indexed by
        ``batchIdx = opShape.m - 1`` at L104). That is an artefact of the AscendC host
        tiling and does not apply to this implementation.

Implementation notes:
    grid = (batch, cdiv(m, BLOCK_M), cdiv(n, BLOCK_N)): one program per
    (batch, row tile, column tile), and the k axis is tiled with a runtime loop, so a
    program only ever holds BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N fp16 tiles
    (BLOCK_K = 64 keeps the operands far below the UB limit that other migrated matmuls
    in this package ran into). Accumulation is fp32 (``out_dtype=tl.float32``) and the
    store casts back to the output dtype. Loads and the store are masked, so m, n and k
    need not be multiples of the tiles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div

__all__ = ["batch_matmul_transpose"]

# Tile sizes. BLOCK_K = 64 is deliberately small: the production k is kv_lora_rank = 512
# and tiling it keeps every live tile tiny (the Ascend UB limit is the usual failure mode
# of matmul kernels migrated in this package).
_BLOCK_M = 64
_BLOCK_K = 64
# BLOCK_N follows the production n (v_head_dim, normally 128); a narrower n just uses the
# 64-wide tile, the masked loads take care of the tail.
_BLOCK_N_WIDE = 128
_BLOCK_N = 64

_FORMAT_MODES = ("ND",)
_QUANT_MODES = ("per_channel_symm", "per_channel_asymm", "per_token_symm")


@triton.jit
def _bmt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ab,
    stride_cm, stride_cb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """tensor_c[m, b, n] = sum_k tensor_a[m, b, k] * tensor_b[b, k, n].

    ``A`` is [M, batch, K] with element strides (stride_am, stride_ab, 1);
    ``B`` is the contiguous [batch, K, N] weight (element row stride N);
    ``C`` is [M, batch, N] with element strides (stride_cm, stride_cb, 1).
    """
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        mask_k = kk < K
        a_tile = tl.load(A + batch * stride_ab + offs_m[:, None] * stride_am + kk[None, :],
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(B + batch * K * N + kk[:, None] * N + offs_n[None, :],
                         mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    tl.store(C + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :],
             acc.to(C.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def batch_matmul_transpose(x, weight, res, format_mode=None, quant_mode=None):
    """Batched ``[m, batch, k] x [batch, k, n] -> [m, batch, n]`` matmul with the batch
    axis of ``x`` in the middle, equivalent to the AscendC operator of the same name
    (see the module docstring for the source lines and the derivation).

    Args:
        x: [m, batch, k] fp16/bf16 -- ``tensor_a``; ``W_UV`` is ``weight`` at the call site.
        weight: [batch, k, n] fp16/bf16 -- ``tensor_b`` (ND layout).
        res: [m, batch, n] fp16/bf16 -- ``tensor_c``, written in place.
        format_mode: "ND" (default, also ``None``) or "NZ" (not implemented).
        quant_mode: "per_channel_symm" (default, also ``None``), "per_channel_asymm" or
            "per_token_symm"; accepted and validated, but it has no effect for the
            fp16/bf16 dtypes this operator supports.

    Returns:
        ``res``. The AscendC schema returns nothing and the call site
        (sfa_v1.py L1201-1202) reads the out-parameter, so both
        ``batch_matmul_transpose(x, w, res)`` and ``res = batch_matmul_transpose(x, w, res)``
        behave correctly.
    """
    fmt = "ND" if format_mode is None else str(format_mode)
    if fmt not in _FORMAT_MODES:
        raise NotImplementedError(
            f"format_mode={fmt!r} is not implemented: the NZ path needs a 4-D fractal "
            "tensor_b (op_host/batch_matmul_transpose.h L73-76) and the call site "
            "records that weight nz is unsupported (sfa_v1.py L689)"
        )
    qm = "per_channel_symm" if quant_mode is None else str(quant_mode)
    if qm not in _QUANT_MODES:
        raise ValueError(f"quant_mode={qm!r} is unsupported (op_host L13-17)")

    if x.dim() != 3:
        raise ValueError(f"tensor_a must be 3-D [m, batch, k], got {tuple(x.shape)}")
    if weight.dim() != 3:
        raise NotImplementedError(
            f"tensor_b must be 3-D [batch, k, n] with format_mode='ND', "
            f"got {tuple(weight.shape)}"
        )
    m, batch, k = (int(s) for s in x.shape)
    n = int(weight.shape[2])
    if k != int(weight.shape[1]):
        raise ValueError(
            f"tensor_a.shape[2]={k} must equal tensor_b.shape[1]={weight.shape[1]}"
        )
    if batch != int(weight.shape[0]):
        raise ValueError(
            f"tensor_a.shape[1]={batch} must equal tensor_b.shape[0]={weight.shape[0]}"
        )

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"tensor_a must be fp16 or bf16, got {x.dtype}")
    if not (x.dtype == weight.dtype == res.dtype):
        raise ValueError(
            f"tensor_a/b/c must share one dtype, got {x.dtype} / {weight.dtype} / {res.dtype}"
        )
    if tuple(res.shape) != (m, batch, n):
        raise ValueError(
            f"tensor_c must be [m, batch, n] = {(m, batch, n)}, got {tuple(res.shape)}"
        )
    if not (x.device == weight.device == res.device):
        raise ValueError("tensor_a/b/c must live on the same device")

    x = x.contiguous()
    weight = weight.contiguous()
    if res.stride(2) != 1:
        raise ValueError(f"tensor_c must have a unit innermost stride, got {res.stride()}")
    if m == 0 or n == 0 or batch == 0:
        return res

    block_n = _BLOCK_N_WIDE if n >= _BLOCK_N_WIDE else _BLOCK_N
    grid = (batch, ceil_div(m, _BLOCK_M), ceil_div(n, block_n))
    _bmt_kernel[grid](
        x, weight, res,
        m, n, k,
        x.stride(0), x.stride(1),
        res.stride(0), res.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=_BLOCK_K,
    )
    return res
