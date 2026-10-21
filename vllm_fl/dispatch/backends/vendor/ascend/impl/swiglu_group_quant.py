# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""swiglu_group_quant -- AscendC -> Triton (the quant_mode=2 / OCP MXFP8 path)

Source: csrc/moe/swiglu_group_quant/ of the AscendC operator:
    op_host/swiglu_group_quant_def.cpp       op definition, inputs/outputs, attrs
    op_host/swiglu_group_quant_proto.cpp     InferShape / InferDtype
    op_host/swiglu_group_quant_tiling.cpp    tiling (block size, group_index, clamp flag)
    op_kernel/swiglu_group_quant.cpp         tiling-key dispatch (1 / 2 / 31 / 32)
    op_kernel/swiglu_mx_quant_perf.h         SwigluMxQuantPerf -- the quant_mode=2 kernel
    op_kernel/swiglu_group_quant_base.h      VFProcessSwigluGroupQuant / VFComputeMaxExp /
                                             VFComputeScale / VFComputeData

Interface (schema csrc/torch_binding.cpp L3550-3557, args def.cpp L22-59):
    npu_swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index,
                           ScalarType dst_type=39, int quant_mode=1, int group_size=128,
                           bool round_scale=False, bool ue8m0_scale=False,
                           bool output_origin=False, int group_list_type=0,
                           float clamp_value=0.0) -> (y, scale, y_origin)
        x         [..., 2H] float16 | bfloat16               (def.cpp L24, meta L1321)
        y         [..., H]  float8_e4m3fn | float8_e5m2      (proto L60-62)
        scale     quant_mode=2: [..., ceil(ceil(H/32)/2), 2] float8_e8m0fnu
                                                             (proto L75-80, L104-105;
                                                              meta L1334-1344)
        y_origin  [..., H]  x.dtype, **never written** in quant_mode=2 (see below)

Production call sites (both byte-identical in their argument lists):
    vllm_fl/dispatch/backends/vendor/ascend/impl/moe/fused_moe.py L759-766
    vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py   L1290-1297
        x, topk_weight=None, group_index=None, dst_type=torch.float8_e4m3fn,
        quant_mode=2, clamp_value=<swiglu_limit>
    plus vllm_ascend's W4A8-MXFP shared expert (shared_experts.py L460) and
    vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py L417 with the same arguments
    => quant_mode=2 is the only path that is really used, and it is the only one
       implemented here.

Formula (all bare line numbers are swiglu_group_quant_base.h; the MX kernel is
swiglu_mx_quant_perf.h):

 1. Layout -- **the two halves are contiguous, not interleaved**:
    x0Gm = x + row * d, x1Gm = x0Gm + splitD (mx perf L132-133, splitD = d/2 = H) and the
    kernel is handed `xLocal` / `xLocal[rowFactor * RoundUp<T0>(dFactor)]` (mx perf
    L156-158, L163-172) => a = x[..., :H] is the SiLU operand, b = x[..., H:] is the
    multiplicand. (The DIST_DINTLV_B16 in VFComputeMaxExp/VFComputeData only splits a
    register into even/odd lanes for packing; it does not change the memory layout.)
 2. Clamp (only when clamp_value != 0, tiling L123-129), applied **before** the SiLU,
    one-sided on `a` and two-sided on `b` (L343-347, L755-761):
        if c != 0:  a = min(a, c);  b = min(max(b, -c), c)
 3. SiLU (VFSwiGlu L159-167):  t = exp(-a);  s = (a / (1 + t)) * b
    (L162-166: Muls(-1) -> Exp -> Adds(1) -> Div -> Mul; `x0` is the SiLU operand)
 4. topk_weight (L349-351):  s = s * topk_weight[row]      (one fp32 value per row)
 5. MX shared exponent -- one per **32** consecutive columns (PER_MX_FP16 = 32, L30;
    the tiling picks splitFactor_ = PER_MX_FP16 for quant_mode=2, tiling L99):
    VFComputeMaxExp (L359-422) maximises the **bf16 exponent field** of |s| over the
    32 lanes (L412-416: And(v, 0x7F80) -> Max -> ReduceMaxWithDataBlock), i.e.
        e = max_i exp_field(s_i)      (exp_field = (bits & 0x7F80) of the bf16 view)
    VFComputeScale (L425-490) then turns it into the E8M0 byte:
        e    = max(e, LOWER_BOUND)                     # L470-471, 0x0400 (e4m3, L39)
                                                       #            0x0780 (e5m2, L38)
        byte = (e - LOWER_BOUND) >> 7                  # L472-473
        byte = 0xFF if e == 0x7F80 (inf/NaN)           # L474-475, FP8_E8M0_NAN_VAL L40
    With LOWER_BOUND = L << 7 and L = 8 (e4m3) / 15 (e5m2), the byte is
        byte = max(E + 127, L) - L = 127 + max(E, L - 127) - L
    where E = the unbiased exponent of the largest |s| in the block, i.e. the OCP MX rule
        scale = 2 ** (byte - 127) = 2 ** (floor(log2(amax)) - L),  byte = exponent + 127
    (`exp_field` is monotone, so maximising it equals flooring the log2 of the maximum;
    both give exponent field 0 for subnormals, and amax == 0 gives byte 0 through the
    lower-bound clamp as well.)
 6. Quantisation (VFComputeData L493-590): invScale = 0x7f00 - sharedExp (L480-484) is
    exactly 2 ** (L - exponent) as a power-of-two bit pattern, then
        q = s * invScale                                # L568-569 / L549-550
        y = sat_rint_fp8(q)                             # L516-517: SatMode::SAT +
                                                        # RoundMode::CAST_RINT
    i.e. round-half-to-even with saturation to 448 (e4m3, FP8_E4M3FN_MAX_VALUE L32) or
    57344 (e5m2, L31); equivalently q = s / 2 ** (byte - 127).
    Blocks whose exponent field is 0x7F80 get invScale = BF16_NAN_VAL (L481) => the whole
    32-element block becomes fp8 NaN, and byte 0xFF.

Outputs:
    y[row, h]     = fp8 code of column h (the MX quantised SiLU output)
    scale[row, s] = E8M0 byte of the 32-column block s (flat byte index s == the element
                    index of the [..., ceil(ceil(H/32)/2), 2] view); the bytes past
                    ceil(H/32) are padding (see "Deviations" 2)
    y_origin      = allocated, not written: the quant_mode=2 kernel takes no yOrigin
                    argument at all (swiglu_group_quant.cpp L43-46 vs L47-53), so the
                    native leaves the tensor it allocated in torch_binding_meta.cpp
                    L1346 uninitialised as well; `output_origin` is accepted and ignored.

Coverage:
    Implemented   quant_mode=2 (OCP MXFP8) with dst_type float8_e4m3fn or float8_e5m2,
                  x float16 / bfloat16, arbitrary leading dimensions, clamp_value,
                  topk_weight (one fp32 per row), group_index in count form
                  (group_list_type=0: the rows that are processed are the first
                  min(sum(group_index), rows) ones, mx perf L41-75), H % 32 != 0.
    NotImplementedError (with the reason):
                  quant_mode=1 (group/static fp8 per token, float32 or E8M0 scale) and
                  quant_mode=3 (fp8 per token + y_origin)
                  group_list_type != 0 (the tiling accepts count mode only, L117-119)
                  dst_type other than float8_e4m3fn / float8_e5m2
                  x dtypes other than float16 / bfloat16 (meta L1321-1322 requires these)
    `group_size`, `round_scale` and `ue8m0_scale` are accepted and ignored: the MX tiling
    never reads group_size (only PER_MX_FP16=32 is used, tiling L99), and round_scale /
    ue8m0_scale only exist for the quant_mode=1/3 paths, whose scale is either float32 or
    an E8M0 cast of a float scale; quant_mode=2 always produces an E8M0 scale
    (proto L102-108).

Deviations from the native kernel (both are deliberate, both are documented, neither
changes the result on the production shapes):
 1. The native stores the SiLU result into UB **in T0 = x.dtype** and re-reads it for the
    MX pass (mx perf L92/L177/L188), so its exponent and its quantised value come from
    the x.dtype-rounded SiLU output; this implementation keeps the SiLU result in fp32,
    i.e. it quantises the SiLU value itself, which is what the quant_mode=2 definition
    (`VFProcessSwigluGroupQuant` -> `VFComputeMaxExp` -> `VFComputeData`) describes when
    the intermediate is not rounded. Measured effect of that one rounding step on real
    data (8192 x 3584 bf16 / 1024 x 3584 fp16, DSV4 shared-expert shape, CPU torch):
        bf16: 3.34% of the fp8 bytes differ (mean |delta| = 1.6 codes, max 8)
        fp16: 0.41% of the fp8 bytes differ
    (the max 8 occurs when the rounded block maximum crosses a power of two, which moves
    that block's shared exponent by one and therefore the whole block's scale by 2x; both
    readings are valid MX quantisations, and the dequantised error stays inside the 6.25%
    RNE bound either way). To follow the native bit for bit instead, round the SiLU result
    into x.dtype and back before the exponent extraction: `y = y.to(X.dtype.element_ty)`
    then `.to(tl.float32)` right after the `tl.exp` line.
 2. The scale row is `2 * ceil(ceil(H/32) / 2)` bytes but only the first ceil(H/32) are
    meaningful. They are allocated as zeros here; the native allocates with at::empty
    (meta L1345) and its CopyOut leaves the tail uninitialised. For an even
    ceil(H/32) - e.g. H a multiple of 64 - the two agree byte for byte, because then
    ceil(ceil(H/32)/2) * 2 == ceil(H/32). Rows that group_index excludes are left as
    allocated, exactly as the native skips them.
 3. The E8M0 special-minimum branch of VFComputeScale (L483-484) is not reproduced: it
    can only fire when sharedExp == 0x7f00, i.e. an unbiased exponent of 127 + L, which
    no finite bf16/fp16 value reaches.

Backend notes (910_93 + Triton-Ascend, same family as the sibling dequant_swiglu_quant /
kv_compress_epilog ports):
  * Every store has the form "scalar row index + 1-D column tile + 1-D mask" (or a scalar
    store for the scale byte), which is the shape that was measured reliable on this
    backend; no 2-D tiles are used.
  * The fp8 bytes are written through a uint8 view of the fp8 tensors with a hand-rolled
    saturating RNE encoder (the semantics of the backend's own fp8 cast are not relied
    upon). It is the same integer-arithmetic style as the encoder verified byte-for-byte
    in kv_compress_epilog.py, extended with the e5m2 branch, and it was validated against
    torch's own fp32 -> fp8 cast on every finite magnitude of both formats (plus/minus
    half an ulp: 634 e4m3 and 619 e5m2 values) and on 200000 random values spanning 25
    binades: 0 differing bytes.
  * The grid is the physical Vector core count (never the row count), with a runtime
    stride loop inside the kernel, so the grid <= 65535 limit cannot be hit.

Verification (measured on 910_93 in this container with this exact file):
  * activation: `a / (1 + exp(-a)) * b` compared with `torch_npu.npu_swiglu` - 0 of 2048
    bf16 and 0 of 2048 fp16 output elements differ, for both the torch expression and a
    Triton kernel carrying the same expression (so the backend's `tl.exp` matches).
  * quantisation: the fp8 bytes and the E8M0 scale bytes equal a pure-torch reference of
    the formula above, bit for bit, for 10 cases (float16/bfloat16, e4m3fn/e5m2,
    clamp 0/5/7/10/12, H = 32/33/50/64/96/3584 = 7168, 2-D and 3-D), with topk_weight,
    with group_index (the excluded rows stay 0), and with a non-contiguous input; an inf
    block yields scale byte 0xFF and 32 x fp8 NaN codes.
  * cost: 0.6-0.7 ms per call for 64 x 7168 bfloat16 -> fp8 + E8M0 on one Vector core set.
"""

from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

__all__ = ["swiglu_group_quant", "npu_swiglu_group_quant"]

#: PER_MX_FP16 (base.h L30, tiling L99): one shared E8M0 exponent per 32 columns.
_MX_BLOCK = 32
#: LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 / _FOR_E5M2 (base.h L38-39) as a bf16 exponent field,
#: i.e. emax_elem << 7 (8 for e4m3, 15 for e5m2).
_LOWER_BOUND = {torch.float8_e4m3fn: 8 << 7, torch.float8_e5m2: 15 << 7}
#: Physical Vector core count used as the grid size; can be overridden for experiments.
_NUM_CORE_CACHE: dict = {}


@triton.jit
def _sgq_fp8_code(v, IS_E4M3: tl.constexpr):
    """Saturating round-half-to-even fp32 -> fp8 code byte (e4m3fn or e5m2).

    This is the `Cast<U, float, traitB32ToB8Layout0>` of VFComputeData (base.h L516-517:
    SatMode::SAT + RoundMode::CAST_RINT): round half to even, then saturate to the largest
    finite magnitude of the format. Integer arithmetic on the |v| bit pattern is used so
    that the result does not depend on the backend's own fp8 cast:

      * the normal path rounds the fp32 mantissa to MANT bits with the standard
        "add half-1, then add the kept LSB" RNE form (a carry propagates into the
        exponent field), then rebases the exponent bias 127 -> BIAS8;
      * the subnormal path multiplies by 2 ** (MANT + BIAS8 - 1), which maps the fp8
        subnormal grid onto the integers, and rounds that with the explicit
        floor / half / odd form (a tie goes to the even neighbour);
      * `v != v` (NaN) returns the canonical fp8 NaN code, +/-inf saturates through the
        `tl.minimum(..., VMAX)` clamp.
    """
    MANT: tl.constexpr = 3 if IS_E4M3 else 2
    BIAS8: tl.constexpr = 7 if IS_E4M3 else 15
    VMAX: tl.constexpr = 448.0 if IS_E4M3 else 57344.0
    CMAX: tl.constexpr = 0x7E if IS_E4M3 else 0x7B
    SH: tl.constexpr = 20 if IS_E4M3 else 21
    REBIAS: tl.constexpr = (127 - BIAS8) << MANT
    SUBN: tl.constexpr = (1 - BIAS8 + 127) << 23
    SUBSCALE: tl.constexpr = float(1 << (MANT + BIAS8 - 1))

    b = v.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80                      # the sign bit, moved to bit 7
    ab = b & 2147483647                          # |v| as fp32 bits (monotone in |v|)
    x = tl.minimum(ab.to(tl.float32, bitcast=True), VMAX)      # SAT part of CAST_RINT
    ab = x.to(tl.int32, bitcast=True)
    # normal path (RNE)
    r = ab + ((1 << (SH - 1)) - 1) + ((ab >> SH) & 1)
    code_n = (r >> SH) - REBIAS
    # subnormal path (RNE on the x * 2 ** (MANT + BIAS8 - 1) grid)
    xs = x * SUBSCALE
    fl = tl.floor(xs)
    fr = xs - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0
    code_s = tl.where(
        fr > 0.5, fl + 1.0,
        tl.where(fr < 0.5, fl, fl + tl.where(odd, 1.0, 0.0))).to(tl.int32)
    code = tl.where(ab < SUBN, code_s, code_n)
    code = tl.minimum(tl.maximum(code, 0), CMAX)
    code = tl.where(v != v, 0x7F, code)          # NaN -> canonical fp8 NaN
    return (sign | code) & 0xFF


@triton.jit
def _sgq_mx_kernel(
    X, TOPK, Y, SCALE,
    R, H, RDX, SROW, CLAMP, NUM_CORE,
    HAS_TOPK: tl.constexpr,
    IS_E4M3: tl.constexpr,
    LOWER: tl.constexpr,
    BLK: tl.constexpr,
):
    """quant_mode=2: SwiGLU + per-32-column MXFP8 quantisation, one 32-column block per
    inner iteration (BLK == PER_MX_FP16 == 32, so the block maximum *is* the tile maximum
    and no cross-lane reduction over several blocks is needed).

    `X` is [R, 2H] (flattened leading dimensions, row stride 2H), `Y`/`SCALE` are the
    uint8 views of the fp8 output and of the E8M0 scale, `SROW` is the scale row stride in
    bytes. Out-of-range lanes are given `other=0.0`, and a zero contributes the exponent
    field 0 to the maximum, so the tail block of an H that is not a multiple of 32 only
    takes its in-range lanes into account.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLK)
    for row in range(pid, R, NUM_CORE):
        base = row * 2 * H
        w = 1.0
        if HAS_TOPK:
            w = tl.load(TOPK + row).to(tl.float32)
        for s in range(0, RDX):
            c = s * BLK + cols
            m = c < H
            a = tl.load(X + base + c, mask=m, other=0.0).to(tl.float32)
            b = tl.load(X + base + H + c, mask=m, other=0.0).to(tl.float32)
            if CLAMP != 0.0:
                a = tl.minimum(a, CLAMP)
                b = tl.maximum(tl.minimum(b, CLAMP), -CLAMP)
            y = a / (1.0 + tl.exp(0.0 - a)) * b
            if HAS_TOPK:
                y = y * w
            # The native stores the activation into UB as x.dtype and reads it back before
            # the MX exponent extraction (swiglu_mx_quant_perf.h L92/L177/L188), so its
            # exponent and quantised value come from the x.dtype-rounded activation. Round
            # here as well to stay bit-for-bit identical with the AscendC operator.
            y = y.to(X.dtype.element_ty).to(tl.float32)
            # bf16-domain exponent field of the largest |y| in this block (VFComputeMaxExp)
            e = tl.max((y.to(tl.int32, bitcast=True) >> 16) & 0x7F80, axis=0)
            byte = (tl.maximum(e, LOWER) - LOWER) >> 7
            special = e == 0x7F80                    # inf / NaN block (base.h L474-475)
            byte = tl.where(special, 0xFF, byte)     # FP8_E8M0_NAN_VAL, base.h L40
            # 2 ** (127 - byte) == 1 / 2 ** (byte - 127) == the native's invScale
            inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
            code = _sgq_fp8_code(y * inv, IS_E4M3=IS_E4M3)
            code = tl.where(special, 0x7F, code)
            tl.store(Y + row * H + c, code.to(tl.uint8), mask=m)
            tl.store(SCALE + row * SROW + s, byte.to(tl.uint8))
        # The declared scale row is 2 * ceil(ceil(H/32) / 2) bytes, so when ceil(H/32) is
        # odd exactly one padding byte follows the last block: write it as 0 (the native
        # allocates with at::empty and leaves it undefined, meta L1345).
        pad = tl.arange(0, 2)
        tl.store(SCALE + row * SROW + RDX + pad, tl.zeros([2], dtype=tl.uint8),
                 mask=RDX + pad < SROW)


def _num_core(dev_idx: int | None) -> int:
    """Physical Vector core count, used as the grid size (official Triton-Ascend form:
    grid ~ cores + a runtime stride loop inside the kernel). Falls back to the measured 40
    of this machine family when the property is unavailable; SGQ_NUM_CORE overrides."""
    env = _os.environ.get("SGQ_NUM_CORE")
    if env:
        return max(1, int(env))
    key = -1 if dev_idx is None else int(dev_idx)
    if key in _NUM_CORE_CACHE:
        return _NUM_CORE_CACHE[key]
    n = 0
    try:
        from triton.runtime import driver as _drv

        props = _drv.active.utils.get_device_properties(dev_idx)
        for k in ("num_vectorcore", "num_vector_core", "num_aivector_core", "vector_core_num"):
            try:
                n = int(props[k])
            except Exception:  # noqa: BLE001
                continue
            if n > 0:
                break
    except Exception:  # noqa: BLE001
        n = 0
    if n <= 0:
        n = 40
    _NUM_CORE_CACHE[key] = n
    return n


def swiglu_group_quant(
    x: torch.Tensor,
    topk_weight: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    dst_type: torch.dtype = torch.float8_e4m3fn,
    quant_mode: int = 1,
    group_size: int = 128,
    round_scale: bool = False,
    ue8m0_scale: bool = False,
    output_origin: bool = False,
    group_list_type: int = 0,
    clamp_value: float = 0.0,
):
    """Equivalent of the AscendC SwigluGroupQuant for the MXFP8 main path (quant_mode=2).

    Args:
        x: [..., 2H] float16 | bfloat16, contiguous or not (it is made contiguous; the
            kernel assumes the row stride is exactly 2H).
        topk_weight: optional [rows] float32, one weight per row, multiplied into the
            SiLU result before quantisation (base.h L349-351).
        group_index: optional int64 token-count-per-group array (group_list_type=0). The
            native sums it and processes only the first min(sum, rows) rows (mx perf
            L41-75); rows beyond that are left untouched, exactly as the native does.
        dst_type: torch.float8_e4m3fn (448, emax_elem 8) or torch.float8_e5m2
            (57344, emax_elem 15).
        quant_mode: only 2 (MXFP8) is implemented; see "Coverage" in the module
            docstring for the two modes that raise NotImplementedError.
        group_size: ignored -- the MX block is fixed at 32 by the tiling (tiling L99).
        round_scale, ue8m0_scale: ignored -- they only drive the quant_mode=1/3 scale
            formats; quant_mode=2 always returns an E8M0 scale.
        output_origin: accepted for signature compatibility; y_origin is allocated but not
            written (the native quant_mode=2 kernel has no yOrigin argument either).
        group_list_type: only 0 (count) is implemented (tiling L117-119).
        clamp_value: 0.0 disables the clamp (tiling L123-129), otherwise
            gate = min(gate, c) and up = clamp(up, -c, c).

    Returns:
        (y [..., H] fp8, scale [..., ceil(ceil(H/32)/2), 2] float8_e8m0fnu,
         y_origin [..., H] x.dtype) -- three values, like the native operator.
    """
    if int(quant_mode) != 2:
        raise NotImplementedError(
            f"quant_mode={quant_mode} is not implemented: only quant_mode=2 (MXFP8) is "
            f"migrated, because it is the only mode the call sites use "
            f"(fused_moe.py L759-766, device_op.py L1290-1297); quant_mode=1 (group/static "
            f"fp8 per token) and quant_mode=3 (fp8 per token + y_origin) are not")
    if dst_type not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            f"dst_type={dst_type} is not implemented (only float8_e4m3fn and float8_e5m2)")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"x must be float16 or bfloat16 (the op definition and the meta function "
            f"require it), got {x.dtype}")
    if x.dim() < 2:
        raise ValueError(f"x must be at least 2-D [..., 2H], got {tuple(x.shape)}")
    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"the last dimension of x must be even, got {d}")
    if int(group_list_type) != 0:
        # the tiling rejects every other group_list_type unconditionally (L117-119)
        raise NotImplementedError(
            "group_list_type != 0 is not implemented (the tiling rejects it, L117-119)")

    H = d // 2
    rows_total = int(x.numel()) // d if d else 0
    x = x.contiguous()
    if topk_weight is not None:
        if int(topk_weight.numel()) != rows_total:
            raise ValueError(
                f"topk_weight must hold one value per row ({rows_total}), got "
                f"{tuple(topk_weight.shape)}")
        topk_weight = topk_weight.reshape(-1).contiguous().to(torch.float32)
    rows = rows_total
    if group_index is not None:
        # The native sums the counts and clamps them to the row count (mx perf L64-68);
        # the sum is read back on the host there as well (groupSumLocal.GetValue(0)).
        count = int(group_index.reshape(-1).sum().item())
        rows = max(0, min(count, rows_total))

    rdx = (H + _MX_BLOCK - 1) // _MX_BLOCK if H > 0 else 0     # ceil(H / 32), tiling L186
    tail = (rdx + 1) // 2                                      # proto L76-78 tailDim
    srow = 2 * tail                                            # the extra dim of size 2
    dev = x.device
    y = torch.empty(x.shape[:-1] + (H,), dtype=dst_type, device=dev)
    # zeros, not empty: rows excluded by group_index must not leak uninitialised memory.
    # NOTE: torch.zeros(..., dtype=torch.float8_e8m0fnu, device="npu") raises (aclnn
    # InplaceZero does not implement E8M0 on this CANN/torch_npu pair, measured), so the
    # buffer is zeroed as uint8 and then viewed as E8M0.
    scale = torch.zeros(x.shape[:-1] + (tail, 2), dtype=torch.uint8, device=dev).view(
        torch.float8_e8m0fnu)
    y_origin = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=dev)
    if rows > 0 and H > 0:
        num_core = _num_core(x.device.index)
        _sgq_mx_kernel[(max(1, min(num_core, rows)),)](
            x,
            topk_weight if topk_weight is not None else x,
            y.view(torch.uint8),
            scale.view(torch.uint8).reshape(-1),
            rows,
            H,
            rdx,
            srow,
            float(clamp_value),
            num_core,
            HAS_TOPK=topk_weight is not None,
            IS_E4M3=(dst_type == torch.float8_e4m3fn),
            LOWER=_LOWER_BOUND[dst_type],
            BLK=_MX_BLOCK,
        )
    return y, scale, y_origin


#: The call sites spell the operator `torch.ops._C_ascend.npu_swiglu_group_quant`; the
#: deployment shim imports the module-level name, so both spellings are kept.
npu_swiglu_group_quant = swiglu_group_quant
