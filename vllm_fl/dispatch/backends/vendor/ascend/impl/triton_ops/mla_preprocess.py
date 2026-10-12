"""mla_preprocess —— AscendC → Triton

来源：csrc/ascend/mla_preprocess/
接口（mla_preprocess_torch_adpt.h L33-42 + op_kernel/mla_preprocess_kernel.cpp L23-28）：
    Input : hiddenState  [N, hidden]                  bf16/fp16
            wdqkv        [qLora+kv+rope, hidden]      bf16   (transB matmul)
            gamma1       [qLora]                     bf16
            wuq          [headNum*(nope+rope), qLora] bf16  (transB matmul)
            gamma2       [kvLora]                    bf16
            wuk          [headNum, nope, kvLora]     bf16
            cos, sin     [N, rope]                   bf16
            slotMapping  [N]                         int32
            kv_cache / kv_cache_rope                 bf16（input 侧，本实现未用）
    Output: q            [N, headNum, kvLora(+rope)]  bf16
            q2           [N*headNum, rope]            bf16（cache_mode="krope_ctkv" 时）
            kv_cache_out0 / kv_cache_out1             bf16
            inner_out                                 （enable_inner_out，本实现未覆盖）
    Attr  : cache_mode ∈ {kvcache, krope_ctkv, int8_nzcache, nzcache}
            quant_mode ∈ {per_tensor_quant_asymm, per_token_quant_symm,
                          per_token_quant_asymm, no_quant}
            enable_inner_out

本文件实现的是 **quant_mode="no_quant" 的 bf16/fp16 路径**
（op_kernel/mla_preprocess_mix_bf16_nq.hpp，tiling key KEY_BF16_CACHEMODE_1_QUANTMODE_3=281），
这是唯一不含 int8 量化分支的路径。量化路径（mix_bf16 / mix_fp16 / mix_bf16_qdown）
未实现（见 docstring 末尾「简化说明」）。

公式（来源：mla_preprocess_mix_bf16_nq.hpp）
    ProcessCube (L1195-1211)：
        mm1 : s3 = hiddenState @ wdqkv^T                    # [N, qLora+kv+rope]
                                                              （L972 mm_1.Init，transB=true）
        mm2 : s2 = rmsnorm_q(s3[:, kv+rope:], gamma1) @ wuq^T # [N, headNum*(nope+rope)]
                                                              （L1001-1003 rmsNormQuant2 + L973 mm_2）
        mm3 : q_nope = einsum('thk,hkn->thn', s2[:, :, :nope], wuk)
                                                              （L974 mm_ein_sum, ein=true）
    RmsNormQuant (L287-407)：
        avg = 1/qLora;  ss = sum(x^2) * avg;  y = x / sqrt(ss + eps) * gamma1   # eps=1e-6
        （L375 Muls(avg) → L377 ReduceSumCustom → L379 Adds(eps) → L381 Sqrt
          → L384 factor=1/sqrt → L387 Muls → L390 Mul(g)）
    RmsNormAndRopeConvergence1 (L1029-1134)：对 s3[:, 0:kv] 做 rmsnorm（gamma2，eps=1e-6），
        对 s3[:, kv:kv+rope] 做 RoPE（HALF 旋转）：
        （L1081 rms = sqrt(sum(x^2)/kv + eps)；L1094-1116 RoPE）
            revert[0:h]  = x[h:2h]
            revert[h:2h] = x[0:h]
            k = cos*x ;  revert = sin*revert ;  revert *= [-1]*h+[+1]*h ;  out = k + revert
        即 out[i]      = cos[i]  * x[i]     - sin[i]  * x[i+h]
           out[h + i] = cos[h+i]* x[h+i]   + sin[h+i]* x[i]
    RopeFp16 (L27-248)：对 s2[:, :, nope:] 做同样的 HALF RoPE（cos2/sin2 按 token 索引），
        结果按 cache_mode 写到 q 行内偏移 concatSize 处（CACHE_MODE_KVCACHE）
        或独立的 q2（CACHE_MODE_KROPE_CTKV）。
    cache 写回（L1122-1130）：
        CACHE_MODE_KVCACHE(0)：kv_cache_out0[slot, 0:kv+rope] = [rmsnorm_kv | rope_k]
        CACHE_MODE_KROPE_CTKV(1)：kv_cache_out0[slot, 0:kv] = rmsnorm_kv
                                  kv_cache_out1[slot, 0:rope] = rope_k
    q 布局：CACHE_MODE_KVCACHE → [N, headNum, kv+rope]（nope 由 mm3 写入前 kv 列，
            rope 由 RopeFp16 写到 +kv 偏移）；CACHE_MODE_KROPE_CTKV → [N, headNum, kv]
            且 rope 单独写 q2[N*headNum, rope]。

简化说明（明确未覆盖的分支）
    1. quant_mode 支持 "no_quant"（原 nq 路径）与 "per_tensor_quant_asymm"（int8 量化路径）。
       per_token_quant_symm(1) / per_token_quant_asymm(2) 未实现。
    2. cache_mode 支持 "kvcache" / "krope_ctkv" / "nzcache"；"int8_nzcache"(2) 及其
       EinSumQuant(q_nope_scale) 未实现。
    3. enable_inner_out / inner_out 未实现；fp16 量化分支（mix_fp16.hpp）未实现。
    4. 输入侧 kv_cache / kv_cache_rope 未参与计算（与内核一致：只写不读）。
    5. 所有未覆盖参数一律显式 NotImplementedError，不静默忽略。

================================================================int8 量化路径（per_tensor_quant_asymm）的完整公式与出处见 mla_preprocess_int8.py 的模块 docstring；
流程编排见 mla_preprocess_int8.mla_preprocess_quant()。

================================================================
⚠️ 后端形态坑（910_93 + BiShengIR/FlagTree，实测复现，见 `_probe_2dstore_*.py`）
================================================================
1. 「2-D block + 2-D 掩码 store + 间接（runtime 载入的）行/列寻址」在本后端上会
   **静默写坏值**：既可能整 lane 不写（保持输出缓冲初值），也可能把**掩码外 lane 读到的
   越界垃圾（fp16 溢出 / NaN）写进掩码内 lane**——不报错、不崩、err 均值看着正常。
   本文件 `_kv_cache_kernel` 原 2-D 写法实测在 KVD=64/BLK_D=256、KVD=512/BLK_D=256 下
   各有 18 / 52 个 lane 被写入 1e3~6e4 / NaN（sentinel 检测：drop@expected=0、
   write@non-expected=0，但值本身是垃圾）。
   ⇒ 新写的 kernel 一律用「**标量行号 + 1-D 列 tile + 1-D 掩码 store**
   （运行时行循环）」，本文件 `_kv_cache_kernel` 已按此改写。
2. `tl.reshape` 把 2-D tile 拉平成 1-D 再 store 在本后端**同样大面积丢 lane**
   （实测 cols=1024、无任何掩码时仍丢 896/1024 个 lane）⇒ **不要用 reshape 当规避手段**。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["mla_preprocess"]


# ---------------------------------------------------------------------------
# kernel 1: OUT[M,N] = A[M,K] @ W[N,K]^T   （transB 的普通 matmul）
# ---------------------------------------------------------------------------
@triton.jit
def _mm_nt_kernel(
    A, W, OUT, M, K, N,
    BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        kmask = rk < K
        a = tl.load(A + rm[:, None] * K + rk[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


# ---------------------------------------------------------------------------
# kernel 2: y = x / sqrt(sum(x^2)/D + eps) * gamma      （沿最后一维）
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel(
    X, GAMMA, Y, M, D, AVG, EPS,
    BLK_M: tl.constexpr, BLK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    ss = tl.zeros((BLK_M,), dtype=tl.float32)
    for d0 in range(0, D, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        m2 = mmask[:, None] & (rd < D)[None, :]
        x = tl.load(X + rm[:, None] * D + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    inv = 1.0 / tl.sqrt(ss * AVG + EPS)
    for d0 in range(0, D, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        dm = rd < D
        m2 = mmask[:, None] & dm[None, :]
        x = tl.load(X + rm[:, None] * D + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
        y = x * inv[:, None] * g[None, :]
        tl.store(Y + rm[:, None] * D + rd[None, :], y.to(Y.dtype.element_ty), mask=m2)


# ---------------------------------------------------------------------------
# kernel 3: s3 的 [kv | rope] 段 —— rmsnorm(gamma2) + RoPE(HALF) + 写 KV cache
# ---------------------------------------------------------------------------
@triton.jit
def _kv_cache_kernel(
    S3, GAMMA, COS, SIN, SLOT, KVOUT, KROPOUT,
    M, KVD, RD, D2, KLINE, AVG, EPS,
    SPLIT_OUT: tl.constexpr, NZ: tl.constexpr,
    BLK_M: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
):
    """s3 的 [kv | rope] 段：rmsnorm(gamma2) + RoPE(HALF) + 写 KV cache。

    ⚠️ 形态约束（910_93 + BiShengIR/FlagTree 实测，见 _probe_2dstore_*.py）：
    这里**必须**用「标量行号 + 1-D 列 tile + 1-D 掩码 store」，不能写成
    `[BLK_M, BLK_D]` 2-D tile + 2-D 掩码 store + 间接 slot 寻址。原 2-D 写法在
    KVD 不是 BLK_D 的整数倍/小于 BLK_D 时会把**越界读到的垃圾值（fp16 溢出 / NaN）
    写进本应写入的 offset**（sentinel 检测：drop@expected=0、write@non-expected=0，
    但 18~52 个 lane 的值是 1e3~6e4/NaN），且不报错、不崩。1-D 改写后 0 个。
    单测 Test 用例：`小配置 bf16 krope_ctkv N=6` / `真实维度 bf16 krope_ctkv N=4`。
    """
    pid = tl.program_id(0)
    rd0 = tl.arange(0, BLK_D)
    h = tl.arange(0, HALF)
    hm = h < HALF
    for i in range(BLK_M):
        row = pid * BLK_M + i
        if row < M:
            ssv = tl.zeros([BLK_D], dtype=tl.float32)
            for d0 in range(0, KVD, BLK_D):
                rd = d0 + rd0
                x = tl.load(S3 + row * D2 + rd, mask=rd < KVD, other=0.0).to(tl.float32)
                ssv += x * x
            inv = 1.0 / tl.sqrt(tl.sum(ssv, axis=0) * AVG + EPS)

            slot = tl.load(SLOT + row)
            for d0 in range(0, KVD, BLK_D):
                rd = d0 + rd0
                dm = rd < KVD
                x = tl.load(S3 + row * D2 + rd, mask=dm, other=0.0).to(tl.float32)
                g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
                y = x * inv * g
                if NZ:
                    # nzcache: off(n) = outer*(128*KVD) + (n//16)*2048 + inner*16 + (n%16)
                    so = (slot // 128) * (128 * KVD) + (rd // 16) * 2048 \
                         + (slot % 128) * 16 + (rd % 16)
                elif SPLIT_OUT:
                    so = slot * KVD + rd
                else:
                    so = slot * KLINE + rd
                tl.store(KVOUT + so, y.to(KVOUT.dtype.element_ty), mask=dm)

            # ---- RoPE(HALF) on s3[row, KVD:KVD+RD] ----
            base = row * D2 + KVD
            xa = tl.load(S3 + base + h, mask=hm, other=0.0).to(tl.float32)
            xb = tl.load(S3 + base + HALF + h, mask=hm, other=0.0).to(tl.float32)
            ca = tl.load(COS + row * RD + h, mask=hm, other=0.0).to(tl.float32)
            cb = tl.load(COS + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
            sa = tl.load(SIN + row * RD + h, mask=hm, other=0.0).to(tl.float32)
            sb = tl.load(SIN + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
            oa = ca * xa - sa * xb
            ob = sb * xa + cb * xb
            if NZ:
                # rope: off(n) = outer*(128*RD) + (n//16)*2048 + inner*16 + (n%16)
                base2 = (slot // 128) * (128 * RD) + (slot % 128) * 16
                soa = base2 + (h // 16) * 2048 + (h % 16)
                sob = base2 + ((HALF + h) // 16) * 2048 + ((HALF + h) % 16)
                tl.store(KROPOUT + soa, oa.to(KROPOUT.dtype.element_ty), mask=hm)
                tl.store(KROPOUT + sob, ob.to(KROPOUT.dtype.element_ty), mask=hm)
            elif SPLIT_OUT:
                tl.store(KROPOUT + slot * RD + h, oa.to(KROPOUT.dtype.element_ty), mask=hm)
                tl.store(KROPOUT + slot * RD + HALF + h, ob.to(KROPOUT.dtype.element_ty), mask=hm)
            else:
                tl.store(KVOUT + slot * KLINE + KVD + h, oa.to(KVOUT.dtype.element_ty), mask=hm)
                tl.store(KVOUT + slot * KLINE + KVD + HALF + h, ob.to(KVOUT.dtype.element_ty), mask=hm)


# ---------------------------------------------------------------------------
# kernel 4: q_nope = s2[:,:, :nope] @ wuk[h]  + q 的 rope 部分
# ---------------------------------------------------------------------------
@triton.jit
def _q_head_kernel(
    S2, WUK, COS, SIN, Q, Q2,
    M, HN, HD, NOPE, RD, KVD, QLINE,
    ROPE_IN_Q: tl.constexpr,
    BLK_M: tl.constexpr, BLK_K: tl.constexpr, BLK_N: tl.constexpr,
    HALF: tl.constexpr,
):
    pm = tl.program_id(0)
    h = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    rk = tl.arange(0, BLK_K)          # BLK_K == NOPE
    kmask = rk < NOPE
    S2STRIDE = HN * HD
    QSTRIDE = HN * QLINE
    # A = s2[rm, h, 0:NOPE]
    a = tl.load(S2 + rm[:, None] * S2STRIDE + h * HD + rk[None, :],
                mask=mmask[:, None] & kmask[None, :], other=0.0)
    for n0 in range(0, KVD, BLK_N):
        rn = n0 + tl.arange(0, BLK_N)
        nmask = rn < KVD
        b = tl.load(WUK + h * NOPE * KVD + rk[:, None] * KVD + rn[None, :],
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b)
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + rn[None, :],
                 acc.to(Q.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])

    hh = tl.arange(0, HALF)
    hm = mmask[:, None] & (hh < HALF)[None, :]
    base = rm[:, None] * S2STRIDE + h * HD + NOPE
    xa = tl.load(S2 + base + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    xb = tl.load(S2 + base + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    ca = tl.load(COS + rm[:, None] * RD + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    cb = tl.load(COS + rm[:, None] * RD + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    sa = tl.load(SIN + rm[:, None] * RD + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    sb = tl.load(SIN + rm[:, None] * RD + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    oa = ca * xa - sa * xb
    ob = sb * xa + cb * xb
    if ROPE_IN_Q:
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + KVD + hh[None, :],
                 oa.to(Q.dtype.element_ty), mask=hm)
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + KVD + HALF + hh[None, :],
                 ob.to(Q.dtype.element_ty), mask=hm)
    else:
        row = rm[:, None] * HN + h
        tl.store(Q2 + row * RD + hh[None, :], oa.to(Q2.dtype.element_ty), mask=hm)
        tl.store(Q2 + row * RD + HALF + hh[None, :], ob.to(Q2.dtype.element_ty), mask=hm)


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------
_CACHE_MODES = {"kvcache": 0, "krope_ctkv": 1, "int8_nzcache": 2, "nzcache": 3}
_QUANT_MODES = {"per_tensor_quant_asymm": 0, "per_token_quant_symm": 1,
                "per_token_quant_asymm": 2, "no_quant": 3}

_UNSUPPORTED_HINT = (
    "\n本 Triton 版 mla_preprocess 目前只覆盖 cache_mode ∈ {kvcache, krope_ctkv} 且 "
    "quant_mode='no_quant' 的路径（tiling key 281 / 256 / 257 的非量化分支）。\n"
    "int8 量化路径（per_tensor_quant_asymm / per_token_quant_symm，含 quant_scale/offset/bias、"
    "descale、ctkv_scale、q_nope_scale、nzcache 的 NZ 写回、enable_inner_out）尚未实现。\n"
    "内核侧公式已提取并记录在本模块 docstring 末尾「int8 量化路径（待实现）」一节。"
)


def _reject_unsupported(**kw):
    """调用点会传但本实现未覆盖的参数：显式 NotImplementedError，绝不静默忽略。"""
    bad = [k for k, v in kw.items() if v is not None and not (isinstance(v, bool) and v is False)]
    if bad:
        raise NotImplementedError(
            "mla_preprocess: 以下参数尚未实现，不能静默忽略: " + ", ".join(sorted(bad))
            + _UNSUPPORTED_HINT)


def mla_preprocess(
    hidden_state: torch.Tensor,
    # ↓ 位置参数顺序**完全对齐真实调用点**（vllm_fl device_op.py:349）：
    #   (hidden_states, wd_qkv, deq_scale_qkv, gamma1, beta1, wu_q, qb_deq_scl,
    #    gamma2, cos, sin, W_UK_T, decode_k_nope, decode_k_pe, slot_mapping)
    wdqkv: torch.Tensor | None = None,
    deq_scale_qkv: torch.Tensor | None = None,
    gamma1: torch.Tensor | None = None,
    beta1: torch.Tensor | None = None,
    wuq: torch.Tensor | None = None,
    qb_deq_scl: torch.Tensor | None = None,
    gamma2: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    wuk: torch.Tensor | None = None,
    decode_k_nope: torch.Tensor | None = None,
    decode_k_pe: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    # ↓ 调用点关键字参数
    *,
    quant_scale0: torch.Tensor | None = None,
    quant_offset0: torch.Tensor | None = None,
    bias0: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    quant_offset1: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    ctkv_scale: torch.Tensor | None = None,
    q_nope_scale: torch.Tensor | None = None,
    cache_mode: str = "krope_ctkv",
    quant_mode: str = "no_quant",
    enable_inner_out: bool = False,
    inner_out: torch.Tensor | None = None,
    q_out0: torch.Tensor | None = None,
    kv_cache_out0: torch.Tensor | None = None,
    q_out1: torch.Tensor | None = None,
    kv_cache_out1: torch.Tensor | None = None,
    # ↓ 本实现内部参数（也接受 snake_case 别名）
    q: torch.Tensor | None = None,
    q2: torch.Tensor | None = None,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    head_num: int = 128,
    num_slots: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """等价于 AscendC mla_preprocess 的 **no_quant** 路径（见模块 docstring）。

    位置参数顺序与真实调用点（`vllm_fl` device_op.py:349 附近）**一致**，
    关键字参数也使用调用点原名；同时兼容 `q=/q2=` 等内部别名。

    ⚠️ 目前只有 `quant_mode="no_quant"` + `cache_mode ∈ {kvcache, krope_ctkv}`
    是可运行的；**任何量化相关参数一旦非空就抛 `NotImplementedError`**（不静默忽略）。

    权重 wdqkv 的行序为 [kv_lora_rank | qk_rope_head_dim | q_lora_rank]（与 tiling 一致：
    input_stride_ = splitSizeOne_ = kv+rope 为 qLoRA 段起点）。

    Returns:
        (q, kv_cache_out0, q2, kv_cache_out1)   —— 对应 (q_out0, kv_cache_out0,
        q_out1, kv_cache_out1)
    """
    # ---- 调用点别名回填 ----
    if q_out0 is not None:
        q = q_out0
    if q_out1 is not None:
        q2 = q_out1
    if slot_mapping is None:
        raise ValueError("mla_preprocess 需要 slot_mapping")

    # ---- 未实现参数：显式报错 ----
    _reject_unsupported(beta1=beta1, decode_k_nope=decode_k_nope, decode_k_pe=decode_k_pe)
    if cache_mode != "int8_nzcache":   # ctkv_scale / q_nope_scale 仅 int8_nzcache 使用
        _reject_unsupported(ctkv_scale=ctkv_scale, q_nope_scale=q_nope_scale)
    # enable_inner_out：inner_out = M1 之后 qLoRA 段的 rmsnorm 结果（bf16, [N, q_lora_rank]）
    if enable_inner_out and (inner_out is None or inner_out.numel() == 0):
        raise ValueError("enable_inner_out=True 需要传入非空 inner_out [N, q_lora_rank]")
    if quant_mode not in _QUANT_MODES:
        raise ValueError(f"未知 quant_mode {quant_mode!r}；合法值 {sorted(_QUANT_MODES)}")
    qm = _QUANT_MODES[quant_mode]
    if qm in (1, 2):
        raise NotImplementedError(
            f"mla_preprocess: quant_mode={quant_mode!r} 尚未实现" + _UNSUPPORTED_HINT)
    is_quant = (qm == 0)
    if not is_quant:
        # no_quant 路径下这些量化参数必须为空，否则就是静默忽略
        _reject_unsupported(
            deq_scale_qkv=deq_scale_qkv, qb_deq_scl=qb_deq_scl,
            quant_scale0=quant_scale0, quant_offset0=quant_offset0, bias0=bias0,
            quant_scale1=quant_scale1, quant_offset1=quant_offset1, bias1=bias1)
    if gamma1 is None or wuq is None or gamma2 is None or wuk is None or cos is None or sin is None:
        raise ValueError("mla_preprocess 需要 gamma1/wuq/gamma2/wuk/cos/sin")
    if cache_mode not in ("kvcache", "krope_ctkv", "nzcache", "int8_nzcache"):
        raise NotImplementedError(
            f"mla_preprocess: cache_mode={cache_mode!r} 未实现" + _UNSUPPORTED_HINT)
    if cache_mode == "int8_nzcache" and (ctkv_scale is None or q_nope_scale is None):
        raise ValueError("cache_mode='int8_nzcache' 需要 ctkv_scale 与 q_nope_scale")
    Dq = int(q_lora_rank)
    Dkv = int(kv_lora_rank)
    Dnope = int(qk_nope_head_dim)
    Drd = int(qk_rope_head_dim)
    HN = int(head_num)
    N, Hd = int(hidden_state.shape[0]), int(hidden_state.shape[1])
    Dkvr = Dkv + Drd
    HD = Dnope + Drd
    dev = hidden_state.device
    dt = hidden_state.dtype
    if num_slots is None:
        num_slots = int(slot_mapping.detach().max().item()) + 1

    nz = (cache_mode in ("nzcache", "int8_nzcache"))
    i8nz = (cache_mode == "int8_nzcache")
    split_out = (cache_mode in ("krope_ctkv", "nzcache", "int8_nzcache"))
    qline = Dkv if split_out else (Dkv + Drd)
    nout = ceil_div(max(num_slots, 1), 128)
    if q is None:
        q = torch.empty((N, HN, Dkv if i8nz else qline),
                        dtype=(torch.int8 if i8nz else dt), device=dev)
    if split_out:
        if q2 is None:
            q2 = torch.empty((N * HN, Drd), dtype=dt, device=dev)
        if kv_cache_out0 is None:
            kv_cache_out0 = torch.empty(
                (nout * 128 * Dkv,) if nz else (num_slots, Dkv),
                dtype=(torch.int8 if i8nz else dt), device=dev)
        if kv_cache_out1 is None:
            kv_cache_out1 = torch.empty((nout * 128 * Drd,) if nz else (num_slots, Drd),
                                        dtype=dt, device=dev)
    else:
        if q2 is None:
            q2 = torch.empty((0,), dtype=dt, device=dev)
        if kv_cache_out0 is None:
            kv_cache_out0 = torch.empty((num_slots, Dkvr), dtype=dt, device=dev)
        if kv_cache_out1 is None:
            kv_cache_out1 = torch.empty((0,), dtype=dt, device=dev)

    if enable_inner_out:
        assert tuple(inner_out.shape) == (N, Dq) and inner_out.dtype == dt, \
            f"inner_out 需为 [{N},{Dq}] {dt}，实际 {tuple(inner_out.shape)} {inner_out.dtype}"
    hs = hidden_state.contiguous()
    wd = wdqkv.contiguous()
    wq = wuq.contiguous()
    wk = wuk.contiguous()
    cs = cos.contiguous()
    sn = sin.contiguous()
    slot = slot_mapping.to(torch.int32).contiguous()
    BLK_M_MM, BLK_N_MM, BLK_K_MM = 16, 128, 64
    MM1N = Dkvr + Dq
    D2 = HN * HD

    if is_quant:
        from .mla_preprocess_int8 import mla_preprocess_quant
        return mla_preprocess_quant(
            hs, wd, wq, wk, cs, sn, slot, dt, dev, N, Hd, Dq, Dkv, Dnope, Drd, HN,
            Dkvr, HD, qline, nz, q, q2, kv_cache_out0, kv_cache_out1,
            gamma1, gamma2, quant_scale0, quant_offset0, bias0, deq_scale_qkv,
            quant_scale1, quant_offset1, bias1, qb_deq_scl, float(eps),
            cache_mode=cache_mode, ctkv_scale=ctkv_scale, q_nope_scale=q_nope_scale,
            inner_out=inner_out if enable_inner_out else None)
    # --- mm1: s3 = hidden @ wdqkv^T  （拆成 qLoRA 段与 kv/rope 段）---
    w_q = wd[Dkvr:Dkvr + Dq].contiguous()
    w_kv = wd[0:Dkvr].contiguous()
    s3q = torch.empty((N, Dq), dtype=dt, device=dev)
    s3kv = torch.empty((N, Dkvr), dtype=dt, device=dev)

    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(Dq, BLK_N_MM))](
        hs, w_q, s3q, N, Hd, Dq,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)
    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(Dkvr, BLK_N_MM))](
        hs, w_kv, s3kv, N, Hd, Dkvr,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)

    # --- rmsnorm(s3q, gamma1) ---
    q_lora = inner_out if (enable_inner_out and inner_out is not None) else torch.empty((N, Dq), dtype=dt, device=dev)
    BLK_M_R = 16
    BLK_D_R = 256
    _rmsnorm_kernel[(ceil_div(N, BLK_M_R),)](
        s3q, gamma1.contiguous(), q_lora, N, Dq, 1.0 / Dq, float(eps),
        BLK_M=BLK_M_R, BLK_D=BLK_D_R)

    # --- mm2: s2 = q_lora @ wuq^T ---
    s2 = torch.empty((N, D2), dtype=dt, device=dev)
    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(D2, BLK_N_MM))](
        q_lora, wq, s2, N, Dq, D2,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)

    # --- mm3 (per-head) + q rope ---
    _q_head_kernel[(ceil_div(N, 16), HN)](
        s2, wk, cs, sn, q, q2,
        N, HN, HD, Dnope, Drd, Dkv, qline,
        ROPE_IN_Q=bool(not split_out),
        BLK_M=16, BLK_K=next_pow2(Dnope), BLK_N=128, HALF=Drd // 2)

    # --- kv / rope 缓存 ---
    _kv_cache_kernel[(ceil_div(N, 16),)](
        s3kv, gamma2.contiguous(), cs, sn, slot, kv_cache_out0, kv_cache_out1,
        N, Dkv, Drd, Dkvr, qline, 1.0 / Dkv, float(eps),
        SPLIT_OUT=bool(split_out), NZ=bool(nz),
        BLK_M=16, BLK_D=256, HALF=Drd // 2)

    return q, kv_cache_out0, q2, kv_cache_out1
