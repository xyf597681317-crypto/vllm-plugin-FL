"""inplace_partial_rotary_mul —— AscendC → Triton

来源：csrc/ascend/attention/inplace_partial_rotary_mul/
调用点签名（csrc/ascend/torch_binding.cpp L278-286 / L1246）：
    inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2,
                               str rotary_mode, int[] partial_slice) -> ()
    其中 rotary_mode 是**字符串**，映射到 AscendC 的 mode 整数：
        {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
    真实调用点示例（vllm_fl/attention/ascend/dsa_v1.py L1762-1768）：
        inplace_partial_rotary_mul(o_proj_input.unsqueeze(1), cos, -sin,
                                   rotary_mode="interleave",
                                   partial_slice=[nope_head_dim, head_dim])
    即 r2 传的是 **-sin**，且只旋转 [512, 576) 这一段。
接口（op_host/inplace_partial_rotary_mul_def.cpp L22-46）：
    Input : x [dim0, dim1, dim2, D_all]  (4D, fp16/fp32/bf16)
            cos [r0, r1, 1, headDim]     (与 x 同 dtype 或 fp32)
            sin [r0, r1, 1, headDim]     (与 cos 同 dtype)
    Output: x（同名输出，原地写回；非切片区间原样保留）
    Attr  : mode         = 0 HALF / 1 INTERLEAVE / 2 QUARTER / 3 DEEPSEEK_INTERLEAVE
            partial_slice = [start, end]  → 只旋转最后一维的 [start, end) 区间
                          → headDim = end - start

shape 规则（op_host/inplace_partial_rotary_mul_a3_tiling.cpp）：
    - x 必须是 4D（L279-281）；cos/sin 必须 4D、形状完全一致（L282-292）
    - cos/sin 的 dim2 必须为 1（L330-332）；dim3 == end - start == headDim（L324-326）
    - dim0 必须与 x 的 dim0 相等（L327-329）
    - A2/A3 路径 mode 只接受 1（INTERLEAVE）（L302-305）
    - dim1 可以是 1 或 x.dim1（L336-346：r1dim1_==1 走广播，否则逐 head）
      → cos/sin 对 x 的 (dim0, dim1, dim2) 做标准广播（dim2 恒为 1，广播到 N 个 head）

公式出处：

  (a) A2/A3 membase 内核 op_kernel/inplace_partial_rotary_mul.h L227-299（ABA Process）
        y   = x * cos                       # L277  ComputeMul(y, x, r1)
        x'  = gather(x, ids)  ids[i] = i^1  # L279-281 相邻元素两两交换
        x'  = x' * sin                      # L283
        x'  = x' * (-1)  where i even       # L286 InterleavedInversion：掩码 0x5555…
        y   = y + x'                        # L288
      → INTERLEAVE：
        out[2k]   = x[2k]*cos[2k]   - x[2k+1]*sin[2k]
        out[2k+1] = x[2k+1]*cos[2k+1] + x[2k]*sin[2k+1]

  (b) arch35 regbase VF op_kernel/apply_rotary_pos_emb_common.h：
        L43-48   RotaryPosEmbeddingMode { HALF=0, INTERLEAVE=1, QUARTER=2, DEEPSEEK_INTERLEAVE=3 }
        L50-53   HALF 公式注释（实现 L82-112）
        L115-120 QUARTER 公式注释（实现 L157-201）
        L205-317 INTERLEAVE（DeInterleave/Muls(-1)/Interleave，与 (a) 完全一致，互为交叉验证）
        L319-451 DEEPSEEK_INTERLEAVE（输入按对交错，输出前半=偶流、后半=奇流）

  四种 mode 的等价写法（本文件实现，P = 每个配对块的长度）：
        out[a] = x[a]*cos[a] - x[b]*sin[a]
        out[wa] = ... 见下表（读下标 (a,b)，写下标 (wa,wb)）
            HALF(0)   : a=k,        b=P+k,       wa=a,          wb=b
            QUARTER(2): HALF 在 x[0:2P] 与 x[2P:4P] 上各做一次
            INTERLEAVE(1)      : a=2k, b=2k+1, wa=a,   wb=b
            DEEPSEEK(3)        : a=2k, b=2k+1, wa=k,   wb=P+k

  注：AscendC 原算子是**原地**写回 x；本 Triton 实现返回一份新的 y（x 未修改），
      语义等价（非切片区间逐元素等于 x）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["inplace_partial_rotary_mul", "MODE_HALF", "MODE_INTERLEAVE",
           "MODE_QUARTER", "MODE_DEEPSEEK_INTERLEAVE"]

MODE_HALF = 0
MODE_INTERLEAVE = 1
MODE_QUARTER = 2
MODE_DEEPSEEK_INTERLEAVE = 3


@triton.jit
def _rope_pair(X, COS, SIN, Y, x_base, cs_base,
               a_off, b_off, wa_off, wb_off, km):
    """一组配对（a,b）→（wa,wb）的 RoPE（一维向量，无 2-D 广播）。

        out[wa] = x[a]*cos[wa] - x[b]*sin[wa]
        out[wb] = x[b]*cos[wb] + x[a]*sin[wb]

    cos/sin 在 **写** 下标处取样（对四种 mode 都成立，见模块 docstring）。
    """
    xa = tl.load(X + x_base + a_off, mask=km, other=0.0).to(tl.float32)
    xb = tl.load(X + x_base + b_off, mask=km, other=0.0).to(tl.float32)
    ca = tl.load(COS + cs_base + wa_off, mask=km, other=0.0).to(tl.float32)
    sa = tl.load(SIN + cs_base + wa_off, mask=km, other=0.0).to(tl.float32)
    cb = tl.load(COS + cs_base + wb_off, mask=km, other=0.0).to(tl.float32)
    sb = tl.load(SIN + cs_base + wb_off, mask=km, other=0.0).to(tl.float32)
    oa = xa * ca - xb * sa
    ob = xb * cb + xa * sb
    tl.store(Y + x_base + wa_off, oa.to(Y.dtype.element_ty), mask=km)
    tl.store(Y + x_base + wb_off, ob.to(Y.dtype.element_ty), mask=km)


@triton.jit
def _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE: tl.constexpr):
    """交错配对 (2k, 2k+1) 的 RoPE —— 避免「一维跨步 + 掩码」形态。

    ⚠️ 为什么要这么写：Ascend/FlagTree 的 FLIR pass
    `InterleaveStatusWithMaskOptimization`（`InterleaveOptimization.cpp:635`）
    对 `tl.load(X + 2*k, mask=km)` + `tl.load(X + 2*k+1, mask=km)` 这种
    「一维跨步 + 掩码」访问直接 assert 失败（`assert(llvm::isa<Attribute>(
    extractSizes.back()))`，动态 size 触发），进程 abort。
    把配对展开成 2-D [BLK, 2] 块、掩码按行广播后，编译与执行都正常
    （2026-10-12 在 910_93 + FlagTree 上用最小复现验证过两侧）。

    ILV_WRITE=True  : 读、写、cos/sin 全部按 (2k, 2k+1) 交错（MODE_INTERLEAVE）
    ILV_WRITE=False : 只读交错，cos/sin 与写按 (k, P+k) 连续（MODE_DEEPSEEK_INTERLEAVE）
    """
    j = tl.arange(0, 2)
    m0 = j[None, :] == 0
    m2 = km[:, None]
    off2 = 2 * k[:, None] + j[None, :]

    x2 = tl.load(X + x_base + off2, mask=m2, other=0.0).to(tl.float32)
    xa = tl.sum(tl.where(m0, x2, 0.0), axis=1)
    xb = tl.sum(tl.where(m0, 0.0, x2), axis=1)

    if ILV_WRITE:
        c2 = tl.load(COS + cs_base + off2, mask=m2, other=0.0).to(tl.float32)
        s2 = tl.load(SIN + cs_base + off2, mask=m2, other=0.0).to(tl.float32)
        ca = tl.sum(tl.where(m0, c2, 0.0), axis=1)
        cb = tl.sum(tl.where(m0, 0.0, c2), axis=1)
        sa = tl.sum(tl.where(m0, s2, 0.0), axis=1)
        sb = tl.sum(tl.where(m0, 0.0, s2), axis=1)
        oa = xa * ca - xb * sa
        ob = xb * cb + xa * sb
        y2 = tl.where(m0, oa[:, None], ob[:, None])
        tl.store(Y + x_base + off2, y2.to(Y.dtype.element_ty), mask=m2)
    else:
        ca = tl.load(COS + cs_base + k, mask=km, other=0.0).to(tl.float32)
        sa = tl.load(SIN + cs_base + k, mask=km, other=0.0).to(tl.float32)
        cb = tl.load(COS + cs_base + P + k, mask=km, other=0.0).to(tl.float32)
        sb = tl.load(SIN + cs_base + P + k, mask=km, other=0.0).to(tl.float32)
        oa = xa * ca - xb * sa
        ob = xb * cb + xa * sb
        tl.store(Y + x_base + k, oa.to(Y.dtype.element_ty), mask=km)
        tl.store(Y + x_base + P + k, ob.to(Y.dtype.element_ty), mask=km)


@triton.jit
def _inplace_partial_rotary_mul_kernel(
    X, COS, SIN, Y,
    N_HEADS, SEQ, D,
    C1, HEAD_DIM, START, P,
    BR: tl.constexpr,      # cos/sin 的 dim0 是否参与广播（即是否等于 x.dim0）
    SR: tl.constexpr,      # cos/sin 的 dim1 是否参与广播（即是否等于 x.dim1）
    MODE: tl.constexpr,
    BLK_P: tl.constexpr,
):
    # 一个 program 处理一行 (dim0, dim1, dim2)：全部是标量/一维，避免 2-D 广播掩码
    row = tl.program_id(0)
    ss_ = (row // N_HEADS) % SEQ
    bb_ = row // (N_HEADS * SEQ)
    if BR:
        cbr = bb_ * C1
    else:
        cbr = 0
    if SR:
        csr = ss_
    else:
        csr = 0
    cs_base = (cbr + csr) * HEAD_DIM
    x_base = row * D + START

    k = tl.arange(0, BLK_P)
    km = k < P

    # MODE 取字面量：@triton.jit 内不能引用模块级全局变量
    if MODE == 1:          # MODE_INTERLEAVE
        _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE=True)
    elif MODE == 3:        # MODE_DEEPSEEK_INTERLEAVE
        _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE=False)
    elif MODE == 0:        # MODE_HALF
        _rope_pair(X, COS, SIN, Y, x_base, cs_base, k, P + k, k, P + k, km)
    else:                  # MODE_QUARTER：长度 4P，前后两半各做一次 HALF 配对
        for j in tl.static_range(2):
            base = 2 * j * P
            _rope_pair(X, COS, SIN, Y, x_base, cs_base,
                       base + k, base + P + k, base + k, base + P + k, km)


def _parse_rotary_mode(rotary_mode) -> int:
    """把调用点的 rotary_mode 归一化成 AscendC 的 mode 枚举值。

    出处：csrc/ascend/torch_binding.cpp L278-286
        static const std::unordered_map<std::string, int> modes{
            {"half", 0}, {"interleave", 1}, {"quarter", 2}, {"interleave-half", 3}};
    """
    if isinstance(rotary_mode, str):
        table = {"half": MODE_HALF, "interleave": MODE_INTERLEAVE,
                 "quarter": MODE_QUARTER, "interleave-half": MODE_DEEPSEEK_INTERLEAVE}
        key = rotary_mode.strip().lower()
        if key not in table:
            raise ValueError(f"rotary_mode 只支持 {sorted(table)}（或 int 0..3），实际 {rotary_mode!r}")
        return table[key]
    if isinstance(rotary_mode, int):
        if rotary_mode not in (MODE_HALF, MODE_INTERLEAVE, MODE_QUARTER, MODE_DEEPSEEK_INTERLEAVE):
            raise ValueError(f"rotary_mode(int) 只支持 0..3，实际 {rotary_mode}")
        return rotary_mode
    raise ValueError(f"rotary_mode 必须是 str 或 int，实际 {type(rotary_mode)}")


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor | None = None,
    r2: torch.Tensor | None = None,
    rotary_mode="interleave",
    partial_slice: tuple[int, int] | list[int] | None = None,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    mode: int | None = None,
) -> torch.Tensor:
    """等价于 AscendC 的 InplacePartialRotaryMul（返回更新后的完整张量）。

    调用点签名（csrc/ascend/torch_binding.cpp L1246）：
        inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2,
                                   str rotary_mode, int[] partial_slice) -> ()
    ⇒ 位置参数顺序为 (x, r1, r2, rotary_mode, partial_slice)，与 vllm_fl 调用点一致
      （如 vllm_fl/attention/ascend/dsa_v1.py L1762-1768：x, cos, -sin,
        rotary_mode="interleave", partial_slice=[nope_head_dim, head_dim]）。
      注意调用点传的 r2 是 **-sin**，本实现按 r1/r2 的通用公式计算，天然正确。

    Args:
        x:   4D [dim0, dim1, dim2, D_all]，fp16 / bf16 / fp32（op_host def.cpp L24/L54 声明）
        r1:  cos，4D [r0, r1, 1, headDim]，与 x 同 dtype 或 fp32（混合精度）
        r2:  sin（调用点常传 -sin）
        rotary_mode: "half"(0) / "interleave"(1) / "quarter"(2) / "interleave-half"(3)，
                     也接受 int 0..3
        partial_slice: [start, end]；缺省或 [0,0] 视为整段 [0, D_all)
        cos/sin: r1/r2 的关键字别名（向后兼容）
        mode:    AscendC 的 mode 枚举别名（向后兼容，优先级最高）

    Returns:
        y: 与 x 同 shape/dtype；仅最后一维 [start, end) 被旋转，其余等于 x。
    """
    if mode is not None:
        rm = _parse_rotary_mode(mode)
        if rotary_mode not in (None, "interleave") and _parse_rotary_mode(rotary_mode) != rm:
            raise ValueError(f"mode={mode} 与 rotary_mode={rotary_mode!r} 冲突")
    else:
        rm = _parse_rotary_mode(rotary_mode)

    if r1 is None:
        r1 = cos
    if r2 is None:
        r2 = sin
    if r1 is None or r2 is None:
        raise ValueError("必须提供 r1/r2（或 cos/sin）")
    cos, sin = r1, r2
    if x.dim() != 4:
        raise ValueError(f"x 必须是 4D（AscendC 接口要求），实际 {tuple(x.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin 形状必须一致: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.dim() != 4:
        raise ValueError(f"cos/sin 必须是 4D，实际 {tuple(cos.shape)}")
    if cos.shape[2] != 1:
        raise ValueError(f"cos/sin 的 dim2 必须为 1，实际 {cos.shape[2]}")
    if cos.shape[0] not in (1, x.shape[0]):
        raise ValueError(f"cos/sin 的 dim0 必须是 1 或 x.dim0={x.shape[0]}，实际 {cos.shape[0]}")
    if cos.shape[1] not in (1, x.shape[1]):
        raise ValueError(f"cos/sin 的 dim1 必须是 1 或 x.dim1={x.shape[1]}，实际 {cos.shape[1]}")
    if rm not in (MODE_HALF, MODE_INTERLEAVE, MODE_QUARTER, MODE_DEEPSEEK_INTERLEAVE):
        raise ValueError(f"mode 必须是 0/1/2/3，实际 {rm}")

    d_all = x.shape[3]
    # AscendC 的 attr 默认值是 {0, 0}，语义上等同于"不显式指定切片"→ 整段
    if partial_slice is None or (int(partial_slice[0]) == 0 and int(partial_slice[1]) == 0):
        start, end = 0, d_all
    else:
        start, end = int(partial_slice[0]), int(partial_slice[1])
    if start < 0 or end > d_all or end <= start:
        raise ValueError(f"partial_slice=[{start},{end}] 非法（x 最后一维 = {d_all}）")
    head_dim = end - start
    if head_dim != cos.shape[3]:
        raise ValueError(f"headDim={head_dim} 必须等于 cos/sin 的最后一维 {cos.shape[3]}")

    if rm == MODE_QUARTER:
        if head_dim % 4 != 0:
            raise ValueError(f"mode=2(QUARTER) 要求 headDim%4==0，实际 {head_dim}")
        p = head_dim // 4
    else:
        if head_dim % 2 != 0:
            raise ValueError(f"mode={rm} 要求 headDim%2==0，实际 {head_dim}")
        p = head_dim // 2

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"x dtype 只支持 fp16/bf16/fp32（op_host def.cpp L24），实际 {x.dtype}")
    if cos.dtype != x.dtype and cos.dtype != torch.float32:
        raise ValueError(f"cos/sin dtype 必须是 x 的 dtype 或 fp32，实际 {cos.dtype}")

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    y = x.clone()

    dim0, dim1, dim2 = x.shape[0], x.shape[1], x.shape[2]
    n_rows = dim0 * dim1 * dim2
    grid = (n_rows,)
    _inplace_partial_rotary_mul_kernel[grid](
        x, cos, sin, y,
        dim2, dim1, d_all,
        cos.shape[1], head_dim, start, p,
        BR=bool(cos.shape[0] != 1),
        SR=bool(cos.shape[1] != 1),
        MODE=rm,
        BLK_P=next_pow2(p),
    )
    return y
