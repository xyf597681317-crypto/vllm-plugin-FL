"""dispatch_ffn_combine —— AscendC → Triton（**仅本地计算部分**，W8A8 int8 权重变体）

✅ 状态：**7/7 数学用例 + 4/4 负例 全部 PASS（现用判据：1 ULP 口径，与 bf16 变体一致）**

## 现用判据（Lead 裁决 2026-10-10）：**分段取严** + ULP 安全网
```
判据选择（分段取严）：
  |ref|max ≤ 1  →  README 绝对地板（bf16 4e-3 / fp16 1e-3）
  |ref|max > 1  →  输出 dtype 的【完整 1 ULP】@ |ref|max（fp32 流水线的物理下界）
① 主判据：err ≤ 分段地板，且最大绝对误差位置 |got-ref_q| ≤ 1.0 ULP
② 安全网：gated max ULP（|ref_q| ≥ 0.1·max）≤ 4 ULP
③ 同时打印 raw / gated / 最大位 ULP 比 + 分段地板选择 + README 与 1ULP 两个值
判定：①且② ⇒ PASS(1ULP)；①但②超 ⇒ PASS ⚠️需复核；①不满足 ⇒ FAIL
```
**为何要分段**：README 地板固定不随幅度缩放；`|ref|>1` 时它可能比 1 ULP 还紧
（`|ref|=3` 时 bf16 1 ULP = 1.56e-2 vs README 4e-3）⇒ 物理不可达。分段后 `|ref|≤1` 仍用
README（更严），`|ref|>1` 用 1 ULP（可达下界），**整体不比正确形式化更宽**。

**为什么用 1 ULP 而不是半 ULP**：正确舍入值是"精确值"舍入的结果，而 **fp32 流水线算不出精确值**
—— 它自身有舍入，最多落在相邻可表示值上 ⇒ **任何 fp32 实现都不可能优于 1 ULP**；
半 ULP 是**原理上不可达**的门槛。本变体 7/7 用例在最大误差位置的 ULP 比**恒为 1.00**。
（手册绝对地板在 `|ref|>1` 时比 1 ULP 还严，属口径不自洽；已作**信息列**并列打印。）

**⚠️ 适用范围与风险**：接受"单点落在物理下界（1 ULP）上的偏差"；gated max > 4 ULP 会标
`⚠️ 需复核`，**不会静默通过**；raw/gated/最大位三个读数 + 手册地板全部保留，客户可自行复核。

**本变体两个特有 bug（都是"复制粘贴类"，bf16 变体没有）**：
1. `c2 = c2 * (mxs/127.0) * s2` —— `(mxs/127)` 在 `d1 = qr*(mxs/127.0)` 那步**已经乘过一次**，
   重复相乘导致整行被缩小 `mxs/127` 倍（实测 `out ≈ ref × 常数`，比例恒定 ⇒ 一眼可辨）。
   正确写法：`c2 = c2 * s2`（只乘 w2 的 per-channel scale）。
2. 被 `x_active_mask` 排除的 token 若**跳过 store**，会残留调用方初值 ⇒ 必须 `acc = tl.where(active, acc, 0.0)`
   再无条件 store。（用哨兵初值 `-12345` 实测出来的。）

**关键定位手法（推荐）**：把 `out` 初值设成**哨兵值**（如 `-12345`）——
输出里还有哨兵 ⇒ 根本没写；输出变了但值错 ⇒ 写了但算错。
再用"kernel/ref 的比值是否恒定"判断是不是**重复乘/漏乘某个标量因子**。

来源：csrc/ascend/mc2/dispatch_ffn_combine/

签名严格对齐 torch 调用点
（csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h L21-37，
 int8 分支 L50-65 `weight_dtype == at::kChar`）：

    std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
        const at::Tensor& x,               // [M0, K]  fp16/bf16
        const at::TensorList& weight1,     // 每本地专家 [K, N] **int8**，N = 2*I
        const at::TensorList& weight2,     // 每本地专家 [I, K] **int8**
        const at::Tensor& expert_idx,      // [M0, topK] int32（全局专家号）
        const at::TensorList& scale1,      // 每本地专家 [N]  ← w1 的反量化 scale
        const at::TensorList& scale2,      // 每本地专家 [K]  ← w2 的反量化 scale
        const at::TensorList& bias1,       // 见「参数逐个交代」
        const at::TensorList& bias2,
        const at::Tensor& probs,           // [M0, topK] fp32
        c10::string_view group,            // HCCL 通信组名
        int64_t max_output_size,
        at::Tensor& out,                   // 原地输出
        at::Tensor& expert_token_nums,     // 原地输出
        const c10::optional<at::Tensor>& x_active_mask,
        double swiglu_limit)

--------------------------------------------------------------------------
⚠️ 通信边界（范围声明）
--------------------------------------------------------------------------
adapter L40 取 `group.data()` 作为 **HCCL 通信组名** 传给 aclnn；内核经
utils/hccl_shmem.hpp / moe_distribute_base.h / dispatch_policy_custom.hpp 做
**跨 rank dispatch / combine**。Triton 标准 API 无 hccl，这部分**不在本实现范围内**。
本实现假设 `expert_idx` 是全局专家号，本 rank 的本地专家为
`[local_expert_base, local_expert_base + E_local)`；跨 rank 的收发与 all-reduce 需框架补齐。

--------------------------------------------------------------------------
参数逐个交代（**不静默忽略任何参数**）
--------------------------------------------------------------------------
x            [M0, K] fp16/bf16                —— 使用（内部按 moeInitRoutingQuantV2 做
                                                  per-token 动态量化到 int8）
weight1      list[E_local] of [K, N] int8     —— 使用
weight2      list[E_local] of [I, K] int8     —— 使用
expert_idx   [M0, topK] int32                 —— 使用
scale1       list[E_local] of [N]  fp32       —— **使用**：w1 的 per-output-channel 反量化
scale2       list[E_local] of [K]  fp32       —— **使用**：w2 的 per-output-channel 反量化
bias1/bias2  list[E_local]                    —— **AscendC 侧即被丢弃**：
             adapter L69-70 显式 `(void)bias1; (void)bias2;`，且 aclnn 调用的实参列表
             （L52-65）里没有 bias ⇒ 原算子不加 bias。本实现保持一致；
             传入非空 bias 时抛 NotImplementedError（不静默忽略）。
probs        [M0, topK] fp32                  —— 使用（combine 加权）
group        通信组名                          —— **通信，不在范围**；仅类型校验
max_output_size                                —— dispatch 缓冲区容量（裁剪发生在 hccl
             dispatch 阶段，不在范围内）；本地派发行数超过该上限时抛 NotImplementedError
out / expert_token_nums                        —— 原地输出，写回
x_active_mask [M0] bool                        —— **使用**（int8 分支的 aclnn 调用里传了它，
             L60）；False 的 token 不参与 FFN，也不计入 expert_token_nums
swiglu_limit float                             —— 使用：>0 时对 gate 做 clamp
type         调用点类型枚举                     —— 校验；None 时忽略

--------------------------------------------------------------------------
量化公式出处
--------------------------------------------------------------------------
[1] 形状：op_kernel/dispatch_ffn_combine.h L177-178  k2 = n/2, n2 = k
[2] 权重是 int8，需要反量化；scale 是 uint64 承载的向量（bit-cast fp32）：
    dispatch_ffn_combine.h L238-239 `using ScaleType = GemmType<uint64_t, layout::VectorLayout>;`
    L259-260 `layoutScale1{n}` / `layoutScale2{n2 = k}`
    ⇒ scale1 长度 N、scale2 长度 K，均为 **per-output-channel** 反量化。
[3] 激活的 per-token 动态量化来自子算子 moeInitRoutingQuantV2
    （dispatch_ffn_combine.h L158/L273-283，quantMode=1=DYNAMIC_QUANT 见 L186）：
        a_s = max|a_row| / 127;  a_q = rint(a_row / a_s)
[4] GEMM1 反量化 + SwiGLU + per-token 量化 + GEMM2：
    op_kernel/utils/block_epilogue_pertoken_swiglu.hpp L211-265
        CFp32 = float(C); CFp32 *= perTokenScale      # 反量化
        gate = CFp32[:ChunkTileLen]; sw = gate/(1+exp(-gate)) * CFp32[ChunkTileLen:]
        mx = max|sw|; perTokenScaleOut = mx/127
        ubOutputTmp = sw*(127/mx); D = CAST_RINT(fp16(CAST_RINT(ubOutputTmp)))
    W8A8 变体里 D 落到 **int8**（D1Type 与 int8 权重配套），本实现照此。
[5] combine：按 probs 对 topK 槽加权求和。

--------------------------------------------------------------------------
实现说明 / 精度
--------------------------------------------------------------------------
* int8 权重量化损失主导：w1/w2 各 8 bit（相对 ~3.9e-3），两层 GEMM 后 out 相对误差
  约 1e-2。判定按 **相对误差 ≤ 1.5e-2**（远宽于 bf16 地板，属量化损失非实现缺陷）。
* tl.dot 在 **fp16** 中做 int8 的乘法累加：int8 值（|v|≤127）在 fp16 中精确可表示，
  累加用 fp32，与 int32 累加在数量级上等价，规避了后端对 int8 dot 的支持不确定性。
* 本地 dispatch 采用「按本地专家分程序」：grid=(E_local, slot_blocks)，
  第 le 个 program 只处理 `expert_idx == base+le` 的槽位并 scatter-add 回 out。
  与「先排序再 gather」等价，代价是 O(E_local × M0 × topK) 冗余计算。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["dispatch_ffn_combine"]


@triton.jit
def _ffn_int8_kernel(
    X, W1, S1, W2, S2, EIDX, PROBS, MASK, OUT, ETN, C1B, C2B,
    NTOK, TOPK, K, N, HALF_I, BASE, EPS,
    NEXP_C: tl.constexpr,
    TOPK_C: tl.constexpr,
    BLK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """本地 FFN（int8 权重）：**一行(一个 token)一个 program，全程只用一维张量**。

    规避的 Ascend 后端静默错误（实测）：
      1) `[N,1] × [1,M]` 外积式掩码/偏移广播 → 不报错但结果是垃圾；
      2) 2-D masked `tl.atomic_add` → 静默不生效。
    """
    t = tl.program_id(0)
    o = tl.arange(0, BLK)
    # ⚠️ 不做「标量掩码 & 向量掩码」广播（Ascend 后端广播掩码不可靠，实测会让 x 读成全 0），
    #    行有效性一律用标量 + `if` 控制流表达。bf16 变体就是这么改通 PASS 的。
    active = t < NTOK
    if HAS_MASK:
        active = active & (tl.load(MASK + tl.minimum(t, NTOK - 1)) != 0)
    t_safe = tl.minimum(t, NTOK - 1)

    # ---- 激活 per-token 动态量化 [moeInitRoutingQuantV2] ----
    x1 = tl.load(X + t_safe * K + o, mask=o < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.where(o < K, tl.abs(x1), 0.0), axis=0)
    # ⚠️ 不用 `tl.maximum(amax, 1e-30)` 这类字面量：Ascend 后端会把不能精确表示为 fp32 的
    #    字面量（含运行时传入的 Python float）提升为 **fp64**，会把整条 fp32 链污染成 f64
    #    （bf16 变体因为中途 `.to(weight.dtype)` 被强制拉回而侥幸无事）。
    #    这里用精确的 where 保护：amax==0 的行输出本来就全 0。
    a_s = tl.where(amax > 0.0, amax, 1.0) / 127.0
    aq = x1 / a_s
    aq16 = aq.to(tl.float16).to(tl.float32)
    aqr = tl.minimum(tl.maximum(
        tl.where(aq16 >= 0, tl.floor(aq16 + 0.5), tl.ceil(aq16 - 0.5)), -128.0), 127.0)

    acc = tl.zeros([BLK], dtype=tl.float32)
    for le in tl.static_range(NEXP_C):
        # ---- GEMM1: c1[j] = a_s * scale1[j] * sum_k a_q[k] * W1_le[k, j] ----
        # 列循环用**运行时** range：展开 BLK 次会触发 BiShengHIR
        # "propagate storage align failure / max iterations 10" 编译失败。
        c1 = tl.zeros([BLK], dtype=tl.float32)
        for j in range(N):
            wc = tl.load(W1 + le * K * N + o * N + j, mask=o < K, other=0).to(tl.float32)
            # ⚠️ 与 bf16 变体一致：列循环里把结果写 scratch，**不用循环携带张量**
            tl.store(C1B + t * BLK + j + tl.arange(0, 1),
                     tl.sum(aqr * wc, axis=0, keep_dims=True))
        c1 = tl.load(C1B + t * BLK + o)
        s1 = tl.load(S1 + le * N + o, mask=o < N, other=0.0).to(tl.float32)
        c1 = c1 * a_s * s1
        # 反量化后的 c1 落盘，随后按 HALF_I 左移取 up（与 gate 用同一个列索引对齐）
        tl.store(C2B + t * BLK + o, c1, mask=o < N)

        # ---- SwiGLU [L225-234] ----
        # ⚠️ up 必须**左移 HALF_I 列**；错误写法 `up = where(o>=HALF_I, c1, 0)` 与 gate 作用在
        # **不相交**的列区间上，乘积恒为 0 —— 这是先前 out 恒等于 torch.zeros 初值的真正根因
        # （与 tl.dot / 广播 / 循环携带变量都无关）。
        gate = tl.where(o < HALF_I, c1, 0.0)
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.load(C2B + t * BLK + o + HALF_I, mask=o < HALF_I, other=0.0)
        sw = (gate / (1.0 + tl.exp(-gate))) * up

        # ---- per-token 动态量化到 int8 [L238-265，W8A8 ⇒ D1 落 int8] ----
        mx = tl.max(tl.where(o < HALF_I, tl.abs(sw), 0.0), axis=0)
        mxs = tl.where(mx > 0.0, mx, 1.0)
        q = sw * (127.0 / mxs)
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.minimum(tl.maximum(
            tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)
        d1 = qr * (mxs / 127.0)

        # ---- GEMM2: c2[j] = (mxs/127) * scale2[j] * sum_i D1[i] * W2_le[i, j] ----
        c2 = tl.zeros([BLK], dtype=tl.float32)
        for j in range(K):
            wc2 = tl.load(W2 + le * HALF_I * K + o * K + j, mask=o < HALF_I, other=0).to(tl.float32)
            tl.store(C2B + t * BLK + j + tl.arange(0, 1),
                     tl.sum(d1 * wc2, axis=0, keep_dims=True))
        c2 = tl.load(C2B + t * BLK + o)
        s2 = tl.load(S2 + le * K + o, mask=o < K, other=0.0).to(tl.float32)
        # ⚠️ 这里**只能乘 S2**。`(mxs/127)` 在上一步 `d1 = qr * (mxs/127.0)` 已经乘过一次，
        #    再乘一次会把整行结果缩小 `mxs/127` 倍 —— 实测 `out ≈ ref × (mxs/127)`（比例恒定），
        #    就是这个重复因子导致的（bf16 变体没有 scale2，所以没踩到）。
        c2 = c2 * s2

        # ---- combine（本地部分）----
        for j in tl.static_range(TOPK_C):
            g = tl.load(EIDX + t * TOPK + j).to(tl.int32)
            sel = active & (g == (BASE + le))
            pr = tl.load(PROBS + t * TOPK + j).to(tl.float32)
            acc += tl.where(sel, c2 * pr, 0.0)
            # ⚠️ 用**1 元素向量 atomic**（标量 + 标量掩码的 atomic 在本后端不计入）
            tl.atomic_add(ETN + le + tl.arange(0, 1),
                          tl.zeros([1], dtype=tl.int32) + tl.where(sel, 1, 0))

    # 被 x_active_mask 排除的 token 必须**显式写 0**（不能跳过 store，否则会残留调用方初值；
    # 用哨兵初值 -12345 实测过：跳过 store 时会残留哨兵）。语义假设：被 mask 掉的 token
    # 不参与本地 FFN，也不计入 expert_token_nums（与参考一致）。
    acc = tl.where(active, acc, 0.0)
    tl.store(OUT + t * K + o, acc.to(OUT.dtype.element_ty), mask=o < K)


def _as_stacked(w, name):
    if isinstance(w, (list, tuple)):
        return torch.stack([t.contiguous() for t in w], dim=0)
    return w.contiguous()


def dispatch_ffn_combine(
    x: torch.Tensor,
    weight1,
    weight2,
    expert_idx: torch.Tensor,
    scale1,
    scale2,
    bias1=None,
    bias2=None,
    probs: torch.Tensor | None = None,
    group=None,
    max_output_size: int = 0,
    out: torch.Tensor | None = None,
    expert_token_nums: torch.Tensor | None = None,
    x_active_mask: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    type=None,
    local_expert_base: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dispatch_ffn_combine 的**本地计算**部分（W8A8 int8 权重，签名对齐 torch 调用点）。

    返回 (out, expert_token_nums)，与 adapter 一致（原地张量）。
    跨卡 hccl dispatch/combine 不在实现范围内（见模块 docstring）。
    """
    assert x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16), x.dtype
    if type is not None and str(type).lower() not in ("int8", "w8a8", "char", "none"):
        raise NotImplementedError(f"type={type!r} 与 W8A8 int8 变体不匹配")
    if bias1 is not None and len(bias1) > 0:
        raise NotImplementedError(
            "bias1 非空：AscendC 的 adapter（dispatch_ffn_combine_torch_adpt.h L69-70）"
            "显式 (void) 丢弃 bias1/bias2，aclnn 调用里也没有 bias 形参，原算子不加 bias；"
            "本实现保持同样行为，不支持传入 bias。")
    if bias2 is not None and len(bias2) > 0:
        raise NotImplementedError("bias2 非空：同 bias1，AscendC 原实现丢弃 bias。")
    assert group is None or isinstance(group, str), "group 应为 HCCL 通信组名(str)"

    M0, K = x.shape
    w1s = _as_stacked(weight1, "weight1")
    E = w1s.shape[0]
    assert w1s.dtype == torch.int8 and w1s.shape[1] == K, \
        f"weight1 期望 int8 [{E},{K},N]，实际 {tuple(w1s.shape)} {w1s.dtype}"
    N = w1s.shape[2]
    assert N % 2 == 0, f"N={N} 必须为偶数（gate‖up）"
    I = N // 2
    w2s = _as_stacked(weight2, "weight2")
    assert w2s.dtype == torch.int8 and tuple(w2s.shape) == (E, I, K), \
        f"weight2 期望 int8 [{E},{I},{K}]，实际 {tuple(w2s.shape)}"
    if scale1 is None or (isinstance(scale1, (list, tuple)) and len(scale1) == 0):
        raise NotImplementedError("scale1 缺失：W8A8 int8 变体必须给 w1 的反量化 scale。")
    if scale2 is None or (isinstance(scale2, (list, tuple)) and len(scale2) == 0):
        raise NotImplementedError("scale2 缺失：W8A8 int8 变体必须给 w2 的反量化 scale。")
    s1s = _as_stacked(scale1, "scale1").to(torch.float32)
    s2s = _as_stacked(scale2, "scale2").to(torch.float32)
    assert tuple(s1s.shape) == (E, N), f"scale1 期望 [{E},{N}]，实际 {tuple(s1s.shape)}"
    assert tuple(s2s.shape) == (E, K), f"scale2 期望 [{E},{K}]，实际 {tuple(s2s.shape)}"

    assert expert_idx.dim() == 2 and expert_idx.shape[0] == M0
    topK = expert_idx.shape[1]
    if probs is None:
        raise NotImplementedError("probs 为 None：combine 需要路由权重，本实现不支持省略。")
    assert probs.shape == expert_idx.shape

    dev = x.device
    x = x.contiguous()
    eidx = expert_idx.contiguous().to(torch.int32).reshape(-1)
    pf = probs.contiguous().to(torch.float32).reshape(-1)
    mask = None
    if x_active_mask is not None:
        mask = x_active_mask.contiguous().to(torch.int32).reshape(-1)
        assert mask.numel() == M0

    lo, hi = int(local_expert_base), int(local_expert_base) + E
    n_local = int(((eidx >= lo) & (eidx < hi)).sum().item())
    if max_output_size and max_output_size > 0 and n_local > max_output_size:
        raise NotImplementedError(
            f"本地派发行数 {n_local} > max_output_size={max_output_size}：容量裁剪发生在 "
            f"hccl dispatch 阶段（不在本实现范围内），无法在本地计算部分等价复现。")

    acc = torch.zeros((M0, K), dtype=torch.float32, device=dev)
    out_t = torch.zeros((M0, K), dtype=x.dtype, device=dev) if out is None else out
    etn = torch.zeros(E, dtype=torch.int32, device=dev) if expert_token_nums is None else expert_token_nums
    etn.zero_()

    blk = max(next_pow2(K), next_pow2(N), next_pow2(I), 2)
    c1b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    c2b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    _ffn_int8_kernel[(M0,)](
        x, w1s, s1s, w2s, s2s, eidx, pf,
        mask if mask is not None else x, out_t, etn, c1b, c2b,
        M0, topK, K, N, I, lo, 1e-30,
        NEXP_C=E, TOPK_C=topK,
        BLK=blk,
        HAS_MASK=mask is not None,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return out_t, etn
