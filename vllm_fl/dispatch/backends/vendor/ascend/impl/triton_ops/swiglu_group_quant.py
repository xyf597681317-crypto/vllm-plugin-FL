"""swiglu_group_quant —— AscendC → Triton（**未实现：目标环境不可用，按规约显式拒绝**）

## 结论（先说重点）

**本算子在目标环境 910_93 上不可用，因此本文件不提供可运行的 Triton 实现，
所有入口一律抛 `NotImplementedError`。** 这不是偷懒，而是避免「猜一个布局然后测出假 PASS」：
其中「两半是连续还是交错」这一分叉在源码里存在两种互斥证据，猜错会得到数值上完全不同的结果
（silu(act)*gate 与 silu(gate)*act 的差别不是精度问题，是算错）。

## 不可用证据

1. `csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp` L61：
   仅 `this->AICore().AddConfig("ascend950");`（**没有** ascend910b / ascend910_93），
   而目标容器是 Ascend910_9382（910_93）。
2. 容器内实测（`/root/tops/probe_sgq.py` → 日志 `/root/dsh_work/probe_sgq.log`，2026-10-09）：
   ```
   torch 2.10.0+cpu npu ok True
   hasattr(torch.ops._C_ascend, npu_swiglu_group_quant) = False
   可用相关算子: []
   ```
   即 **`torch.ops._C_ascend.npu_swiglu_group_quant` 在容器里根本不存在**，
   Lead 建议的「用可辨识输入反推布局」实验无法执行。

## 已从源码确认的语义（供 ascend950 环境实现时直接使用）

调用点 schema（`csrc/torch_binding.cpp` L3550-3558）：
    npu_swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index,
                           ScalarType dst_type=39, int quant_mode=1, int group_size=128,
                           bool round_scale=False, bool ue8m0_scale=False,
                           bool output_origin=False, int group_list_type=0,
                           float clamp_value=0.0) -> (Tensor y, Tensor scale, Tensor y_origin)

1. **形状**（`op_host/swiglu_group_quant_proto.cpp` L48-87）：
   x [.., 2H] → y [.., H]；`scale` = x.shape[:-1] + [tailDim]，
   `quant_mode==2`（MX）时 `tailDim = ceil(ceil(H/32)/2)` 且再补一维 2；
   否则 `tailDim = ceil(H/128)`。
2. **group_index 语义 = 每组 token 数（count），且本算子只用到它的「求和」**：
   调用点原注释 `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` L408-416：
   「currently only supports the `count` type … the operator performs a **summation** calculation
   on `group_index`. Therefore, under the cumsum type, the last value is directly taken」
   → 即 `Σ group_index = 有效 token 总数`。（与 `dequant_swiglu_quant` 的「每组 token 数」一致，
   但这里更进一步：只需该数组的和。）
3. **SwiGLU**（`op_kernel/swiglu_group_quant_base.h` L162-166）：
   `silu(t) = t / (1 + exp(-t))`，`y = silu(其一) * 另一`；`clamp_value>0` 时两侧都 clamp 到
   [-c, c]（L238-244）；`topk_weight` 非空时按行乘到该行上（L249-250）。
4. **per-token fp8 量化**（L777-822）：`amax = max|y|` → `clampScale = max(amax, 1e-4)`（L789）
   → `scale = clampScale / 448.0`（L791，448 = E4M3FN max，L32）
   → `y_q = y / scale`（L793/L817-822，即乘 `448/clampScale`）。
5. **E8M0 共享指数（MX / ue8m0_scale）—— `round_scale=True` 的规则已读出（L795-813）**：
   ```
   exp   = (bits(scale) >> 23) & 0xFF          # L797-798
   man   = bits(scale) & 0x7FFFFF              # L799
   t     = (man != 0) ? 1 : 0                  # L800-801
   e     = (exp - 127) + t                     # L802-803 ← **尾数非 0 就进位**
   scale_e8m0 = e + 127                        # L805/L808（e8m0 存的就是 e+127 的裸位）
   invScale   = ((127 - e) << 23) as float32   # L812-813 ← 正好 2^-e
   y_q        = y * invScale                   # L817/L821
   ```
   → **`e = ceil(log2(sf))`**（向上取到 2 的幂），量化时乘 `2^-e`；
   `round_scale=False` 时则直接 `y * (448/clampScale)`（L793）。
   **仍待确认**：MX 模式下 block=32 的分块方式、scale 末维再 reshape 成 (-1,2)
   （proto L76-80）与 `maybe_normalize_mxfp_scale_layout()` 的对齐方式——
   这两点需要有硬件或明确的布局结论才能落地。

## 两个未解决的分叉（**必须硬件实测才能定，故不实现**）

**分叉 A：两半布局 = 连续 还是 交错？**
- 支持「交错」：`swiglu_group_quant_base.h` L399 / L542 用
  `LoadDist::DIST_DINTLV_B16`（**去交错**：偶/奇下标拆成 x0/x1）取两半；
- 支持「连续」：infershape（proto L60-62）只是把最后一维对半，与 `dequant_swiglu_quant`
  的连续对半约定一致；L551-586 的 `Interleave(...)` 可能只是 fp8 输出的打包
  （`StoreDist::DIST_PACK4_B32`，4 个 fp8 塞 1 个 32bit 块）需要。
- 两种解释在数值上完全不同，读汇编也无法百分之百确定（DINTLV 可能作用在 UB 内的另一份排布上）。

**分叉 B：MX(E8M0) 的取整方向与 scale 布局**（见上 §5）。

## 为什么不留一个"能跑"的实现

按 Lead 的明确要求：**「猜错布局的 PASS 是假 PASS」**。本目标环境无法验证布局与 E8M0 规则，
故本文件只保留规格与拒绝入口；待有 ascend950 环境（或能提供布局结论）时按上面 §1-§5 实现即可。
"""

from __future__ import annotations

import torch

__all__ = ["swiglu_group_quant", "SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON"]

SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON = (
    "swiglu_group_quant 仅注册 ascend950（csrc/moe/swiglu_group_quant/op_host/"
    "swiglu_group_quant_def.cpp L61），目标环境 910_93 上 torch.ops._C_ascend."
    "npu_swiglu_group_quant 不存在（容器实测 hasattr=False），且内核里『两半连续 vs 交错』"
    "存在互斥证据（DIST_DINTLV_B16 vs infershape 对半），无法在无硬件的情况下判定布局；"
    "按规约不提供未经实测的实现。详见本模块 docstring。"
)


def swiglu_group_quant(*args, **kwargs):
    """未实现：见模块 docstring（目标环境 910_93 不可用 + 布局分叉未定）。"""
    raise NotImplementedError(SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON)
