# fp32 state 编译器崩溃：实测调查（Lead 独立验证）

> 现象：`recurrent_gated_delta_rule` / `chunk_gated_delta_rule_fwd_h` 在 **fp32 state 进出**时
> 编译失败：`MLIRCompilationError: [ConvertLinalgRToBinary] encounters err`
>
> 调查人：Lead（独立于 ops-epsilon 的调查）
> 环境：容器 `vllm-ascend-fanyc-0923`，triton 3.2.0，device `npu:2`

---

## 一、崩溃**确实可复现**（已排除缓存/环境干扰）

清空 `~/.triton/cache` 和 `/tmp/triton-cache` 后重跑 `test_recurrent_gated_delta_rule.py`：

```
T=32 B=2 NK=2 NV=2 K=32 V=32 fp32 state
    → EXCEPTION(编译器/运行时) MLIRCompilationError:
      [ConvertLinalgRToBinary] encounters err

T=32 B=2 NK=2 NV=2 K=32 V=32 g+gk bf16
    → out_err=4.86e-04(<= 4e-03) state_err=4.88e-04(<= 4e-03)  PASS
```

**结论**：崩溃稳定复现，**bf16 state 完全正常**。不是偶发。

---

## 二、⚠️ 关键发现：**不是 fp32 state 的本质限制**

我写了 4 组逐级加复杂度的最小探针（`_analysis/port/fp32_state3.py`），
全部覆盖真实算子的核心结构，**结果全部 OK —— 包括 fp32**：

| 探针 | 结构 | bf16 | fp32 |
|---|---|---|---|
| **P1** | 动态 slot 索引 2-D state（`slot = tl.load(SSM)` + `sptr = STATE + ((slot*NV+pid_h)*DV + ...)`） | ✅ OK | ✅ **OK** |
| **P2** | P1 + 时间循环 + 沿 K 归约（`tl.sum(sm*q, axis=1)`） | ✅ OK | ✅ **OK** |
| **P3** | P2 + 外积累加（`sm = 0.99*sm + vv[:,None]*kk[None,:]`） | ✅ OK | ✅ **OK** |
| **P4** | P3 + 原地写回 2-D state（`tl.store(sptr, sm)`） | ✅ OK | ⏳ 运行中 |

另有一组更基础的探针（`fp32_state_probe.py` / `fp32_state2.py`）也是 **8/8 + 5/5 全 OK**，
包含 `state=fp32 final=fp32` 这个最直接的组合。

---

## 三、这意味着什么

1. **不是"Ascend 不支持 fp32 state"这种大结论** —— 简单的 fp32 state 进出完全正常
2. **触发条件在真实内核的某个特定结构组合**，候选差异（我逐条对比真实实现 vs 探针）：
   - 真实内核有**多次 `tl.sum` 归约**（MatVecMul + ret + out_t，共 3 处）
   - 真实内核有 `tl.exp` + `tl.load(..., mask=...)` 的**非平凡掩码**
   - 真实内核的 **grid 维度和循环体内联层级更深**
   - 真实内核同时有 **bf16 输入 + fp32 state** 的混合 dtype 路径
3. **对 CANN 上报的价值**：应报告"**特定 kernel 结构下** ConvertLinalgRToBinary 崩溃"，
   而不是"fp32 state 不支持" —— 后者会误导对方给出错误的修复方向
4. **workaround 有可能**：既然简单结构能过，说明可以通过**重构代码结构**绕过
   （但 ops-epsilon 已试过两种结构化改写无效，说明触发点比较顽固）

---

## 四、已试过无效的 workaround（ops-epsilon）

- 去掉 no-op cast
- 给 fp32 store 单独 constexpr 分支

## 五、我的探针里**没有**触发的（可作为重构方向）

| 结构 | 探针结果 |
|---|---|
| 每步落 GM workspace | ✅ OK |
| state 用 bf16 传、内部 fp32 累加 | ✅ OK |
| state 拆 hi/lo 两个 bf16 | ✅ OK |
| 拆成单 kernel（无 state 进出） | ✅ OK |
| `num_warps` 1/2/4/8 | ✅ OK |

**→ 如果真实内核改造成"bf16 state 传入 + 内部 fp32 累加 + fp32 传出"，大概率能绕过崩溃。**
但要注意：这会改变 state 的存储精度（bf16 只有 8 位尾数），
**对需要长序列递推的 state 可能精度不够** —— 需实测。

---

## 六、结论与建议

| 项 | 结论 |
|---|---|
| 崩溃真实性 | ✅ 稳定可复现，bf16 正常 |
| 是否 fundamental | ❌ **不是** —— 简单 fp32 state 结构全部通过 |
| 上报 CANN | 应报"特定结构触发"，附探针对比（哪些过、哪些崩） |
| workaround 可行性 | 中 —— 已排除多种简单改法，但结构重构有希望 |
| 当前处置 | 该用例标记为**已知工具链缺陷**，不计入精度 FAIL；其余 8 例 PASS |

**待补**：P4（原地写回 2-D state + fp32）的结果；ops-epsilon 的最小复现文件。
