# dequant_swiglu_quant —— AscendC 语义发现记录（task-6）

> 目的：记录本次补全（`swiglu_mode=1` / `group_index` 多组 / `activate_left`）时**实际读到的公式出处**、
> 关键判断依据，以及**仍然存在的缺口**。所有行号对应
> `_repos/vllm-plugin-FL-xmd/vllm-plugin-FL-feature-v024-ascend-platform-runtime/csrc/ascend/moe/dequant_swiglu_quant/`。

---

## 1. 接口与形状

| 项 | 出处 | 结论 |
|---|---|---|
| Input/Output/Attr 列表 | `op_host/dequant_swiglu_quant_def.cpp` L30-102 | x / weight_scale / activation_scale / bias / quant_scale / quant_offset / group_index → y(int8) + scale(fp32) |
| 输出形状 | `op_host/dequant_swiglu_quant_infershape.cpp` L44-69 | y 与 x 同形，`activate_dim`（默认最后一维）长度减半；`scale` 形状 = y 去掉最后一维 |
| 默认属性 | `dequant_swiglu_quant_def.cpp` L22-24 | `dst_type=2`(INT8)、`clamp_limit=0.0`、`glu_alpha=1.00`（注释里 1.702 是历史值） |
| `quant_mode` 映射 | `op_host/dequant_swiglu_quant_tiling.cpp` L82 附近 / `_tiling_base.cpp` L342-353 | `"dynamic"→1`、`"static"→0`（**调用点传 1 即 dynamic**） |

---

## 2. 反量化（仅 x=int32）

`op_kernel/dequant_swiglu_quant.h`：

```
L401  xf = fp32(x)
L405  xf = weight_scale * xf          # 逐通道[2H]广播
L413  xf = activation_scale * xf      # 逐行[M]广播（pertoken_scale），仅当 activationScale 非空
L424  xf = xf + bias                  # 浮点 bias（int32 bias 例外：L386-390 在 cast 前先加）
```
注意 L403 的 `if constexpr (std::is_same_v<TXGm, int32_t>)`：**只有 int32 输入才做上述缩放**；
bf16/fp16 输入直接 cast 后进 SwiGLU（本实现因此**不复现** bf16/fp16 直通路径，见 §6）。

---

## 3. SwiGLU 两条路径

### 3.1 切分（`activate_left`）

```
L182-183  actOffset_ = actRight * UbFactorDimy ; gateOffset_ = UbFactorDimy - actOffset_
_tiling_base.cpp L155   "false <-> 0: activate right"
```
→ `activate_left=true` ：act = 左半 `x[:, :H]`，gate = 右半 `x[:, H:]`
→ `activate_left=false`：act = 右半 `x[:, H:]`，gate = 左半 `x[:, :H]`

**两个生产调用点都传 `activate_left=True`**（`moe_mlp.py` L249、`fused_moe.py` L720）。

### 3.2 `swiglu_mode=0`（普通）

```
L459-465  exp(-act) → +1 → act/(1+exp(-act))
L472      out = gate * act_silu
```
即 `y = gate * act * sigmoid(act)`；**不涉及** clamp / glu_alpha / glu_bias。

### 3.3 `swiglu_mode=1`（SwiGLU-OAI / `swigluoai_uninterleave`）

`SwiGluGate()` L630-674，逐条对应：

```
L650-656  if (clampLimit > 0) { gate = min(gate, +c); gate = max(gate, -c); }   # gate 双侧 clamp
L657      gate = gate + gluBias                                                 # ★ 无条件（在 if 之外）
L659-663  if (clampLimit > 0) { act = min(act, +c); }                           # ★ act 只有上限
L664-666  t = act * (-gluAlpha) ; t = exp(t)
L668      t = t + 1
L670      s = act / t                                                           # 用 clamp 后的 act
L672      out = gate * s
```
即：

```
gate = clamp(gate, -c, +c)  (c>0)  →  gate += glu_bias
act  = min(act, +c)         (c>0)
y    = gate * act / (1 + exp(-glu_alpha * act))
```

**容易踩的点**：
1. `glu_bias` 加在 **gate** 上（不是 act），且 `clamp_limit=0` 时**仍然加**。
2. `act` **只做上限 clamp**，没有下限（与 gate 不对称）。
3. `glu_alpha` 只出现在指数的 `-gluAlpha*act` 里，分母的分子仍是 `act` 本身（不是 `alpha*act`）。

---

## 4. MoE 多组 `group_index`

### 4.1 语义：**每组 token 数**，不是前缀和

`op_kernel/dequant_swiglu_quant_cut_group.h` L38-58 与 `dequant_swiglu_quant.h` L224-238：

```
groupOffset_ = 0
for groupIdx in 0..inGroupNum-1:
    realGroupIdx = speGroupType==0 ? groupIdx : groupIndexGm_(groupIdx*2)      # [G,2] 时列0=真实组号
    realDimx_    = speGroupType==0 ? groupIndexGm_(groupIdx) : groupIndexGm_(groupIdx*2+1)  # 该组 token 数
    if realDimx_ <= 0 && speGroupType: break
    ProcessSingleGroup(realGroupIdx, realDimx_, groupOffset_)                  # 取 weight_scale[realGroupIdx]
    groupOffset_ += realDimx_
```
→ `group_index[g]` 被当作**第 g 组的 token 数**，组 g 覆盖输出行 `[Σ_{k<g} cnt_k, Σ_{k≤g} cnt_k)`；
`GROUPINDEX_STRIDE = 2`（`_cut_group.h` L25）对应 `[G,2]`（列0=组号、列1=token 数）。

### 4.2 调用点确实传计数

`vllm_fl/dispatch/backends/vendor/ascend/impl/moe/moe_mlp.py` L248：

```python
"group_index": cumsum_group_list(group_list, group_list_type, 1),
```
`cumsum_group_list()` 定义在同文件 L54-83，`dst_list_type=1` 分支即「转成 per-group 计数」
（L60-61 同型直接返回；L64-67 由前缀和做 diff；L68-79 的 [.,2] 形式也归到计数语义）。
→ **与本实现按「计数」解释一致**（若传前缀和会与 kernel 不符）。

### 4.3 weight_scale 形状

`op_host/dequant_swiglu_quant_tiling_arch35.cpp` L326：
`wScaleShape.GetDim(0) == groupIndexShape_.GetDim(0)` 且最后一维 == x 的最后一维
→ 多组时 `weight_scale` 形状为 `[G, 2H]`，行号 = 该组的 `realGroupIdx`。

---

## 5. 动态量化（`quant_mode=dynamic`）

```
L681-694  if (needSmoothScale) y = y * quant_scale[H]        # 逐通道 smooth scale
L696-711  rowmax = max_d |y|
L714      scale  = rowmax * DYNAMIC_QUANT_FACTOR (1/127，L33)
L732-735  yq     = y / scale
L789-800  CastFloatToInt8：RINT(fp32→int32) → CAST_ROUND(int32→half) → TRUNC(half→int8)
```
→ 取整是 **就近偶数**（CAST_RINT，等价 `torch.round`）；`scale` 是**反量化 scale**（乘回去）。

---

## 6. 仍然存在的缺口（未实现 / 未验证）

| 缺口 | 说明 |
|---|---|
| `quant_mode="static"`(0) | `StaticQuant()` L740-786：`y*quant_scale+quant_offset` 或 `y/quant_scale+quant_offset`；`quantIsOne` 分支未实现 |
| x 为 bf16/fp16（直通路径） | L403 的 constexpr 分支不做缩放；本实现直接 assert 拒绝，避免未验证语义 |
| int32 bias | L386-390 在 cast 前做整数加，未实现（生产调用点 `bias=None`） |
| `activation_scale` 的 `[G,M]` 形状 | 本实现只接受 `[M]`（调用点传 pertoken `[M]`） |
| `dst_type` 非 int8 | int4/fp8(`FLOAT4_E2M1/E1M2`、`FLOAT8_E4M3FN/E5M2`、`HiFloat8`) 未实现 |
| `activate_dim != -1` | 只实现最后一维对半切 |
| `group_index` 为 [G,2] 且组号乱序 | 本实现按列0取组号，已测「[G,2] 顺序组号」；乱序/重复组号未覆盖 |
| `round_mode` 非 "rint" | 只实现 CAST_RINT |
| `swiglu_mode=1` 的 half/interleave 细节 | `CreateOffsetLocalTensor()` L622-627 注释说明「直接用前一半和后一半」，故两模式切分一致；若上游 `swigluoai_uninterleave` 语义要求交错布局，需要在图侧确认张量已重排 |

---

## 7. 实测结论（容器 `vllm-ascend-fanyc-0923`，NPU Ascend910_93）

`test_dequant_swiglu_quant.py` 覆盖三组参数组合（详见该文件输出与
`/root/dsh_work/dequant_swiglu_quant_test.txt`）：组合 3 即 `moe_mlp.py` 的真实路径
（`group_index=<每组 token 数>` + `activate_left=True` + `swiglu_mode=1` + clamp/alpha/bias）。
判定：int8 输出 max|Δ|≤1 且逐元素相等率 ≥99.9%；scale 相对误差 ≤2e-6。
