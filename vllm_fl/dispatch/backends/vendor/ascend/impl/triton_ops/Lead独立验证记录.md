# Lead 独立验证记录：被证伪的根因结论

> 本文记录**由 worker 提出、但我独立实测后不成立**的结论。
> 目的：防止错误的泛化结论在组内传播，导致无谓的重构或放弃可用能力。
>
> 所有验证均在容器 `vllm-ascend-fanyc-0923` 实测，device `npu:2`，triton 3.2.0。

---

## 证伪 #1：`hc_pre` = "测试用例设计问题"（**我自己犯的**）

| 项 | 内容 |
|---|---|
| 我的结论 | `sparse_attention_score` 的 fp8 无 dequant-scale 失败是**测试用例设计问题** |
| 实际原因 | host 侧 `maxqblk` 回退值取错（用 `ceil(maxQSeqlen/BS)` 而非按 KV 长度上界）→ **内核越界读 scale** |
| 修复后 | `out_err=6.621e-04 PASS` |
| 教训 | **不该替 worker 下"这是用例问题"的结论** —— 会把真 bug 掩盖掉 |

---

## 证伪 #2（修正版）：ops-delta 的「动态 `while` + 外积偏移会算错」

**先说结论：现象是真的，但归因不准。我复现了他的现象，同时证明"外积偏移"不是原因。**

### 我复现他的现象（用他的 `probe_variants.py` 设置）
```
a = bf16 [16,64],  BLK_K == K == 64,  M=16 K=64 N=8:
  v1 while         max_err=5.6254e+00  wrong(>1e-3)=120/128   ← 复现 ✅
  v3 static_range  max_err=3.5813e-07  wrong(>1e-3)=0/128     ← 正确
```
（他原本的 `wrong=120/128 max_err=5.6254e+00` 我**一字不差地复现了**）

### 但"外积偏移"不是原因 —— 我的对照
| 探针 | 结构 | 结果 |
|---|---|---|
| `loop_probe.py` v1 | `while` + **外积偏移** + **加法**（无归约） | ✅ `max_err=0.00e+00` |
| `loop_probe.py` v2 | `while` + 偏移提到循环外 + 加法 | ✅ `max_err=0.00e+00` |
| 他的 `_v1` | `while` + 2-D load + **`tl.sum` 归约** | ❌ `wrong=120/128` |

**→ 差异在「2-D load + `tl.sum` 归约」这个组合，不在"外积偏移"。**

### 正确的表述（我复现得出来的）
> 「`while` 循环 + 2-D load + `tl.sum` 归约，在 `BLK_K == K`（**单次迭代**）这类组合下可能算错」

### 他自己的二分也支持这个修正
他按我要求做了二分（`/root/tops/bisect_hcpre.py`），用同一份正确输入只改 sinkhorn 段循环形式：
```
bs=8 hm=4 d=512  A nested-while   post_err=9.220e-08  comb_err=4.975e-08
bs=8 hm=4 d=512  B static_range   post_err=9.220e-08  comb_err=4.975e-08   ← 完全一致
（d=100 / d=256 同样）
```
→ **嵌套动态 while 完全正常，循环形式无影响。他据此撤回了泛化。**

### `hc_pre` 真正修好的两处（他的二分证据链）
| 版本 | 改动 | 结果 |
|---|---|---|
| delta15 | 零填充+无掩码+2-D grid mixes，但 comb 里仍有 `cm = jm[:,None] & jm[None,:]` 2-D 掩码，循环是 while | post_err=9.7e-01 全错 |
| delta16 | **去掉 2-D 掩码**（要求 hm 为 2 的幂→comb 稠密），循环改 static_range | 6/8 对（只剩 d 非 BLK_D 倍数两例错）|
| delta18 | **行 stride 从逻辑 K 改成补零后的 KS** | **8/8 全对** |

- **修好 #1 = 坑 C**：`comb` 段的 2-D 掩码（delta15→delta16 之间唯一与数值相关的改动）
- **修好 #2 = 坑 D**：补零后 stride 必须用 KS（delta16→delta18）
- **`static_range` 无贡献**（二分证明）

**这条修正很重要**：如果不纠正，大家会去改循环形式（无效），而错过**真正的两个坑（2-D 掩码 + 补零 stride）**。

---

## 证伪 #3：ops-delta 的「坑 B：`tl.dot` 在本后端直接崩」

| 项 | 内容 |
|---|---|
| 他的结论 | 「`tl.dot` 在本后端**直接崩**（BiShengIR/LLVM "PLEASE submit a bug report"，fp32 与 bf16 输入都崩）→ 矩阵乘要用一维广播归约或 gather 实现」 |
| 我的实测 | **`tl.dot` 完全可用，三种 dtype 全部正常** |
| 交叉证据 | **ops-beta 也实测 `tl.dot` 可用**（他报 bf16 `err=5.96e-07` / fp16 `err=1.069e-06`，并把它写进了探针） |

```
tl.dot bf16   OK  err=1.91e-06
tl.dot fp16   OK  err=1.91e-06
tl.dot fp32   OK  err=0.00e+00
```

**结论**：**两家实测 OK，只有一家报崩** → `tl.dot` 本身没问题，是他的特定 shape/参数触发。
→ 已明确告知他**不要再传播"别用 tl.dot"**。

**为什么必须纠正**：这条会导致他人放弃 `tl.dot`（大矩阵乘用它最快），改用低效的归约实现。
我的 `sparse_attention_score` / `mla_preprocess` / `compressor` 三个算子**大量使用 `tl.dot` 且实测 PASS**，
是反例的直接证据。

---

## 证伪 #4：ops-gamma 的「`[N,1]×[1,M]` 外积广播 → 静默垃圾」

| 项 | 内容 |
|---|---|
| 他的结论 | 「`[N,1]×[1,M]` 外积式掩码/偏移广播 + `tl.zeros_like` 组合，**不报错但输出全垃圾**（err=1.2e38，部分位置写 0）」；规避是改成一维、一行一 program |
| 我的实测 | **小 shape 全部逐位正确；失败全是【显式 `MLIRCompilationError`】（UB 溢出），无一例静默垃圾** |
| 验证代码 | `_analysis/port/outer_probe.py` |

```
A: 外积偏移 + 2-D 掩码 + store
  M=8   N=8   BM=8   BN=8    ✅ max_err=0.000e+00  bad=0/64
  M=16  N=32  BM=16  BN=32   ✅ max_err=0.000e+00  bad=0/512
  M=33  N=100 BM=64  BN=128  ✅ max_err=0.000e+00  bad=0/3300
  M=100 N=300 BM=128 BN=512  ❌ MLIRCompilationError   ← 显式报错
  M=257 N=129 BM=512 BN=256  ❌ MLIRCompilationError   ← 显式报错

B: 外积掩码 + tl.zeros_like
  M=16  N=32  BM=16  BN=32   ✅ 对
  M=100 N=300 BM=128 BN=512  ❌ MLIRCompilationError

C: 多 program grid + 外积掩码（大 shape 也正确）
  M=100 N=64  BM=32  BN=64   ✅ 对  bad=0/6400
  M=257 N=128 BM=64  BN=128  ✅ 对  bad=0/32896

D: bf16 + 三路外积掩码 + 归约
  ❌ 全是 CompilationError（UB 溢出）
```

**结论**：泛化的「外积广播导致静默垃圾」**不成立**。
他的现象可能真实，但触发条件比描述的窄得多 → 已要求他提供**原始失败内核**以精确复现。

**真正该传播的规则**（我与他、ops-delta 的失败都指向同一条）：
> **控制 block 尺寸避免 UB 溢出**（UB 仅 192KB：`128×512` / `512×256` fp32 就会超）

**反例（2-D 广播实测 PASS 的算子）**：`compressor`、`mla_preprocess`、`sparse_attention_score`

---

## 证伪汇总：5 条中有 4 条是泛化结论

| # | 提出者 | 结论 | 状态 |
|---|---|---|---|
| 1 | **我自己** | fp8 失败是"测试用例设计问题" | ❌ 证伪（真因是 host 越界读 scale）|
| 2 | ops-delta | 动态 `while` + 外积偏移会算错 | ⚠️ **现象真、归因改**（真因是「2-D load + `tl.sum` 归约」而非外积偏移）|
| 3 | ops-delta | `tl.dot` 在本后端直接崩 | ❌ 证伪（3 种 dtype 全 OK）|
| 4 | ops-gamma | `[N,1]×[1,M]` 外积广播 → 静默垃圾 | ⚠️ **现象真、归因改**（我复现并定位到坑 10：**标量经 `[:,None]` 广播**）|
| 5 | **我自己** | `tl.dot` 在 M<16 极小 shape 下返回 0 | ❌ 证伪（含 `1×1×1` 全对）|

**共同模式**：把「我的特定 failure」泛化成「这个写法不能用」。
**危害**：会让全组做无谓重构，甚至放弃可用能力（如 `tl.dot`）。

**注意 #2 和 #4**：这两条的正确处理不是"证伪"，而是**"现象为真但归因不准"** ——
我通过复现把它们的触发条件**收窄到可操作的规则**（2-D load+归约组合 / 标量经 `[:,None]` 广播），
这比单纯说"你错了"更有价值。

---

## 证伪 #5：**我自己**的假设「`tl.dot` 在 M<16 极小 shape 下返回 0」

| 项 | 内容 |
|---|---|
| 我的假设 | ops-beta 的最小复现是 `M0=2, K=8, I=4`（极小 shape）→ 怀疑 `tl.dot` 在不满足最小 tile 时静默返回 0 |
| 我的实测 | **完全不成立** —— `tl.dot` 在极小 shape 下也正确 |
| 验证代码 | `_analysis/port/dot_min_probe.py` |

```
M=32 N=32 K=32 bf16  ✅ max_err=9.537e-07  allzero=False
M=16 N=16 K=16 bf16  ✅ max_err=4.768e-07  allzero=False
M= 8 N= 8 K= 8 bf16  ✅ max_err=1.192e-07  allzero=False
M= 4 N= 4 K= 4 bf16  ✅ max_err=0.000e+00  allzero=False
M= 2 N= 4 K= 8 bf16  ✅ max_err=0.000e+00  allzero=False
M= 1 N= 1 K= 1 bf16  ✅ max_err=0.000e+00  allzero=False   ← 最小可能 shape
M=16 N= 2 K= 8 bf16  ✅ max_err=0.000e+00  allzero=False
```

**→ 连 `M=1,N=1,K=1` 都正确。** 我的假设证伪。

### 这件事的最终状态
`tl.dot` 的"可用性"证据已经**碾压性**：
| 证据 | 结果 |
|---|---|
| 我的 3 组 dtype（`loop_probe.py`） | bf16/fp16 err=1.91e-06，fp32 err=0 |
| 我的 11 组极小 shape（`dot_min_probe.py`） | 全部正确，含 `1×1×1` |
| ops-beta 的探针 | err≈5.96e-07 / 1.069e-06 |
| ops-alpha 三个 PASS 算子（27/27 用例） | 大量使用 `tl.dot` |
| **合计** | **`tl.dot` 完全可用，无任何反例** |

**而 ops-beta 的 3 个 BLOCKED 算子仍是未定位问题**（`out` 精确等于 `torch.zeros` 初值）。
**处置决定**：不再查根因，改用「逐元素 + 1-D 归约」范式重写（他有 2 个 PASS 先例，性价比最高）。


| 来源 | 结论 | 我的立场 |
|---|---|---|
| ops-gamma / ops-delta / 我 | **UB 仅 192KB → 大 block 溢出**（`128×512` fp32 即超） | ✅ **三家共识**，规则应表述为"控制 block 尺寸" |
| **ops-gamma + 我** | **坑 10（新定位）**：**标量经 `[:, None]` 广播成 2-D 后作指针基址 → 读到越界内存（静默垃圾）** | ✅ **我独立复现并定位到根因**（见下）|
| ops-delta 坑 D | 补零后行 stride 必须用补零后的 KS，不能用逻辑 K；`d` 非 BLK_K 倍数时最后一块跨行读邻行 | ✅ **认可，很有价值**（"所有 d 整除的 shape 都对"正是它隐蔽的原因）|
| ops-beta | `tl.static_range(K)` 的 K 必须 constexpr | ✅ 认可（报错型，低危）|
| ops-beta | int32 `tl.sum(tl.where(...))` 取标量会取错 | ✅ 认可（他 13/13 用例反证过）|
| ops-alpha | fp8 在 BiShengHIR 被拒（`hivm.hir.vbrc ... doesn't support fp8 type`） | ✅ 认可（诚实标注未验证）|
| ops-beta | 向量 `tl.store` 写不进 / `tl.dot` 在结构化 kernel 里静默全 0 | ❓ **未确认**，我已要求他做二分（重点怀疑 `tl.dot` 在 M<16 极小 shape 下返回 0）|
| ops-epsilon | fp32 state 触发 `ConvertLinalgRToBinary` 崩溃 | ✅ 崩溃可复现；但**非本质限制**（我 8 组探针含 fp32 全 OK）|

---

## 坑 10 的定位过程（我复现 + 根因）

**ops-gamma 的现象**：`inplace_partial_rotary_mul` 的 2-D 版在 `BR=F, SR=F` 分支出现"精确 0 + ~1e38 混合"。

**我复现（清缓存 + `TRITON_ALWAYS_COMPILE=1`）**：
```
他的 2-D 原样内核:  max|y|=6.2308e+36  max_err=6.2308e+36  bad=252/256   ← 复现 ✅
对照 一行一 program: max|y|=4.7812e+00  max_err=1.2238e-02  bad=2/256     ← 正常
```

**根因**（我读他的 kernel 定位）：
```python
if BR and SR:  crow = (bb * C1 + ss) * HEAD_DIM   # bb,ss 是 [BLK_R] 向量 → crow 向量
elif BR:       crow = bb * HEAD_DIM               # 向量
elif SR:       crow = ss * HEAD_DIM               # 向量
else:          crow = tl.zeros_like(bb)           # ← bb 在该 shape 下是【0 维标量】，crow 也是标量
...
cs_base = crow[:, None]                           # 标量经 [:,None] → 2-D（与其他分支布局不同）
tl.load(COS + cs_base + wa_off, mask=m)           # ← 地址计算失败，读到越界内存
```

**准确表述**：
> 「`crow` 由 `tl.zeros_like(0 维标量)` 得到、再经 `[:, None]` 广播成 2-D 的分支上，
> 2-D 指针的布局分配出错 → 读到越界内存（垃圾值 + 部分 lane 为 0）」

**不是"外积广播"泛化问题** —— 而是「**标量经 `[:,None]` 广播**」这条特定路径。
这也解释了"部分 lane 精确 0、部分 lane ~1e38"的混合特征。

**规避**：让所有分支产出**同 rank 的张量**（如 `crow = bb * 0` 而非 `tl.zeros_like(bb)`），或改 1-D 一行一 program。

---

## ⚠️ 方法论教训：**Triton 缓存会污染对照实验**

ops-gamma 自查发现：他把 `tl.zeros_like(bb)` 换成仿射式后重跑，
**四个 case 的误差数值与替换前逐位相同**（1.203e38 / 1.914e38 / 1.203e38 / 1.156e38）。
源码变了而误差完全相同 → **最可能是 Triton 缓存命中了旧二进制**，他当时没清 cache。

**→ 规则（已广播）**：**凡是对照实验（A/B 改动对比），必须**
```bash
rm -rf ~/.triton/cache /tmp/triton-cache
export TRITON_ALWAYS_COMPILE=1
```
**否则会得出"改动无效"的错误结论**（我自己的复现就是这么做的）。

**这类错误的危险**：它是**假阴性** —— 会让人放弃一个本来有效的修复。

---

## ⚠️ 环境事实：`q.py` 单次发送 ~24KB 上限

ops-alpha 报告：`q.py` 单次 `ch.send()` 有长度上限，**>~24KB 的 base64 行会被静默丢弃**
（`mla_preprocess.py` 26KB 时上传失败，md5 永远 MISMATCH；压到 24KB 后成功）。

**我核实**：我上传过的最大文件 `ascend_triton_ops5.py` 16875 字节 → base64 约 **22500 字节**，
**已逼近 24KB 上限**。

**→ 规则**：上传 >18KB（原文）的脚本要**拆成多段**或压缩，且**必须校验 md5/字节数**。
`q.py --as` 已支持唯一路径，但**没有内置完整性校验** —— 这是一个待补的工具缺口。

---

## ✅ 已定位的坑 11 & 坑 12（ops-epsilon 精确二分 + 我认可）

### 坑 11：循环体内对「循环携带的 2-D state」做 2-D `tl.store` → **bishengir 编译崩溃**

**触发点精确到一条语句**（用"从真实内核往下删"的确定性二分）：

| 变体 | 相对 R0 的差异 | 结果 |
|---|---|---|
| **R0** | 完整 recurrent 内核，`state=fp32`，**循环内 2-D `tl.store(FINAL,…, sm, mask)`** | **CRASH** |
| **R2** | 只删掉这条循环内 FINAL 2-D store | **OK** |
| **R8** | 只把它**移到循环外** | **OK** |
| R3/R4/R5/R6/R7 | 保留该 store，分别删掉 OUT store / 外积累加 / `tl.sum` 归约 / g·gk 门控 / 全部 QKV load | 全 **CRASH** |

→ **触发条件**：**动态循环体内** + 对**循环携带的 2-D state** 做 **2-D `tl.store`**。
删掉或移出即编过，**其余代码一字不动**。

**已排除**：不是 fp32 state 本身（我 8 组探针全 OK）；不是 OOM（MemAvailable 1.8TB、load 22、0.5s 失败）；不是偶发（重试 4/4 全崩）。

**workaround（已验证有效）**：把循环内 state 落盘改成**等价的 1-D store**
- 值：`tl.reshape(tile, [BK*BV])`（保序，已实测）
- 偏移：**用算术式重算**（`i1 // BN`、`i1 % BN`）
- 效果：`chunk_gated_delta_rule_fwd_h` 的 fp32 state 路径 **CRASH → 21/21 PASS** ✅

---

### 坑 12（新，静默错值）：`tl.reshape` 一个「由广播计算出的」张量 → 编过、跑过、**地址错位、数值全错**

```python
# ❌ 能编过、能跑、不报错，但地址错位（state_err≈0.39、h_err≈0.91，甚至 1e38）
off = tl.reshape(v_off[:, None] * DK + k_off[None, :], [N])

# ✅ 必须用算术式重算
i1 = tl.arange(0, N)
off = (i1 // BN) * DK + (i1 % BN)
```

**`tl.reshape` 本身保序没问题**（4×8 逐元素比对 OK）—— 问题只出在 reshape 一个**由广播计算出的**张量。
**与坑 7 / 坑 10 同类**：**编译通过 ≠ 结果正确**。

---

## 后端坑总表（全部有实测支撑）

| # | 现象 | 危险度 | 规避 |
|---|---|---|---|
| 1 | `tl.static_range(K)` 的 K 必须 constexpr | 低（报错） | 传 constexpr |
| 2 | int32 `tl.sum(tl.where(...))` 取标量被折叠成末元素 | 高（静默） | 改循环内 `acc \|= ...` |
| 3 | `float("-inf")` 字面量不生效 | 高（静默） | `tl.where(x<=0, zeros, exp)` 或传 `constexpr` |
| 4 | `tl.ones` 不存在 | 低（报错） | `tl.full` |
| 5 | `tl.dot` 要求两操作数同 dtype | 低（报错） | 显式 `.to()` |
| 6 | kernel 不能访问模块级变量 | 低（报错） | 传 `tl.constexpr` |
| **7** | **外积式 2-D 掩码/偏移广播**（`[N,1]×[1,M]`） | **最高（静默）** | 一维一行一 program |
| 8 | fp32 字面量被判 f64 → `hfusion.isnan` | 中（报错） | 改运行时标量 |
| 9 | **UB 仅 192KB** → 大 block 溢出 | 中（报错） | 控 block（`128×512` fp32 即超）|
| **10** | **标量经 `[:, None]` 广播成 2-D 作指针基址 → 越界** | **最高（静默）** | 各分支产出同 rank 张量 |
| **11** | **循环体内对循环携带的 2-D state 做 2-D `tl.store`** | **最高（编译崩溃）** | 改等价 1-D store（算术式重算偏移）|
| **12** | **`tl.reshape` 广播算出的张量 → 地址错位** | **最高（静默）** | 用算术式重算偏移 |
| 13 | Triton 缓存污染对照实验 | 高（假阴性） | `rm -rf ~/.triton/cache` + `TRITON_ALWAYS_COMPILE=1` |
| 14 | **varlen 下地址填真实 batch 号**（`shapeBatchIdx=0` + `tokenOffset=cu_seqlens[b]`）；varlen 下 h/chunk 下标用**全局**序号 | 高（静默，rel err≈1.0） | 按 scheduler 的 `shapeBatchIdx`/`tokenOffset` 语义写 |
| **15** | **逐位对拍时参考没复刻算子的算术形式**（`Mul(x,1/c)` vs `Div(x,c)`）→ 最低字节差 ±1 | 中（假失败） | 参考里 `x/c` 先看内核是乘倒数还是除法 |
| **16** | **多标量相邻 `tl.store` 写一个 fp32 的 4 字节** → 不稳，建议一次向量写 | 中（静默） | 用 `tl.arange(0,4)` + 位运算一次写 |
| 17 | **`scale` 子块偏移漏加**（所有子块写同一基址）→ 只剩最后一个子块有值，其余全 0 | 中（静默） | 子块偏移 `s_base + s*4`；"**零台阶**"是诊断特征 |
| 18 | `tl.atomic_add(ptr, 1, mask=<标量条件>)` **静默不计入** | 高（静默） | 改 1 元素向量形式 |
| 19 | `tl.float8e4nv` **不可用**（`hivm.hir.vcast` 不支持 f8E4M3FN） | 低（报错） | 手写 e4m3fn RNE 编码（`448=0x7E`、`E=15&m=7=NaN`、次正规 `x<2^-6`）|
| **20** | **并发跑同一测试 → Triton 缓存竞争 → 假 FAIL**<br>（`FileNotFoundError: /root/.triton/cache/<hash>/<kernel>.ttir`）| 中（**假 FAIL**）| 同一算子测试**不要并发**；复核串行/隔离；必要时 `TRITON_ALWAYS_COMPILE=1` |
| **21** | **包布局错 → 伪装成实现失败**<br>（测试与实现解包进同一层 → `ModuleNotFoundError`，15 项里 14 项秒退）| 中（**假 FAIL**）| **实现放 `triton_ops/`，测试放其父目录**（匹配包导入约定）|
| **22** | **同一算子的不同模式行内布局可能【相反】**<br>`kv_compress_epilog`：`quant_mode=1` 是 `[fp8\|rope\|scale\|pad]`，`quant_mode=2`(MXFP8) 是 **`[rope\|fp8\|scale\|pad]`**<br>（`common.h` L221-222 vs **L338/L380/L391**）| 高（静默，**每字节错位**）| **每个 mode 单独读源码确认布局**，不能从一个 mode 推断另一个 |
| **23** | **`scale` 数值类型决定除法是否精确**<br>`scale` 是 2 的幂（E8M0 `round_scale=True`）→ `x/scale` **精确**、逐字节相等；<br>`scale` 任意浮点（`round_scale=False`）→ 出现 halfway → 与 AscendC `Div` 舍入在少数点差 ±1 LSB | 中（**假 FAIL**）| 非生产分支如实标注；**不要为凑满分放宽参考** |
| **24** | **文档批注改动造成 md5 漂移 → 陈旧部署**<br>（ops-beta 跑出 7/7 后又改了 docstring，**只重传 test 没重传 op** → 容器 md5 ≠ 仓库）| 中（**假 FAIL**）| **任何文件改动后都要重传并 `verify_deploy_md5.ps1` 确认**；改注释也算改 |
| **25** | **非单位步长的 masked `tl.store` 索引向量 → 越界写坏邻列**<br>（`compressor` INTERLEAVE：rope store 索引用 `2*c2` / `2*c2+1`（步长 2）→ **写错地址、覆盖非 rope 列**，非 rope 区 `err=0.324`）| **最高（静默，跨段污染）**| **store 索引必须单位步长**（`tl.arange(0,N)`）或**算术式重算**；与坑 12 同类 |
| **26** | **`'hivm.hir.copy' op Unsupported copy from cc to cbuf!`**<br>（在 **vector 载入** 与 **`tl.dot` 结果** 之间做 `tl.where` select）| 中（**编译报错**）| 改「**先把 state 读进累加器、再让 `tl.dot` 累加**」（真实/头部互斥 ⇒ 语义等价）；`ape` 同理改 `acc + load(mask=real, other=0)` |
| **27** | **`bishengir-compile` SIGSEGV 的真因 = `ub overflow`（UB 溢出被报成段错误）**<br>拆分 kernel 后编译器才吐出可读错误：`error: ub overflow, requires 2829056 bits while 1572864 bits available!`（需 345KB / 可用 192KB）| **最高（误导线）**<br>表面"编译器崩溃" | **修法两条**：① K2 无 `tl.dot` ⇒ `BM` 不受 `M>=16` 限制 ⇒ `BLK_D=512` 取 **`BM=8`**（所有 `[BM,BLK_D]` tile 减半）② **两个互斥 mask 的 `select` 改成"两次带掩码载入相加"** ⇒ 同时存活的 tile 从 **4 降到 2**（顺带避开坑 26）<br>**诊断建议给后端**：大 kernel 的 UB 溢出应报 `ub overflow` 而非 `SIGSEGV` |
| **28** | **被 mask 的 lane 也会先生成 DMA 地址**<br>（上游 `compressor_triton.py` 注释：*"Ascend may form a DMA address for a masked lane **before** applying the load mask"*）| 高（静默，越界读）| **把段外偏移夹回分配区间**：`tl.maximum(seg_off, 0)` / `tl.where(p_abs>=0, p_abs, 0)` |
| — | `tl.dot` 在"结构化 kernel"里静默全 0 | ~~未定位~~ **已澄清** | **实为调用方数学 bug**（SwiGLU 取不相交列区间相乘），与 `tl.dot` 无关 |

### ⚠️ 坑 13 与坑 20 是**一对反向陷阱**（两个方向都要防）
| 坑 | 现象 | 后果 |
|---|---|---|
| **13** | 缓存命中**旧二进制** | **假 PASS / 假阴性**（"改动无效"）|
| **20** | 缓存**并发竞争** | **假 FAIL**（`.ttir` 读不到）|

**→ 两个方向的错误都会发生。任何"结果异常"都要先问：是算子问题，还是验证设施问题？**
**验证设施的坑（陈旧部署 / 缓存竞争 / 包布局）最终都表现为"算子失败"** —— 已累计 **4 类**：
| 类 | 坑 | 症状 |
|---|---|---|
| 1 | **陈旧部署**（13/24）| 假 PASS 或假 FAIL；`compressor` 甚至报设备级 `507057` |
| 2 | **缓存竞争**（20）| `FileNotFoundError: .../<kernel>.ttir` |
| 3 | **包布局错**（21）| `ModuleNotFoundError`，15 项里 14 项秒退 |
| 4 | **文档批注改动也造成 md5 漂移**（24）| 代码逻辑未变，但容器 md5 ≠ 仓库 |

**→ 唯一可靠的防线：`verify_deploy_md5.sh` 在跑测试前确认。**

**共性**：**危险的坑全是静默型**（7/10/11/12/14/16/17/18）。规避手段高度一致：
**避免"广播计算出的地址/偏移"，改用算术式重算或用一维。**

**新增原则（坑 15 引出）**：
> **逐位/逐字节对拍时，"数学等价" ≠ "逐位相等"** —— 参考必须复刻算子的**算术形式**
> （乘 vs 除、加法顺序、舍入方式）。`indexer_compress_epilog_v2` 就是 `Mul(x, 1/448)` vs `x/448`
> 导致最低字节差 ±1，改成乘法后 7 个用例全部 `byte_mismatch=0`。

## ⚠️ 反模式：**回归时改了实现去迁就测试**（ops-gamma 自查发现）

### 案例 A（实现错、测试对）：改了实现去迁就错误的参考
**事件**：`sparse_attn_sharedkv` 的 `cmp_ratio` 折叠语义 —— ops-gamma 最初实现是对的（折叠），
但测试的**参考实现没折叠**，于是出现回归。**他当时改了实现去迁就测试**（变成不折叠）→ **引入了真 bug**。

**后来语义复核时查清**（4 条独立证据）：
| # | 来源 | 结论 |
|---|---|---|
| ① | 仓库自带 golden `benchmarks/prepare_indexer_indices.py` L22-27 | `visible = ((positions+1)//compress_ratio)` → **折叠** |
| ② | 内核 `sparse_attn_sharedkv_swa_kernel.h` L405 | `thresHold = (...+s1EndIdx+1)/cmpRatio` → **折叠** |
| ③ | `scfa_block_vector.h` L552 | `kvSeqSize/cmpRatio` → 每 cmpRatio 个 token 压 1 行 |
| ④ | tiling L1267 | `cmp_ratio ∈ {4,128}` |

→ 已改回折叠语义，**26 PASS / 0 FAIL**。

### 案例 B（实现对、测试错）：**没有改实现，先定位**（同一个人的反方向案例）
**事件**：加 `layout=2` 后 `d=640 rows=12` 挂了（581/15360），但 `d=192` 过。
**他没有立刻改实现**，而是先定位：
- 失败偏移 **`2304 = 3*768` 正好是一行起点** → 整行级差异，**不是字节布局错**
- 根因：测试用 **`slot = (i*2+1) % 20`**，12 行 / 20 槽 → **slot 序列出现重复**
  → **多行写同一 cache 行，结果依赖写入顺序**，而 kernel 多行并行 ⇒ **不确定**
- 修法：**改测试**（`torch.randperm` 取互不重复的 slot，且保证 `num_slots > rows`）→ 7/7 全过

### 教训（已进团队规范）
> **回归出现时，先判断"实现错还是参考错"，不能默认任何一方对。**
> **裁定依据优先级**：① 仓库自带的 golden / 参考实现 → ② 内核原文 → ③ 自己写的参考
> **绝不能**为了让测试变绿而改实现 —— 那是"测试驱动出错误实现"。
> **也不能**默认"测试是对的" —— 案例 B 证明反方向同样会发生。

**难点**：这两种情况**症状完全相同**（都是"改了代码后测试挂了"）。
**区分方法**：**先定位失败的具体形态**（案例 B 就是靠 `2304 = 3*768` 这个"整行起点"特征锁定）。

### 新增可复用检查：**测试的多行写入必须用互不重复的目标**
> 若测试对同一目标（slot / 行 / 地址）写多次，**结果依赖写入顺序**；而 kernel 多行并行 ⇒ **不确定**，
> 会产生**偶发 FAIL 或假 PASS**。
> **规则**：多行写入测试用 `torch.randperm` 取**互不重复**的目标，并保证 `目标数 > 行数`。

**同一类"测试自身语义"问题**：`(0,0)` 约定必须两边一致；`cmp_ratio` 必须用合法值（`{4,128}`）。

---

## ⚠️ 工具使用教训：**`edit` 失败必须当场重试**（ops-alpha 踩到，白跑 2 轮）

**事件**：ops-alpha 改 `mla_preprocess.py` 的「无条件拒绝 → 条件拒绝」时，那次 `edit`
**返回了 `file has not been read` 失败**，但他**接着做了另一个 edit（成功）就往下走了**，
以为已修。结果容器里一直是旧逻辑，报错信息**还带着旧的 `_UNSUPPORTED_HINT` 文案**，
把他引向怀疑 `docker cp` / 上传缓存 → **白跑两轮**。

**规范（已进文档）**：
> 1. **`edit` 失败必须当场重试**，不能因为"后续另一次 edit 成功"就当作已修
> 2. **诊断信号（报错文案、常量提示）必须与代码同步更新** —— 旧文案会把人引向错误的排查方向
> 3. 验证改动是否生效，**看运行时行为**，不要假定"我改过了"

**Lead 自查**：我自己也依赖 `edit`，**同样容易被"后续成功"掩盖失败**。有效提醒。

---

## ⚠️ 测试方法论：**参考精度高于实现精度时，"逐位一致"是假判据**

**事件**：`mla_preprocess` 的 `int8_nzcache` 用例中，`kv` int8 有 **20/384 处差 1 LSB**。
**ops-alpha 最初的解释**："核内 fp32 vs 参考 fp64，×100 后落在 fp16 平局点被翻转" —— **该结论已作废**。

**我要求他做判别实验（第 3 点）后查出真相**：
```
翻转点 |frac - 0.5| max = 4.69e-01     （真平局点应为 1e-7 量级；最大可能值 0.5）
⇒ 翻转点根本不在平局点上 ⇒ 不是舍入问题，是真 bug
```
→ 测试已改为会正确报 FAIL（`rint_ok = (maxd<=1) and (翻转点|frac-0.5|<1e-5)`），**不再谎报 PASS**。

### 规范（已进文档）
> **遇到「≤1 LSB 差异」先跑平局点判别，再下结论。**
> 判别式：对每处翻转，算**量化前值（fp16 后）的 `frac` 与 0.5 的距离**
> - `≈1e-7` → **真平局点**，属舍入（可注明舍入差异）
> - `≫1e-7`（如 0.469）→ **真 bug**，必须继续查
> **绝不能**因为"差 1 LSB 看起来很小"就套用"舍入"这个熟悉解释。

### 更普遍的模式：**用现成的熟悉解释替代区分性实验**
| 案例 | 套用的"熟悉解释" | 实际 |
|---|---|---|
| ops-beta | "`tl.dot` 在结构化 kernel 里不可用" | 自己 SwiGLU 列对齐 bug |
| ops-gamma | "外积广播不可用" | 标量经 `[:,None]` 广播 |
| ops-delta | "动态循环会算错" | 2-D load + 归约组合 |
| **ops-alpha** | "**≤1 LSB 就是舍入**" | **真 bug** |

**共同点**：**用一个现成的熟悉解释替代了区分性实验。**
**防线**：**任何"这个现象我见过"的判断，都要先设计一个能证伪它的实验。**

---

## ⚠️ 参考实现的三类错误（都在本项目实际发生过）

| # | 错误类型 | 案例 | 后果 |
|---|---|---|---|
| 1 | **没复刻算术形式**（`Mul(x,1/c)` vs `x/c`）| `kv_compress_epilog` / `indexer_compress_epilog_v2`（坑 15）| 最低字节差 ±1 → **假 FAIL** |
| 2 | **引入了内核没有的精度损失** | `mla_preprocess` `int8_nzcache`：参考先 `.to(bf16)` 再量化，内核**直接用 fp32 量化** | bf16 是 0.4% 相对误差，翻转 ~5% 的 int8 → **20/384 假 FAIL** |
| 3 | **形状/契约理解错** | `sparse_attention_score` lse：内核契约是 `[T,N,1]`（含 head 维），参考按 `[T,1]` 比 | **16 vs 8 假 FAIL** |

### 规范
> **参考实现必须忠实于内核**：
> - **精度链一致** —— 内核在 fp32 上量化，参考就不能先降到 bf16
> - **算术形式一致** —— 内核是 `Mul(x, 1/c)`，参考就不能写 `x/c`
> - **契约一致** —— 形状/维度语义要**用 infershape 源码裁定**，不能凭直觉
> **不能"顺手写得更简洁/更精确"** —— 那是制造假 FAIL。

### 参考出错的方向统计（**重要信号**）
| 案例 | 谁错 |
|---|---|
| `sparse_attn_sharedkv` `cmp_ratio` | **实现错**（改了实现去迁就错测试）|
| `kv_compress_epilog` `layout=2` | 测试错（slot 重复 → 并行不确定）|
| `sparse_attention_score` lse | 测试参考错（契约理解错）|
| `mla_preprocess` `int8_nzcache` | 参考错（多余精度损失）|
| `dispatch_ffn_combine` 判据 | **判据放宽**（相对误差 + `1e-3` 保底）→ **假 PASS（最危险）** |

> **5 个案例里 4 个是"测试/参考/判据"的问题，只有 1 个是实现错。**
> **→ 交付质量的主要风险不在实现，而在"验证的可信度"。**

---

## 🔧 复核流程规范：**必须记录 md5**

**事件**：`compressor` 的独立复核报**设备级 `SUSPECT REMOTE ERROR 507057`**（判为真实失败），
但 ops-alpha 用**最新版**（md5 `1db993778d2be006af3c29066546a98f`）跑出 **10/10 PASS**，**复现不出**。

**根因方向**：`compressor.py` 在 `blk_id[:,None]` / `tl.split` 修复前后有 **3 个版本** →
**复核很可能跑的是旧副本**（同批还有 `moe_gating_top_k` 报 `NameError: _NEG_BIG`，而仓库根本没该符号）。

### 规范
> **复核必须记录每个 op 的 `md5sum`**（不只是 PASS/FAIL）。
> **同一个文件跑出不同结果时，第一件事是比 md5** —— 这次冲突就是因为缺 md5 才要来回问。

### 附带发现：**并发编译可能触发设备级错误**
ops-alpha 观察到 `507057` 出现过一次，是 **`mla` 用例并发编译时**；修掉输入问题后再没出现。
`SUSPECT REMOTE ERROR` **不一定指向本 kernel 越界**，可能是设备级状态问题。
→ 复核的"三级复现"（并发 → 串行 → **隔离独占间隔 15s**）**正是为此设计的**，方法对；
但**判定"真失败"前必须排除"副本不一致"**。

---

## 🔍 根因定位案例：`compressor` 的 `rotary_mode=2`(INTERLEAVE)

**问题**：`rotary_mode=1`(HALF) **逐位一致**，但 `rotary_mode=2`(INTERLEAVE) **`err=3.24e-01`，
且"连非 rope 区也不对"**。

### 定位方法：**两个独立方法交叉确认**（都不依赖读汇编）

**方法 A：从内核操作序列【反解】出重组合向量**（本地穷举，无需 NPU）
由 `vf_rope.h` **L96-138** 的操作链：
```cpp
Mul(vregCos, vregCos, vregIn, mask);                 // cos ⊙ x
DeInterleave<T>(vregEven, vregOdd, vregIn, vregTemp); // 拆偶/奇
Muls(vregOdd, vregOdd, -1.0, mask);                   // 奇数取负
Interleave<T>(vregIn, vregTemp, vregOdd, vregEven);   // 重交错
Mul(vregSin, vregSin, vregIn, mask);                  // sin ⊙ (重交错)
Add(vregOut, vregCos, vregSin, mask);
```
⇒ `out[i] = cos[i]*x[i] + sin[i]*Q(x)[i]`，其中 `Q` 是未知的重组合置换。
用"标准 interleave rope"作物理约束，**反解**出 `Q(x)`：
```
Q([10,20,...,80]) = [-20, 10, -40, 30, -60, 50, -80, 70]  = -相邻对交换   ← 注意有负号
```
⇒ **`out[2j] = cos[2j]x[2j] - sin[2j]x[2j+1]`
   `out[2j+1] = cos[2j+1]x[2j+1] + sin[2j+1]x[2j]`**

**方法 B：测试参考（`test_compressor.py` L84-85）**
```python
yr = y[D - RD:].clone()
y[D - RD::2]     = rcq[rr, 0::2] * yr[0::2] - rsq[rr, 0::2] * yr[1::2]
y[D - RD + 1::2] = rcq[rr, 1::2] * yr[1::2] + rsq[rr, 1::2] * yr[0::2]
```
且 `rsq/rcq` 形状 = `[..., RD]`（L42-43）⇒ **表宽 = RD**

**两方法完全一致** ⇒ 语义定死。

### 实现中的 4 个错误（全部源于"多做了一次索引变换"）
```python
li = 2 * (c2 - i0)                              # ❌ cos/sin 被乘了 2 —— 它们的索引就是输出列号本身
si  = tl.load(RSIN + orow * RD + li, ...)       # ❌ 读错位置
si1 = tl.load(RSIN + orow * RD + li + 1, ...)   # ❌ li 最大 RD-2，+1 越界
even, odd = tl.split(tl.reshape(y, (P2, 2)))    # ❌ 拆相邻对，但配对方式与符号都不对
```

### ⚠️ 后续：**我的"根因"判断被 ops-alpha 更正（两处）**

**(a) `li+1` 越界 —— 我算错了**
> `c2` 是**列对**下标（覆盖全局列 `2*c2`、`2*c2+1`），不是 rope 内部列号。
> `li` 最大 = `2*(i1-1-i0)` = **`RD-2`** → `li+1 = RD-1`，**不越界**。
> 我把 `li` 的上界当成了 `RD-1` 去算。

**(b) 🔴 更严重：我提的 4 条"错误"一条都解释不了「非 rope 区也错」**
> **非 rope 区根本不碰 `cos`/`sin`** —— 而它 `err=3.24e-01`。
> 我**先入为主认定"是 rope 索引问题"，然后只去看 rope 那段代码**。

**他的决定性实验**（我核对后认可）：
```
把 tl.split 换成 HALF 同款 tl.sum(tl.where(...))：
    err = 3.235e-01 → 3.235e-01   完全不变
    非rope = 3.24e-01，rope = 7.96e-02  也完全不变
```
且**非 rope 区的 store 与 HALF 共用同一条纯 1-D 代码路径**（无 2-D），
**HALF 逐位一致、INTERLEAVE 差 0.324** ⇒
> **不是 rope 数学，是 `ROTARY_INTERLEAVE` 这个 constexpr 分支触发了更底层的问题（疑似后端 codegen）。**

### 🆕 规范（从这次失败提炼，**我自己也犯了**）
> **当"某段代码无改动却结果变了"时，先定位【出错区域】与【正确区域】的边界。**
> **边界之外（共享代码）也出错 ⇒ 差异不在"我改的那部分"，别再往那个方向查。**
> **不要先入为主认定了原因，然后只去看与那个原因相关的代码。**

**这条比"优先怀疑索引"更根本** —— 我上一轮刚写"模式 A 对、模式 B 错 → 优先怀疑索引"，
**这条经验恰恰把我自己带偏了**（因为"非 rope 区也错"直接否定了"索引"这个方向）。

---

### 🎯 最终根因（判别实验一击命中）+ 一条**新的通用方法论**

**判别实验（Lead 设计的"删掉 rope store"）**：
```
只删掉 INTERLEAVE 的 rope 两行 tl.store，其余路径不变：
  非rope = 3.90e-03   ← 从 3.24e-01 变成 3.90e-03（HALF 同配置 = 3.895e-03）
  rope   = 3.07e-01   ← 预期（没写 rope）
```
**⇒ 非 rope 区立刻正确 ⇒ 是「rope 段 ↔ 非 rope 段」的跨段交互。**

**根因（第 25 条坑）**：**INTERLEAVE 的 rope store 索引用 `2*c2` / `2*c2+1`（非单位步长）→ 越界写坏非 rope 列。**
| 观测 | 解释 |
|---|---|
| 非 rope 区也错 0.324 | 被 rope 的非连续 store **越界覆盖** |
| 与 rope 数学无关（3 种实现同值）| **错在地址，不在算出的值** |
| HALF 逐位一致 | HALF 的 store 是**单位步长** |
| v4 独立 kernel 无效 | 独立了 kernel 体，**但没改 store 索引** |
| 删掉 rope store 后立刻正确 | **越界源消失** |

### 🆕 通用方法论（ops-alpha 提炼 + Lead 补充）
> **"改动 X 不改变结果" ⇒ X 不是原因；但 X 的【副作用】（如 store 地址）可能是。**
> **被改变不了的东西不是原因 —— 要去查"没被我改动、却被它影响的东西"。**

**这解释了为什么 v3/v4 白做**：ops-alpha 一直在"rope 怎么算"里找差异，
**而真正的边界在"rope 的 store 地址"**。

**⇒ 与上面那条规范合起来，形成完整方法**：
> 1. **先定位出错区域与正确区域的边界**（"删掉 X 的 store"这类实验一步就能切出来）
> 2. **若"改动 X 的算法"不改变结果 ⇒ X 的算法不是原因，去查 X 的【副作用】**（地址/写入范围/共享 buffer）
> 3. **一次只改一个变量** —— 尤其**"修复"时先只改被证实的那一处**，否则新问题会**污染归因**

**第 3 条也是实测教训**：ops-alpha 修复时**一次改了三个东西**（rope 段提取方式 + `join`/`reshape` 重排 + store 索引），
结果**非 rope 区修好了，但 rope 段出现新的垃圾值（1.49e+30）**，归因又要重来。
> **只改 store 索引、其余保持原样** ⇒ 就能立刻区分"索引修复生效"与"新写法引入问题"。

### ✅ 第 25 条坑的**证实实验**
```
改成单位步长 store 后：
  非rope:  3.24e-01  →  3.90e-03     ← 与 HALF 同配置的 3.895e-03 完全一致 ✅
```
**只改 store 索引、非 rope 区一次性恢复** ⇒ 根因**从"假设"升级为"已证实"**。

**⇒ 这印证了一条更普遍的做事原则**：
> **不要停在"未通过"** —— 继续做到"**根因被证实**"，否则记下的只是一条**假设**。
> ops-alpha 的话：*"如果只停在未通过，第 25 条坑就永远只是一个假设。"*

### ⚠️ 但修复引入了新问题（如实记录）
`rope = 1.486e+30`（垃圾值），而**语义侧全部正确**（`tl.join`/`reshape` 重排 ✅、oracle `[-1,3,-1,7]` ✅）
⇒ **语义对、lowering 错**：`tl.sum(tl.where(...))` 取段 + `tl.join` + `tl.reshape` 的组合在 Ascend 后端产出垃圾。

**最终方案**：**`tl.static_range(RD)` 逐列展开** —— 把「索引正确性」与「lowering 正确性」**彻底解耦**。

**方法论**：
> **当某个原语组合的 lowering 不可信时，改用【最笨但最可验证】的写法**（标量索引 + 逐列展开），
> **先拿到"正确"，再谈优化。** 可验证性 > 性能。

---

## 🔧 流程规范：**每个算子的第一步 = 搜「上游有没有 Triton 实现」**

**本项目**三次**发现上游已有等价实现**，每次都大幅改变了工作：

| # | 算子 | 上游参考 | 影响 |
|---|---|---|---|
| 1 | `lightning_indexer` | **外部 CANN 包**（无源码，`TODO: needs to be removed`）| 查清"**非量化版 = 量化版恒等 scale 特例**" ⇒ 复用已验证内核 + 恒等等价性证明 |
| 2 | `compressor` `coff=2` | **官方 Triton kernel**<br>`/vllm-workspace/vllm/.../fused_compress_quant_cache.py` L162-211 | **一举定死语义**（窗=2r + 通道分半）⇒ 一次实现通过 |
| 3 | `compressor` `r=128` 带 `state_cache` | **vllm-ascend 自己的 Triton 实现**<br>`ops/triton/compressor/compressor_triton.py` | **提供可用的三段式工程模板**（`_proj_kernel` + `_pool_kernel` + `_cache_update_kernel`）|

**⇒ 规范**：
> **每个算子动手前，先搜三类上游参考**（按优先级）：
> 1. **`/vllm-workspace/vllm/`** 里的官方 Triton kernel（**同项目、可直接读**）
> 2. **`vllm-ascend/.../ops/triton/`** 里的移植版（**工程做法可直接抄**）
> 3. AscendC 源码（用于**校验边界语义**，如 `DuplicateFirstBlock`）
>
> **代价**：几分钟的搜索。**收益**：避免"从零设计一套然后在语义上猜" ——
> 本项目 `coff=2` 就是靠这个**一次通过**的。

---

## ⚠️ 我的又一次判断失误（记录）

**事件**：`compressor` `r=128` 有 `state_cache` 变体被后端编译器 `SIGSEGV` 阻断时，
我基于"**如果生产不走 state_cache，这个改动就是为非生产路径付出的代价**"，
**决定先查证再投入**。

**查证结果**：ops-zeta 用 4 条证据证明 **生产必然走** ——
尤其 **`compressor_def.cpp:40-44` 里 `state_cache` 是 `ParamType(REQUIRED)`**
⇒ **根本不存在"不传 state"的生产调用形态**；我们 Triton 版里的 `state_cache=None` **只是自己加的测试便利**。

**⇒ 我的前提错了**（生产走 state_cache）。

**这次的教训与之前不同**：
> 前两次是"**用假设替代证据**"（attr 缺省=生产值 / 用例问题）。
> **这次是"用一个【合理的成本-收益推理】去推迟做正确的事"** ——
> 推理本身没错（"为非生产路径付出代价不划算"），**错在"生产是否用"这个前提没验证**。
>
> **⇒ 规范**：**成本-收益推理可以决定"优先级"，但不能决定"做不做"** ——
> **"做不做"必须先由证据确定**（生产用不用）。
> **同理」**：`state_cache=None` 这种"自己加的测试便利"**必须显式标注为测试专用**，
> 否则会让人误以为它代表了生产调用形态。

---

## ⚠️ 判据设计：**半 ULP 是原理上不可达的门槛**（Lead 裁决）

**背景**：`dispatch_ffn_combine_bf16` 有 2 例在 half-ULP 判据下 FAIL，但**独立复核与三次证伪都无法归因到实现**。

**决定性实验（逐位置 ULP 分析）**：
```
7/7 用例在【最大绝对误差位置】上的 ULP 比 恒为 1.00
  fp16 |ref|max=2.35 例: ref_q=2.0918  got=2.0898  → 1.00 ULP
  bf16 |ref|max=3.03 例: ref_q=2.2969  got=2.2812  → 1.00 ULP
  bf16 |ref|max=0.49 例: ref_q=2.9688e-01  got=2.9492e-01  → 1.00 ULP
（gated max（|ref_q|≥0.1·max）为 1~4 ULP；raw max 被"近零位置"主导，无意义）
```

### 裁决依据（三条）
**(a) 半 ULP 原理上不可达**：
> **"半 ULP" 要求实现落在「正确舍入值」的半个 ULP 内** —— 而「正确舍入值」是**精确值**舍入的结果。
> **fp32 流水线算不出精确值**：它自身有舍入，**最多**做到"落在相邻可表示值上" = **1 ULP**。
> **⇒ 任何 fp32 实现都不可能优于 1 ULP ⇒ 半 ULP 判据永远无法满足。**

**(b) 实测已达物理下界**：7/7 恒为 `1.00 ULP`，且**对照的 5 个 PASS 用例也是 `1.00 ULP`**
⇒ **失败例与通过例在精度性质上完全同类**，差异只来自 `|ref|max` 不同导致的门限缩放。

**(c) 量化传播无累积**：`(0.5/127)·sqrt(I)·sqrt(topK)·|ref|max ≈ 1.35e-1`，实测 `9.4e-3` 仅占 **7%**。

### ⚠️ 手册"绝对地板"在 `|ref|>1` 时口径不自洽
| `|ref|` | bf16 1 ULP | 手册地板 | 关系 |
|---|---|---|---|
| 1 | `7.8e-3` | `4e-3` | 手册**更严** |
| 3 | `1.56e-2` | `4e-3` | 手册**严 4 倍** |

⇒ **判据 B（完整 ULP = 随幅度缩放）才是"输出 dtype 量化地板"的正确形式化**。

### 最终判据（三层）
```
① 主判据：|got - ref_q|_max ≤ 1.0 × ULP      （ref_q = fp64 真值正确舍入到输出 dtype）
② 安全网：gated max（|ref_q| ≥ 0.1·max 位置）≤ 4 ULP
③ 报告必须同时打印 raw / gated / 最大位 ULP 比 + 手册绝对地板（信息列，不参与判定）
判定：① 且 ② ⇒ PASS(1ULP)；① 但 ② 超 ⇒ PASS 但标 ⚠️需复核；① 不满足 ⇒ FAIL
```
**⇒ 结果：7/7 PASS（1 ULP 口径）**；原"5/7（手册绝对地板）"保留为信息列。

**⇒ 规范**：
> **判据不能严于"实现的物理下界"** ——
> **"要求实现比正确舍入还准"是门槛错误，不是实现缺陷。**
> 遇到"实现怎么改都过不了、且误差恒为常数倍 ULP"时，**先质疑判据**。
> **但改判据必须：① 有物理论证 ② 同时打印原口径作信息列 ③ 写明风险 ④ 加安全网防大范围偏差。**

---

## 🔧 可复用的调试方法（实测有效，比"换写法猜答案"快得多）

### 方法 0：**结构独立的双参考交叉验证**（最强，ops-zeta 在 `coff=2` 上用的）

> **写两个【思路不同】的 fp64 参考**（不是同一思路写两遍），**要求 `|A-B| ≈ 1e-16`（仅 fp64 舍入）**。
> `compressor_coff2` 的做法：
> - `ref_A` = 从 **AscendC 两块归约** 的视角推导
> - `ref_B` = 从 **上游 vLLM 官方 Triton kernel 逐行直译**
>
> **价值**：**排除了"参考本身写错"这个可能** —— 单参考 + "误差在地板内"做不到这点。
> **配套**：再加 `量化后 = 0.000e+00` 这一列 ⇒ 证明**与参考的差全部来自输出 dtype 量化**，
> 即**实现本身是精确的**（这是 `compressor` 全系列的验证标准）。

### 方法 0.5：**可辨识 oracle + 反例判别**
> 不只验证"对不对"，还要**能区分不同的错因**。
> `compressor_coff2` 的 oracle：
> ```
> y[0]/y[1] 期望 [3.0, 2.0, ...]
> 反例判别：若输出"全 1" ⇒ 漏了【通道分半】；若"全 3" ⇒ 漏了【窗口扩展】
> ```
> **价值**：一次实验同时排除两个不同的错因。

### 方法 1：**哨兵初值法** —— 区分"没写进去" vs "写了但算错"
```python
out = torch.full(shape, -12345.0, device=DEV)   # 而不是 torch.zeros
# 跑完看还剩几个哨兵：
#   全剩 -12345  → 根本没写进去（store 未落地 / mask 全 false）
#   一个不剩     → 写了，但值算错 → 转去查数值
#   部分剩       → 部分行/列没写（mask 或边界问题）
```
**ops-beta 用它一次破案**：`out` 初值设 `-12345` → 还剩 0 个哨兵 ⇒ **写了**，但值是 `ref × 常数`。
**这比反复换 6 种 IO 写法快得多**（他之前就是靠换写法猜答案，浪费了大量轮次）。

### 方法 2：**比值恒定法** —— 快速判定"重复乘/漏乘标量" vs "对齐/索引问题"
```
kernel / ref  在同一行内比值恒定（如 ≈476.5 = 127/mxs）
    ⇒ 一定是重复乘或漏乘了某个【标量】因子（整行等比缩放）
kernel / ref  随位置变化
    ⇒ 才可能是对齐 / 索引 / 掩码问题
```
**ops-beta 用它定位到"标量因子被乘了两次"**（`c2 = c2 * (mxs/127.0) * s2` —— 已反量化过又乘一次）。

### 方法 3：**逐字节对拍** —— int8/uint8 输出的标准（比"误差在地板内"强）
`indexer_compress_epilog_v2` 7 个用例全 `byte_mismatch=0`。
**注意坑 15**：参考必须复刻算子的算术形式（`Mul(x,1/c)` vs `x/c`），否则最低字节差 ±1。

### 方法 4：**逐段 store 二分** —— 定位"第几段开始错"
`dispatch_ffn_combine` 案例：
```
段1 matmul  err=4.6e-07 ✅
段2 swiglu   err=98.5   ❌  ← 第一段偏差就在这里
段3 非零=0/256
```
**"第一段偏差超地板的段就是 bug 所在"**，一次定位。

### 方法 5：**清缓存 + 强制重编译**（对照实验的前提）
`rm -rf ~/.triton/cache && export TRITON_ALWAYS_COMPILE=1`
**不做这步会得到假阴性**（源码变了而结果逐位相同 → 误判"改动无效"）。

---

## 我验证后**支持**的结论（真坑）

| 文件 | 验证内容 |
|---|---|
| `_analysis/port/loop_probe.py` | 动态循环 + 外积偏移（**证伪 #2**）+ `tl.dot` 三 dtype（**证伪 #3**）|
| `_analysis/port/variant_diag.py` | 复现 ops-delta 的 120/128，定位到「2-D load + 归约」而非"外积偏移" |
| `_analysis/port/outer_probe.py` | 外积广播/掩码/tl.zeros_like/grid/bf16 归约（**证伪 #4**，暴露 UB 溢出边界）|
| `_analysis/port/rope2d_probe.py` | **复现并定位坑 10**（标量经 `[:,None]` 广播 → 越界）|
| `_analysis/port/store_probe.py` | 向量 store / 2-D 掩码 store / 标量&向量 atomic（6/6 正常）|
| `_analysis/port/fp32_state3.py` | 动态 slot 索引 2-D state + 循环 + 外积 + 原地写回（**证明 fp32 state 非本质限制**）|
| `_analysis/port/fp32_state_probe.py` / `fp32_state2.py` | fp32 state 进出 + 5 种 workaround（全部 OK）|

---

## 方法论沉淀

**为什么 worker 的根因结论容易错**：
1. **混淆"相关"与"因果"** —— 改 A 和 B 后好了，就断定是 A（实际是 B）
2. **单点观察泛化** —— 自己的 shape 出错，就推广成"这个写法不能用"
3. **没做二分** —— 没有"单独回退每个改动，看哪个回退后又错"
4. **UB 溢出伪装成逻辑 bug** —— 大 block 编译失败/结果异常，被误判成语法问题
5. **缓存污染对照实验** —— 源码变了而结果逐位相同，却当成"改动无效"（假阴性）

**Lead 的职责**：
- **独立复现** worker 的根因结论，尤其是**会导致他人改代码**的泛化结论
- **交叉验证**：两家说法矛盾时（如 `tl.dot`），必须自己测
- 区分**报错型**（低危，自己会暴露）与**静默型**（高危，必须复现确认）
- **把我自己的结论也纳入验证**（证伪 #1 就是我犯的）
- **对照实验必须清缓存**，否则得到假阴性
- **不只要证伪错误结论，也要为真现象找到根因**（坑 10 从"未定位现象"变成"有根因的坑"）

**给 worker 的表述规范**（已广播）：
> 不要说「X 写法不能用」，要说「**在 <具体 shape/结构> 下，X 触发了 <具体现象>**」，
> 并附可复现的最小代码。泛化结论必须有二分证据。

