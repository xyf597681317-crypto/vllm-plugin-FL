# 附录 A：踩坑总表与独立复核证据链

> **本附录的目的**：把"多少个 PASS"从**一串数字**变成**可复查的证据链**。
> 每条结论都给出：**出处（源码行号）/ 判据口径 / 原始日志路径 / 复现命令**，任何人可自行复查。
>
> **维护者**：ops-gamma ｜ **最后更新**：本轮复核结束
> **容器**：`vllm-ascend-fanyc-0923` ｜ **triton** 3.2.0（ascend 后端）｜ **UB** 192KB

---

## 零、证据可追溯性的三条硬要求（本轮的实际教训）

任何一条"PASS/FAIL"结论，必须同时记录以下三项，**缺一不可**：

| 项 | 为什么必须记录 | 本轮的实际反例 |
|---|---|---|
| **① 取件来源（md5）** | 同一算子名可能对应**不同修订**；跑的是旧副本会让结论完全反向 | ops-epsilon 12:36 跑的是 10:14 的部署副本，ops-alpha 13:23 又更新了 → 3 项"失败"经 md5 裁定为**跑了旧副本** |
| **② 判据口径** | 手册绝对地板 vs 推导 half-ULP 的**通过数不同** | `dispatch_ffn_combine_bf16`：改判前"全部 PASS"，改判后 **5/7（2 例真实不达标）** |
| **③ 缓存策略** | 缓存命中旧二进制 → **假阴性**；并发编译 → **假 FAIL** | 见坑 13 / 坑 20 |

**工具**：73 文件 md5 基线 `_analysis/_verify/verify_deploy_md5.sh`（容器内 `/root/verify_deploy_md5.sh`），
用法 `bash /root/verify_deploy_md5.sh /root/tops/triton_ops`。

---

## 一、后端坑总表（21 条 + 1 条澄清，全部有实测支撑）

**危险度**：🔴静默（编译通过、结果错）/ 🟠假结论（假 FAIL / 假 PASS）/ 🟡报错（编不过）

| # | 现象 | 危险度 | 规避 | 发现者 | 证据出处 |
|---|---|---|---|---|---|
| 1 | `tl.static_range(K)` 的 K 必须 constexpr | 🟡 | 传 constexpr | ops-beta | `Lead独立验证记录.md` 坑表 |
| 2 | int32 `tl.sum(tl.where(...))` 取标量被折叠成末元素 | 🔴 | 循环内 `acc \|= (gof==gi)` | ops-beta | 同上 |
| 3 | `float("-inf")` 字面量不生效（退化成 `exp(min(x,0))`） | 🔴 | `tl.where(x<=0, zeros, exp)` 或传 constexpr | 手册 | `README_作业说明.md` §5.1 |
| 4 | `tl.ones` 不存在 | 🟡 | `tl.full` | 全组 | 坑表 |
| 5 | `tl.dot` 要求两操作数同 dtype | 🟡 | 显式 `.to()` | 全组 | 坑表 |
| 6 | kernel 不能访问模块级变量 | 🟡 | 传 `tl.constexpr` | ops-gamma | `inplace_partial_rotary_mul` 首次编译报错原文 |
| **7** | **外积式 2-D 掩码/偏移广播**（`[N,1]×[1,M]`）"导致静默垃圾" | — | **⚠️ 该泛化已被证伪**（见下） | ops-gamma 提出 → **已撤回** | `_analysis/port/outer_probe.py`（Lead 实测小 shape 全部逐位正确） |
| 8 | fp32 字面量被判 f64 → `hfusion.isnan` | 🟡 | 改运行时标量 | ops-delta | 坑表 |
| 9 | **UB 仅 192KB** → 大 block 溢 | 🟡 | 控 block（`128×512` fp32 即超） | 三方共识 | ops-gamma 实测 `ub overflow, requires 1576960 bits while 1572864 available!`（`BLK_S2=128×BLK_D=128` → 降 64 通过） |
| **10** | **标量经 `[:, None]` 广播成 2-D 作指针基址 → 读越界（静默垃圾）**<br>特征：**同一份输出里既有精确 0 又有 ~1e38** | 🔴 | 各分支产出**同 rank** 张量（如 `crow = bb*0` 而非 `tl.zeros_like(bb)`），或 1-D 一行一 program | ops-gamma 现象 / **Lead 定位根因** | `_analysis/port/rope2d_probe.py`；Lead 复现 `max\|y\|=6.2308e+36 bad=252/256` |
| **11** | **循环体内对"循环携带的 2-D state"做 2-D `tl.store`** → bishengir LLVM crash | 🟡 | 改**等价 1-D store**（值 `tl.reshape` 展平 + **偏移算术式重算**） | ops-epsilon | `后端崩溃最小复现.md` |
| **12** | **`tl.reshape` 一个"由广播算出"的张量 → 地址错位**（编过/跑过/不报错） | 🔴 | 偏移用算术式重算（`i1//BN`、`i1%BN`） | ops-epsilon | `后端崩溃最小复现.md`（err≈0.39 / 1e38） |
| 13 | **Triton 缓存污染对照实验**（源码变了结果逐位相同） | 🟠**假阴性** | `rm -rf ~/.triton/cache` + `TRITON_ALWAYS_COMPILE=1` | ops-gamma | `Lead独立验证记录.md` §"Triton 缓存会污染对照实验" |
| 14 | **varlen 下地址填真实 batch 号**（应 `shapeBatchIdx=0` + `tokenOffset=cu_seqlens[b]`）；h/chunk 下标用**全局**序号 | 🔴 | 按 scheduler 语义写 | ops-epsilon | `后端崩溃最小复现.md`（rel err≈1.0） |
| **15** | **逐位对拍时参考没复刻算子的算术形式**（`Mul(x,1/c)` vs `Div(x,c)`）→ 最低字节差 ±1 | 🟠**假 FAIL** | 参考里凡 `x/c` 先看内核是**乘倒数**还是除法 | **ops-gamma** | 本附录 §三.1 |
| **16** | **多标量相邻 `tl.store` 写一个 fp32 的 4 字节** → 不稳 | 🔴 | `tl.arange(0,4)` + 位运算**一次向量写** | **ops-gamma** | `indexer_compress_epilog_v2` 修对子块偏移后仍残差 |
| **17** | **`scale` 子块偏移漏加**（所有子块写同一基址）→ 只剩最后一个子块有值，其余全 0 | 🔴 | 子块偏移 `s_base + s*4`；**"零台阶"是诊断特征** | **ops-gamma** | `kv_compress_epilog` / `indexer_compress_epilog_v2` |
| 18 | `tl.atomic_add(ptr, 1, mask=<标量条件>)` **静默不计入** | 🔴 | 改 1 元素向量形式 | ops-beta | 坑表 |
| **19** | **`tl.float8e4nv` 不可用**（`hivm.hir.vcast` 不支持 `f8E4M3FN`） | 🟡 | 手写 e4m3fn RNE 编码 | Lead 预判 / **ops-gamma 实测确认** | 报错原文见 §三.2 |
| **20** | **并发跑同一测试 → Triton 缓存竞争 → 假 FAIL**<br>`FileNotFoundError: /root/.triton/cache/<hash>/<kernel>.ttir` | 🟠**假 FAIL** | 同一算子测试**不要并发**；复核串行/隔离 | **ops-gamma** | 本附录 §三.3 |
| **21** | **包布局错 → 伪装成实现失败**（测试与实现解包进同一层 → `ModuleNotFoundError`） | 🟠**假 FAIL** | **实现放 `triton_ops/`，测试放其父目录** | **ops-gamma** | 本附录 §三.4（15 项里 14 项秒退） |
| — | `tl.dot` 在"结构化 kernel"里静默全 0 | — | **已澄清**：实为调用方数学 bug（SwiGLU 取不相交列区间相乘恒 0），与 `tl.dot` 无关 | ops-beta 撤回 | `Lead独立验证记录.md` 证伪 #3 |

### ⚠️ 坑 7 的最终状态（重要，防止误传）
- **ops-gamma 原始结论**：「`[N,1]×[1,M]` 外积广播 → 静默垃圾」→ **已正式撤回**（我自己的 A/B 对照被缓存污染，证据链不成立）
- **Lead 独立实测**：小 shape 外积广播**全部逐位正确**；大 shape 是**显式 `MLIRCompilationError`（UB 溢出）**，无一例静默垃圾
- **定位到的真根因 = 坑 10**（**标量**经 `[:, None]` 广播，而非"外积"本身）
- **反例（2-D 广播实测 PASS）**：`compressor`、`mla_preprocess`、`sparse_attention_score`
- **该传播的规则**：**控制 block 尺寸避免 UB 溢出**（不是"别用 2-D 广播"）

### ⚠️ 坑 13 与坑 20 是**一对反向陷阱**
| 坑 | 现象 | 后果 |
|---|---|---|
| **13** | 缓存命中**旧二进制** | **假 PASS / 假阴性**（"改动无效"）|
| **20** | 缓存**并发竞争** | **假 FAIL**（`.ttir` 读不到）|

**→ 任何"结果异常"先问：是算子问题，还是验证设施问题？** 验证设施的坑（陈旧部署 / 缓存竞争 / 包布局）**已累计 3 类，全部表现为"算子失败"**。

---

## 二、两张独立复核表交叉比对

### 2.1 方法与来源差异（**必须先看这张表，否则会误读为矛盾**）

| 项 | **表 A：ops-epsilon** | **表 B：ops-gamma（本表）** |
|---|---|---|
| 取件来源 | **容器里的部署副本** | **工作区当前版本**（重新打包上传） |
| 缓存策略 | **清空缓存 + 每项 `TRITON_ALWAYS_COMPILE=1`**（全部重编译） | 复用缓存（但**隔离串行**，避免坑 20） |
| 覆盖范围 | 31 项（全组所有算子） | **15 项**（ops-delta 9 + ops-beta 6） |
| 结论 | **31/31 通过** | **13 项数值 PASS + 1 项"如实标注不可用" + 1 项 2 例真实不达标** |
| 产物 | `真实验收状态.md` §零 | `/root/dsh_work/gamma_verify/RESULT2.txt` + `*_run.log` |

**两者不矛盾** —— 差异来源是"**取件快照时点不同**"与"**判据口径**"，逐项对照见下。

### 2.2 结论一致（置信度最高，**交叉验证通过**）

ops-delta 的 8 项数值算子：`compressor_metadata` / `rms_norm_dynamic_quant` / `hc_pre` /
`dequant_swiglu_quant` / `causal_conv1d` / `moe_gating_top_k` / `add_rms_norm_bias` / `gemma_rms_norm`
→ **两张表都判 PASS**，且我的复核**从工作区取件**、ops-epsilon **清缓存重编译** —— 两条独立路径一致。

ops-beta 的 4 项：`moe_init_routing_custom` / `moe_gating_top_k_hash` /
`grouped_matmul_swiglu_quant_v2` / `grouped_matmul_swiglu_quant_weight_nz_tensor_list`
→ **两张表都判 PASS** ✓

### 2.3 结论不一致（**已定位原因**）

| 项 | 表 A（ops-epsilon） | 表 B（ops-gamma） | 原因裁定 |
|---|---|---|---|
| `dispatch_ffn_combine` | PASS **但对旧修订**（md5 `7eae9afb…` vs 仓库 `813b30ff…`） | PASS（当前版） | **两个修订**，不矛盾；md5 铁证 |
| `dispatch_ffn_combine_bf16` | PASS **但对旧修订**（`3f4aca8f…` vs `775c6ce5…`） | 见下（**已修正**） | 同上 + **判据口径**（见 2.4） |

### 2.4 ⚠️ **我这次复核自身的修正（诚实记录）**
**问题**：我第一次跑 `dispatch_ffn_combine_bf16` 报"全部 PASS（12 条 PASS 行）"。
**Lead 质疑**：ops-beta 自报该算子有 **2 例真实不达标**（half-ULP 口径 5/7），要求确认判据口径。

**我核查的结果**（md5 对照我打包时的副本 vs 当前工作区）：
```
test_dispatch_ffn_combine.py        我打包时=66B88927  现在=3830CABE  **DIFF**
test_dispatch_ffn_combine_bf16.py   我打包时=6EB3F5D8  现在=71048E76  **DIFF**
dispatch_ffn_combine.py             我打包时=813B30FF  现在=813B30FF  SAME
dispatch_ffn_combine_bf16.py        我打包时=775C6CE5  现在=775C6CE5  SAME
```
→ **实现文件相同，但我跑的是改判前的测试**（旧判据：相对误差 + `1e-3` 保底）。

**用当前工作区版本重跑后的确定性结论**（`/root/dsh_work/dispatch_ffn_combine_bf16_run.log`）：
```
fp16 M0=16 topK=2 K=64 I=32 E=3      out_err=1.012e-03 (<=9.766e-04* half-ULP) etn_diff=0  FAIL
bf16 M0=24 topK=4 K=64 I=32 E=4      out_err=9.409e-03 (<=7.812e-03* half-ULP) etn_diff=0  FAIL
（其余 5 例 PASS；4 个"负例：不许静默忽略参数"用例 PASS）
总体: 存在 FAIL ❌（2 例真实不达标，两种口径下都 FAIL）
```
→ 与 ops-beta 自报的 **5/7** **完全一致** ✓ **表 B 以本条为准**。
→ **教训**：我先前那句"12 PASS"是**宽松（旧）判据**的结果；**判据口径必须与数字一起记录**。

### 2.5 `swiglu_group_quant` 的准确定性（**不计入数值 PASS**）
其日志结论为：
```
总体: 全部通过 ✅（不可用已如实标注，且入口不会静默算错）
  入口抛 NotImplementedError 且理由正确? True
  理由: 仅注册 ascend950（swiglu_group_quant_def.cpp L61），910_93 上
        torch.ops._C_ascend.npu_swiglu_group_quant 不存在（容器实测 hasattr=False）
说明：本算子无数值 PASS 结论；在 ascend950 环境实现后再补精度测试。
```
**定性**：**无 910_93 实现 / 无数值 PASS / 不可计入 PASS**。
⚠️ **措辞差异会让自动统计出错**：它写的是"全部**通过**"，`grep "全部 PASS"` **匹配不到**。
→ 统计脚本请同时匹配 `全部 PASS` 与 `全部通过`，并**人工确认"通过"背后是"数值 PASS"还是"如实拒绝"**。

---

## 三、本轮 ops-gamma 新发现的三条坑（附复现细节）

### 3.1 坑 15：参考没复刻算术形式 → 逐位对拍差 1 ulp（假 FAIL）
**场景**：`indexer_compress_epilog_v2` / `kv_compress_epilog` 的动态 fp8 量化。
AscendC 内核是 `Mul(scale, scale, coeff)`，`coeff = (float)(1/448)` —— **乘倒数**。
我的参考最初写 `mx / 448.0`（**除法**）→ 每个 scale 的**最低字节差 ±1**：
```
got=147 ref=146 ; got=220 ref=219 ; got=74 ref=73 ; got=184 ref=183   (10/16896 处)
```
改成 fp32 **乘法**后 → **`byte_mismatch=0`**。
**规范**：逐位/逐字节对拍时，**"数学等价" ≠ "逐位相等"**；参考必须复刻**算术形式**（乘/除、加法顺序、舍入）。

### 3.2 坑 19：`tl.float8e4nv` 不可用 → 手写 e4m3fn RNE
**报错原文**：
```
error: 'hivm.hir.vcast' op currently don't support cast float_to_UNKNOWN_rintmode
error: unrecognized float type: 'f8E4M3FN'
```
**规避**：手写编码（已逐字节对拍 torch 的 `float8_e4m3fn` 验证）：
- sign1/exp4/man3、bias=7、**无 inf**、`E=15 & m=7` 为 **NaN**、最大有限值 **448 = 0x7E**
- 正规：`q = x*2^(3-e) ∈ [8,16)`，`m = round_rne(q)-8`，`m==8` 进位
- 次正规（`x < 2^-6`）：`code = round_rne(x / 2^-9)`
- RNE 用 `diff>0.5 / <0.5 / ==0.5 取偶` 三分支
- `NO_SAT` 溢出我做了**饱和到 448**（`scale=max/448` 保证不触发，**此选择未读 HW 文档确认**）

### 3.3 坑 20：并发跑同一测试 → Triton 缓存竞争 → 假 FAIL
**症状**：`FileNotFoundError: /root/.triton/cache/<hash>/<kernel>.ttir`（**不是** UB overflow、**不是**编译错误）
**原因**：我的复核进程与 ops-delta 自己的进程**同时编译同一 cache key**，一方读到另一方未写完/正在清理的文件。
**判定过程**：**没有当成实现问题** → **隔离重跑**（我独占）→ `add_rms_norm_bias` **8/8 全部 PASS** ✓
**规避**：同一算子的测试不要并发；复核串行/隔离；必要时 `TRITON_ALWAYS_COMPILE=1`。

### 3.4 坑 21：包布局错 → 伪装成实现失败
**症状**：15 项复核里 14 项**秒退**，`ALLDONE_1`、`PASS=0 FAIL=0`，日志只有 `ModuleNotFoundError`。
**原因**：我把 `test_*.py` 和实现文件**一起解包进 `/root/tops/`**，而测试 `import triton_ops.<op>`。
**规避**：**实现放 `/root/tops/triton_ops/`，测试放 `/root/tops/`**（与包导入约定一致）。
**同类**：与"陈旧部署"一样，**验证设施的问题伪装成算子缺陷**。

---

## 四、ops-gamma 交付的 9 个算子：证据表

> **判据**：手册绝对地板（fp32 2e-6 / bf16 4e-3 / fp16 1e-3）；**索引类零容差（逐元素相等）**；
> **逐字节输出**用 `byte_mismatch=0`（比"误差在地板内"强）。
> **取件来源**：与工作区一致（我推送前用 `wc -c` 校验字节数）。
> **日志**：`/root/dsh_work/<op>_run.log`（`--cat` 可看结果 txt）。

| 算子 | Triton 行 | 判据 | 关键实测 | 判定 |
|---|---|---|---|---|
| `inplace_partial_rotary_mul` | 235 | 手册地板 | 调用点 `ds=576 rot=64 bf16` err=1.884e-03 (≤4e-03) | ✅ 12/12 |
| `vllm_quant_lightning_indexer` | 313 | **索引零容差** | TND+PA `cmp_ratio=4` `idx_mismatch=0` | ✅ 17/17 |
| `sparse_flash_attention` | 343 | 手册地板 | TND+PA_BSND `err=1.220e-04` (≤1e-03) | ✅ 15/15 |
| `sparse_attn_sharedkv` | 421 | 手册地板 | TND+PA_ND 双分页 `err=1.183e-04` | ✅ 26/26 |
| `store_kv_block` | 122 | **逐位相等** | `vs_grouped=OK vs_scatter=OK`（**双独立参考**） | ✅ 7/7 |
| `indexer_compress_epilog_v2` | 174 | **逐字节相等** | `byte_mismatch=0/16896` | ✅ 6/6 |
| `kv_compress_epilog` | 254 | **逐字节相等** | **生产配置**（`quant_mode=2` MXFP8 + `group=64` + `round_scale=1` + `layout=1`）`byte_mismatch=0/15360` | ✅ **16/17**（1 例非生产分支，见 §4.3） |
| `lightning_indexer` | 107 | **索引零容差** + 恒等等价 | `idx_mismatch=0` **且** `vs_quant_identity=0` | ✅ 5/5 |
| `lightning_indexer_quant` | 242 | **索引零容差** | TND+PA `idx_mismatch=0`（5 例） | ✅ 5/5 |

### 4.1 两个方法论要点（可复用）
1. **恒等 scale 等价性**（`lightning_indexer`）：把新实现**钉死在已验证的内核**上 ——
   `lightning_indexer(x) ≡ lightning_indexer_quant(x, qs=1, ks=1)` **逐元素相同**。
   依据：**fp16 里 `x * 1.0 == x` 精确**，所以等价性是**逐位**的。
   这比"新实现自己测着 PASS"强得多（后者无法排除"两套逻辑都错但错得一致"）。
2. **双独立参考**（`store_kv_block`）：除 AscendC 语义参考外，再加一个**调用点等价物**参考
   （`indexer.py` 非 reshape-optim 路径走的是 `scatter_cache`）。两条都逐位吻合才算过。

### 4.3 ⚠️ 生产路径覆盖度（**两列必须分开看**）

> **"生产配置是否已实现"** 与 **"是否在本容器（910_93）验证过"** 是**两个不同的问题**。
> 生产调用点多为 **A5（ascend950）/ A3** 路径，而验收容器是 **910_93** —— 不可混为一谈。

| 算子 | 生产配置 | 生产是否已实现 | 910_93 是否验证过 | 证据（file:line） |
|---|---|---|---|---|
| `kv_compress_epilog` | **`quant_mode=2` + `quant_group_size=64` + `round_scale=1` + `layout=1`** | ✅ **已实现（本轮）** | ✅ **逐字节 PASS** | `dsa_attn_kv_plan.py` L195-203；fl `device_op.py` L1542-1550 |
| ↳ 同算子 `quant_mode=1`（非生产） | — | ✅ 已实现 | ✅ 逐字节 PASS（**保留为"给定配置下的实现正确性"**） | attr 缺省，无调用点 |
| ↳ `round_scale=0` 分支 | ❌ 生产不用 | ✅ 已实现 | ⚠️ **5/15360 字节差 1**（见下） | 生产固定 `round_scale_flag=True` |
| `indexer_compress_epilog_v2` | `layout=2` | ✅ 已实现 | ✅ 逐字节 PASS | `device_op.py` L1440 |
| ↳ `fp8_e5m2` | ❌ **生产不用** | ❌ 未实现（显式报错） | — | `attention_c8_mxfp.py` L188 `dtype=torch.float8_e4m3fn` |
| `compressor` | `rotary_mode=2`(INTERLEAVE) **生产在用** | ⚠️ **未覆盖**（根因未定位） | — | `models/deepseek_v4/compressor.py` L221；`dsa_cp.py` L1940/L2093 |
| ↳ `coff=2` | ✅ 生产在用（overlap 配置） | ⚠️ **未覆盖**（需 overlap 跨 chunk + workspace，属新功能） | — | `compressor.py` L219；`dsa_cp.py` L1919/L2067 |
| `mla_preprocess` | `per_tensor_quant_asymm` | ✅ 已实现 | ✅ PASS（`mla_preprocess_int8.py`，4 输出全 0.000e+00） | `sfa_v1.py` L1561 |
| ↳ `enable_inner_out=True` | ✅ 生产在用 | ⚠️ **未覆盖**（显式 `NotImplementedError`） | — | `sfa_v1.py` L1562 |

**⚠️ 教训（Lead 自查）：**
> **我曾假设"attr 缺省值 = 生产值"，没查调用点。**
> 这与"**用熟悉解释替代区分性实验**"是同一类错误 —— **都是用假设替代证据**。
> `kv_compress_epilog` 的 attr 缺省是 `quant_mode=1`，而**唯一生产调用点传的是 `quant_mode=2`**。

**`round_scale=0` 的 5 字节差异（非生产分支，如实记录、不计 PASS）**：
```
d=576 group=64 rscale=0 rows=9   byte_mismatch=5/15360
  offset 1690: got=123 ref=124 ; offset 1725: got=105 ref=106 ; ...
```
位于 fp8 值区（如 `1690 - 2*768 - 128 = 26`），是**除法舍入边界**的 ±1 LSB：
`rscale=1` 时 scale 是 2 的幂 → 除法**精确** → 逐字节相等；
`rscale=0` 时 scale 是任意浮点 → `x/scale` 出现 halfway 值，与 AscendC 的 `Div` 舍入在极少数点不同。
**生产固定 `round_scale_flag=True`**，故不影响生产路径。

### 4.2 语义假设的最终处置
| 假设 | 处置 | 依据 |
|---|---|---|
| `cmp_ratio` 索引含义 | ✅ **移出"语义假设"**（已查清并要求**折叠**） | ① 仓库自带 golden `benchmarks/prepare_indexer_indices.py` L22-27 `visible=(positions+1)//compress_ratio`；② 内核 `swa_kernel.h` L405 / `scfa_kernel.h` L373,589,600 `thresHold=(cmpMaskRight+s1EndIdx+1)/cmpRatio`；③ 物理含义 `scfa_block_vector.h` L552 `kvSeqSize/cmpRatio`；④ 合法取值 `tiling.cpp` L1267 `cmp_ratio ∈ {4,128}` |
| `block_stride` 缺省值的 scale/value 重叠 | ⚠️ **降级为"已知陷阱"**（生产不触发） | `torch_binding.cpp` L1902 传 **`cache.stride(0)`**，attr 缺省值生产走不到；且该 fused 路径是 A5 **在跑的出货路径** → `stride(0)` 实际足够大 |
| `indexer_compress_epilog_v2` 的 `NO_SAT` 溢出行为 | ⚠️ **实现选择，未读 HW 文档确认** | 我做了饱和到 448；`scale=max/448` 保证 |v|≤448，此路不触发 |

---

## 五、反模式案例（防"测试驱动出错误实现"）

| 案例 | 情形 | 结果 | 教训 |
|---|---|---|---|
| **A（实现错、测试对）** | `sparse_attn_sharedkv` 的 `cmp_ratio`：加入 cmp 掩码后 BSND 用例回归，而**参考当时没折叠** → 我**改实现去迁就错误的参考** | **测试全绿、实现是错的**（最危险） | **绝不能为了让测试变绿而改实现** |
| **B（实现对、测试错）** | `kv_compress_epilog` 加 layout=2 后 `d=640 rows=12` 挂：测试 `slot=(i*2+1)%20` **重复** → 多行写同一行，并行下不确定 | 定位到 `2304 = 3*768`（**整行起点**）→ 判定为测试问题 → 改 `randperm` 后 7/7 PASS | **先定位失败的具体形态**（整行 vs 单字节）再判方向 |
| **C（判据口径）** | `dispatch_ffn_combine_bf16`：我跑旧判据得"全部 PASS"，当前判据 **5/7（2 例真实不达标）** | 见 §2.4 | **判据口径必须与数字一起记录** |

**裁定优先级**：① **仓库自带 golden / 参考实现** → ② **内核原文** → ③ **自己写的参考**。

**同类"测试自身语义"问题清单**（已累计）：
- `partial_slice=(0,0)` 的"整段"约定必须**两边一致**
- `cmp_ratio` 测试值必须在**合法值域** `{4,128}` 内（用非法值测试，即使 PASS 也不代表生产）
- **多行写入必须用互不重复的目标**（slot/行/地址），否则并行下不确定 → 偶发 FAIL 或**假 PASS**

---

## 六、日志与出处索引（可复查）

| 内容 | 路径 |
|---|---|
| 本附录的复核表 | `/root/dsh_work/gamma_verify/RESULT2.txt` |
| 本附录的复核原始日志 | `/root/dsh_work/gamma_verify/<op>.log` |
| ops-gamma 9 个算子的测试日志 | `/root/dsh_work/<op>_run.log`、`/root/dsh_work/<op>_test.txt` |
| 坑 10 的 Lead 复现 | `_analysis/port/rope2d_probe.py`、`_analysis/port/outer_probe.py` |
| 坑 11/12/14 的最小复现 | `_analysis/triton_ops/后端崩溃最小复现.md` |
| 坑表与证伪记录 | `_analysis/triton_ops/Lead独立验证记录.md` |
| 验收状态与两张复核表 | `_analysis/triton_ops/真实验收状态.md` |
| md5 基线工具 | `_analysis/_verify/verify_deploy_md5.sh`（73 文件基线） |

---

## 七、一句话总结

> **"多少个 PASS"本身没有意义；有意义的是一条"取件来源 + 判据口径 + 缓存策略 + 原始日志"齐备的证据链。**
> 本项目的 3 类"验证设施坑"（**陈旧部署 / 缓存竞争 / 包布局**）**全部伪装成算子缺陷**，
> 而 4 类"静默坑"（**10/11/12/14**）**全部伪装成算子正确** —— 两者都要防。
