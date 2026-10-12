# Ascend 后端坑清单（910_93 + BiShengIR / FlagTree / triton 3.2.0）

面向后来者。每条都按「现象 → 复现命令 → 原始输出 → 规避写法 → 已验证/未验证范围」写。
**本文件只写实测过的东西**；凡属推断一律显式标注「推断」。

环境约定
- 硬件/软件：910_93（910C），容器 `flaggems114`（在 114 上），`triton 3.2.0`（BiShengIR/FlagTree 后端）。
- 单测/探针运行目录：容器内 `/workspace/FlagGems/src`（单测就是从这里 import）。
- 设备：除 1.1 的亲自复跑在 `npu:0` 外，本文其余输出均为 `npu:3`；备选 `npu:0/4/5/6/7`。
  撞到 `507057 / HasMemUceErr / 507035` 时**该次结果作废、换卡重跑**（实测 npu:3 上偶发：
  跑 fix-moe 复现脚本时连续两次撞 507057，换 npu:0 即正常；Lead 的逐 die 扫描已证明没有坏卡）。
- 部署三跳（少一跳就会跑到旧文件）：`up.py` 到跳板机 → `ssh root@172.16.10.114 "cat > ..."` →
  `docker cp` 到容器 `/workspace/FlagGems/src/triton_ops/`（部分场景还需
  `.../vllm_fl/dispatch/backends/vendor/ascend/impl/triton_ops/`），每跳 `md5sum` 核对。
- 跑单测（`RC=0` 不等于通过，判定看日志里的 `总体: 全部 PASS`）：
  ```bash
  docker exec flaggems114 bash -c 'cd /workspace/FlagGems/src && DSH_NPU=npu:3 timeout 900 \
      python3 -W ignore triton_ops/test_<name>.py > /tmp/t2d_<name>.log 2>&1; echo RC=$? >> /tmp/t2d_<name>.log'
  ```
  本文里的 `.sh` 都在跳板机 `/root/dsh_work/handoff_1011/`：`t2d_run.sh`（部署+后台跑）、
  `t2d_poll.sh`（取日志）、`t2d_push.sh`（只部署不跑）、`t2d_final.sh`/`t2d_finpoll.sh`（跑 4 个单测）。

---

## 目录（先看这里）

**最常用的一条**：用原生实现当裁判之前，**先用手算例在目标形状上验裁判本身** ——
见「工具页 · 第 0 条：裁判本身也要先验」。

| 节 | 一句话 | 症状关键词 |
|---|---|---|
| 坑 1 | `[BLK_R,BLK_C]` 2-D tile + 2-D 掩码 store（叠加间接寻址）**静默出错** | 保持初值不写（1.1）／把越界垃圾写进掩码内 lane（1.2） |
| 坑 2 | 1-D **跨步 + 掩码** store ⇒ BiShengIR 编译期 **SIGABRT(RC=134)** | `InterleaveOptimization.cpp:635` 断言 |
| 坑 2′ | `tl.reshape` 2-D→1-D 再 store ⇒ **大面积丢 lane** | cols=1024 无掩码仍丢 896/1024 |
| 坑 3 | BiShengIR **UB 容量**（限额 1,572,864 bits）与实测边界 | `ub overflow, requires N bits`；GMM 生产宽度 |
| 坑 4 | **判据/工具本身不可信**（三条，都会把"正确"判成"错"）：4.1 设备端 bool 归约返回垃圾值；4.2 一次崩溃污染同进程；4.3 **参考实现自己会错、"两份参考"可能共享同一个 bug** | `isfinite().sum().item()` 得 4.3e18；"全 0"其实是污染；参考对第二组给全 0 |
| 坑 5 | 计算结果**随输出缓冲内容漂移**（症状 C，fix-moe） | 只改预填值 `-7.5→+7.5`，row0 结果就变 |
| 工具页 | **怎么用原生实现当裁判**（厂商 aclnn 不可用 / FL 原生 payload 可用）；**第 0 条先验裁判** | `aclnnXxx not in libopapi.so` |
| 方法论 | sentinel 双填充判定丢写的三条硬要求 | 单次全 0 不能否证 |
| 对照表 | 已验证／未验证一览（按文件/kernel） | — |

> 注：坑 5 是 `fix-moe` 追加的（原误编为"坑 4"，2026-10-12 由 `fix-2dstore` 重编号以消除重号）。
> 新增小节时请顺号往后排，并在本目录加一行。

---

## 坑 1：`[BLK_R, BLK_C]` 2-D tile + **2-D 掩码 store**（尤其叠加间接寻址）会静默出错

两个症状方向**都**出现过，都不报错、不崩、误差均值看着正常：

### 1.1 症状 A：整 lane 不写 —— 该 lane 保持输出缓冲的初值
提出者 fix-moe，最小复现脚本（**旧 2-D kernel 源码被原样冻结在探针里**，因此不再依赖生产文件）：
`triton_ops/_probe_moe_2dstore.py`（容器内，md5 `8d25b3823eea2af211e14e82e8b2e812`）。
```bash
docker exec flaggems114 bash -c 'cd /workspace/FlagGems/src && MOE_DEV=npu:0 \
    timeout 900 python3 -W ignore triton_ops/_probe_moe_2dstore.py'
```
`V0` = 旧 2-D `_apply_quant_kernel` 原样；`V1` = 基础写法 + src 间址；`V2` = V1 + 除法；
`V3` = V2 + fp16 往返 + rint + 饱和（= V0 等价物）。**本人在 npu:0 亲自复跑 3 次（2026-10-12 08:41~08:43）**：
```
run1: cols=33 BD=8: V0=62 V1=0  V2=31 V3=31 | cols=33 BD=2: V0=62 V1=31 V2=0  V3=0
      cols=48 BD=8: V0=0  V1=16 V2=0  V3=16
run2: cols=33 BD=2: V0=31 V1=0  V2=19 V3=31 | cols=48 BD=8: V0=0  V1=0  V2=0  V3=16
run3: cols=33 BD=8: V0=0  V1=0  V2=63 V3=0  | cols=33 BD=2: V0=31 V1=31 V2=0  V3=31
      cols=48 BD=8: V0=32 V1=0  V2=16 V3=16
cols=64 BD=8（2 的幂对照）: 我完整看到的 2 次运行（batch1-run2、batch2-run3）与 fix-moe 那次，
                          V0=V1=V2=V3=0 全部为 0
```
fix-moe 自己的首次输出（同脚本，08:15）又是另一套：`cols=33 BD=8: V0=31 V1=31 V2=0 V3=1` /
`cols=48 BD=8: V1=V2=V3=16`。
**⇒ 两个可复现的结论：① 该脚本覆盖的非 2 幂列宽（cols=33 / 48）下，3 次运行每次都至少有一个变体丢写；
② 每次运行「哪个变体丢、丢哪些 lane」都会变（3 次运行 3 套结果），而 2 的幂对照 cols=64 每次都是 0。**
丢写形态通常是**逐行连续前缀**（如 `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX..`，31 丢 + 2 写），
但不总是前缀（run2 的 V2 出现过 `.............XXXXXXXXXXXXXXXXXX..` 这种中段形态）。
用同一脚本改写 1-D 形态后该算子单测从 5 FAIL 变为 14/14 PASS（连跑 3 次）。

### 1.2 症状 B：把**掩码外 lane 越界读到的垃圾写进掩码内 lane**（更隐蔽）
探针：`triton_ops/_probe_2dstore_kvcache.py`（本仓库 `_analysis/triton_ops/`）。
受害 kernel：`mla_preprocess.py::_kv_cache_kernel`（`[BLK_M, BLK_D]` 2-D tile + 2-D 掩码 store
＋**由 runtime 载入的 `slot` 做间接寻址**）。改前 `test_mla_preprocess.py` 8 个用例 FAIL，
**全部**错在 `kv_ctkv` / `k_rope`。原始输出（npu:3）：
```
[prod2d]     N=6 KVD=64 RD=32 D2=96 split=True nz=False: drop@expected=0 write@non-expected=0
  prod2d: huge_or_nan=18 (in_expected=18, other=0) 首 8 个 offset=[70,74,78,80,82,98,102,112]
     off=70 slot=1 d=6  val=1.0990e+03  expected_offset?=True
     off=78 slot=1 d=14 val=6.0096e+04  expected_offset?=True
     off=82 slot=1 d=18 val=nan         expected_offset?=True
[prod2d_bd64]（同一输入，仅 BLK_D 256→64）: huge_or_nan=0
[fixed1d]    （1-D 改写形）:               huge_or_nan=0
[cmp] kv: prod2d huge/nan=17  fixed1d huge/nan=0
```
注意这个二分**不能**简单归结为「列掩码部分为真」：KVD=64 时 BLK_D 256→64 会归零，
但 `KVD=512, RD=64` 那一档列掩码全是真值、BLK_D=256 仍复现（`huge_or_nan=52`）。
⇒ **sentinel 判据在这里会误判为「clean」**（drop=0 且无越界写），必须同时看值的量级。

### 1.3 触发条件（实测支持到什么程度）
- **两个确证的受害者都同时具备：2-D tile + 2-D 掩码 store + 由 runtime 载入的值参与地址计算（间接寻址）**：
  MoE 的 `row = src // K`（`src` 从内存载入）、`_kv_cache_kernel` 的 `slot`（从内存载入）。
- **只有「2-D 掩码 + 长值链」而没有任何间接寻址的，我这边实测都没丢**：
  `_probe_2dstore_form.py`（13 形状 × 多个变体，含 `FORM_BLK_R=8 FORM_REPEAT=2` 共 52 个 clean 用例）、
  `_probe_2dstore_rmsdq.py`、`_probe_2dstore_gmmv2.py`、`_probe_2dstore_mlai8.py` 全部 `dropped=0`。
  ⇒ **「2-D 掩码 + 长值链」不是充分条件**，不要据此把所有 2-D store 都改写。
- fix-moe 的二分显示值链本身也会影响命中：V1（无除法/无 fp16 往返）与 V2（+除法）之间会来回切换，
  同一结构下每次运行的受害变体还在变。
- 「推断」（未做正交实验）：**间接寻址是必要条件之一；值链长度/具体运算改变命中概率**。
  我这份矩阵复现不到、fix-moe 的配置稳定复现，也支持「与具体编译产物相关」这一点。

**可靠写法（已被两个独立修复验证）**：
> **标量行号 + 1-D 列 tile + 1-D 掩码 store + 运行期行/列循环**
```python
row = tl.program_id(0)                      # 或 for i in range(BLK_M): row = pid*BLK_M + i; if row < M:
dr = tl.arange(0, BLK_C)
for j in range(tl.cdiv(COLS, BLK_C)):
    d = j * BLK_C + dr
    dm = d < COLS                           # 1-D 掩码
    x = tl.load(X + row * COLS + d, mask=dm, other=0.0)
    tl.store(Y + row * COLS + d, ..., mask=dm)
```
`_kv_cache_kernel` 按此改写后（**签名与 grid 不变**，故两个调用点无需改动）
`test_mla_preprocess.py` 由 8 FAIL 变 `总体: 全部 PASS ✅`，连跑 3 次稳定。

---

## 坑 2：1-D **跨步 + 掩码** 的 store ⇒ BiShengIR 编译期 SIGABRT（RC=134）

> 注意与下文「坑 2′（tl.reshape）」区分：两条都是编译期问题，但触发结构不同。

现象：`sparse_flash_attention` 在 `sparse_block_size=2` 的用例上直接
`RC=134(SIGABRT)`，栈里是 FLIR：
```
/root/ft/FT061/third_party/flir/lib/UtilsIncubated/InterleaveOptimization.cpp:635:
  llvm::LogicalResult mlir::triton::InterleaveStatusWithMaskOptimization(...):
  Assertion `llvm::isa<Attribute>(extractSizes.back())' failed.
```
注意 pass 名字里就写着 **WithMask**。最小复现 `_probe_sfa_ilv.py`（**每个变体一个进程**，
因为崩溃是进程级 abort；跑法见文件头注释），npu:12 实测：
```
stride2_masked   store 偏移 = slot*BS + sub（1-D 步长 2）+ mask   → RC=134（复现，报错原文同上）
stride2_nomask   同上但**去掉 mask**                            → RC=0 OK
contig_masked    改成连续 slot（步长 1）+ mask                    → RC=0 OK
contig_nomask    连续 + 无 mask                                  → RC=0 OK
twod             改成 2-D [BLK, BS] + 2-D mask                    → RC=0 OK
real             真实算子（B2 S1=4 S2=48 N1=4 N2=1 K=8 fp16 bs=2）→ RC=134（复现）
```
⇒ **触发条件是「跨步(步长>1) + 掩码」同时成立**；只要去掉任一条（连续化 或 去掩码 或 改 2-D）即可。

修法（已实施，`sparse_flash_attention.py`）：该处是两趟算法的**内部 scratch `SB`**，
布局可自由选择，于是把交织布局 `SB[row, slot, sub]`（偏移 `slot*BS + sub`）改成
**块状布局 `SB[row, sub, slot]`**（偏移 `sub*K_IDX + slot`，连续写），
`_sfa_out_kernel` 的 lane→(slot, sub) 映射同步改成 sub-major（`slot = lane % K_IDX`、
`sub = lane // K_IDX`）。求和/取最大值与 lane 顺序无关 ⇒ 数值语义不变。
验收：`test_sparse_flash_attention.py` 两次 `总体: 全部 PASS`（npu:0，第 5 例
`B2 S1=4 S2=48 N1=4 N2=1 K=8 fp16 block_size=2` err=1.944e-04 ≤ 1e-3）。

⚠️ 排障过程中的两个教训（都写下来，避免后人重复踩）：

1. **换卡跑单测必须 `torch.npu.set_device(DEV)`**。原本 `test_sparse_flash_attention.py` 写死
   `DEV="npu:0"`，我为了多 agent 并发改成读 `DSH_NPU` 却没加 `set_device`，结果张量分配在 `npu:12/13`
   而 kernel 发到**当前设备（默认 0）**，指针跨设备 → 设备侧 `507035`（"vector core execution is
   abnormal"，日志里还有整片 `aivec error / D-cache reads and writes data to the UB`）。
   表现极具误导性：**npu:0 上 2/2 PASS、npu:12 和 npu:13 都在第一个用例就挂**，看起来像"坏卡"。
   加上 `torch.npu.set_device(DEV)` 后 npu:13 立刻 `总体: 全部 PASS`。
   ⇒ 单测里出现"换卡就挂"时，先查 `set_device`，再怀疑硬件。
2. **探针自己也要先算好边界**。本探针第一版的无掩码变体让 `slot*BS` 写出了缓冲区
   （64 元素缓冲写到 offset 79）——虽然事后用 `_probe_dev_health.py` 复检
   npu:0/12/13/14/15 全部 `OK`（`sum=3145728.0`）、**没有证据表明卡被写坏**，
   但这属于探针自身的 bug，已修（无掩码变体令 `BLK == N`，缓冲区按 `rows*max(N,BLK)*BS` 分配）。

---

## 坑 2′：`tl.reshape` 把 2-D tile 拉平成 1-D 再 store ⇒ 大面积丢 lane

探针：`_probe_2dstore_form.py` 的 `2D_CHAIN_RESHAPE` 变体（值链与 `2D_CHAIN` 完全相同，只多一步
`tl.reshape(qi, [BLK_R*BLK_C])` + 1-D store）。原始输出（npu:3）：
```
[2D_CHAIN         ] cols= 1024 blk=(1,128) dropped=0    wrong=0     clean
[2D_CHAIN_RESHAPE ] cols= 1024 blk=(1,128) dropped=7168 wrong=0     DROP!
[2D_CHAIN_RESHAPE ] cols=   64 blk=(1,128) dropped=448  wrong=0     DROP!
[2D_CHAIN_RESHAPE ] cols=   32 blk=(1,128) dropped=224  wrong=0     DROP!
```
规律（实测地图）：**每个 128 列分块里只有最前面一小段被写入**——cols=1024 时每块写前 16 列、后 112 列全丢
（每行丢 896 = 87.5%）；cols=64 时每行写前 8 列、丢 56；cols=32 时写前 4 列、丢 28。
**cols=1024 且完全无掩码时仍然丢**，说明这与掩码无关，是 reshape 本身的布局问题。
⇒ `tl.reshape` **不能**当作 2-D store 的规避手段。已在 `mla_preprocess.py` 与
`mla_preprocess_int8.py` 的 docstring 里写了这条告警。

---

## 坑 3：BiShengIR **UB 容量**约束（限额 1,572,864 bits）

报错原文形如：
```
error: ub overflow, requires 2359808 bits while 1572864 bits available!
(possible reason: tiling basic block is too large or block number is more than what
 user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
error: Failed to run BiShengIR pipeline
```
探针：`_probe_2dstore_ub.py`（扫 BLK_N/BLK_M/BLK_K）、`_probe_2dstore_ub2.py`（按生产形状打两个 host 函数）。

### 3.1 实测边界（`grouped_matmul_swiglu_quant_v2`，M=16/64，K=128，BLK_M=16，BLK_K=128）
```
I=100/128 (BLK_N=128)  → OK
I=129/200/256 (BLK_N=256) → OK
I=257/300/512 (BLK_N=512) → UB! requires 2,359,808
I=1024     (BLK_N=1024)   → UB! requires 4,719,104
I=1536/2048(BLK_N=2048)   → UB! requires 9,437,696
I=2304     (BLK_N=4096)   → UB! requires 18,874,880
K=2048, I=512             → UB! requires 25,952,768
K=4096/7168, I≥512        → CompilationError（更早，tl.arange 阶段就挂）
```
**只缩 BLK_M 救不回来**：`I=512, BLK_M=1` 仍 `requires 1,704,192`；必须沿 **N 轴**分块。
大 K 也危险：`K=512/I=256` → 3,539,456；`K=1024/I=256` → 6,685,184（即 BLK_K 也要分块）。

### 3.2 生产形状到底是多少（`/public-flash/models/*/config.json` 的 `moe_intermediate_size`）
```
DeepSeek-V4-Flash-w8a8-mtp 2048   DeepSeek-V4.1-Flash 2304   GLM-5.2/5.3-Flash-w8a8 2048
Qwen3.8-2.4T-A95B 2048            Hy4-preview(-W8A8) 2048    DeepSeek-V2-Lite 1408
ERNIE-4.5-21B-A3B / Hy3 1536      Qwen3.5/3.6-35B 512        Qwen3.5-397B 1024
⇒ 最小 512，最大 2304，没有一个 ≤ 256
```
这两个融合 GMM 的 `I = weight_scale 长度 / 2`（gate‖up），host 侧 `blk_n = max(next_pow2(I), 16)`
⇒ **生产 BLK_N ∈ {512, 1024, 2048, 4096}，全部超过实测上限 256**。
`_probe_2dstore_ub2.py` 按生产形状调 host 函数实测：
```
[OK   ] gmm_v2  M=64 K=128 I=256(N=512)
[UB!  ] gmm_v2  M=64 K=128 I=512(N=1024)   requires=2,359,808
[UB!  ] gmm_v2  M=64 K=2048 I=512(N=1024)  requires=25,952,768
[UB!  ] gmm_v1  M=64 K=128 I=512(N=1024)   requires=2,359,808
[UB!  ] gmm_v1  M=64 K=2048 I=256(N=512)   requires=6,686,208
（gmm_v1 = grouped_matmul_swiglu_quant_weight_nz_tensor_list，host tiling 与之逐字相同）
```

### 3.3 这条对生产意味着什么（**已核实的事实**，推断另标）
- `grouped_matmul_swiglu_quant_v2` 在 `_C_ascend_sigs.GROUP_B_OPS` 里，且**不在**
  `patches/patch_triton_ops.py::_DEFAULT_SKIPPED`（该名单只有 4 个：`npu_sparse_attn_sharedkv_metadata`、
  `npu_sparse_attn_sharedkv`、`npu_moe_init_routing_custom`、`npu_vllm_quant_lightning_indexer`）
  ⇒ **Triton adapter 默认已注册生效**（`VLLM_FL_TRITON_OPS=0` 才能关）。
- 容器内 `site-packages/vllm_fl/` 里 `_C_ascend.grouped_matmul_swiglu_quant_v2` 只有**一处**调用点：
  `impl/moe/moe_mlp.py:343`（`torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2`），
  门控 `use_w4a8_per_channel_gmm_swiglu`（= `quant_type == QuantType.W4A8 and is_per_channel_weight`，
  `moe_stage_params.py:88`；函数形参默认 `False`），并且总是传非空
  `weight_assist_matrix=bias1` ⇒ 走到这里会在跑 kernel 前抛 `NotImplementedError`（**响铃**，不是静默）。
- W8A8 的融合分支 `moe_mlp.py:360` 调的是同族 **v1**
  （`grouped_matmul_swiglu_quant_weight_nz_tensor_list`），它同样注册了 Triton adapter、tiling 相同。
- 「推断」：**只要**任一分支按生产宽度（I≥512，K=2048/7168）真正走到这两个 Triton 实现，
  就会在 BiShengIR 阶段编译失败。当前 W8A8 部署是否走到，取决于 `_custom_gmm_swiglu_enabled` /
  `use_w4a8_per_channel_gmm_swiglu` 这两个开关 —— **我没有跑模型验证开关状态**，不要当作已确认。
- 修法方向（**已在 3.5 实施**）：沿 N 轴运行期分块（`BLK_N ≤ 256`）+ 大 K 时同样分块 BLK_K；
  只缩 BLK_M 无效（见 3.1 实测）。

### 3.5 已在 Triton 侧解决（task-6，2026-10-12）
`grouped_matmul_swiglu_quant_v2.py` 与 `grouped_matmul_swiglu_quant_weight_nz_tensor_list.py`
已改为「**N 轴 + K 轴运行期分块**」：host 侧 `blk_n = min(next_pow2(I), 256)`、
`blk_k = min(next_pow2(K), 128)`（小形状下这两个值仍覆盖整个 I/K ⇒ 只有 1 块，
**与分块前数值行为完全一致**，所以原有小形状单测逐位不变，实测 y_diff 仍为 0）。

因为 per-token 量化需要**整行**的 `max|sw|`，分块后单块看不到整行，所以走
「**趟 1 求行内最大值 / 趟 2 重算并量化写出**」——**tl.dot 计算量 ×2**，这是明确记录的代价
（不用 scratch 暂存 `sw` 是因为它要 `M×I×4` 字节，生产 M 可达数万 token → 几百 MB）。

验收（`_probe_gmm_prod.py`，npu:3）：
```
v2/v1 各 K ∈ {2048,7168} × I ∈ {512,1024,2048,2304}（G=2, M=32）：16 组全部
  编译+执行通过，且与 fp64 CPU 参考**逐元素一致**：max|Δ|=0、>1LSB=0、exact=100.0000%、
  scale_err ≤ 1.1e-06（判据 2e-6×max(1,|ref|)）→ A(PASS)=True B(PASS)=True
两个 test_*.py：总体: 全部 PASS ✅
```
UB 余量（同一探针段 C，K=2048 / I=512，直接调 kernel 强制指定 tile）：
```
[OK   ] BM=16 BK=128 BN=256  预测 1,182,464 ≤ 1,572,864（余量 ~25%）
[OK   ] BM=16 BK=128 BN=128  预测 591,232
[OK   ] BM=16 BK=64  BN=256  预测 1,182,464
[UB!  ] BM=16 BK=128 BN=512  实测 3,409,920   ← 拟合式只预测 2,364,928，**低预测 44%**
[UB!  ] BM=16 BK=512 BN=256  实测 3,540,992   ← 拟合式 3,541,760，一致
[UB!  ] BM=16 BK=2048 BN=512 实测 25,954,304  ← 拟合式 25,957,888，一致
```
⚠️ 新结论：**加了 K/N 运行期循环后，BLK_N 很大时比旧的单块写法更费 UB**（同一 BLK_N=512 从
2,359,808 涨到 3,409,920）⇒ 不能只靠拟合式兜底，真正的保护是 `BLK_N ≤ 256` / `BLK_K ≤ 128`
两个硬上限；拟合式只当粗网，超限时 host 侧**显式抛 `NotImplementedError`（带预算与限额）**，
不会静默错值。

### 3.4 另一个 UB 受害算子（他方证据）
`moe_gating_top_k` 生产形状报 `ub overflow 2638848 bits`，根因是**容器里那份文件是旧的**
（`BLK_R_CAP=4`，仓库里已调成 1，kernel body 逐字节相同）；修好并复验 9/9 + 单测 14/14。
⇒ 教训：**UB 边界对 constexpr 极敏感，先核对容器里的文件 md5 再怀疑 kernel**。

---

## 坑 4：**判据/工具本身不可信**（三条，都会把"正确"判成"错"）

> 来源：`fix-tndpa` 在 `dequant_swiglu_quant` / `moe_gating_top_k` 的生产形状扫描里实测
> （探针 `_probe_cov_verify.py` A/A2 段、`_probe_cov_dsq0.py`）；由 `fix-2dstore` 合并进本文件。

### 4.1 设备端 bool 归约（`isfinite`/`isnan` 的 `.sum().item()`）会返回垃圾值
```python
t = op_output                                   # (4096, 8) bf16，实际全部有限
(~torch.isfinite(t.float())).sum().item()       # → 4302524403630753799（≈4.3e18，numel 只有 32768）
torch.isnan(t.float()).sum().item()             # → 同量级垃圾值
(~torch.isfinite(t.detach().cpu().float())).sum().item()   # → 0（正确）
```
后果：批量扫描把 5 个**正常**用例全标成 `NAN`，差点当成"大面积 nan"上报。
**规避：判 nan/inf 一律先 `.detach().cpu()` 再判；不要把设备端 bool 归约当健全性判据。**
表述要准确：同族的 `torch.isnan(tf).sum().item()` 在 `sparse_attn_sharedkv` 探针里又正常
⇒ 准确说法是「**该归约在此后端上不可靠、不能当判据**」，不是"永远错"。

### 4.2 一次内核失败/崩溃会**污染同进程后续全部结果**（把"正常"读成"丢写"）
```
M=65536 → RuntimeError: KernelLaunch failed because value 65536 for parameter coreDim is
          invalid (≤65535)
其后同进程内：M=6656 / 8192 / 16384 的**正常**输出全部变成"非零元素=0"（看着像大面积丢写）
换独立进程逐档重测：M∈{48,512,4096,16384} 非零率 44.0%~45.5%（正常，swiglu 输出天然近半为 0）
```
同类现象（本文件维护者实测）：`_probe_extra_nv.py` 里原生 `npu_recurrent_gated_delta_rule`
触发 `507035` 之后，**同进程后续所有调用**（含本该正常的 Triton 路径）全部报 507035；
换成「每阶段重建输入 / 崩溃项默认跳过」的 `_probe_extra_nv2.py` 才拿到干净结果。
⇒ **规则：判"大面积丢写 / 全 0 / 整片未写 / nan"之前，必须换独立进程复测一次**；
同进程里前面只要有过一次 launch 失败、SIGABRT 或 `507035/507057`，后面所有结果一律作废。
（与「sentinel 双填充法」互补：哨兵法解决"怎么判丢写"，这条解决"判出来的丢写是不是真的"。）
同族补充（`fix-tndpa` F3）：`dumped core` / `UNREACHABLE` 类崩溃同样会让**同进程后续用例全废**
⇒ 批量扫描必须**按组分段**（他们的框架用 `COV_ONLY=<group>` 控制）。

### 4.3 参考实现自己会错，而且"两份参考"可能共享同一个 bug（`fix-tndpa` F7）
现象：`grouped_matmul_swiglu_quant_v2` 的 `glist_type=1` 用例里，扫描器的 `ref_rows_v2`
对**第二组返回全 0 参考**，于是把一个**正确**的 kernel 判成
`max|Δ|=127 / 一致率 66.5% / scale_rel=1e30` —— 典型的"看起来像大面积错值"。
规避（三条）：
1. **判 FAIL 之前先交叉验证参考**：手工逐行 fp64 对照（`_probe_cov_j3_gmm3.py` 的 E1/E2/E3 模式）
   + **重写一份自包含、不共享行切片代码的参考**（`_probe_cov_j3_gmm4.py`）。
2. **两份"独立"参考若共用分组/切片代码 ⇒ 同一个错误被复制两次，不构成交叉验证。**
3. 同族（F8）：三方对照里"取采样行 + 比较"写了两份 ⇒ 同时产出假阳性与假阴性；
   **取行/比较只允许一个入口**（`take(t, rows, shape)` + shape 断言），
   关键结论必须用**重写过的干净版**复核（`_probe_cov_j3_hcpre3.py`）。

**这一节的共同主题**：坑 4 的三条都不是算子/后端的问题，而是**判据本身**的问题 ——
它们会把正确的东西判成错的（比漏报更危险，因为它会让人去"修"没坏的东西）。

---

## 工具：**怎么用原生实现当裁判**（`fix-2dstore` 实测，取代"没有裁判"的旧结论）

> ⚠️ **先看第 0 条**（裁判本身也要先验）—— 它决定下面所有对照是否成立。
>
> 与 `生产形状覆盖表.md` §4 的 F1~F9 的关系：本节的「原生 payload 可用」**取代**了那里的 F6
> （"不存在 AscendC 裁判"，已被 F9 与本节更正）；F2 / F5 / F7 已并入本文件坑 4.1 / 4.2 / 4.3；
> F1（PowerShell 读写毁 UTF-8）、F4（编造生产参数）、F8（取行/比较写两份）仍以那张表为准。

结论先行：本容器里**厂商 aclnn 那条路不可用，但 FL 的原生 payload 可用**。
```
厂商 vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so       → 只是 aclnn 薄封装，
    aclnnHcPost / aclnnHcPreInvRms / aclnnCausalConv1d not in libopapi.so ⇒ 一调就报错
    （`vllm_fl.copied107/` 那份旧 .so dlopen 还会 core dump，别用）
FL 原生 payload：vllm_fl.ascend_custom_ops.enable_custom_op()  → True，11 个 _C_ascend 算子
    全部注册且**真能跑**（实测原生 npu_hc_post 输出 shape=(1,4,4,64) |y|max=0.4824）
    —— 它就是 `patch.py:63 enable_ascend_native_ops()` 在模型进程里调的那条。
```
用法（同进程两阶段对照；探针 `_probe_extra_nv2.py`）：
```python
from vllm_fl.ascend_custom_ops import enable_custom_op
assert enable_custom_op() is True                    # ① 原生就位
run_native()                                         # ② 用 torch.ops._C_ascend.<op> 跑原生

schemas, adapters = triton_ops._C_ascend_extra.build()
lib = torch.library.Library("_C_ascend", "IMPL")     # ③ 按 patch 的方式覆盖成 Triton
for n, fn in adapters.items():
    lib.impl(n, fn, "PrivateUse1")                   #    与 patch 的 _DISPATCH_KEY 一致
run_triton()                                         # ④ 同一批输入再跑一遍 → 逐元素比对
```
### 第 0 条（最重要，先看这条）：**裁判本身也要先验**

> **原生实现只在"被验证过的形状区间"内可信。用原生当裁判之前，必须用手算例在"目标形状"上
> 先验裁判本身对不对。** 这条决定**所有**基于原生对照的结论是否成立 —— 别拿"它在别的形状上是对的"
> 当通行证。

证据（`_probe_native_r2.py`；手算图案取最干净的两种：**全 1** ⇒ 文档值 `1/sqrt(1+eps)=1.0`，
**单个 1** ⇒ 文档值 `sqrt(R)`）：

| shape | R | 图案 | 手算(文档) | Triton | 原生 AscendC | 原生可信？ |
|---|---|---|---|---|---|---|
| (2,4,32) | 128 | ones | 1.000000 | 1.0 | 0.603168 | ❌ |
| (64,4,512) | 2048 | ones | 1.000000 | 1.0 | [0.9562, **0.6963**, 0.9562] | ❌ 逐行不一致 |
| (64,4,512) | 2048 | one_first | 45.208564 | 45.208565 | [31.98, 26.12, 27.15] | ❌ |
| (64,4,512) | 2048 | one_last | 45.208564 | 45.208565 | [45.208565 ✓, 31.98, 33.93] | ❌ |
| (16,4,4096) | 16384 | ones | 1.000000 | 1.0 | [1.0, 1.0, 1.0] | ✅ |
| (8,4,7168) | 28672 | ones | 1.000000 | 1.0 | [1.0, 1.0, 1.0] | ✅ |
| (1,64,4,512) | 2048 | ones | 1.000000 | 1.0 | [0.9561, 0.6963, 0.9546] | ❌ |

⇒ `npu_hc_pre_inv_rms` 的**可信区间实测是 `d ≥ 4096`（R ≥ 16384）；`d ≤ 2048` 不可信**
（同一图案 row0 对、row1 错）。它的**公式**与文档一致（见下面定案），坏的是小 d 下的访存/tiling。
需要在小 d 区间对照时，改用 fp64 手算参考（我们的 Triton 在那些形状上与 fp64 逐元素精确一致）。

**四条注意事项（都踩过）**：
1. **每阶段重建输入**（同 seed 即可）：`causal_conv1d` 之类的算子会**原地更新 state**，
   复用同一份输入会让第二阶段从"已被推进过的状态"起步，比较无效。
2. **别在同进程跑会 507035 的原生 op**：原生 `npu_recurrent_gated_delta_rule` 实测触发 507035 后，
   同进程后续**所有**调用（含本该正常的 Triton 路径）全部报 507035 ⇒ 崩溃项单独进程或直接跳过。
3. **先证调用约定，再判谁错**：用**能手算的整数输入**核对形状与 eps 敏感性
   （例：`x[2,4,32]`、R=128、row0 只有 3/4 两个非零 ⇒ Σx²=25，手算 1/sqrt(25/128+eps)=2.262736）。
   形状或 eps 敏感性对不上，说明是**我的调用约定错了**，此时不得下"原生语义不一致"的结论；
   对上了也还要过第 0 条（在目标形状上验裁判）。
4. `enable_custom_op()` 需要 `SOC_VERSION` 匹配（本容器 `ascend910_9391`），失败会抛
   `RuntimeError: FL Ascend Qwen GDN native payload is unavailable`。

### 用它跑出来的三个结果（探针 `_probe_native_min.py` / `_probe_native_r.py` / `_probe_native_r2.py`）
```
npu_hc_post           原生 vs Triton max|Δ|=0（n=2048）                    ← 原生裁判 PASS
npu_causal_conv1d_custom  单 batch decode：手算 = 原生 = Triton（fp16 精度内逐元素同值）  ← PASS
                      （此前 _probe_extra_nv2.py 报的 2.165 是 harness 假象，已撤回）
```

**`npu_hc_pre_inv_rms`：已定案 —— 公式两边一致，是原生侧在小 `d` 上有访存/tiling 缺陷。**
先读容器里的原生 AscendC 源码（`_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/`
`custom_transformer_impl/ascendc/hc_pre_inv_rms/hc_pre_inv_rms_full_load_large_d.h` L203-216）：
```cpp
AscendC::Cast(castLocal, xLocal, CAST_NONE, R);
AscendC::Mul(castLocal, castLocal, castLocal, R);                 // x²
ReduceSumHalfInterval(yLocal[idx], castLocal[idx*rAlignB32], R);  // Σ_{R} x²
float meanCof = 1.0f / R;  Muls(y, y, meanCof);  Adds(y, y, epsilon);
Duplicate(one, 1.0f);  Sqrt(y, y);  Div(y, one, y);               // y = 1/sqrt(Σx²/R + eps)
```
⇒ **原生的公式与我们的文档/Triton 语义逐字相同**（`1/sqrt(Σx²/R + eps)`，`R = hc*d`）。
再用"全 1 / 单个 1"的手算图案在不同形状上对量（`_probe_native_r2.py`）：
```
shape              R      图案        手算(文档)   Triton      原生
(2,4,32)           128    ones        1.000000     1.0         0.603168        ← 原生错
(64,4,512)        2048    ones        1.000000     1.0         [0.9562, 0.6963, 0.9562]  ← 逐行不一致
(64,4,512)        2048    one_first  45.208564    45.208565   [31.98, 26.12, 27.15]     ← 原生错
(64,4,512)        2048    one_last   45.208564    45.208565   [45.208565 ✓, 31.98, 33.93]
(16,4,4096)      16384    ones        1.000000     1.0         [1.0, 1.0, 1.0]  ← **原生正确**
(8,4,7168)       28672    ones        1.000000     1.0         [1.0, 1.0, 1.0]  ← **原生正确**
(1,64,4,512)      2048    ones        1.000000     1.0         [0.9561, 0.6963, 0.9546]  ← 原生错
```
**结论（三条，都实测）**：
1. **Triton/文档口径是对的**：`one_first/one_last` 这种手算例上 Triton 给出精确的 `45.208565`，
   而原生在小 `d` 下逐行给出不同值（同一图案 row0 对、row1 错）⇒ 是**原生侧的访存/tiling 缺陷**，
   **不是文档错** ⇒ **不需要回头审"按文档写的参考"**。
2. **原生的适用边界**：`d ≥ 4096`（R ≥ 16384）实测与文档完全一致；`d = 512 / 32`（R ≤ 2048）不可信。
   （这也正好解释了我先前在 `[64,4,512]` 上看到的"原生差 0.6"。）
3. ⚠️ **用原生当裁判时，必须先用手算例在"目标形状"上验裁判本身**——它在别的形状上对，不代表在这个形状上对。
   这条已并入上面的注意事项 3。

---

## 复现方法论（sentinel 双填充）—— 三条硬要求

1. **判定用「同一 kernel 跑两遍、两个不同初值」**：输出缓冲先 `torch.full(-99)` 跑一遍，
   再 `torch.full(-77)` 跑一遍；某 lane 两遍分别仍等于各自初值 ⇒ 该 lane 没被写。
   **不能**用「与参考值比较」判定 —— kernel 合法写出的 `-99` 会被误判成丢写。
2. **必须含 load + store 的组合**：纯 store（值是寄存器常量）实测 0 丢写；把 2-D 掩码 load 加回来才出现。
3. **掩码/列宽要是 runtime 标量**、且**位置会漂移**：同一脚本不同次运行丢写位置会变
   ⇒ 单次全 0 **不能**否证；同一配置至少重复 2~3 轮，并保留 2 幂形状作为对照（2 幂实测恒干净）。
4. 症状 B 还要额外看**值的量级**（`|v| > 1e3` 或 `isnan`），因为它的 `drop@expected` 是 0。

---

## 已验证 / 未验证对照表

| 文件 / kernel | store 形态 | 实测结论 | 依据（探针/单测） | 是否已改 |
|---|---|---|---|---|
| `rms_norm_dynamic_quant.py` L148/L160 | 2-D 掩码 store（BLK_R=1） | 非 2 幂 D∈{17…7168} 全 dropped=0/wrong=0；2-D 原形与 1-D 改写形逐元素一致；e2e run2diff=0 | `_probe_2dstore_rmsdq.py`；`test_rms_norm_dynamic_quant.py` 全 PASS | 未改（仅注释） |
| `dequant_swiglu_quant.py` L168/L201 | **1-D** 掩码 store（本来就安全） | H∈{17…7168} dropped=0；e2e wrong>1LSB=0、run2diff=0 | `_probe_2dstore_dsq.py`；`test_dequant_swiglu_quant.py` 全 PASS | 未改（仅注释） |
| `mla_preprocess_int8.py` L65/L133/L172 | 2-D 掩码 store ×3 | dropped=0、write@non-expected=0（D/KVD 非 2 幂全覆盖） | `_probe_2dstore_mlai8.py`；`test_mla_preprocess.py` 全 PASS | 未改（仅注释） |
| `mla_preprocess.py::_kv_cache_kernel` | 2-D 掩码 + **间接 slot 寻址** | **写垃圾**：drop@expected=0 但 18~52 个 lane 是 1e3~6e4/NaN | `_probe_2dstore_kvcache.py` | **已改 1-D**，8 FAIL → 全 PASS（连跑 3 次） |
| `grouped_matmul_swiglu_quant_v2.py` L129 | 2-D 掩码 store + dot/长值链 | dropped=0、wrong>1LSB=0（I∈{17…127}，smooth col/row、limit） | `_probe_2dstore_gmmv2.py`；单测全 PASS | store 未改（仅注释）；**坑 3 的 UB 已按 3.5 分块修复** |
| `grouped_matmul_swiglu_quant_weight_nz_tensor_list.py` | 同族，host tiling 逐字相同 | 生产宽度同样 UB | `_probe_gmm_prod.py` | **已按 3.5 分块修复**（两算子单测全 PASS） |
| `moe_init_routing_custom.py`（fix-moe） | 2-D 掩码 + 间接行寻址 | 丢写（v0 dropped=31 等） | 容器 `_probe_moe_2dstore.py` | 已改 1-D，5 FAIL → 14/14（他方复验） |
| `sparse_flash_attention.py` L125 | 1-D **跨步 + 掩码** store（`slot*BS + sub`） | 编译期 SIGABRT（RC=134），pass 名 `InterleaveStatusWithMaskOptimization` | `_probe_sfa_ilv.py`（5 变体逐进程隔离） | **已改块状布局**，`test_sparse_flash_attention.py` 2/2 `总体: 全部 PASS` |
| 一般新 kernel | — | — | — | **建议一律用 1.3 的可靠写法** |

未验证（不要当成结论）：
- 其他 die（npu:0/4/5/6/7）上的触发概率差异；
- `grouped_matmul_swiglu_quant_weight_nz_tensor_list`（v1）与 `grouped_matmul_swiglu_quant_v2`（v2）
  在**真实模型**里是否真的被调到（取决于 `_custom_gmm_swiglu_enabled` / `use_w4a8_per_channel_gmm_swiglu`；
  我没有跑模型）；
- 坑 1 的精确触发条件：目前证据只支持「2-D 掩码 store + 间接寻址」是充分性很高的组合，
  「值链长度/具体算子」与命中率的关系只有相关证据，没有正交实验；
- 坑 1 症状 B 在 `KVD=512`（列掩码全真）下仍复现的原因未定位（不是单纯「列掩码部分为真」）。

---

## 坑 5：计算结果**随输出缓冲的分配/内容变化**（症状 C，2026-10-12，fix-moe）

**现象**：同一个 kernel、同一组输入、同一个进程内连跑两遍，**只把输出缓冲的预填值从 `-7.5` 改成 `+7.5`**
（`torch.full`），`compressor_coff2` 的**第 0 个输出行**结果就变：`1.65→0.44`、`NaN→0.0`；
其余所有输出行**逐位完全相同**。既不是「保持初值」（双填充判定 dropped=0），也不是「写垃圾」，
而是**算出来的值本身随缓冲分配漂移**。

**受害 kernel**：`compressor_coff2.py::_compressor_coff2_kernel`（生产 `coff=2`：r=4 / D=512 /
20 层 DeepSeek-V4 走它）。坏的是 **ct=0（每个 batch 的第一个 chunk）**那个 program ——
它的窗口有一段是「整块无历史」（`p_abs<0` 的前缀 lane），`c≥1` 的 program 窗口完整、逐位正确。

**探针**（不改算子文件，直接调 kernel + 双填充；同时显式 `torch.npu.set_device`）：
```bash
# 文件：_analysis/triton_ops/_probe_compressor_sentinel.py
docker exec flaggems114 bash -c 'cd /workspace/FlagGems/src && MOE_DEV=npu:0 \
    timeout 900 python3 -W ignore triton_ops/_probe_compressor_sentinel.py'
```
实测（npu:0 与 npu:2 各一遍，结论一致）：
```
S=8  : row0 两遍不同=128/128  run1=[1.6484,-0.5625,…]  run2=[0.4395,0.8242,…]   row1 逐位相同 ✅
S=16 : row0 两遍不同=128/128  run1=[nan,nan,…]         run2=[0.0,-0.0,…]        row1/2/3 逐位相同 ✅
S=24 D=512(生产) : row0 = 0.0（两遍都是，稳定）; row1..5 逐位相同 ✅
所有用例：无任何元素保留填充值（未写=0/0）
```
**辅助证据**：`_probe_compressor_row0.py`（identity-RoPE + 均匀 softmax + 解析候选）显示
c≥1 与解析值误差 = bf16 地板（5e-3），只有 row0 错 —— 说明**窗口/通道/softmax/RMSNorm/RoPE 的数学都是对的**，
坏的只是 ct=0 那个 program 的 codegen/访存。

**规避尝试与其 UB 代价（重要，别重复踩）**：把「2r 行扁平窗 + 前缀 mask」改成
「两个独立 r 行块各自归约后合并」（对齐 AscendC `MulReduceSumbaseVF`）后，
- D=128/H=64 **编译+运行正常**；
- **D=512（生产）UB 溢出：`requires 1982464 bits`（1.26×，限额 1,572,864）** ——
  两种写法都撞同一个数：`for blk in range(2)` 把投影 h-loop **展开成两份**（每份 4 个权重 tile 的
  multi-buffer）是主要来源；D=128 时没问题只是因为 tile 小。
- **⇒ 结论/建议**：在 `BLK_D=512` 下不要复制投影 h-loop；要两块结构就保持**单条融合 h-loop**
  （一次算两个通道的 4 个 dot，如原实现），或先做 D 轴分块把 `[M, BLK_D]` 减半，
  再改归约形态。实验版本已归档为 `_probe_compressor_coff2_2block.py`（**未采用**），
  容器 `compressor_coff2.py` 已回滚到基线 `c2fca68d063f517d451d815aa890ec19`（回滚后 D=512 复核可编译）。

**顺带澄清设备坑（与坑 1 的 507057 记录冲突，以此为准）**：
脚本/探针把设备做成可配（`npu:N`）时**必须显式 `torch.npu.set_device(DEV)`**，
否则张量在 `npu:N`、kernel 发到当前设备（默认 0）→ `507035/507057`（"vector core execution is abnormal" /
"D-cache reads and writes data to the UB"），**看起来像坏卡**。本人在 npu:2 上加上 `set_device` 后同 kernel 完全正常
⇒ 之前「npu:2/3 被 compressor 打挂」的说法**已撤回**；没有证据表明任何 die 坏了。

**未验证**：① 症状 C 的精确触发条件（只知道与「窗口含整块无历史的前缀」相关，未做正交实验）；
② 本 kernel 在 `c0>0`（decode、首块含 state 历史行）下是否也有同样漂移（本轮只测了 `sp=0`）。
