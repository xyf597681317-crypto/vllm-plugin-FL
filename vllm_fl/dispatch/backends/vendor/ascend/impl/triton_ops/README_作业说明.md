# AscendC → Triton 迁移作业说明（teammate 必读）

> 本文件是**自包含**的作业手册。所有路径、命令、规范以本文件为准，不要凭猜测。
> 任务：把 vllm-fl 仓库 `csrc/ascend/` 下的 AscendC 算子用 **Triton** 重写，精度对齐。

---

## 一、目标与产出

**把 AscendC 算子重写为纯 Triton 实现**，要求：

1. **只用标准 `triton` / `triton.language`**
   - ❌ 禁止 `import vllm`、`import triton_ascend`、`import flag_gems`
   - ❌ 禁止 `triton.language.extra.cann.extension`（`insert_slice`/`extract_slice`/`get_element` 都不许用）
   - ✅ 只用 `tl.load`/`tl.store`/`tl.dot`/`tl.arange`/`tl.where`/`tl.exp` … 标准 API
2. **精度对齐** AscendC 原实现的语义（用 **fp64 参考**对拍）
3. 产出两个文件：
   - `_analysis/triton_ops/<op_name>.py` —— Triton 实现
   - `_analysis/triton_ops/test_<op_name>.py` —— 精度测试

---

## 二、源码位置（本地，只读）

AscendC 源码在本地克隆里：

```
C:\Users\笑笑\Desktop\公司\flaggems\_repos\vllm-plugin-FL-xmd\
    vllm-plugin-FL-feature-v024-ascend-platform-runtime\
        csrc\ascend\<category>\<op_name>\
            op_host\      ← 接口定义（输入/输出/属性）+ tiling
            op_kernel\    ← AscendC 内核实现（要读这个来提取公式）
```

**读代码的要点**：
1. 先读 `op_host/<op>_def.cpp` → 拿到 **Input/Output/Attr 的名字、个数、dtype、shape 规则**
2. 再读 `op_host/<op>_tiling*.cpp` / `_infershape.cpp` → 拿到 shape 推导规则
3. 最后读 `op_kernel/<op>.h` 或 `<op>.cpp` 的 **`CoreCompute` / `Process`** → 提取**数学公式**
   - AscendC 内核文件顶部常有**注释直接写出公式**（如 `g = -exp(A_log) * softplus(...)`）——优先看
   - 注意 `__aicore__` 只是标记，`extern "C" __global__ __aicore__` 才是入口

---

## 三、环境与部署（重要）

### 3.1 目标环境

| 项 | 值 |
|---|---|
| 机器 | `10.55.0.166`（主机名 `bm-jn-zs-zone1-910C-64G-10-105`） |
| 容器 | `vllm-ascend-fanyc-0923` |
| Python | 3.12.13 |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| triton | 3.2.0（ascend 后端，毕昇编译） |
| 容器工作目录 | `/root/tops/`（已建好，`triton_ops/` 包在里面） |

### 3.2 SSH 连接方式

用仓库里现成的工具脚本（**不要自己写 paramiko**）：

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:JMP_PASS  = 'mFZZBZ7x&5KZDU'
$env:JSSH_USER = 'xinyufeng@root@10.55.0.166'
$A = "C:\Users\笑笑\Desktop\公司\flaggems\_analysis"

# 跑命令（可多条，按顺序）
python "$A\q.py" "cmd1" "cmd2"

# 先上传本地脚本再跑
python "$A\q.py" --up "$A\port\xxx.sh" "nohup bash /root/dsh_work/_probe.py > /root/dsh_work/out.txt 2>&1 &" "sleep 60; echo ok"
python "$A\q.py" "cat /root/dsh_work/out.txt"
```

**关键注意**：
- `q.py --up` 把文件传到宿主机 `/root/dsh_work/_probe.py`（**不是容器里**）
- ⚠️ **宿主机 `cp` 被 alias 成 `cp -i`**（会交互确认卡住）→ 必须用 **`\cp -f`** 或 **`mv -f`**
- Docker 命令输出不要用嵌套引号（PowerShell 会吃）→ **写成 .sh 脚本再执行**
- 容器里跑 Python 要 `docker exec -d`（后台）+ 轮询产物文件

### 3.3 部署测试的标准动作

```bash
# 在宿主机（通过 q.py 执行）：
C=vllm-ascend-fanyc-0923
\cp -f /root/dsh_work/_new.py /root/dsh_work/tops/triton_ops/<op>.py
docker cp /root/dsh_work/tops/triton_ops/<op>.py $C:/root/tops/triton_ops/<op>.py
docker cp /root/dsh_work/tops/test_<op>.py      $C:/root/tops/test_<op>.py
docker exec -d $C bash -lc 'cd /root/tops && python3 -W ignore test_<op>.py > /root/dsh_work/<op>.log 2>&1'
sleep 70
docker exec $C cat /root/dsh_work/<op>.log
```

### 3.4 已有成果（可直接参考写法）

- `_analysis/triton_ops/common.py` —— 公共工具（`npu_device`/`ceil_div`/`next_pow2`）
- `_analysis/triton_ops/__init__.py`
- **`_analysis/triton_ops/fused_gdn_gating.py`** ← **范例，照它的结构写**
- **`_analysis/triton_ops/test_fused_gdn_gating.py`** ← **测试范例，照它的结构写**

---

## 四、精度验收标准（不许放松，也不许过严）

用 **fp64 参考**（`.detach().cpu().double()`）对拍，误差 ≤ **输出 dtype 的量化地板**：

| 输出 dtype | 地板 |
|---|---|
| fp32 | **2e-6**（数学等价写法差异导致的 fp32 舍入）|
| bf16 | **4e-3** |
| fp16 | **1e-3** |
| int32/int64（索引类） | **必须完全相等** |

### ⚠️ 重要修订（实测发现）：**上面的绝对地板在 `|ref|max > 1` 时物理不可达**

**实测证据**（`dispatch_ffn_combine_bf16` / `moe_gating_top_k_hash`）：
```
7/7 用例在【最大绝对误差位置】上的 ULP 比 恒为 1.00
    fp16 |ref|max=2.35 例: ref_q=2.0918  got=2.0898  → 1.00 ULP
    bf16 |ref|max=3.03 例: ref_q=2.2969  got=2.2812  → 1.00 ULP
⇒ 实现结果与"fp64 真值【正确舍入】到输出 dtype"的值恰好相差 1 个 ULP
```

**原因**：
> **"半个 ULP" 或"比 1 ULP 更严的固定地板"要求实现落在「正确舍入值」的更近处** ——
> 而「正确舍入值」是**精确值**舍入的结果。
> **fp32 流水线算不出精确值**（它自身有舍入）⇒ **最多**做到"落在相邻可表示值上" = **1 ULP**。
> **⇒ 任何 fp32 实现都不可能优于 1 ULP。**

**而固定地板不随幅度缩放**：
| `\|ref\|max` | bf16 输出 1 ULP | 原地板 | 关系 |
|---|---|---|---|
| 1.0 | `7.8e-3` | `4e-3` | 地板更严（**但可达**）|
| 3.03 | **`1.56e-2`** | `4e-3` | **地板比 1 ULP 还紧 4 倍 ⇒ 不可达** |

### ✅ 修订后的判据：**按幅度分段**（README 原意的正确形式化）

```
|ref|max ≤ 1  →  用本表绝对地板（bf16 4e-3 / fp16 1e-3 / fp32 2e-6）
|ref|max > 1  →  用输出 dtype 的【完整 1 ULP】@ |ref|max
```
其中 `1 ULP @ v = 2^(floor(log2|v|) - (p-1))`，`p` 为含隐含位的有效位数（bf16 `p=8` / fp16 `p=11`）。

**⚠️ 不要写成"两者取较严者（`min()`）"** —— 字面 `min()` 会退化成"永远用 README 地板"，
在 `|ref|>1` 时仍不可达。**`|ref|=1` 是两条曲线的交叉点**：
- `|ref|≤1`：README 地板**更严** ⇒ 用它
- `|ref|>1`：1 ULP **更严** ⇒ 用它

**为什么"分段"是正确口径（三条理由）**：
1. **README 标题原文就是「误差 ≤ 输出 dtype 的量化地板」** —— 而**量化地板在物理上就是 ULP 量级**
   （`|ref|≈1` 时 bf16 的 1 ULP = `7.8e-3`，与 README 的 `4e-3` 同量级）。
   **README 的固定数值只是"`|ref|≈1` 处的近似"**，不是"对所有幅度都成立的绝对约束"。
2. **固定地板在 `|ref|>1` 时比输出可表示精度还严** ⇒ **不可达**（`|ref|=3.03` 时 bf16 1 ULP = `1.56e-2`，README 是 `4e-3`）。
3. **本节的标题就写着「不许放松，也不许过严」** ⇒ 固定地板在 `|ref|>1` 时属于**"过严"**，按本节自己的要求就不该用。

**⇒ 这是 README 原意的正确形式化，不是放宽。**

**附加要求（防"万能放行"）**：
1. **必须同时打印**：`raw max ULP` / `gated max ULP`（`|ref_q| ≥ 0.1·max` 的位置）/ **最大绝对误差位置的 ULP 比**
   + **原绝对地板（信息列，不参与判定）** + **字面 `min()` 值（供核对口径）**
2. **安全网**：`gated max ULP ≤ 4`；超出 ⇒ **标 `⚠️ 需复核`（不静默通过）**
3. **判据变更必须写明理由与风险**（文件头）
4. **每个 FAIL 在改判前必须先用 ULP 分析证明它是"1 ULP 伪影"** ——
   **不允许"口径统一了就都算过"**（反例：`moe_gating_top_k_hash` 的 `rsf=2.5` 例实测 `ULP@maxerr = 0.00`，
   即实现是输出 dtype 的**最优表示**，才判通过）

**⇒ 依据**：`1 ULP` 是 fp32 流水线的**物理可达下界**（fp32 算不出精确值，最多落在相邻可表示值上）。

**⚠️ 注意事项**：
- ⚠️ **`Tensor.double()` 在 NPU 上会静默降级为 fp32** → 必须 `.detach().cpu().double()`
- ⚠️ 对拍时**输入只生成一次**，算子与参考共用同一份（否则是伪误差）
- ⚠️ **`raw max ULP` 会被"近零位置"主导**（`out` 有抵消时 `ref_q≈0`，`ulp_at` 极小，微小绝对差折算成几百~几千 ULP）—— **不能当判据**，只有 `gated` 与"最大误差位置"有信息量
- 若某算子误差**系统性**超过地板，先怀疑**自己的参考实现写错了**（这个坑踩过 5 次）
- 调试技巧：用**可辨识数据**（`input[i]=i`、分块不同常数）定位是哪一段算错

---

## 五、输出格式规范

### 5.1 Triton 实现文件（`<op>.py`）

```python
"""<op_name> —— AscendC → Triton

来源：csrc/ascend/<category>/<op_name>/
接口（op_host/<op>_def.cpp）：
    Input : ...
    Output: ...
    Attr  : ...

公式（op_kernel/<op>.h L?? 的注释/实现）：
    ...
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
from .common import ceil_div, next_pow2

__all__ = ["<op_name>"]

@triton.jit
def _<op>_kernel(...):
    ...

def <op_name>(...) -> ...:
    """等价于 AscendC 的 <op_name>。"""
    ...
```

**要求**：
- 模块 docstring 必须写清：**来源路径 + 接口 + 公式出处行号**
- 一个 `<op>.py` 只放一个算子

---

## 五之二、⚠️ Ascend Triton 后端已知坑（实测踩过，务必避开）

### 坑 1：`tl.static_range(K)` 的 K 必须是 `tl.constexpr`
传运行时参数会直接 `CompilationError`。循环上界若依赖输入 shape：
要么改成编译期常量，要么用普通 `range`。

### 坑 2（严重，后端 bug）：`tl.sum(tl.where(...))` 在 int32 向量上取标量会取错
```python
# ❌ 这个写法在 Ascend 后端会错误地返回向量的【最后一个元素】
val = tl.sum(tl.where(tl.arange(0, BLK) == j, vec, 0), axis=0)
```
实测后果：用它做分组 gating 候选集时，候选集只剩最后一组。
**规避**：改成融合循环，或换别的取元素方式。
**不要用 `tl.sum(tl.where(...))` 做 int32 标量提取。**

### 坑 3：`float("-inf")` 字面量不生效
会退化成 `exp(min(x, 0))`。规避：
```python
tl.where(x <= 0, tl.zeros_like(y), tl.exp(y))
```
或把大负数作为 `tl.constexpr` 传入（如 `NEG_BIG: tl.constexpr`）。

### 坑 4：`tl.ones` 不存在
用 `tl.full([N], 1.0, tl.float32)`。

### 坑 5：`tl.dot` 要求两个操作数 dtype 相同
不同 dtype 会 `AssertionError('Both operands must be same dtype')` → 显式 `.to(...)`。

### 坑 6：kernel 里不能访问普通模块级变量
`NameError: Cannot access global variable X from within @jit'ed function`
→ 作为 `tl.constexpr` 参数传入。

### 🔴 坑 7（最危险，**静默数据损坏，不报错**）：`[N,1] × [1,M]` 外积式广播
```python
# ❌ 不报错，但输出全是垃圾（实测 err=1.2e38，部分位置被写成 0）
rm = tl.arange(0, BLK_R)[:, None]
km = tl.arange(0, BLK_P)[None, :]
m    = rm_mask[:, None] & km_mask[None, :]
offs = rm * stride + km                      # [N,1] × [1,M] 广播
```
换 `tl.zeros_like` 也无效。**改成一维、一行一个 program 后立刻全对。**

**规避**：**避免 `[N,1]×[1,M]` 的外积式掩码/偏移广播**，宁可一行一 program 用纯一维。

> **为什么最危险**：它不报错。你只会看到精度不对，
> 很容易误判成"自己的参考实现写错了"而浪费大量时间。
> **如果掩码恰好恒真、或 shape 恰好为 1，垃圾路径不会暴露** →
> 已测 PASS 的算子也要复查这些位置，**用真实 shape 复测**。

### 坑 8：fp32 字面量被判成 f64
`-3.0e38` / `1e-30` 这类**不能精确表示为 fp32** 的字面量会被 triton 判成 f64 →
编译报 `hfusion.isnan ... tensor<1xf64>`。
**规避**：改成**运行时标量参数**传入。

### 坑 9：Ascend UB 只有 192 KB
大 block（如 `[4,1024]` fp32 链式运算）会 `ub overflow`。
**规避**：分块 + 多趟循环（实测 rms 用 `BLK_D=128`、3 趟才通过）。

### ✅ 实测可用的操作
`tl.cumsum`、2-D 归约、`tl.sort`、gather、scatter、`tl.make_block_ptr`、`tl.dot`。

### 5.2 测试文件（`test_<op>.py`）

照 `test_fused_gdn_gating.py` 的结构：
- 多个 shape 用例（含**非整除**的边界，如 `H=17`）
- 多种 dtype（bf16 / fp16，视算子支持）
- fp64 参考对拍
- 每个用例打印 `PASS/FAIL` + 误差 + 地板
- 结尾打印总体结论
- **结果同时写到 `/root/dsh_work/<op>_test.txt`**

---

## 六、交付与汇报

完成后向 Lead 汇报：
1. 算子名 + 文件路径
2. **AscendC 行数 → Triton 行数**
3. **精度实测数据表**（每个用例的误差 + 判定）
4. 遇到的问题 / 不确定的语义假设

**不要**在没跑通精度的情况下报"完成"。
