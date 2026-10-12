"""验证 `npu_hc_pre_v2` **注册入口**的语义（锁住 norm_eps / hc_eps 的传参顺序）。

## 为什么要单独测这一条
`test_hc_pre.py` 只直调 `triton_ops.hc_pre`（实现层），**没有覆盖注册入口**。
而注册 schema 与实现层的后两个形参顺序**相反**：

| 层 | 签名末两参 |
|---|---|
| `csrc/ascend/torch_binding.cpp` L1226（schema） | `... int hc_sinkhorn_iters, float norm_eps, float hc_eps` |
| `_analysis/triton_ops/hc_pre.py` L163-164（实现） | `... hc_sinkhorn_iters, hc_eps, norm_eps` |

`_C_ascend_ops_a.py::npu_hc_pre_v2` 是**按关键字**转发的，所以顺序是对的；
但这条"正确性"此前**没有任何测试保护** —— 谁把 adapter 改成按位置转发、
或在 schema 侧调整顺序，都会静默把两个 eps 弄反（数值只轻微偏移，不会崩）。

生产调用点：`vllm_fl/models/deepseek_v4_ascend.py:969-971`，
**全位置参数**：`(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)`。
本测试就是照这个形态调用的。

## 覆盖内容
1. 走**注册入口**（`torch.ops._C_ascend.npu_hc_pre_v2`，位置参数，同生产形态）
   vs **实现直调**（按关键字给正确的 eps），逐元素比；
2. 走 **adapter 直调**（不依赖 torch.ops / vllm 可导入）再比一次；
3. **顺序回归**：故意用"把两个 eps 反过来"的直调做对照，确认本用例**真的能抓到**
   顺序错误（否则测试可能是假绿）。

## 运行
```bash
DSH_NPU=npu:0 python3 -W ignore triton_ops/test_hc_pre_v2_entry.py
```
⚠️ 第 1 项需要 `vllm` 可导入（生产环境满足）；容器里独立跑 `vllm` 会因
"Only one platform plugin can be activated" 失败，此时**该项记为 SKIP**并按
"注册入口 SKIP" 计入总结（**不算 PASS**），第 2、3 项仍会跑。
"""
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path.insert(0, _SRC)
DEV = "npu:" + os.environ.get("DSH_NPU", "npu:0").split(":")[-1]
torch.npu.set_device(int(DEV.split(":")[-1]))

L = []


def emit(s=""):
    L.append(str(s))


# 用**包导入**（hc_pre.py 里有 `from .common import ...`，按文件路径加载会失败）
from triton_ops.hc_pre import hc_pre as hc_pre_impl  # noqa: E402

# ---- adapter：直接从 _C_ascend_ops_a.py 取（它只依赖同目录的 _C_ascend_sigs，不需要 vllm）----
# 容器里集成包不在 FlagGems 源码树内，所以再兜一个部署路径 /workspace/flag_gems_integration
_INTG_CANDS = [
    os.path.normpath(os.path.join(_HERE, "..", "flaggems_integration")),
    "/workspace/flag_gems_integration",
]
_INTG = next((p for p in _INTG_CANDS
              if os.path.exists(os.path.join(p, "_C_ascend_ops_a.py"))), _INTG_CANDS[0])
_A_PATH = os.path.join(_INTG, "_C_ascend_ops_a.py")
adapter = None
ADAPTER_ERR = None
if os.path.exists(_A_PATH):
    try:
        if _INTG not in sys.path:
            sys.path.insert(0, _INTG)
        from _C_ascend_ops_a import npu_hc_pre_v2 as adapter  # noqa: E402
    except Exception as e:  # noqa: BLE001
        ADAPTER_ERR = f"{type(e).__name__}: {str(e)[:140]}"
else:
    ADAPTER_ERR = f"{_A_PATH} 不存在（容器里需部署 flaggems_integration）"

# ---- torch.ops **真实注册入口**（生产形态）----
# 生产走 `vllm_fl/.../patches/patch_triton_ops.py::apply_triton_ops_patch()`，
# 但那条路径会 import vllm，独立进程里必然失败
# （`RuntimeError: Only one platform plugin can be activated, but got: ['fl','ascend']`）。
# 这里改为**复刻它的注册序列**（`_C_ascend_ops_a.py` 头部记录的做法）：
#     torch.library.define("<lib>::<op>", schema); lib.impl(op, adapter, "PrivateUse1")
# schema 用 `_C_ascend_sigs.py` 的**真值**，adapter 用 `_C_ascend_ops_a.py` 的**真函数**，
# 所以验的就是生产那两个对象的组合，而不是我另写的等价物。
REG_OP = None
REG_ERR = None
DISPATCH_KEY = "PrivateUse1"
_TEST_LIB = "_C_ascend"                     # 独立进程里该 namespace 还没被占用
_SIGS_PATH = os.path.join(_INTG, "_C_ascend_sigs.py")


def _register_like_production():
    """用 `torch.library.define` + `torch.library.impl(..., 'PrivateUse1')` 注册。

    ⚠️ **不要**用 `torch.library.Library(lib, 'IMPL').impl(op, fn, 'PrivateUse1')`：
    实测在这套 torch_npu 上，那种写法会被 torch_npu 的
    `VariableFallbackKernel.cpp:250 npu_cpu_fallback` 截获 —— 张量先被搬到 CPU
    才进 adapter（adapter 收到 `device=cpu`），必然失败：
        [W] The operator 'X::y' is not currently supported on the NPU backend
            and will fall back to run on the CPU.
        >>> adapter called, first tensor device = cpu
    `torch.library.impl` 注册的 kernel 能拿到 `npu:0` 张量（已实测，见 `_probe_ops_6.py`）。
    """
    if adapter is None:
        raise RuntimeError(f"adapter 不可用（{ADAPTER_ERR}）")
    if not os.path.exists(_SIGS_PATH):
        raise RuntimeError(f"{_SIGS_PATH} 不存在")
    if _INTG not in sys.path:
        sys.path.insert(0, _INTG)
    from _C_ascend_sigs import _C_ASCEND_SIGS  # noqa: E402
    schema = _C_ASCEND_SIGS["npu_hc_pre_v2"]
    if not isinstance(schema, str):
        schema = schema[0]
    torch.library.define(f"{_TEST_LIB}::npu_hc_pre_v2", schema)
    torch.library.impl(f"{_TEST_LIB}::npu_hc_pre_v2", DISPATCH_KEY)(adapter)
    return torch.ops._C_ascend.npu_hc_pre_v2, schema


try:
    REG_OP, _reg_note = _register_like_production()
except Exception as e:  # noqa: BLE001
    REG_ERR = f"{type(e).__name__}: {str(e)[:160]}"

try:
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_triton_ops import (  # noqa: E402,F401
        apply_triton_ops_patch,
    )
    _PROD_PATH_OK = True
except Exception:  # noqa: BLE001
    _PROD_PATH_OK = False


def _mk(bs, hm, d, seed=0, sinkhorn=3, b=1, s=1):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm
    if b * s == bs and (b, s) != (1, 1):
        x = torch.randn(b, s, hm, d, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.randn(bs, hm, d, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    fn = torch.randn(hmix, hm * d, device=DEV, dtype=torch.float32)
    sc = torch.randn(3, device=DEV, dtype=torch.float32)
    bs_ = torch.randn(hmix, device=DEV, dtype=torch.float32)
    return x, fn, sc, bs_, hm, sinkhorn


def _cmp(a, b, tag, tol=0.0):
    """逐元素比较元组；返回 (ok, 最大绝对差)。tol=0 表示要求完全一致。"""
    if not isinstance(a, (tuple, list)) or not isinstance(b, (tuple, list)):
        raise AssertionError(f"{tag}: 返回值不是 tuple")
    if len(a) != len(b):
        raise AssertionError(f"{tag}: 长度不同 {len(a)} vs {len(b)}")
    worst = 0.0
    for i, (u, v) in enumerate(zip(a, b)):
        if tuple(u.shape) != tuple(v.shape):
            raise AssertionError(f"{tag}: 第{i}项 shape 不同 {tuple(u.shape)} vs {tuple(v.shape)}")
        dd = float((u.float() - v.float()).abs().max())
        worst = max(worst, dd)
    return worst <= tol, worst


# 故意取两个**数值上差异明显**的 eps，这样顺序反了会立刻现形
NORM_EPS, HC_EPS = 1e-5, 1e-2
CASES = [
    ("bs=8 hm=4 d=16 (生产 hc_mult=4)", dict(bs=8, hm=4, d=16)),
    ("bs=4 hm=1 d=32 (hc_mult=1)", dict(bs=4, hm=1, d=32)),
    ("b=2 s=2 hm=2 d=8 4D 输入", dict(bs=4, hm=2, d=8, b=2, s=2)),
]

emit("=" * 100)
emit("npu_hc_pre_v2 注册入口语义（锁 norm_eps / hc_eps 顺序）")
emit(f"  DEV={DEV}; 位置参数形态同 deepseek_v4_ascend.py:969-971;"
     f" 故意取 norm_eps={NORM_EPS} hc_eps={HC_EPS}（差异明显，顺序反了必然现形）")
if REG_OP is None:
    emit(f"  [SKIP] 注册入口不可用（{REG_ERR}）—— 第 1 项记为 SKIP（不计 PASS）")
else:
    emit(f"  注册入口：torch.library.define(真 schema) + "
         f"torch.library.impl('{DISPATCH_KEY}')(真 adapter)")
    emit(f"  生产路径 patch_triton_ops 可导入 = {_PROD_PATH_OK}"
         f"（独立进程里 False 属预期：它 import vllm）")
emit("=" * 100)

ok = True
reg_checked = 0
reg_skipped = 0

for name, kw in CASES:
    x, fn, sc, bs_, hm, sinkhorn = _mk(**kw)
    call_x = x
    # ---- 期望：实现直调，按**关键字**给正确的 eps（hc_eps=HC_EPS, norm_eps=NORM_EPS）
    exp = hc_pre_impl(call_x, fn, sc, bs_, hc_mult=hm, hc_sinkhorn_iters=sinkhorn,
                      hc_eps=HC_EPS, norm_eps=NORM_EPS)
    # ---- 反例：把两个 eps 反过来（用于验证本测试真的能抓到顺序错误）
    wrong = hc_pre_impl(call_x, fn, sc, bs_, hc_mult=hm, hc_sinkhorn_iters=sinkhorn,
                        hc_eps=NORM_EPS, norm_eps=HC_EPS)
    _, d_wrong = _cmp(exp, wrong, f"{name} 反例")
    detectable = d_wrong > 0.0
    emit(f"\n  [{name}]")
    emit(f"    反例可检测性: 顺序反了 max|Δ|={d_wrong:.4e}"
         f"  {'OK(能抓到)' if detectable else '**警告：两个 eps 太接近，测试抓不到**'}")
    if not detectable:
        ok = False
    # ---- 第 2 项：adapter 直调（生产形态：位置参数）
    if adapter is not None:
        got_a = adapter(call_x, fn, sc, bs_, hm, sinkhorn, NORM_EPS, HC_EPS)
        good, dd = _cmp(exp, got_a, f"{name} adapter")
        emit(f"    [2] adapter 直调 vs 实现直调: max|Δ|={dd:.4e}  "
             f"{'PASS' if good else 'FAIL'}（tol=0，要求完全一致）")
        ok &= good
    else:
        emit(f"    [2] adapter 直调: SKIP（{ADAPTER_ERR}）")
        ok = False
    # ---- 第 1 项：torch.ops 注册入口（生产形态）
    if REG_OP is not None:
        got_r = REG_OP(call_x, fn, sc, bs_, hm, sinkhorn, NORM_EPS, HC_EPS)
        good, dd = _cmp(exp, got_r, f"{name} 注册入口")
        emit(f"    [1] torch.ops 注册入口 vs 实现直调: max|Δ|={dd:.4e}  "
             f"{'PASS' if good else 'FAIL'}（tol=0，要求完全一致）")
        ok &= good
        reg_checked += 1
    else:
        emit(f"    [1] torch.ops 注册入口: SKIP（{REG_ERR}）")
        reg_skipped += 1

emit()
emit(f"注册入口实测 {reg_checked} 例 / SKIP {reg_skipped} 例")
if reg_skipped and not reg_checked:
    emit("总体: 注册入口全部 SKIP（未验证 torch.ops 路径）—— 需在可导入 vllm 的环境复跑")
    ok = False
else:
    emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/hc_pre_v2_entry_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
