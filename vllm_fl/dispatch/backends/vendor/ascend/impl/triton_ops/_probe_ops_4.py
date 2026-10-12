# -*- coding: utf-8 -*-
"""_probe_ops_4.py —— 不 import vllm，直接加载 extra/bridge 模块，看注册表里有什么。

回答：
  1. `npu_hc_pre_v2` 到底有没有被接管（在 EXTRA_OPS / schemas 里吗）？
  2. `_adapter_npu_hc_pre` 的入口确实把 (norm_eps, hc_eps) 反过来了吗？（用实现直调对比）
  3. 顺带确认三个孤儿算子在 extras/bridge 注册表里是否存在。

跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_4.py
"""
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
DEV = f"npu:{IDX}"
SRC = "/workspace/FlagGems/src/triton_ops"
sys.path.insert(0, "/workspace/FlagGems/src")
print("device =", IDX, flush=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


extra = load(os.path.join(SRC, "_C_ascend_extra.py"), "kx_extra")
print("\n== EXTRA_OPS 条目 ==", flush=True)
for k in extra.EXTRA_OPS:
    print("   ", k, flush=True)
schemas, adapters = extra.build()
print(f"\n== build() -> schemas={len(schemas)} adapters={len(adapters)} ==", flush=True)
print("   schemas:", sorted(schemas), flush=True)

print("\n== 关心 ==", flush=True)
for p in ("npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_inv_rms", "npu_hc_pre_sinkhorn",
          "npu_add_rms_norm_bias", "npu_gemma_rms_norm", "npu_swiglu_group_quant"):
    print(f"   {p:34s} in_EXTRA_OPS={p in extra.EXTRA_OPS}  in_build={p in schemas}",
          flush=True)

# ---- 验证 hc_pre 的 (norm_eps, hc_eps) 顺序差异 ----
print("\n== _adapter_npu_hc_pre vs 实现直调 ==", flush=True)
hc_pre = getattr(load(os.path.join(SRC, "hc_pre.py"), "kx_hc_pre"), "hc_pre")
ad = extra._adapter_npu_hc_pre
torch.manual_seed(0)
rows, HM, D = 6, 2, 8
x = torch.randn(rows, D, device=DEV, dtype=torch.float32)
fn = torch.randn(HM * D, device=DEV, dtype=torch.float32)
sc = torch.randn(HM, device=DEV, dtype=torch.float32)
bs = torch.randn(HM, device=DEV, dtype=torch.float32)
NORM_EPS, HC_EPS = 1e-5, 1e-3           # 故意取两个不同的值，便于分辨
try:
    a = ad(x, fn, sc, bs, HM, 3, NORM_EPS, HC_EPS)          # 走 adapter（注册入口）
    b = hc_pre(x, fn, sc, bs, HM, 3, norm_eps=NORM_EPS, hc_eps=HC_EPS)   # 顺序正确的直调
    c = hc_pre(x, fn, sc, bs, HM, 3, norm_eps=HC_EPS, hc_eps=NORM_EPS)   # 顺序反的直调
    def dmax(u, v):
        if isinstance(u, (tuple, list)):
            return max(float((ui.float() - vi.float()).abs().max()) for ui, vi in zip(u, v))
        return float((u.float() - v.float()).abs().max())
    print(f"  adapter vs 直调(正确顺序) max|d| = {dmax(a, b):.6e}", flush=True)
    print(f"  adapter vs 直调(反向顺序) max|d| = {dmax(a, c):.6e}", flush=True)
    print("  => adapter 用的是反向顺序" if dmax(a, c) < dmax(a, b) else "  => adapter 顺序正确",
          flush=True)
except Exception as e:  # noqa: BLE001
    import traceback
    print("  失败:", type(e).__name__, str(e)[:200], flush=True)
    traceback.print_exc()
print("PROBEOPS4DONE", flush=True)
