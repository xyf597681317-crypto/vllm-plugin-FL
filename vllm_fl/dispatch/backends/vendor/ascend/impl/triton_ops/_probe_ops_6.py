# -*- coding: utf-8 -*-
"""_probe_ops_6.py —— 找 torch_npu 下**真能生效**的注册写法。

_probe_ops_5 已证明：`torch.library.define` + `Library(LIB,"IMPL").impl(op, fn, "PrivateUse1")`
在这套 torch_npu 上会被 `VariableFallbackKernel.cpp:250 npu_cpu_fallback` 截获，
张量被搬到 CPU 后才进 adapter（adapter 收到 device=cpu）→ 必然失败。
本探针试几种替代写法，找出能真正走 NPU 的那一种。
跑法：DSH_NPU=npu:0 python3 -W ignore triton_ops/_probe_ops_6.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:0").split(":")[-1])
torch.npu.set_device(IDX)
DEV = f"npu:{IDX}"
print("torch:", torch.__version__, flush=True)

SCHEMA = ("(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, "
          "int hc_sinkhorn_iters, float norm_eps, float hc_eps) "
          "-> (Tensor out0, Tensor out1, Tensor out2)")

seen = []


def _probe_fn(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    seen.append(("called", str(getattr(x, "device", None)), str(getattr(x, "dtype", None))))
    return x, x, x


def mk():
    hm, d, bs = 4, 16, 8
    hmix = 2 * hm + hm * hm
    torch.manual_seed(0)
    x = torch.randn(bs, hm, d, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    return (x, torch.randn(hmix, hm * d, device=DEV, dtype=torch.float32),
            torch.randn(3, device=DEV, dtype=torch.float32),
            torch.randn(hmix, device=DEV, dtype=torch.float32), hm, 3, 1e-5, 1e-2)


def attempt(tag, fn):
    seen.clear()
    try:
        out = fn()
        print(f"  [OK ] {tag}: called={len(seen)} {seen[:1]}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {tag}: {type(e).__name__}: {str(e)[:110]}"
              f"  called={len(seen)} {seen[:1]}", flush=True)
        return False


# A) torch.library.impl 装饰器 + PrivateUse1
LIB_A = "_C_ascend_pa"
try:
    torch.library.define(f"{LIB_A}::op", SCHEMA)

    @torch.library.impl(f"{LIB_A}::op", "PrivateUse1")
    def _a(*args, **kwargs):
        return _probe_fn(*args, **kwargs)
    print("A 注册完成", flush=True)
    attempt("A torch.library.impl + PrivateUse1",
            lambda: torch.ops._C_ascend_pa.op(*mk()))
except Exception as e:  # noqa: BLE001
    print("A 注册失败:", type(e).__name__, str(e)[:140], flush=True)

# B) torch.library.impl 装饰器 + "npu"（若 torch_npu 注册了该 key）
LIB_B = "_C_ascend_pb"
try:
    torch.library.define(f"{LIB_B}::op", SCHEMA)
    for k in ("npu", "NPU", "PrivateUse1"):
        try:
            torch.library.impl(f"{LIB_B}::op", k)(_probe_fn)
            print(f"B 注册 key={k} 成功", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"B 注册 key={k} 失败: {type(e).__name__}: {str(e)[:80]}", flush=True)
    attempt("B torch.library.impl(multi-key)", lambda: torch.ops._C_ascend_pb.op(*mk()))
except Exception as e:  # noqa: BLE001
    print("B 注册失败:", type(e).__name__, str(e)[:140], flush=True)

# C) torch.library.custom_op（torch 2.4+ 推荐的自定义算子写法）
LIB_C = "_C_ascend_pc"
try:
    @torch.library.custom_op(f"{LIB_C}::op", mutates_args=())
    def _c(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
           hc_base: torch.Tensor, hc_mult: int, hc_sinkhorn_iters: int,
           norm_eps: float, hc_eps: float) -> list[torch.Tensor]:
        seen.append(("called", str(x.device), str(x.dtype)))
        return [x, x, x]

    @_c.register_fake
    def _c_fake(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
        return [torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)]
    print("C 注册完成", flush=True)
    attempt("C torch.library.custom_op", lambda: torch.ops._C_ascend_pc.op(*mk()))
except Exception as e:  # noqa: BLE001
    print("C 注册失败:", type(e).__name__, str(e)[:140], flush=True)

print("\n== 可用 dispatch key 里含 PrivateUse1 的注册情况 ==", flush=True)
for lib in ("_C_ascend_pa", "_C_ascend_pb", "_C_ascend_pc"):
    try:
        ok = torch._C._dispatch_has_kernel_for_dispatch_key(f"{lib}::op", "PrivateUse1")
        print(f"   {lib}: PrivateUse1 kernel = {ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"   {lib}: {type(e).__name__}", flush=True)
print("PROBEOPS6DONE", flush=True)
