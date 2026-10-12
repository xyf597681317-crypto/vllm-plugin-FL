# -*- coding: utf-8 -*-
"""_coverage_j3.py —— J3 判据示例：**dequant_swiglu_quant**（D1 的修复验证）

判据主体在 `_coverage_sweep.py::run_j3`（canonical，**只此一份**）。
本文件只提供"被测算子的输入构造 + 主机 fp64 采样参考"，避免两份实现漂移。

## ⚠️ 调用约定（改前必读，我在这里反复踩过）
`run_j3(name, fn, build_inputs, ref_rows, n_rows, ...)` 内部：
    ins = build_inputs()          # **参数元组**
    out = fn(*ins)                # 展开
    ref = ref_rows(*ins, rows=rows[, want_scale=True])   # 同样展开
⇒ `build_inputs()` 必须返回**参数元组本身**（不是 `(tuple,)`），
  `fn` 与 `ref_rows` 必须收**同样多的位置参数**（这里都是 4 个：x, ws, a, q）。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_coverage_j3.py
"""
import importlib.util
import os
import sys
import traceback

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX

_spec = importlib.util.spec_from_file_location(
    "cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)
run_j3 = cov.run_j3
probe_rows = cov.probe_rows
J3 = []


# =========================================================================
# 用例：dequant_swiglu_quant（D1 的修复验证）
# =========================================================================
def build_dsq(M, H, qs, ascale, seed=9):
    """返回**参数元组** (x, ws, a, q)。"""
    torch.manual_seed(seed)
    x = torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8
    ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
    a = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25) if ascale else None
    q = (torch.rand(H, dtype=torch.float32, device=DEV) + 0.5) if qs else None
    return (x, ws, a, q)


def dsq_fn(x, ws, a, q):
    """位置参数与 `ref_rows_dsq` 逐个对应（4 个）。"""
    from triton_ops.dequant_swiglu_quant import dequant_swiglu_quant
    y, sc = dequant_swiglu_quant(x, ws, a, None, q, None, None, True, "dynamic",
                                 torch.int8, "rint", -1, 0, None, None, None)
    return y, sc


def ref_rows_dsq(x, ws, a, q, rows=None, want_scale=False):
    """只算**采样行**的主机 fp64 独立复算（与 AscendC 语义一致）。"""
    xd = x.detach().cpu().double()[rows]
    wsd = ws.detach().cpu().double()
    H = wsd.numel() // 2
    xf = xd * wsd[None, :]
    if a is not None:
        xf = xf * a.detach().cpu().double()[rows][:, None]
    act, gate = xf[:, :H], xf[:, H:]
    y = gate * (act / (1.0 + torch.exp(-act)))
    if q is not None:
        y = y * q.detach().cpu().double()[None, :]
    mx = y.abs().amax(dim=1)
    sc = mx / 127.0
    if want_scale:
        return sc
    return torch.round(y / sc[:, None])


def sweep_dsq_j3(H=512):
    print("\n===== J3: dequant_swiglu_quant（含 D1 修复验证 M=65535/65536/65537）=====",
          flush=True)
    for M in [48, 512, 4096, 65535, 65536, 65537]:
        run_j3(f"dsq M={M} H={H} qs=T", dsq_fn,
               lambda M=M, H=H: build_dsq(M, H, True, True),
               ref_rows_dsq, n_rows=M, out_index=0, scale_index=1,
               tol=1, exact_min=0.999, n_sample=8, results=J3)
    for M in [4096, 65536]:
        run_j3(f"dsq M={M} H={H} qs=F ascale=F", dsq_fn,
               lambda M=M, H=H: build_dsq(M, H, False, False),
               ref_rows_dsq, n_rows=M, out_index=0, scale_index=1,
               tol=1, exact_min=0.999, n_sample=8, results=J3)


if __name__ == "__main__":
    print(f"J3 framework | DEV={DEV} current={torch.npu.current_device()}", flush=True)
    which = os.environ.get("J3_ONLY", "dsq")
    if which in ("all", "dsq"):
        try:
            sweep_dsq_j3()
        except Exception:
            print("GROUP FAILED:\n" + traceback.format_exc()[-800:], flush=True)
    print("\n===== J3 汇总 =====", flush=True)
    for r in J3:
        print(f"  {r['status']:8s} {r['name']:46s} " +
              (f"max|Δ|={r.get('max_d')} exact={r.get('exact')} 边界异常={r.get('boundary_bad')}"
               if r["status"] in ("PASS", "FAIL") else r.get("detail", "")), flush=True)
    nbad = sum(1 for r in J3 if r["status"] not in ("PASS",))
    print(f"J3 TOTAL={len(J3)} NOT_PASS={nbad}", flush=True)
    print("J3DONE", flush=True)
