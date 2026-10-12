# -*- coding: utf-8 -*-
"""_probe_cov_dsq0.py —— 查清 dequant_swiglu_quant 在我构造的参数下输出是否真的全 0。

对照设计（同一份代码、同一台设备，只改一个变量）：
  A) 复刻单测已知 PASS 的用例（M=48, H=512, qs=True ⇒ quant_scale 非空）
  B) 同一形状但 quant_scale=None（我的探针用法）
  C) quant_scale=None 下逐档放大 M/H
  D) 直接检查中间张量 ybuf（saskv 式做法：用 SASKV_* 无用；这里改为直接调两个 kernel 太侵入，
     所以改为「检查 y 与 scale 是否全 0」+「用 numel 级抽样打印」

判据：如果 A 非 0 而 B 全 0 ⇒ 是我构造的输入（quant_scale 缺省）导致的**用法问题**，
      不是算子缺陷；如果 A 也全 0 ⇒ 单测的 PASS 判据本身有问题（更严重，需上报）。
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.dequant_swiglu_quant import dequant_swiglu_quant  # noqa: E402

print(f"DEV={DEV} current={torch.npu.current_device()}", flush=True)


def stats(tag, y, sc):
    tc = y.detach().cpu()
    nz = int((tc != 0).sum().item())
    scc = sc.detach().cpu().float()
    fin = scc[torch.isfinite(scc)]
    print(f"  [{tag}] y.shape={tuple(tc.shape)} dtype={tc.dtype} 非零={nz}/{tc.numel()} "
          f"({100.0 * nz / max(1, tc.numel()):.2f}%)  y样本={[int(v) for v in tc.flatten()[:6].tolist()]}",
          flush=True)
    print(f"        scale: |max|={float(fin.abs().max().item()) if fin.numel() else float('nan'):.4e} "
          f"zero={int((scc == 0).sum().item())}/{scc.numel()} 样本={[round(float(v), 4) for v in scc.flatten()[:6].tolist()]}",
          flush=True)


def build(M, H, qs, ascale=True, gidx=None, seed=9):
    torch.manual_seed(seed)
    x = torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8
    if gidx is None:
        ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        gi = None
    else:
        ws = torch.rand(len(gidx), 2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        gi = torch.tensor(gidx, dtype=torch.int64, device=DEV)
    a = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25) if ascale else None
    q = (torch.rand(H, dtype=torch.float32, device=DEV) + 0.5) if qs else None
    return x, ws, a, q, gi


def run(tag, M, H, qs, ascale=True, gidx=None):
    try:
        x, ws, a, q, gi = build(M, H, qs, ascale, gidx)
        y, sc = dequant_swiglu_quant(x, ws, a, None, q, None, gi, True, "dynamic",
                                     torch.int8, "rint", -1, 0, None, None, None)
        torch.npu.synchronize()
        stats(tag, y, sc)
    except Exception as e:
        print(f"  [{tag}] EXC {type(e).__name__}: {str(e)[:160]}", flush=True)


print("\n===== A) 复刻单测已知 PASS：M=48 H=512 qs=True =====", flush=True)
run("A M=48 H=512 qs=T", 48, 512, True)

print("\n===== B) 同形状、quant_scale=None（我的探针用法）=====", flush=True)
run("B M=48 H=512 qs=F", 48, 512, False)

print("\n===== C) 放大扫描（qs=False）=====", flush=True)
for M, H in [(48, 512), (512, 512), (4096, 512), (4096, 2048), (16384, 512)]:
    run(f"C M={M} H={H}", M, H, False)

print("\n===== C2) 放大扫描（qs=True）=====", flush=True)
for M, H in [(512, 512), (4096, 512), (4096, 2048)]:
    run(f"C2 M={M} H={H}", M, H, True)

print("\n===== D) 单测里的另一种用法：无 activation_scale =====", flush=True)
run("D M=512 H=512 ascale=F", 512, 512, False, ascale=False)

print("DSQ0DONE", flush=True)
