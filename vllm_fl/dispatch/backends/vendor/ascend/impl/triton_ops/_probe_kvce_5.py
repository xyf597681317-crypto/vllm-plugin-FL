# -*- coding: utf-8 -*-
"""_probe_kvce_5.py —— 判定"独立运行必崩" vs "需要预热/设备相关"。

把失败用例跑三次：
  W0: 完全不预热（第一次就调算子）
  W1: 先做 20 次简单 NPU 运算 + 一次小张量 matmul 预热
  W2: 再跑一次（同进程第二次调用，排除首次编译因素）
每一步都打印，便于看到崩在第几行。
跑法：DSH_NPU=npu:3 python3 -W ignore triton_ops/_probe_kvce_5.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:3").split(":")[-1]
print("DEV =", DEV, flush=True)
print("torch_npu:", torch_npu.__version__, flush=True)


def geom2(d, group):
    qc = d - 64
    sc = (qc + group - 1) // group
    cc = qc + 128 + sc
    return qc, sc, cc, ((cc + 127) // 128) * 128


def case(tag, rows=9, d=576, group=64, rscale=0, seed=9):
    qc, sc, cc, kc = geom2(d, group)
    row_stride = kc
    nslots = max(20, rows + 2)
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    perm = torch.randperm(nslots)[:rows].tolist()
    slots = [int(t) for t in perm]
    slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(nslots * row_stride, dtype=torch.uint8, device=DEV)
    print(f"[{tag}] begin slots={slots}", flush=True)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    print(f"[{tag}] launched, before sync", flush=True)
    torch.npu.synchronize()
    print(f"[{tag}] OK nz={int((buf.detach().cpu() != 0).sum())}", flush=True)


print("--- 预热 ---", flush=True)
for i in range(20):
    a = torch.ones(256, device=DEV) * (i + 1)
    _ = float(a.sum())
print("预热(简单运算)完成", flush=True)
m1 = torch.randn(64, 64, device=DEV)
m2 = torch.randn(64, 64, device=DEV)
_ = (m1 @ m2).sum().item()
torch.npu.synchronize()
print("预热(matmul)完成", flush=True)

print("--- W1: 预热后第一次调算子 ---", flush=True)
case("W1")
print("--- W2: 同进程第二次 ---", flush=True)
case("W2")
print("KVCE5DONE", flush=True)
