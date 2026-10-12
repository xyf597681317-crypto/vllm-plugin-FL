# -*- coding: utf-8 -*-
"""_probe_kvce_4.py —— 隔离"独立运行就崩"这件事。

顺序跑 5 个变体，每个用**全新分配的 cache**，任何一步崩就停在那一行：
  A. 全零 x，slot=0，1 行
  B. 随机 x，slot=0，1 行
  C. 随机 x，9 行全有效 slot（无 -1）
  D. 随机 x，9 行含 -1（失败用例的形状，但 group=64 rscale=1）
  E. 失败用例原样（rscale=0）
另：把 kv_compress_epilog.py 的 md5 打出来，确认容器里是哪份。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_4.py
"""
import hashlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
import triton_ops.kv_compress_epilog as M  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]
print("DEV =", DEV, flush=True)
print("module:", M.__file__, flush=True)
print("module md5:", hashlib.md5(open(M.__file__, "rb").read()).hexdigest(), flush=True)
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402


def geom2(d, group):
    qc = d - 64
    sc = (qc + group - 1) // group
    cc = qc + 128 + sc
    return qc, sc, cc, ((cc + 127) // 128) * 128


def call(tag, rows, d, group, rscale, seed=9, with_neg=True, zeros=False):
    qc, sc, cc, kc = geom2(d, group)
    row_stride = kc
    nslots = max(20, rows + 2)
    torch.manual_seed(seed)
    if zeros:
        x = torch.zeros(rows, d, dtype=torch.bfloat16, device=DEV)
    else:
        x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    slots = list(range(rows))
    if with_neg and rows >= 5:
        slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(nslots * row_stride, dtype=torch.uint8, device=DEV)
    print(f"[{tag}] 调用前 slots={slots}", flush=True)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    torch.npu.synchronize()
    print(f"[{tag}] OK 非零字节={int((buf.detach().cpu() != 0).sum())}", flush=True)


if __name__ == "__main__":
    call("A 全零 1行 slot0", 1, 576, 64, 1, zeros=True, with_neg=False)
    call("B 随机 1行 slot0", 1, 576, 64, 1, with_neg=False)
    call("C 9行 无-1 rscale=1", 9, 576, 64, 1, with_neg=False)
    call("D 9行 含-1 rscale=1", 9, 576, 64, 1, with_neg=True)
    call("E 失败用例原样 rscale=0", 9, 576, 64, 0, with_neg=True)
    print("KVCE4DONE", flush=True)
