"""第四轮：真实算子 fp32 路径 —— 崩溃是否可重试恢复（+ 机器负载相关性）。"""
import os
import subprocess
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/tops")
from triton_ops.recurrent_gated_delta_rule import recurrent_gated_delta_rule  # noqa: E402

dev = "npu:0"
T, B, NK, NV, DK, DV = 32, 2, 2, 2, 32, 32


def load_info(tag):
    try:
        la = os.getloadavg()
    except Exception:  # noqa: BLE001
        la = (0, 0, 0)
    mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()
    avail = mem[1].split()[6] if len(mem) > 1 else "?"
    print(f"  [{tag}] load1/5/15={la[0]:.1f}/{la[1]:.1f}/{la[2]:.1f}  MemAvailable={avail}MB", flush=True)


def mk():
    torch.manual_seed(0)

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=dev) * 2 - 1) * lo

    q = U(T, NK, DK).to(torch.bfloat16)
    k = U(T, NK, DK).to(torch.bfloat16)
    v = U(T, NV, DV).to(torch.bfloat16)
    b = U(T, NV).to(torch.bfloat16)
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=dev)
    ssm = torch.arange(T, dtype=torch.int32, device=dev) * NV
    st = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.float32)
    g = U(T, NV, lo=0.1) - 0.05
    return q, k, v, b, st, cu, ssm, g


print("=" * 88, flush=True)
print("真实算子 recurrent_gated_delta_rule：state=fp32 连续尝试 4 次", flush=True)
print("=" * 88, flush=True)
load_info("start")
args = mk()
succ = 0
for i in range(4):
    load_info(f"try{i}")
    t0 = time.time()
    try:
        o, f = recurrent_gated_delta_rule(*args)
        torch.npu.synchronize()
        succ += 1
        print(f"  try{i}: OK  ({time.time()-t0:.1f}s) out.dtype={o.dtype} final.dtype={f.dtype} "
              f"|final|max={float(f.abs().max()):.4f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  try{i}: {type(e).__name__} ({time.time()-t0:.1f}s) "
              f"{str(e).replace(chr(10),' ')[:70]}", flush=True)
print(f"  => {succ}/4 成功", flush=True)
print("done", flush=True)
