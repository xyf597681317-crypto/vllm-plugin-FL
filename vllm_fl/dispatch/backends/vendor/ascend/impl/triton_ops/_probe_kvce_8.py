# -*- coding: utf-8 -*-
"""_probe_kvce_8.py —— 同进程内：先按测试流程跑，再直接调算子，看是否与"进程状态"有关。

跑法：DSH_NPU=npu:3 python3 -W ignore triton_ops/_probe_kvce_8.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

SRC = "/workspace/FlagGems/src/triton_ops/test_kv_compress_epilog.py"
sys.path.insert(0, "/workspace/FlagGems/src")
src = open(SRC, encoding="utf-8").read()
head = src[:src.index('emit("=" * 96)')].replace(
    'DEV = "npu:0"',
    'DEV = "npu:" + os.environ.get("DSH_NPU", "npu:3").split(":")[-1]')
ns = {"__name__": "kvce_ref", "__file__": SRC}
exec(compile(head, SRC, "exec"), ns)  # noqa: S102
run_case = ns["run_case"]
DEV = ns["DEV"]
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

print("DEV =", DEV, flush=True)

print("--- P1: run_case 小用例（先让进程进入和测试相同的状态）---", flush=True)
try:
    print("P1 结果:", run_case("p1 d=192 rows=4", 4, 192, num_slots=8), flush=True)
except Exception:
    traceback.print_exc()
    print("P1 崩了", flush=True)

print("--- P2: 直接调算子（同一个进程，紧接着）---", flush=True)
try:
    d, group, rows = 576, 64, 9
    kc = 768
    torch.manual_seed(9)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    slots = [-1, 0, 18, 13, 17, -1, 19, 5, 2]
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(20 * kc, dtype=torch.uint8, device=DEV)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=0, block_stride=kc)
    torch.npu.synchronize()
    print("P2 OK", flush=True)
except Exception:
    traceback.print_exc()
    print("P2 崩了", flush=True)

print("--- P3: 再跑一次 run_case 失败用例 ---", flush=True)
try:
    print("P3 结果:", run_case("p3 失败用例", 9, 576, qmode=2, group=64, rscale=0),
          flush=True)
except Exception:
    traceback.print_exc()
    print("P3 崩了", flush=True)
print("KVCE8DONE", flush=True)
