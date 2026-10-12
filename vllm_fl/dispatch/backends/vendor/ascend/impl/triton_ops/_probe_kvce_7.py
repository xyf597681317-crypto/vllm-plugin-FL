# -*- coding: utf-8 -*-
"""_probe_kvce_7.py —— 用测试文件自己的 run_case 跑前 N 个用例（证明它在我的进程里也能跑），
然后跑失败用例并 dump 细节。

这是"测试 vs 我的探针"矛盾的判定实验：
  * 若前面用例都 PASS 而失败用例也复现 → 我的 harness 不是原因，矛盾消失；
  * 若第一个用例就崩 → 说明 test 文件跑起来和我跑起来确实不同（再查环境）。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_7.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

SRC = "/workspace/FlagGems/src/triton_ops/test_kv_compress_epilog.py"
sys.path.insert(0, "/workspace/FlagGems/src")
src = open(SRC, encoding="utf-8").read()
cut = src.index('emit("=" * 96)')
head = src[:cut].replace(
    'DEV = "npu:0"',
    'DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]')
ns = {"__name__": "kvce_ref", "__file__": SRC}
exec(compile(head, SRC, "exec"), ns)  # noqa: S102

run_case = ns["run_case"]
print("DEV =", ns["DEV"], flush=True)

print("=== step1: 用测试自己的 run_case 跑第 1 个用例 ===", flush=True)
r1 = run_case("d=640 rows=16", 16, 640)
print("step1 结果:", r1, flush=True)

print("=== step2: 跑失败用例 ===", flush=True)
r2 = run_case("d=576 group=64 rscale=0 rows=9 含-1", 9, 576, qmode=2, group=64, rscale=0)
print("step2 结果:", r2, flush=True)
print("KVCE7DONE", flush=True)
