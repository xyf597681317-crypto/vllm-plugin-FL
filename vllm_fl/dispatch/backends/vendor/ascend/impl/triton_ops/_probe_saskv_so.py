# -*- coding: utf-8 -*-
"""查 _C_ascend.so 里到底注册/导出了哪些算子（AscendC 裁判可用性）。"""
import os
import subprocess
import torch
import torch_npu  # noqa: F401

print("=== 1) find _C_ascend / libtorch_binding so ===", flush=True)
r = subprocess.run(["bash", "-lc",
                    "python3 -c \"import torch;print(torch.__file__)\"; "
                    "find / -xdev -name '_C_ascend*' 2>/dev/null | head -40; "
                    "find / -xdev -name '*torch_binding*' 2>/dev/null | head -20"],
                   capture_output=True, text=True, timeout=600)
print(r.stdout, r.stderr[:2000], flush=True)

print("=== 2) all ops in _C_ascend (dir) ===", flush=True)
print(sorted([n for n in dir(torch.ops._C_ascend) if not n.startswith('_')])[:200], flush=True)

print("=== 3) all _C_* namespaces ===", flush=True)
import torch.utils.cpp_extension  # noqa: F401
try:
    import torch._C
    ns = [x for x in dir(torch.ops) if not x.startswith('__')]
    print(ns, flush=True)
except Exception as e:
    print("ERR", e, flush=True)

print("=== 4) which .so loaded ===", flush=True)
r = subprocess.run(["bash", "-lc", "cat /proc/self/maps | grep -oE '/[^ ]+\\.so[^ ]*' | sort -u | head -60"],
                   capture_output=True, text=True)
print(r.stdout, flush=True)
print("SOFILE_DONE", flush=True)
