# -*- coding: utf-8 -*-
"""深度环境探针：_C_ascend 里到底有没有 sparse_attn_sharedkv；设备占用分布。"""
import os
import subprocess
import torch
import torch_npu  # noqa: F401

print("=== devices ===", flush=True)
busy = []
for d in range(16):
    try:
        free, total = torch.npu.mem_get_info(d)
        used = total - free
        line = f"  npu:{d:2d} free={free/2**30:6.1f}GiB used={used/2**30:6.1f}GiB"
        print(line, flush=True)
        if free < 20 * 2**30:
            busy.append(d)
    except Exception as e:
        print(f"  npu:{d:2d} ERR {type(e).__name__}: {e}", flush=True)
print("busy(<20GiB free):", busy, flush=True)

print("=== _C_ascend namespace ===", flush=True)
names = [n for n in dir(torch.ops._C_ascend) if "sparse" in n or "attn" in n]
print("  sparse/attn ops:", names, flush=True)

print("=== try loading vllm_fl adapters ===", flush=True)
try:
    import vllm_fl  # noqa: F401
    print("  imported vllm_fl", flush=True)
except Exception as e:
    print("  vllm_fl import ERR", type(e).__name__, e, flush=True)
try:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_ops import _C_ascend_ops_b as ob
    print("  got _C_ascend_ops_b, has sparse_attn_sharedkv:", hasattr(ob, "sparse_attn_sharedkv"), flush=True)
except Exception as e:
    print("  _C_ascend_ops_b ERR", type(e).__name__, e, flush=True)

print("=== grep so/binding for symbol ===", flush=True)
for pat in ["npu_sparse_attn_sharedkv"]:
    r = subprocess.run(["bash", "-lc",
                        "grep -rl '%s' /usr/local/python3.12.13/lib/python3.12/site-packages/ 2>/dev/null | head -20" % pat],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    r2 = subprocess.run(["bash", "-lc",
                         "find / -name '*.so' -path '*fl*' 2>/dev/null | head -20"], capture_output=True, text=True)
    print("so files:\n", r2.stdout, flush=True)
print("DEEPDONE", flush=True)
