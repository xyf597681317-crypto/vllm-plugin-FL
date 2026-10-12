# -*- coding: utf-8 -*-
"""隔离测试 AscendC `_C_ascend` 的装载路径（每步一个进程，崩溃隔离）。

用法（容器内）：DSH_NPU=npu:14 python3 -W ignore _probe_extra_so2.py <step>
  step ∈ {pkg, sub, lib, lib107}
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"
PROBE = ("npu_hc_post", "npu_hc_pre_inv_rms", "npu_causal_conv1d_custom",
         "npu_recurrent_gated_delta_rule")


def chk(tag):
    out = []
    for n in PROBE:
        try:
            out.append(f"{n}={hasattr(torch.ops._C_ascend, n)}")
        except Exception as ex:  # noqa: BLE001
            out.append(f"{n}=EXC({type(ex).__name__})")
    print(f"  [{tag}] " + "  ".join(out), flush=True)


step = sys.argv[1] if len(sys.argv) > 1 else "pkg"
print(f"[so2] dev={DEV} step={step}", flush=True)
if step == "pkg":
    print("  import vllm_fl ...", flush=True)
    import vllm_fl  # noqa: F401
    chk("after import vllm_fl")
elif step == "sub":
    print("  import vllm_fl._C_ascend ...", flush=True)
    import vllm_fl._C_ascend  # noqa: F401
    chk("after import vllm_fl._C_ascend")
elif step in ("lib", "lib107"):
    p = (SP + "/vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so" if step == "lib"
         else SP + "/vllm_fl.copied107/_C_ascend.cpython-312-aarch64-linux-gnu.so")
    print(f"  load_library {p}", flush=True)
    torch.ops.load_library(p)
    chk(f"after load {step}")
else:
    raise SystemExit(f"unknown step {step}")
print(f"[so2] {step} DONE", flush=True)
