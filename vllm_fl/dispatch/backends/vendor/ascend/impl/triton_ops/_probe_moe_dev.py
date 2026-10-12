# -*- coding: utf-8 -*-
"""最小设备健全性检查：纯 copy kernel + 本算子，逐个 device 试。

用法（容器内）：python3 -W ignore _probe_moe_dev.py [dev1 dev2 ...]
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@triton.jit
def _copy_kernel(X, Y, N, BLK: tl.constexpr):
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < N
    v = tl.load(X + o, mask=m, other=0.0)
    tl.store(Y + o, v * 2.0, mask=m)


def main():
    print("ASCEND_RT_VISIBLE_DEVICES =", repr(os.environ.get("ASCEND_RT_VISIBLE_DEVICES")), flush=True)
    print("DSH_NPU =", repr(os.environ.get("DSH_NPU")), flush=True)
    try:
        print("device_count =", torch.npu.device_count(), flush=True)
        for i in range(torch.npu.device_count()):
            print("  ", i, torch.npu.get_device_name(i), flush=True)
    except Exception as ex:  # noqa: BLE001
        print("device_count EXC:", type(ex).__name__, str(ex)[:200], flush=True)

    devs = sys.argv[1:] or ["npu:1", "npu:2", "npu:3"]
    for dev in devs:
        try:
            x = torch.arange(1024, dtype=torch.float32, device=dev)
            y = torch.empty_like(x)
            _copy_kernel[(4,)](x, y, 1024, BLK=256)
            torch.npu.synchronize()
            ok = (y[0].item() == 0.0) and (y[1023].item() == 2046.0)
            print(f"  [{dev}] copy kernel OK ({y[1023].item()}) {'PASS' if ok else 'FAIL'}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{dev}] EXC {type(ex).__name__}: {str(ex)[:220]}", flush=True)
        try:
            from triton_ops.moe_init_routing_custom import (DYNAMIC_QUANT, GATHER,
                                                            moe_init_routing_custom)
            torch.manual_seed(0)
            x = torch.randn(8, 33, dtype=torch.float32, device=dev).to(torch.float16)
            ei = torch.randint(0, 4, (8, 2), dtype=torch.int32, device=dev)
            ex_, ridx, cnt, esc = moe_init_routing_custom(
                x, ei, None, None, expert_num=4, quant_mode=DYNAMIC_QUANT,
                expert_tokens_num_flag=True, row_idx_type=GATHER)
            torch.npu.synchronize()
            print(f"  [{dev}] moe op OK (esc[0]={esc[0].item():.6g})", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{dev}] moe EXC {type(ex).__name__}: {str(ex)[:220]}", flush=True)


if __name__ == "__main__":
    main()
