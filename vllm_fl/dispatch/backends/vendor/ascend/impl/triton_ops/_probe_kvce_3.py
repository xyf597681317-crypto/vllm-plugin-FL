# -*- coding: utf-8 -*-
"""_probe_kvce_3.py —— 用**测试文件自己的函数**跑失败用例，避免我自己搭 harness 引入的差异。

做法：exec 测试文件的源码（但要挡住它顶层的自动执行），只取 run_case/ref_bytes_mxfp8，
然后单独跑失败用例，并把每个不一致字节的详情打出来。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_3.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

SRC = "/workspace/FlagGems/src/triton_ops/test_kv_compress_epilog.py"
sys.path.insert(0, "/workspace/FlagGems/src")

# 只取函数定义：截到第一个 emit(...) 调用之前
src = open(SRC, encoding="utf-8").read()
cut = src.index('emit("=" * 96)')
head = src[:cut]
head = head.replace('DEV = "npu:0"',
                    'DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]')
ns = {"__name__": "kvce_ref", "__file__": SRC}
exec(compile(head, SRC, "exec"), ns)  # noqa: S102

ref_bytes_mxfp8 = ns["ref_bytes_mxfp8"]
geom2 = ns["geom2"]
DEV = ns["DEV"]
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

print("DEV =", DEV, flush=True)


def detailed(rows, d, group, rscale, seed=9, num_slots=20, row_stride_extra=0):
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    _, _, _, kv_col = geom2(d, group)
    row_stride = kv_col + row_stride_extra
    num_slots = max(num_slots, rows + 2)
    perm = torch.randperm(num_slots)[:rows].tolist()
    slots = [int(t) for t in perm]
    if rows >= 5:
        slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    cache_fp8 = buf.view(torch.float8_e4m3fn)
    print(f"  calling op slots={slots} row_stride={row_stride}", flush=True)
    kv_compress_epilog(x, sm, cache_fp8, quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    torch.npu.synchronize()
    print("  op done, synced", flush=True)
    ref = ref_bytes_mxfp8(x, slots, d, row_stride, num_slots, group, bool(rscale))
    got = buf.detach().cpu()
    bad = (got != ref).nonzero().flatten().tolist()
    quant_col = d - 64
    scale_col = (quant_col + group - 1) // group
    print(f"  mismatch={len(bad)}/{ref.numel()}", flush=True)
    for i in bad:
        row_i, col = divmod(i, row_stride)
        slot = slots[row_i] if row_i < len(slots) else -99
        kind = ("rope" if col < 128 else
                "quant" if col < 128 + quant_col else
                "scale" if col < 128 + quant_col + scale_col else "pad")
        extra = ""
        if kind == "quant":
            g = (col - 128) // group
            extra = f" g={g} 组内列={col - 128 - g * group}"
        print(f"    offset={i} buf_row={row_i}(slot={slot}) col={col} {kind}{extra}"
              f"  got={int(got[i])}(0x{int(got[i]):02X}) ref={int(ref[i])}"
              f"(0x{int(ref[i]):02X})", flush=True)
    # 打印相关源数据
    xc = x.detach().cpu()
    for i in bad:
        row_i, col = divmod(i, row_stride)
        if row_i >= len(slots) or slots[row_i] < 0 or col < 128:
            continue
        src_row = [k for k, s in enumerate(slots) if s == slots[row_i]][0]
        if col >= 128 + quant_col:
            continue
        j = col - 128
        print(f"    [源] 行{src_row} 列{j}: bf16={float(xc[src_row, j]):.9e}", flush=True)
    return len(bad)


if __name__ == "__main__":
    print("=== 失败用例 ===", flush=True)
    n = detailed(9, 576, 64, 0)
    print("KVCE3DONE mismatch=", n, flush=True)
