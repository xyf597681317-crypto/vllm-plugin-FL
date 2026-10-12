# -*- coding: utf-8 -*-
"""_probe_kvce_6.py —— 在**同一形状**下只改算子配置，二分崩的条件。

每个配置独立进程跑（用 argv 选），避免相互污染：
    python3 triton_ops/_probe_kvce_6.py <case>
  case = m2g128 | m2g64 | m1 | m2g64_small | m2g64_r1
跑法示例：DSH_NPU=npu:3 python3 triton_ops/_probe_kvce_6.py m2g128
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:3").split(":")[-1]


def geom(d, group, qmode):
    qc = d - 64
    sc = (qc + group - 1) // group
    cc = qc + 128 + (sc * 4 if qmode == 1 else sc)
    return qc, sc, cc, ((cc + 127) // 128) * 128


CASES = {
    "m2g128": dict(rows=9, d=576, group=128, qmode=2, rscale=1),
    "m2g64": dict(rows=9, d=576, group=64, qmode=2, rscale=1),
    "m1": dict(rows=9, d=576, group=128, qmode=1, rscale=1),
    "m2g64_small": dict(rows=2, d=192, group=64, qmode=2, rscale=1),
    "m2g64_r1": dict(rows=9, d=576, group=64, qmode=2, rscale=1),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "m2g128"
    c = CASES[name]
    rows, d, group, qmode, rscale = c["rows"], c["d"], c["group"], c["qmode"], c["rscale"]
    qc, sc, cc, kc = geom(d, group, qmode)
    row_stride = kc
    nslots = 20
    print(f"CASE={name} DEV={DEV} rows={rows} d={d} group={group} qmode={qmode}"
          f" rscale={rscale} qc={qc} sc={sc} cc={cc} kc={kc}", flush=True)
    torch.manual_seed(9)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    slots = list(range(rows))
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(nslots * row_stride, dtype=torch.uint8, device=DEV)
    print("calling...", flush=True)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=qmode, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    print("launched, syncing...", flush=True)
    torch.npu.synchronize()
    print(f"OK nz={int((buf.detach().cpu() != 0).sum())}", flush=True)
    print("CASE_DONE", flush=True)


if __name__ == "__main__":
    main()
