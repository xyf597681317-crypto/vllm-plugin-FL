# -*- coding: utf-8 -*-
"""_probe_kvce_16.py —— 用一个"受控行"走真实算子，复现 q=11.5 的平局。

不依赖随机数：手工构造一行，使某个组的 gmax 固定、且组内某元素恰好落在平局点。
- 取 group=0（列 0..63），gmax = 4.0（放一个 4.0 进去）
- 组内某列放 x，使 |y| = 448*|x|/4 = 112*|x| 在 [256,512) 区间且 q 为平局
  要 q = 11.5：|y| ∈ [256,512)，ex=8 ⇒ q = |y| * 2^-5 = |y|/32 ⇒ |y| = 368
  ⇒ |x| = 368/112 = 3.28571428... 不是 bf16 精确值。
  改用"让 gmax = 3.285714…"不可行，所以反过来：直接选一个 bf16 的 x，
  算出它的 y，检查 q 的 frac；只要不是平局就继续扫，直到找到平局点。

更直接：**扫一整行 128 个 bf16 值**（组 0 与组 1 各 64 个），
逐个打印 kernel 字节 vs torch 字节，并与手工 RNE 对照，定位第一处不一致。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_16.py
"""
import os
import struct
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

ROPE, D, GROUP = 64, 192, 64
QC = D - ROPE
KC = 384


def f32(v):
    return struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]


def main():
    print("device =", DEV, flush=True)
    kc = KC
    # 组 0：gmax 固定 4.0；组 1：gmax 固定 4.8125（与失败用例同）
    bits = torch.arange(0, QC, dtype=torch.int64)
    # 线性铺一张 bf16 网格：从 0.5 到 4.8
    vals = (0.5 + (torch.arange(QC, dtype=torch.float32) / QC) * 4.3)
    row = vals.clone()
    row[0] = 4.0            # 组 0 的 gmax
    row[GROUP] = 4.8125     # 组 1 的 gmax
    x = torch.zeros(1, D, dtype=torch.bfloat16)
    x[0, :QC] = row.to(torch.bfloat16)
    x = x.to(DEV)
    sm = torch.tensor([0], dtype=torch.int32, device=DEV)
    for rscale in (0, 1):
        buf = torch.zeros(2 * kc, dtype=torch.uint8, device=DEV)
        kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                           quant_group_size=GROUP, round_scale=rscale, block_stride=kc)
        torch.npu.synchronize()
        got = buf.detach().cpu()
        xc = x.detach().cpu()[0].float()
        nbad = 0
        print(f"===== rscale={rscale} =====", flush=True)
        for g in range(QC // GROUP):
            lo, hi = g * GROUP, g * GROUP + GROUP
            seg = xc[lo:hi]
            gmax = float(seg.abs().max())
            sf = max(gmax, 1e-4) * (1.0 / 448.0)
            bits_sf = struct.unpack("<I", struct.pack("<f", sf))[0]
            e0 = (bits_sf >> 23) & 0xFF
            eb = (e0 + (1 if (bits_sf & 0x7FFFFF) else 0)) if rscale else e0
            scale = (f32(eb << 23) if rscale else float(sf))
            y = (seg / scale).clamp(-448.0, 448.0)
            ref = y.to(torch.float8_e4m3fn).view(torch.uint8)
            for j in range(GROUP):
                off = 128 + lo + j
                gv, rv = int(got[off]), int(ref[j])
                if gv != rv:
                    nbad += 1
                    if nbad <= 8:
                        yy = float(y[j])
                        ab = abs(yy)
                        ex = ((struct.unpack("<I", struct.pack("<f", ab))[0] >> 23)
                              & 0xFF) - 127
                        q = ab * (2.0 ** (3 - ex))
                        print(f"  BAD g={g} j={j} x={float(seg[j]):.9e} scale={scale:.9e}"
                              f" y={yy:.9e} q={q:.9e} frac={q - int(q):.9e}"
                              f" kernel={gv}(0x{gv:02X}) torch={rv}(0x{rv:02X})"
                              f" scaleByte={int(got[128 + QC + g])}/{eb & 0xFF}", flush=True)
        print(f"  rscale={rscale} 不一致 {nbad}/{QC}", flush=True)
    print("KVCE16DONE", flush=True)


if __name__ == "__main__":
    main()
