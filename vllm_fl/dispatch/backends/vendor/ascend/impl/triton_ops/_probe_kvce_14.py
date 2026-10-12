# -*- coding: utf-8 -*-
"""_probe_kvce_14.py —— 用**受控输入**走真实算子路径，验证平局取整。

构造：x[0, group0] = [2.0, 0, 0, ...]，于是 gmax = 2.0，
sf = 2.0/448，y = 2.0/sf = 448*?  → 需要精确 q=11.5 的平局点：
   q = |y| * 2^(3-ex)，取 |y| = 2.0 时 ex=1 ⇒ q = 2.0*4 = 8.0（不是平局）
改为让 gmax = 2.0（保证 scale 固定），再看组内各元素：
   y = |x_i| * 448/2.0 = |x_i| * 224
   q = |y| * 2^(3-ex)，|y| ∈ [1,2) 时 ex=0 ⇒ q = 8|y|
   要 q = 11.5 ⇒ |y| = 1.4375 ⇒ |x_i| = 1.4375/224 = 0.006417...（不是 bf16 精确值）
更稳的办法：直接扫描一个 bf16 值域，找 kernel 与 torch 不一致的元素。
这里就做扫描：x[0] 里放各种 bf16 值，gmax 固定为 4.0，逐字节比对。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_14.py
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

ROPE = 64
D, GROUP = 192, 64          # 小形状：quant_col = 128，group=64 → 2 组
QC = D - ROPE               # 128
KC = 384


def main():
    print("device =", DEV, flush=True)
    vals = []
    # 扫描大量 bf16 值，覆盖 [0, 4] 区间，制造各种 ratio
    import random
    random.seed(0)
    for _ in range(100):
        v = struct.unpack("<f", struct.pack("<I", random.getrandbits(23) | 0x3F800000))[0]
        vals.append(round(v, 6))
    # 加入精确的 2 的幂 / 半值
    vals += [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    n = min(len(vals), QC)
    row0 = torch.zeros(D, dtype=torch.float32)
    for i, v in enumerate(vals[:n]):
        row0[i] = v
    row0[QC:QC + ROPE] = 0.0
    x = torch.zeros(1, D, dtype=torch.bfloat16)
    x[0] = row0.to(torch.bfloat16)
    x = x.to(DEV)
    sm = torch.tensor([0], dtype=torch.int32, device=DEV)
    buf = torch.zeros(2 * KC, dtype=torch.uint8, device=DEV)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=GROUP, round_scale=1, block_stride=KC)
    torch.npu.synchronize()
    got = buf.detach().cpu()
    # 主机参考
    xc = x.detach().cpu()[0].float()
    bad = 0
    total = 0
    for g in range(QC // GROUP):
        lo = g * GROUP
        hi = lo + GROUP
        seg = xc[lo:hi]
        gmax = float(seg.abs().max())
        sf = max(gmax, 1e-4) * (1.0 / 448.0)
        scale = sf
        y = (seg / scale).clamp(-448.0, 448.0)
        ref = y.to(torch.float8_e4m3fn).view(torch.uint8)
        for j in range(GROUP):
            off = 128 + lo + j
            gv, rv = int(got[off]), int(ref[j])
            total += 1
            if gv != rv:
                bad += 1
                if bad <= 12:
                    yy = float(y[j])
                    q_ex = ((struct.unpack("<I", struct.pack("<f", abs(yy)))[0] >> 23)
                            & 0xFF) - 127
                    q = abs(yy) * (2.0 ** (3 - q_ex))
                    print(f"  BAD g={g} j={j} x={float(seg[j]):.9e} y={yy:.9e}"
                          f" q={q:.9e} frac={q - int(q):.9e}"
                          f" kernel={gv}(0x{gv:02X}) torch={rv}(0x{rv:02X})", flush=True)
    print(f"不一致 {bad}/{total}", flush=True)
    print("KVCE14DONE", flush=True)


if __name__ == "__main__":
    main()
