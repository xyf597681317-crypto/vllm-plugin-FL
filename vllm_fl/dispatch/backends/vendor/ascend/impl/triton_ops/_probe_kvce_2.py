# -*- coding: utf-8 -*-
"""_probe_kvce_2.py —— 只看那 5 个字节：为什么 fp8 编码差 1。

复刻失败用例的**精确输入**（seed=9, d=576, group=64, rscale=0, rows=9, slot 逻辑同测试），
对每个不一致字节打印：
  * 它属于哪个 slot / 哪个 group / 组内第几列
  * 该组 gmax、sf、scale（rscale=0 时 = sf）
  * y = x/scale 的 fp32 精确值、十六进制
  * 我们 kernel 给的字节、参考给的字节，以及它们各自的 fp32 反解
  * torch 用 fp32→fp8 的"权威"结果（torch.float8_e4m3fn 转换）
  * 手算 RNE 期望（用 host 端复刻 _to_e4m3_bits 的位运算）

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_2.py
"""
import os
import struct
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]
ROPE = 64
D, GROUP, RSCALE, ROWS, SEED, NSLOTS = 576, 64, 0, 9, 9, 20


def geom2(d, group):
    qc = d - ROPE
    sc = (qc + group - 1) // group
    cc = qc + 128 + sc
    kc = ((cc + 127) // 128) * 128
    return qc, sc, cc, kc


def e8m0(sf, round_scale):
    bits = struct.unpack("<I", struct.pack("<f", float(sf)))[0]
    e0 = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    eb = (e0 + (1 if man != 0 else 0)) if round_scale else e0
    scale = (struct.unpack("<f", struct.pack("<I", (eb << 23) & 0xFFFFFFFF))[0]
             if round_scale else float(sf))
    return float(scale), eb & 0xFF


def f32_hex(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def e4m3_decode(b):
    """e4m3fn 字节 → fp32。"""
    b = int(b)
    s = -1.0 if (b >> 7) & 1 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return float("nan")
    return s * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def host_to_e4m3_bits(yv):
    """host 端逐位复刻 kernel 里的 _to_e4m3_bits（fp32 输入）。"""
    import numpy as np
    b = np.float32(yv).view(np.uint32) if False else struct.unpack(
        "<I", struct.pack("<f", float(np.float32(yv))))[0]
    sign = (b >> 24) & 0x80
    x = struct.unpack("<f", struct.pack("<I", b & 0x7FFFFFFF))[0]
    x = min(x, 448.0)
    ex = ((b >> 23) & 0xFF) - 127
    sc = ((3 - ex) + 127) << 23
    q = x * struct.unpack("<f", struct.pack("<I", sc & 0xFFFFFFFF))[0]
    import math
    fl = math.floor(q)
    diff = q - fl
    odd = (fl - math.floor(fl * 0.5) * 2.0) != 0.0
    if diff > 0.5:
        r = fl + 1.0
    elif diff < 0.5:
        r = fl
    else:
        r = fl + (1.0 if odd else 0.0)
    m = r - 8.0
    carry = m >= 8.0
    e_final = (ex + 8) if carry else (ex + 7)
    m_final = 0.0 if carry else m
    code_norm = (int(e_final) << 3) + int(m_final)
    qs = x * 512.0
    fls = math.floor(qs)
    difs = qs - fls
    odds = (fls - math.floor(fls * 0.5) * 2.0) != 0.0
    if difs > 0.5:
        rs = fls + 1.0
    elif difs < 0.5:
        rs = fls
    else:
        rs = fls + (1.0 if odds else 0.0)
    code_sub = int(rs)
    code = code_sub if x < 0.015625 else code_norm
    if x == 0.0:
        code = 0
    return (sign | code) & 0xFF


def main():
    qc, sc, cc, kc = geom2(D, GROUP)
    row_stride = kc
    nslots = max(NSLOTS, ROWS + 2)
    torch.manual_seed(SEED)
    x = (torch.randn(ROWS, D) * 2.0).to(torch.bfloat16).to(DEV)
    perm = torch.randperm(nslots)[:ROWS].tolist()
    slots = [int(t) for t in perm]
    slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(nslots * row_stride, dtype=torch.uint8, device=DEV)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=GROUP, round_scale=RSCALE, block_stride=row_stride)
    torch.npu.synchronize()
    got = buf.detach().cpu()
    xc = x.detach().cpu()
    print(f"slots={slots} row_stride={row_stride} qc={qc} sc={sc} cc={cc} kc={kc}",
          flush=True)

    for row_i, s in enumerate(slots):
        if s < 0:
            continue
        base = s * row_stride
        xr = xc[row_i].float()
        for g in range(sc):
            lo = g * GROUP
            hi = min(qc, lo + GROUP)
            seg = xr[lo:hi]
            gmax = float(seg.abs().max())
            sf = max(gmax, 1e-4) * (1.0 / 448.0)
            scale, sb = e8m0(sf, bool(RSCALE))
            y = (seg / scale).clamp(-448.0, 448.0)
            ref_u8 = y.to(torch.float8_e4m3fn).view(torch.uint8)
            for j in range(hi - lo):
                off = base + 128 + lo + j
                gg, rr = int(got[off]), int(ref_u8[j])
                if gg == rr:
                    continue
                yv = float(y[j])
                hb = host_to_e4m3_bits(yv)
                print(f"--- offset={off} slot={s} 源行={row_i} g={g} 列={lo + j} ---",
                      flush=True)
                print(f"    x_bf16={float(xc[row_i, lo + j]):.6e}  x_f32={float(xr[lo + j]):.6e}"
                      f" ({f32_hex(xr[lo + j])})", flush=True)
                print(f"    gmax={gmax:.6e} sf={sf:.6e} ({f32_hex(sf)}) scale={scale:.6e}"
                      f" scale_byte={sb}", flush=True)
                print(f"    y=x/scale={yv:.9f} ({f32_hex(yv)})", flush=True)
                print(f"    KERNEL字节={gg} → 反解={e4m3_decode(gg):.9f}", flush=True)
                print(f"    REF   字节={rr} → 反解={e4m3_decode(rr):.9f}", flush=True)
                print(f"    host 复刻 _to_e4m3_bits={hb}"
                      f"  {'==KERNEL' if hb == gg else '==REF' if hb == rr else '==都不是!'}",
                      flush=True)
                # 该字节对应的解量化值（用 ref 的 scale）
                print(f"    量化候选: ref反解*scale={e4m3_decode(rr) * scale:.9f}"
                      f" kernel反解*scale={e4m3_decode(gg) * scale:.9f}", flush=True)
    print("KVCE2DONE", flush=True)


if __name__ == "__main__":
    main()
