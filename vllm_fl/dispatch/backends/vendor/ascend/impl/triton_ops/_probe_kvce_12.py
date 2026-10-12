# -*- coding: utf-8 -*-
"""_probe_kvce_12.py —— 独立探针（正确 set_device）+ 打印 5 个坏字节的精确中间值。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_12.py
"""
import math
import os
import struct
import sys

import torch
import torch_npu  # noqa: F401

IDX = int(os.environ.get("DSH_NPU", "npu:2").split(":")[-1])
torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
print("DEV =", DEV, "current =", torch.npu.current_device(), flush=True)

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

ROPE = 64


def geom2(d, group):
    qc = d - ROPE
    sc = (qc + group - 1) // group
    cc = qc + 128 + sc
    return qc, sc, cc, ((cc + 127) // 128) * 128


def ref_bytes_mxfp8(x, slots, d, row_stride, num_rows, group, round_scale):
    qc, sc, cc, kc = geom2(d, group)
    buf = torch.zeros(num_rows * row_stride, dtype=torch.uint8)
    for r, s in enumerate(slots):
        if s < 0:
            continue
        base = s * row_stride
        row = x[r]
        rope = row[qc:qc + ROPE].contiguous()
        buf[base:base + ROPE * 2] = rope.view(torch.uint8)
        for g in range(sc):
            lo = g * group
            hi = min(qc, lo + group)
            seg = row[lo:hi].float()
            gmax = float(seg.abs().max()) if seg.numel() else 0.0
            sf = max(gmax, 1e-4) * (1.0 / 448.0)
            bits = struct.unpack("<I", struct.pack("<f", sf))[0]
            e0 = (bits >> 23) & 0xFF
            eb = e0
            scale = float(sf)
            y = (seg / scale).clamp(-448.0, 448.0)
            buf[base + 128 + lo:base + 128 + hi] = y.to(torch.float8_e4m3fn).view(torch.uint8)
            buf[base + 128 + qc + g] = eb & 0xFF
    return buf


def e4m3_dec(b):
    b = int(b)
    s = -1.0 if (b >> 7) & 1 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * (m / 8.0) * (2.0 ** -6)
    return s * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def f32hex(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def host_bits(yv):
    """host 端逐位复刻 kernel 的 _to_e4m3_bits。"""
    b = struct.unpack("<I", struct.pack("<f", float(yv)))[0]
    sign = (b >> 24) & 0x80
    xx = struct.unpack("<f", struct.pack("<I", b & 0x7FFFFFFF))[0]
    xx = min(xx, 448.0)
    ex = ((b >> 23) & 0xFF) - 127
    sc = ((3 - ex) + 127) << 23
    q = xx * struct.unpack("<f", struct.pack("<I", sc & 0xFFFFFFFF))[0]
    fl = math.floor(q)
    diff = q - fl
    odd = (fl - math.floor(fl * 0.5) * 2.0) != 0.0
    r = (fl + 1.0) if diff > 0.5 else (fl if diff < 0.5 else fl + (1.0 if odd else 0.0))
    m = r - 8.0
    carry = m >= 8.0
    ef = (ex + 8) if carry else (ex + 7)
    mf = 0.0 if carry else m
    code = (int(ef) << 3) + int(mf)
    qs = xx * 512.0
    fls = math.floor(qs)
    difs = qs - fls
    odds = (fls - math.floor(fls * 0.5) * 2.0) != 0.0
    rs = (fls + 1.0) if difs > 0.5 else (fls if difs < 0.5 else fls + (1.0 if odds else 0.0))
    code = int(rs) if xx < 0.015625 else code
    if xx == 0.0:
        code = 0
    return {"byte": (sign | code) & 0xFF, "x": xx, "ex": ex, "scale2": sc, "q": q,
            "fl": fl, "diff": diff, "odd": odd, "r": r, "code": code,
            "abs_bits": b & 0x7FFFFFFF}


def main():
    rows, d, group, rscale, seed, num_slots = 9, 576, 64, 0, 9, 20
    qc, sc, cc, kc = geom2(d, group)
    row_stride = kc
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    num_slots = max(num_slots, rows + 2)
    perm = torch.randperm(num_slots)[:rows].tolist()
    slots = [int(t) for t in perm]
    slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    print("slots =", slots, "row_stride =", row_stride, flush=True)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale, block_stride=row_stride)
    torch.npu.synchronize()
    print("op done", flush=True)
    ref = ref_bytes_mxfp8(x.detach().cpu(), slots, d, row_stride, num_slots, group,
                          bool(rscale))
    got = buf.detach().cpu()
    bad = (got != ref).nonzero().flatten().tolist()
    print(f"mismatch = {len(bad)}/{ref.numel()}", flush=True)
    xc = x.detach().cpu()
    for i in bad:
        buf_row, col = divmod(i, row_stride)
        slot = buf_row
        if slot not in slots:
            print(f"--- offset={i} buf_row={buf_row} slot={slot} 不在 slots 里，跳过", flush=True)
            continue
        src = slots.index(slot)
        j = col - 128
        g = j // group
        lo = g * group
        seg = xc[src, lo:min(qc, lo + group)].float()
        gmax = float(seg.abs().max())
        sf = max(gmax, 1e-4) * (1.0 / 448.0)
        yv = float(xc[src, j].float() / float(sf))
        hb = host_bits(yv)
        print(f"--- offset={i} slot={slot} src_row={src} g={g} j={j} ---", flush=True)
        print(f"    x_bf16={float(xc[src, j]):.9e} ({f32hex(float(xc[src, j]))})"
              f"  gmax={gmax:.9e} sf={float(sf):.9e} ({f32hex(float(sf))})", flush=True)
        print(f"    y = x/sf = {yv:.9e} ({f32hex(yv)})", flush=True)
        print(f"    KERNEL={int(got[i])}(0x{int(got[i]):02X})->{e4m3_dec(got[i]):.9e}"
              f"   REF={int(ref[i])}(0x{int(ref[i]):02X})->{e4m3_dec(ref[i]):.9e}", flush=True)
        print(f"    host复刻={hb['byte']}(0x{hb['byte']:02X})  "
              f"x={hb['x']:.9e} ex={hb['ex']} q={hb['q']:.9e} fl={hb['fl']}"
              f" diff={hb['diff']:.9e} odd={hb['odd']} r={hb['r']}", flush=True)
        print(f"    diff-0.5 = {hb['diff'] - 0.5:.3e}   "
              f"中点(两相邻fp8)：ref={e4m3_dec(ref[i]):.9e} 下邻="
              f"{e4m3_dec(int(ref[i]) - 2 if int(ref[i]) % 2 == 0 else int(ref[i]) - 1):.9e}",
              flush=True)
    print("KVCE12DONE", flush=True)


if __name__ == "__main__":
    main()
