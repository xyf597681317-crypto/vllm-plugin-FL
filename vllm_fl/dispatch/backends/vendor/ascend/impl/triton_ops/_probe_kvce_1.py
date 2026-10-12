# -*- coding: utf-8 -*-
"""_probe_kvce_1.py —— kv_compress_epilog 5/15360 字节不一致的最小复现 + 逐字节定位。

只做一件事：复现 `d=576 group=64 rscale=0 rows=9 含-1`，把**每一个**不一致字节的
下标、行号、行内偏移、got/ref 取值、以及该字节的语义（quant / rope / scale / pad）
全部打印出来。

跑法（容器内）：
    DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_1.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]
ROPE = 64


def geom2(d, group):
    quant_col = d - ROPE
    scale_col = (quant_col + group - 1) // group
    concat_col = quant_col + 128 + scale_col
    kv_col = ((concat_col + 127) // 128) * 128
    return quant_col, scale_col, concat_col, kv_col


def _e8m0_bytes(sf, round_scale):
    """common.h L361-372 / L389-391 的逐位复刻（纯 python 位运算，不用 struct 花样）。"""
    import struct
    bits = struct.unpack("<I", struct.pack("<f", float(sf)))[0]
    e0 = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    eb = (e0 + (1 if man != 0 else 0)) if round_scale else e0
    if round_scale:
        scale = struct.unpack("<f", struct.pack("<I", (eb << 23) & 0xFFFFFFFF))[0]
    else:
        scale = float(sf)
    return float(scale), eb & 0xFF


def ref_bytes_mxfp8(x, slots, d, row_stride, num_rows, group, round_scale):
    """逐行抄测试文件的参考（不改任何口径），另返回每字节语义标签。"""
    quant_col, scale_col, concat_col, kv_col = geom2(d, group)
    buf = torch.zeros(num_rows * row_stride, dtype=torch.uint8)
    label = ["pad"] * (num_rows * row_stride)
    for r, s in enumerate(slots):
        if s < 0:
            continue
        base = s * row_stride
        row = x[r]
        rope = row[quant_col:quant_col + ROPE].contiguous()
        buf[base:base + ROPE * 2] = rope.view(torch.uint8)
        for i in range(ROPE * 2):
            label[base + i] = f"row{r}:rope[{i}]"
        for g in range(scale_col):
            lo = g * group
            hi = min(quant_col, lo + group)
            seg = row[lo:hi].float()
            gmax = float(seg.abs().max()) if seg.numel() else 0.0
            sf = max(gmax, 1e-4) * (1.0 / 448.0)
            scale, sb = _e8m0_bytes(sf, round_scale)
            y = (seg / scale).clamp(-448.0, 448.0)
            buf[base + 128 + lo:base + 128 + hi] = y.to(torch.float8_e4m3fn).view(torch.uint8)
            for j in range(hi - lo):
                label[base + 128 + lo + j] = f"row{r}:quant[g{g},c{lo + j}]"
            buf[base + 128 + quant_col + g] = sb
            label[base + 128 + quant_col + g] = f"row{r}:scale[g{g}]"
    return buf, label


def run(rows=9, d=576, group=64, rscale=0, seed=9, num_slots=20,
        row_stride_extra=0, verbose=True):
    quant_col, scale_col, concat_col, kv_col = geom2(d, group)
    row_stride = kv_col + row_stride_extra
    num_slots = max(num_slots, rows + 2)

    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    perm = torch.randperm(num_slots)[:rows].tolist()
    slots = [int(t) for t in perm]
    if rows >= 5:
        slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    cache_fp8 = buf.view(torch.float8_e4m3fn)
    kv_compress_epilog(x, sm, cache_fp8, quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    torch.npu.synchronize()
    ref, label = ref_bytes_mxfp8(x.detach().cpu(), slots, d, row_stride, num_slots,
                                 group, bool(rscale))
    got = buf.detach().cpu()
    bad = (got != ref).nonzero().flatten().tolist()
    print(f"== rows={rows} d={d} group={group} rscale={rscale} seed={seed} ==", flush=True)
    print(f"   quant_col={quant_col} scale_col={scale_col} concat_col={concat_col}"
          f" kv_col={kv_col} row_stride={row_stride} num_slots={num_slots}", flush=True)
    print(f"   slots={slots}", flush=True)
    print(f"   不一致字节数 = {len(bad)} / {ref.numel()}", flush=True)
    for i in bad:
        row_i = i // row_stride
        col = i % row_stride
        slot = -1 if row_i >= len(slots) else slots[row_i]
        print(f"     offset={i:6d}  buffer_row={row_i:2d}(slot={slot:3d}) col={col:4d}"
              f"  语义={label[i]:24s} got={int(got[i]):3d} ref={int(ref[i]):3d}"
              f" diff={int(got[i]) - int(ref[i]):+4d}", flush=True)
    if verbose and bad:
        # 把涉及到的 scale 与原始数据也打出来，便于判定谁对
        seen = sorted({(i // row_stride) for i in bad})
        for row_i in seen:
            if row_i >= len(slots) or slots[row_i] < 0:
                continue
            r = slots.index(slots[row_i]) if False else None  # noqa: F841
            # 找到该 slot 对应的源行号
            src = [k for k, s in enumerate(slots) if s == slots[row_i]]
            src = src[0]
            xr = x[src].detach().cpu().float()
            print(f"   --- buffer_row={row_i} slot={slots[row_i]} 源行={src} ---", flush=True)
            for g in range(scale_col):
                lo = g * group
                hi = min(quant_col, lo + group)
                seg = xr[lo:hi]
                gmax = float(seg.abs().max())
                sf = max(gmax, 1e-4) * (1.0 / 448.0)
                scale, sb = _e8m0_bytes(sf, bool(rscale))
                marks = [i for i in bad
                         if i // row_stride == row_i and 128 + lo <= i % row_stride < 128 + hi]
                smark = [i for i in bad if i // row_stride == row_i
                         and i % row_stride == 128 + quant_col + g]
                print(f"     g={g} lo={lo} gmax={gmax:.6e} sf={sf:.6e}"
                      f" scale={scale:.6e} scale_byte={sb}"
                      f" 量化段不符={len(marks)} scale不符={bool(smark)}", flush=True)
    return len(bad)


if __name__ == "__main__":
    n = run()
    print("REPRODONE mismatch=", n, flush=True)
