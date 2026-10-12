# -*- coding: utf-8 -*-
"""_probe_kvce_18.py —— dump 生产 kernel 里的 mx / sf / scale，与主机逐位比对。

若 kernel 的 mx 与主机不同 → reduce 顺序/精度问题（量化 scale 的依据）。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_18.py
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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROPE, D, GROUP = 64, 192, 64
QC = D - ROPE


@triton.jit
def _k_mx(X, SLOT, DMX, DSF, DSCALE, DDIV,
          ROWS, DD, QUANT_COL, SCALE_COL, ROW_STRIDE,
          ROUND_SCALE: tl.constexpr, BLK_G: tl.constexpr):
    r = tl.program_id(0)
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            for s in range(0, SCALE_COL):
                cols = s * BLK_G + tl.arange(0, BLK_G)
                m = cols < QUANT_COL
                v = tl.load(X + r * DD + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                sf = tl.maximum(mx, 1.0e-4) * (1.0 / 448.0)
                bits = sf.to(tl.int32, bitcast=True)
                e0 = (bits >> 23) & 0xFF
                man = bits & 0x7FFFFF
                if ROUND_SCALE:
                    eb = e0 + tl.where(man != 0, 1, 0)
                    scale = (eb << 23).to(tl.float32, bitcast=True)
                else:
                    eb = e0
                    scale = sf
                y = v / scale
                tl.store(DMX + s * BLK_G + tl.arange(0, BLK_G), mx + cols * 0.0)
                tl.store(DSF + s * BLK_G + tl.arange(0, BLK_G), sf + cols * 0.0)
                tl.store(DSCALE + s * BLK_G + tl.arange(0, BLK_G), scale + cols * 0.0)
                tl.store(DDIV + s * BLK_G + tl.arange(0, BLK_G), y)


def f32hex(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def main():
    print("device =", DEV, flush=True)
    vals = (0.5 + (torch.arange(QC, dtype=torch.float32) / QC) * 4.3)
    row = vals.clone()
    row[0] = 4.0
    row[GROUP] = 4.8125
    x = torch.zeros(1, D, dtype=torch.bfloat16)
    x[0, :QC] = row.to(torch.bfloat16)
    x = x.to(DEV)
    sm = torch.tensor([0], dtype=torch.int32, device=DEV)
    Z = lambda: torch.zeros(QC, dtype=torch.float32, device=DEV)  # noqa: E731
    dmx, dsf, dscale, ddiv = Z(), Z(), Z(), Z()
    _k_mx[(1,)](x, sm, dmx, dsf, dscale, ddiv, 1, D, QC, QC // GROUP, 384,
                ROUND_SCALE=False, BLK_G=GROUP)
    torch.npu.synchronize()
    mxk = dmx.detach().cpu()
    sfk = dsf.detach().cpu()
    sck = dscale.detach().cpu()
    divk = ddiv.detach().cpu()
    xc = x.detach().cpu()[0].float()
    for g in range(2):
        lo, hi = g * GROUP, g * GROUP + GROUP
        seg = xc[lo:hi]
        mx_h = float(seg.abs().max())
        sf_h = max(mx_h, 1e-4) * (1.0 / 448.0)
        print(f"--- group {g} ---", flush=True)
        print(f"  mx   kernel={float(mxk[lo]):.9e} ({f32hex(float(mxk[lo]))})"
              f"  host={mx_h:.9e} ({f32hex(mx_h)})"
              f"  {'MATCH' if float(mxk[lo]) == mx_h else 'DIFF'}", flush=True)
        print(f"  sf   kernel={float(sfk[lo]):.9e} ({f32hex(float(sfk[lo]))})"
              f"  host={sf_h:.9e} ({f32hex(sf_h)})"
              f"  {'MATCH' if float(sfk[lo]) == sf_h else 'DIFF'}", flush=True)
        print(f"  scale kernel={float(sck[lo]):.9e} ({f32hex(float(sck[lo]))})"
              f"  host={sf_h:.9e} ({f32hex(sf_h)})", flush=True)
        yk = float(divk[lo])
        yh = float(seg[0] / sf_h)
        print(f"  y[0] kernel={yk:.9e} ({f32hex(yk)})  host={yh:.9e} ({f32hex(yh)})"
              f"  {'MATCH' if yk == yh else 'DIFF'}", flush=True)
    # 关键点 j=103
    jj = 103
    print(f"--- 关键列 j={jj} (归组 {jj // GROUP}) ---", flush=True)
    yk = float(divk[jj])
    sf_h = max(float(xc[jj // GROUP * GROUP:(jj // GROUP + 1) * GROUP].abs().max()),
               1e-4) * (1.0 / 448.0)
    yh = float(xc[jj] / sf_h)
    print(f"  x={float(xc[jj]):.9e}   y_kernel={yk:.9e} ({f32hex(yk)})"
          f"   y_host={yh:.9e} ({f32hex(yh)})", flush=True)
    print("KVCE18DONE", flush=True)


if __name__ == "__main__":
    main()
