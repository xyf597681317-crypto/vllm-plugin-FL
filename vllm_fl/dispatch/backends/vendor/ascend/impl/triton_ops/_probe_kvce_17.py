# -*- coding: utf-8 -*-
"""_probe_kvce_17.py —— 在**生产 kernel 内部**dump 编码中间量。

复制一份 _kce_mxfp8_kernel 的代码（不改原文件），加上调试输出：
  D_Y[j]  = v / scale
  D_Q[j]  = q = |y| * 2^(3-ex)
  D_FL[j] = fl
  D_DIFF[j]= diff
  D_ODD[j]= odd
  D_BYTE[j]= 最终字节
然后与主机同样的计算逐项比对。
跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_17.py
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
from triton_ops.kv_compress_epilog import _to_e4m3_bits  # noqa: E402

ROPE, D, GROUP = 64, 192, 64
QC = D - ROPE
KC = 384


@triton.jit
def _k_dbg(X, SLOT, CACHE, DY, DQ, DFL, DDIFF, DODD, DBYTE,
           ROWS, DD, QUANT_COL, SCALE_COL, CONCAT_COL, KV_COL, ROW_STRIDE,
           ROUND_SCALE: tl.constexpr, BLK_G: tl.constexpr, BLK_PAD: tl.constexpr):
    r = tl.program_id(0)
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            base = slot * ROW_STRIDE
            off = tl.arange(0, 64)
            rv = tl.load(X + r * DD + QUANT_COL + off)
            u = rv.to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(CACHE + base + off * 2, (u & 0xFF).to(tl.uint8))
            tl.store(CACHE + base + off * 2 + 1, ((u >> 8) & 0xFF).to(tl.uint8))
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
                # ---- 编码中间量 ----
                yb = y.to(tl.int32, bitcast=True)
                sign = (yb >> 24) & 0x80
                xa = (yb & 0x7FFFFFFF).to(tl.float32, bitcast=True)
                xa = tl.minimum(xa, 448.0)
                ex = ((yb >> 23) & 0xFF) - 127
                sc_bits = ((3 - ex) + 127) << 23
                q = xa * sc_bits.to(tl.float32, bitcast=True)
                fl = tl.floor(q)
                diff = q - fl
                odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0
                byte = _to_e4m3_bits(y)
                tl.store(DY + s * BLK_G + tl.arange(0, BLK_G), y, mask=m)
                tl.store(DQ + s * BLK_G + tl.arange(0, BLK_G), q, mask=m)
                tl.store(DFL + s * BLK_G + tl.arange(0, BLK_G), fl, mask=m)
                tl.store(DDIFF + s * BLK_G + tl.arange(0, BLK_G), diff, mask=m)
                tl.store(DODD + s * BLK_G + tl.arange(0, BLK_G),
                         tl.where(odd, 1.0, 0.0), mask=m)
                tl.store(DBYTE + s * BLK_G + tl.arange(0, BLK_G), byte, mask=m)
                tl.store(CACHE + base + 128 + cols, byte.to(tl.uint8), mask=m)
                tl.store(CACHE + base + 128 + QUANT_COL + s, (eb & 0xFF).to(tl.uint8))


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
    buf = torch.zeros(2 * KC, dtype=torch.uint8, device=DEV)
    Z = lambda: torch.zeros(QC, dtype=torch.float32, device=DEV)  # noqa: E731
    dy, dq, dfl, ddiff, dodd, dbyte = Z(), Z(), Z(), Z(), Z(), Z()
    _k_dbg[(1,)](x, sm, buf, dy, dq, dfl, ddiff, dodd, dbyte,
                 1, D, QC, QC // GROUP, QC + 128 + QC // GROUP, KC, KC,
                 ROUND_SCALE=False, BLK_G=GROUP, BLK_PAD=128)
    torch.npu.synchronize()
    y_k = dy.detach().cpu()
    q_k = dq.detach().cpu()
    fl_k = dfl.detach().cpu()
    dif_k = ddiff.detach().cpu()
    odd_k = dodd.detach().cpu()
    by_k = dbyte.detach().cpu()
    xc = x.detach().cpu()[0].float()
    print(f"{'j':>4} {'x':>14} {'y_kernel':>16} {'y_host':>16} {'q_k':>14} {'q_host':>14}"
          f" {'odd_k':>6} {'byte_k':>7} {'byte_ref':>9}", flush=True)
    nshow = 0
    for g in range(2):
        lo, hi = g * GROUP, g * GROUP + GROUP
        seg = xc[lo:hi]
        gmax = float(seg.abs().max())
        sf = max(gmax, 1e-4) * (1.0 / 448.0)
        y_host = (seg / sf)
        ref = y_host.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        for j in range(GROUP):
            gj = lo + j
            yk = float(y_k[gj])
            yh = float(y_host[j])
            if abs(yk - yh) > 1e-6 * max(1.0, abs(yh)) or int(by_k[gj]) != int(ref[j]):
                if nshow < 12:
                    nshow += 1
                    print(f"{gj:>4} {float(seg[j]):>14.6e} {yk:>16.9e} {yh:>16.9e}"
                          f" {float(q_k[gj]):>14.9e} {'':>14} {float(odd_k[gj]):>6.0f}"
                          f" {int(by_k[gj]):>7d} {int(ref[j]):>9d}", flush=True)
    print("KVCE17DONE", flush=True)


if __name__ == "__main__":
    main()
