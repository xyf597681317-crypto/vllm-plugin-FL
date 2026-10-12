# -*- coding: utf-8 -*-
"""_probe_kvce_9.py —— 把探针逻辑**追加到 test 文件副本的尾部**执行。

动机：独立探针调用算子必崩（507035），但 test 文件正常。为了绕开这个未知差异，
这里生成一个"test 文件 + 我追加的诊断代码"的临时脚本，用和 test 完全相同的执行方式跑。

诊断代码做的事：重跑失败用例，并对 5 个不一致字节打印
  y=x/scale 的精确 fp32、十六进制、got/ref 字节、各自反解值、以及 host 端 RNE 复刻结果。

跑法：DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_kvce_9.py
"""
import os
import sys
import tempfile

BASE = "/workspace/FlagGems/src/triton_ops"
TEST = os.path.join(BASE, "test_kv_compress_epilog.py")

# ⚠️ 必须在**分配任何张量之前**设置当前设备：否则张量在 npu:N、kernel 却发到设备 0
#   → 指针跨设备 → 507035 "vector core exception"（今天已误导三个人）
import torch  # noqa: E402
import torch_npu  # noqa: E402

_DEV = "npu:" + os.environ.get("DSH_NPU", "npu:0").split(":")[-1]
torch.npu.set_device(_DEV)
print("set_device ->", _DEV, flush=True)

APPEND = r'''

# ======================= appended diagnostic (probe_kvce_9) =======================
def _diag_failing_case():
    import struct
    import math
    rows, d, group, rscale, seed, num_slots = 9, 576, 64, 0, 9, 20
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    _, _, _, kv_col = geom2(d, group)
    row_stride = kv_col
    num_slots = max(num_slots, rows + 2)
    perm = torch.randperm(num_slots)[:rows].tolist()
    slots = [int(t) for t in perm]
    slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale, block_stride=row_stride)
    torch.npu.synchronize()
    ref = ref_bytes_mxfp8(x, slots, d, row_stride, num_slots, group, bool(rscale))
    got = buf.detach().cpu()
    bad = (got != ref).nonzero().flatten().tolist()
    print("DIAG slots=", slots, "row_stride=", row_stride, "mismatch=", len(bad), flush=True)
    xc = x.detach().cpu()
    quant_col = d - 64

    def e4m3_dec(b):
        b = int(b); s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF; m = b & 0x7
        if e == 0:
            return s * (m / 8.0) * (2.0 ** -6)
        return s * (1.0 + m / 8.0) * (2.0 ** (e - 7))

    def f32hex(v):
        return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]

    def host_bits(yv):
        b = struct.unpack("<I", struct.pack("<f", float(yv)))[0]
        sign = (b >> 24) & 0x80
        xx = struct.unpack("<f", struct.pack("<I", b & 0x7FFFFFFF))[0]
        xx = min(xx, 448.0)
        ex = ((b >> 23) & 0xFF) - 127
        sc = ((3 - ex) + 127) << 23
        q = xx * struct.unpack("<f", struct.pack("<I", sc & 0xFFFFFFFF))[0]
        fl = math.floor(q); diff = q - fl
        odd = (fl - math.floor(fl * 0.5) * 2.0) != 0.0
        if diff > 0.5:
            r = fl + 1.0
        elif diff < 0.5:
            r = fl
        else:
            r = fl + (1.0 if odd else 0.0)
        m = r - 8.0; carry = m >= 8.0
        ef = (ex + 8) if carry else (ex + 7)
        mf = 0.0 if carry else m
        code = (int(ef) << 3) + int(mf)
        qs = xx * 512.0; fls = math.floor(qs); difs = qs - fls
        odds = (fls - math.floor(fls * 0.5) * 2.0) != 0.0
        if difs > 0.5:
            rs = fls + 1.0
        elif difs < 0.5:
            rs = fls
        else:
            rs = fls + (1.0 if odds else 0.0)
        code = int(rs) if xx < 0.015625 else code
        if xx == 0.0:
            code = 0
        return (sign | code) & 0xFF, xx, ex, q, fl, diff, odd, r

    for i in bad:
        row_i, col = divmod(i, row_stride)
        if col < 128 or col >= 128 + quant_col:
            continue
        src_row = [k for k, s in enumerate(slots) if s == slots[row_i]][0]
        j = col - 128
        g = j // group
        lo = g * group
        hi = min(quant_col, lo + group)
        seg = xc[src_row, lo:hi].float()
        gmax = float(seg.abs().max())
        sf = max(gmax, 1e-4) * (1.0 / 448.0)
        bits = struct.unpack("<I", struct.pack("<f", sf))[0]
        e0 = (bits >> 23) & 0xFF
        man = bits & 0x7FFFFF
        scale = float(sf)   # rscale=0
        yv = float(xc[src_row, j].float() / scale)
        ref_u8 = (seg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        hb, xx, ex, q, fl, diff, odd, r = host_bits(yv)
        print(f"--- offset={i} slot={slots[row_i]} src_row={src_row} g={g} j={j} ---", flush=True)
        print(f"    x_bf16={float(xc[src_row, j]):.9e} scale={scale:.9e}({f32hex(scale)})"
              f" y={yv:.9e}({f32hex(yv)})", flush=True)
        print(f"    KERNEL={int(got[i])}(0x{int(got[i]):02X})->{e4m3_dec(got[i]):.9e}"
              f"  REF={int(ref[i])}(0x{int(ref[i]):02X})->{e4m3_dec(ref[i]):.9e}"
              f"  ref_u8[j]={int(ref_u8[j])}", flush=True)
        print(f"    host复刻={hb}(0x{hb:02X})  内部: x={xx:.9e} ex={ex} q={q:.9e}"
              f" fl={fl} diff={diff:.9e} odd={odd} r={r}", flush=True)
        print(f"    y 是否精确平局: diff-0.5={diff - 0.5:.3e}   "
              f"ref反解*scale={e4m3_dec(ref[i]) * scale:.9e}"
              f"  kernel反解*scale={e4m3_dec(got[i]) * scale:.9e}", flush=True)
    print("KVCE9DIAGDONE", flush=True)


_diag_failing_case()
'''

tmp = os.path.join(tempfile.gettempdir(), "kvce_runner.py")
src = open(TEST, encoding="utf-8").read()
cut = src.index('emit("=" * 96)')
head = src[:cut].replace(
    'DEV = "npu:0"',
    'DEV = "npu:" + os.environ.get("DSH_NPU", "npu:2").split(":")[-1]')
open(tmp, "w", encoding="utf-8").write(head + APPEND)
print("runner =", tmp, flush=True)
os.execv(sys.executable, [sys.executable, "-W", "ignore", tmp])
