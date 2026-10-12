"""验证 kv_compress_epilog 的 Triton 实现 vs AscendC 语义（quant_mode=1 + layout=1）。

一行输出字节布局（kvCacheCol 字节）：
  [0, quantCol)                    fp8 量化值（每 128 元素一个 scale）
  [quantCol, quantCol+128)         x 的最后 64 列 BF16 原样（小端）
  [quantCol+128, +scaleCol*4)      各子块 fp32 scale（小端）
  [concatCol, kvCacheCol)          pad 0
量化：scale = max|x_sub| * Mul(1/448)（乘法！）; y = x/scale → fp8_e4m3fn(RNE)
逐字节相等才算 PASS。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.kv_compress_epilog import kv_compress_epilog  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:0").split(":")[-1]
L = []
ROPE = 64


def emit(s=""):
    L.append(str(s))


def geom(d):
    quant_col = d - ROPE
    scale_col = (quant_col + 127) // 128
    concat_col = quant_col + 128 + scale_col * 4
    kv_col = ((concat_col + 127) // 128) * 128
    return quant_col, scale_col, concat_col, kv_col


def ref_bytes(x, slots, d, row_stride, num_rows):
    """quant_mode=1 (GROUP_FP8) 参考。scale 同样必须在 fp32 里算
    （AscendC common.h L207/L235 `Mul(scale, scale, coeffReg)`，coeff 是 fp32 的 1/448）。"""
    quant_col, scale_col, concat_col, kv_col = geom(d)
    coeff32 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    buf = torch.zeros(num_rows * row_stride, dtype=torch.uint8)
    xd = x.detach().cpu()                     # bf16
    for r, s in enumerate(slots):
        if s < 0:
            continue
        base = s * row_stride
        row = xd[r]
        # 量化段
        for sb in range(scale_col):
            lo = sb * 128
            hi = min(quant_col, lo + 128)
            seg = row[lo:hi].float()
            mx = seg.abs().max()
            if float(mx) > 0.0:
                scale_t = torch.tensor(float(mx), dtype=torch.float32) * coeff32
                y = seg.double() / float(scale_t)
            else:
                scale_t = torch.tensor(0.0, dtype=torch.float32)
                y = torch.zeros_like(seg)
            buf[base + lo:base + hi] = y.to(torch.float8_e4m3fn).view(torch.uint8)
            sb_bytes = scale_t.reshape(1).view(torch.uint8)
            buf[base + quant_col + 128 + sb * 4:base + quant_col + 128 + sb * 4 + 4] = sb_bytes
        # rope 透传（bf16 原样小端）
        rope = row[quant_col:quant_col + ROPE].contiguous()
        buf[base + quant_col:base + quant_col + ROPE * 2] = rope.view(torch.uint8)
        # pad 已是 0
    return buf



def geom2(d, group):
    """quant_mode=2 (MXFP8)：scale 为 1 字节 E8M0。"""
    quant_col = d - ROPE
    scale_col = (quant_col + group - 1) // group
    concat_col = quant_col + 128 + scale_col
    kv_col = ((concat_col + 127) // 128) * 128
    return quant_col, scale_col, concat_col, kv_col


def _e8m0_scale_bytes(sf, round_scale):
    """复刻 common.h L361-372 / L389-391：sf=max(gmax,1e-4)*(1/448)；
    round_scale → 2^ceil(log2(sf))（man!=0 进位）；scale 字节 = (scale_bits>>23)&0xFF。"""
    import struct
    bits = struct.unpack("<I", struct.pack("<f", float(sf)))[0]
    e0 = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    eb = (e0 + (1 if man != 0 else 0)) if round_scale else e0
    scale = struct.unpack("<f", struct.pack("<I", (eb << 23) & 0xFFFFFFFF))[0] if round_scale else float(sf)
    return float(scale), eb & 0xFF


def ref_bytes_mxfp8(x, slots, d, row_stride, num_rows, group, round_scale):
    """quant_mode=2 (MXFP8) 参考。

    ⚠️ **scale 必须在 fp32 里算**（2026-10-12 修）：AscendC 是
        `Maxs(max2, max2, 1e-4f); Muls(max2, max2, coeff)`    （common.h L361-362）
    其中 `coeff = INV_FP8_E4M3_MAX_VALUE = 0x3b124925`，即 **fp32 表示**的 1/448，
    且乘法就是 fp32 的 `Muls`（不是 fp64 再截断）。
    旧参考写成 `max(gmax,1e-4) * (1.0/448.0)` —— Python 这里是 **fp64** 运算，
    结果比 kernel 高 1 ULP（实测 mx=4.8125 时 fp64→0x3c300000、fp32→0x3c300001），
    于是 `y = x/scale` 也差 ~1 ULP，**恰好落在 fp8 平局点上的元素就会差 1 个字节**。
    证据：`d=576 group=64 rscale=0 rows=9 含-1` 用例的 5 个不一致字节
    （offset 1690/1725/13225/13517/13531）**全是精确平局**（q=11.5/9.5/10.5/…，
    frac 恰好 0.5），改成 fp32 后对齐；`rscale=1` 时 scale 被取整成 2 的幂，
    这条 1 ULP 差异被吸收，所以之前只有 rscale=0 暴露。
    """
    quant_col, scale_col, concat_col, kv_col = geom2(d, group)
    # fp32 的 1/448（与 AscendC INV_FP8_E4M3_MAX_VALUE 同位模式 0x3b124925）
    coeff32 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    buf = torch.zeros(num_rows * row_stride, dtype=torch.uint8)
    xd = x.detach().cpu()
    for r, s in enumerate(slots):
        if s < 0:
            continue
        base = s * row_stride
        row = xd[r]
        # rope 透传在前 128 字节
        rope = row[quant_col:quant_col + ROPE].contiguous()
        buf[base:base + ROPE * 2] = rope.view(torch.uint8)
        # 量化段从字节 128 开始
        for g in range(scale_col):
            lo = g * group
            hi = min(quant_col, lo + group)
            seg = row[lo:hi].float()                     # bf16 → fp32
            gmax = float(seg.abs().max()) if seg.numel() else 0.0
            # 先各步都在 fp32 完成（与 kernel 的 Maxs/Muls 一致），再交给 fp64 做除法
            sf32 = (torch.tensor(max(gmax, 1e-4), dtype=torch.float32) * coeff32)
            sf = float(sf32)
            scale, sb = _e8m0_scale_bytes(sf, round_scale)
            y = (seg.double() / scale).clamp(-448.0, 448.0)
            buf[base + 128 + lo:base + 128 + hi] = y.to(torch.float8_e4m3fn).view(torch.uint8)
            buf[base + 128 + quant_col + g] = sb
    return buf


def run_case(name, rows, d, num_slots=20, row_stride_extra=0, seed=9, with_neg=True, layout=1, qmode=1, group=128, rscale=1):
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 2.0).to(torch.bfloat16).to(DEV)
    _, _, _, kv_col = geom(d) if qmode == 1 else geom2(d, group)
    # layout=1：可用 block_stride 拉大行距；layout!=1：行距固定 kvCacheCol
    row_stride = (kv_col + row_stride_extra) if layout == 1 else kv_col
    num_slots = max(num_slots, rows + 2)
    # slot 必须**互不重复**：重复会让多行写同一行，结果依赖写入顺序（并行下不确定）
    num_slots = max(num_slots, rows + 2)
    perm = torch.randperm(num_slots)[:rows].tolist()
    slots = [int(t) for t in perm]
    if with_neg and rows >= 5:
        slots[::5] = [-1] * len(slots[::5])
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)

    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    # 用 fp8 dtype 的视图调用（真实调用点传 fp8 张量）
    cache_fp8 = buf.view(torch.float8_e4m3fn)
    kv_compress_epilog(x, sm, cache_fp8, quant_mode=qmode, layout=layout,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride if layout == 1 else 0)
    torch.npu.synchronize()

    ref = (ref_bytes(x, slots, d, row_stride, num_slots) if qmode == 1
           else ref_bytes_mxfp8(x, slots, d, row_stride, num_slots, group, bool(rscale)))
    got = buf.detach().cpu()
    bad = int((got != ref).sum().item())
    ok = bad == 0
    line = f"  {name:52s} byte_mismatch={bad}/{ref.numel()}  {'PASS' if ok else 'FAIL'}"
    emit(line)
    print(line, flush=True)
    if bad:
        idx = (got != ref).nonzero()[:6].flatten().tolist()
        for i in idx:
            emit(f"        offset {i}: got={int(got[i])} ref={int(ref[i])}")
        print("\n".join(L[-7:]), flush=True)
    return ok


emit("=" * 96)
emit("kv_compress_epilog: Triton vs AscendC（quant_mode=1/layout=1，逐字节相等）")
emit("=" * 96)

ok = True
ok &= run_case("d=640 (quant=576,scaleCol=5) rows=16", 16, 640)
ok &= run_case("d=576 (quant=512,scaleCol=4) rows=12", 12, 576)
ok &= run_case("d=192 (quant=128,scaleCol=1) rows=10", 10, 192)
ok &= run_case("d=320 (quant=256,scaleCol=2) rows=8 rowStride 加 pad", 8, 320, row_stride_extra=64)
ok &= run_case("d=576 rows=9（含 -1 跳过）", 9, 576, with_neg=True)
ok &= run_case("d=192 layout=2（quant_mode=1 行距=kvCacheCol）", 8, 192, layout=2)
ok &= run_case("d=640 layout=2 rows=12", 12, 640, layout=2)

emit()
emit("--- quant_mode=2 (MXFP8/E8M0)：**生产配置**（dsa_attn_kv_plan.py L199-202）---")
ok &= run_case("生产: d=640 group=64 rscale=1 layout=1 rows=12", 12, 640, qmode=2, group=64)
ok &= run_case("生产: d=576 group=64 rscale=1 layout=1 rows=10", 10, 576, qmode=2, group=64)
ok &= run_case("生产: d=192 group=64 rscale=1 layout=1 rows=8", 8, 192, qmode=2, group=64)
ok &= run_case("d=640 group=128 rscale=1 rows=12", 12, 640, qmode=2, group=128)
ok &= run_case("d=576 group=64 rscale=0（不取整）rows=9 含-1", 9, 576, qmode=2, group=64, rscale=0)
ok &= run_case("d=320 group=64 rscale=1 layout=2 rows=8", 8, 320, qmode=2, group=64, layout=2)

# ======================= 回归：平局取整 + fp32 scale =======================
# 背景（2026-10-12）：`d=576 group=64 rscale=0 rows=9 含-1` 曾报 5/15360 字节不一致
# （offset 1690/1725/13225/13517/13531，全部 got=ref-1）。根因是**参考**把
# `sf = max(gmax,1e-4) * (1.0/448.0)` 算在 fp64 里，比 AscendC 的 fp32 `Muls`
# 高 1 ULP，恰好把落在 fp8 平局点的元素推过中点。
# 下面两条用例把这个语义钉住：一旦有人把参考改回 fp64、或 kernel 的 scale 不用
# fp32 的 1/448（INV_FP8_E4M3_MAX_VALUE = 0x3b124925），就会立刻变红。
TIE_COEFF = torch.tensor(1.0 / 448.0, dtype=torch.float32)   # 与 AscendC 同位模式


def _tie_band_values(group, n=64, rnd=7919):
    """构造恰好落在 fp8 (e4m3) 编码**正中间**的 |y| 候选。

    取 fp8 值 v（指数 e∈[0,15]、尾数 m∈[0,7]），则相邻两个 e4m3 值的算术中点为
    `v * (1 + 1/16)`。用它反推 |y|（= mid * 448/|gmax| 那步由 gmax 决定），
    使 `|y| * 2^(3-ex(y))` 的分数部分恰为 0.5 ⇒ 只有 RNE（就近取偶）
    与 fp32 精度的 scale 才能对齐 AscendC。
    """
    import math
    out = []
    for e in range(0, 12):
        for m in range(0, 8):
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
            if v * 1.0625 <= 448.0:
                out.append(v * 1.0625)
    out = out or [1.0]
    return [out[(i * rnd) % len(out)] for i in range(group)]


def run_tie_regression(name, d=192, group=64, rows=2, rscale=0):
    """受控平局用例：gmax 固定为 448.0（⇒ scale = 1.0），组内塞满平局候选。

    scale = fp32(448.0) * fp32(1/448) = 1.0，因此 y = x 且 q = x * 2^(3-ex)，
    平局点全部保留，逐字节判定。
    """
    qc, sc, cc, kc = geom2(d, group)
    torch.manual_seed(20261012)
    x = torch.zeros(rows, d, dtype=torch.float32)
    vals = _tie_band_values(group)
    for r in range(rows):
        for i, v in enumerate(vals):
            x[r, i] = v
        x[r, 0] = 448.0                       # 固定 gmax（保证 scale=1.0）
        x[r, qc:qc + ROPE] = 0.0
    x = x.to(torch.bfloat16).to(DEV)
    slots = list(range(rows))
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    row_stride = kc
    num_slots = max(20, rows + 2)
    buf = torch.zeros(num_slots * row_stride, dtype=torch.uint8, device=DEV)
    kv_compress_epilog(x, sm, buf.view(torch.float8_e4m3fn), quant_mode=2, layout=1,
                       quant_group_size=group, round_scale=rscale,
                       block_stride=row_stride)
    torch.npu.synchronize()
    ref = ref_bytes_mxfp8(x, slots, d, row_stride, num_slots, group, bool(rscale))
    got = buf.detach().cpu()
    bad = int((got != ref).sum().item())
    okc = bad == 0
    line = (f"  {name:52s} byte_mismatch={bad}/{ref.numel()}  {'PASS' if okc else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    if bad:
        for i in (got != ref).nonzero()[:6].flatten().tolist():
            emit(f"        offset {i}: got={int(got[i])} ref={int(ref[i])}")
    return okc


emit()
emit("--- 回归：fp8 平局取整（RNE）+ fp32 scale（offset 1690/1725/13225/13517/13531 曾错）---")
ok &= run_tie_regression("平局回归 d=192 group=64 rscale=0 受控平局行", 192, 64, rows=2,
                         rscale=0)
ok &= run_tie_regression("平局回归 d=192 group=64 rscale=1 受控平局行", 192, 64, rows=2,
                         rscale=1)
ok &= run_case("失败用例原样复跑 d=576 group=64 rscale=0 rows=9 含-1（固定 seed）",
               9, 576, qmode=2, group=64, rscale=0)
ok &= run_case("同形状 rscale=1 对照 rows=9 含-1", 9, 576, qmode=2, group=64, rscale=1)

emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/kv_compress_epilog_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
