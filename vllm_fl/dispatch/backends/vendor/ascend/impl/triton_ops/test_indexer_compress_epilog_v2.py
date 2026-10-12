"""验证 indexer_compress_epilog_v2 的 Triton 实现 vs AscendC 语义。

语义：逐行（slot != -1）做 **每 128 元素子块** 的动态 fp8_e4m3 量化：
        scale = max(|x|_sub) / 448 ； y = x / scale → fp8_e4m3（CAST_RINT = 就近取偶）
      cache 字节布局：每个 block = [cacheBs*d 个 fp8 字节][cacheBs*scaleCol 个 fp32 小端]
      block_stride 取调用点传的 indexer_compress_cache.stride(0)（csrc/torch_binding.cpp L1902）
参考：torch 的 float8_e4m3fn 转换（CPU）。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.indexer_compress_epilog_v2 import indexer_compress_epilog_v2  # noqa: E402

DEV = "npu:0"
L = []
FP8_MAX = 448.0


def emit(s=""):
    L.append(str(s))


def ref_bytes(x, slots, num_blocks, cache_bs, d, stride):
    """AscendC 语义参考：返回整块 (numBlocks*stride) 的 uint8 期望值。

    注意：AscendC 是 `Mul(scale, scale, coeff)`，coeff = (float)(1/448) —— **乘**倒数而非除以 448，
    两者会差 1 ulp；这里严格用 fp32 乘法以逐位对齐。
    """
    rows = x.shape[0]
    scale_col = (d + 127) // 128
    coeff = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    buf = torch.zeros(num_blocks * stride, dtype=torch.uint8)
    xd = x.detach().cpu().float()
    for r in range(rows):
        s = int(slots[r])
        if s < 0:
            continue
        blk_num, blk_off = s // cache_bs, s % cache_bs
        v_base = blk_num * stride + blk_off * d
        s_base = blk_num * stride + cache_bs * d + blk_off * scale_col * 4
        for sb in range(scale_col):
            lo = sb * 128
            hi = min(d, lo + 128)
            seg = xd[r, lo:hi]
            mx = seg.abs().max()                       # fp32 0-dim tensor
            if float(mx) > 0.0:
                scale_t = mx * coeff                   # fp32 乘法（与 kernel 一致）
                y = seg / scale_t                      # fp32 除法（与 kernel 一致）
            else:
                scale_t = torch.tensor(0.0, dtype=torch.float32)
                y = torch.zeros_like(seg)
            buf[v_base + lo:v_base + hi] = y.to(torch.float8_e4m3fn).view(torch.uint8)
            sb_bytes = scale_t.reshape(1).view(torch.uint8)
            buf[s_base + sb * 4:s_base + sb * 4 + 4] = sb_bytes
    return buf


def run_case(name, rows, d, dtype, cache_bs=4, num_blocks=8, seed=5, with_zero_row=False):
    torch.manual_seed(seed)
    x = (torch.randn(rows, d) * 3.0).to(dtype).to(DEV)
    if with_zero_row:
        x[0] = 0                                    # 全 0 行 → scale=0，y 必须为 0
    n_slots = num_blocks * cache_bs
    slots = [(i * 3 + 1) % n_slots for i in range(rows)]
    if rows >= 7:
        slots[::7] = [-1] * len(slots[::7])          # 掺入 -1（跳过）
    slot = torch.tensor(slots, dtype=torch.int32, device=DEV)

    scale_col = (d + 127) // 128
    stride = cache_bs * d + cache_bs * scale_col * 4
    # 底层缓冲 = numBlocks*stride 字节；cache 是它的 [numBlocks, cacheBs, 1, d] 视图
    buf = torch.zeros(num_blocks * stride, dtype=torch.uint8, device=DEV)
    view4 = buf[:num_blocks * cache_bs * d].reshape(num_blocks, cache_bs, 1, d)

    # 调用点风格：block_stride 传 cache.stride(0)（真实 binding 就是这么传的）
    indexer_compress_epilog_v2(x, slot, view4, layout=2, block_stride=stride)
    torch.npu.synchronize()

    ref = ref_bytes(x, slots, num_blocks, cache_bs, d, stride)
    got = buf.detach().cpu()
    bad = int((got != ref).sum().item())
    ok = bad == 0
    line = f"  {name:50s} byte_mismatch={bad}/{ref.numel()}  {'PASS' if ok else 'FAIL'}"
    emit(line)
    print(line, flush=True)
    if bad:
        idx = (got != ref).nonzero()[:6].flatten().tolist()
        for i in idx:
            emit(f"        offset {i}: got={int(got[i])} ref={int(ref[i])}")
        print("\n".join(L[-7:]), flush=True)
    return ok


emit("=" * 96)
emit("indexer_compress_epilog_v2: Triton vs AscendC（每 128 元素动态 fp8_e4m3 + 字节布局，逐位相等）")
emit("=" * 96)

ok = True
F16 = torch.float16
BF = torch.bfloat16

ok &= run_case("fp16 rows=16 d=128 cacheBs=4 nb=8", 16, 128, F16)
ok &= run_case("bf16 rows=16 d=128 cacheBs=4 nb=8", 16, 128, BF)
ok &= run_case("fp16 rows=12 d=256 cacheBs=8 nb=6 scaleCol=2", 12, 256, F16)
ok &= run_case("fp16 rows=10 d=512 cacheBs=4 nb=8 scaleCol=4", 10, 512, F16)
ok &= run_case("bf16 rows=9  d=384 cacheBs=2 nb=8 scaleCol=3", 9, 384, BF)
ok &= run_case("fp16 rows=8  d=512 cacheBs=4 nb=8 含全0行", 8, 512, F16, with_zero_row=True)

emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/indexer_compress_epilog_v2_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
