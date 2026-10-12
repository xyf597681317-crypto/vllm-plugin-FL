"""验证 compressor_metadata 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import importlib
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")   # 容器内可用独立包名隔离
_M = importlib.import_module(_PKG + ".compressor_metadata")
SLOT_MAPPING_BLOCK_OFFSET = _M.SLOT_MAPPING_BLOCK_OFFSET
SLOT_MAPPING_FLAT = _M.SLOT_MAPPING_FLAT
compressor_metadata = _M.compressor_metadata

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
INT32_MAX = 2147483647
L = []


def emit(s=""):
    L.append(str(s))


def build_case(seq_lens, start_positions, cmp_ratio):
    """由每个请求的 (seqLen, startPos) 造 cuSeqlens / startPos。"""
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return np.asarray(cu, dtype=np.int64), np.asarray(start_positions, dtype=np.int64)


def ref_impl(rope_cos, rope_sin, cu, sp, bt, num_rows, kv_block_size,
             slot_format, cmp_ratio, actual, tile_rows):
    """严格按 compressor_metadata.h 的 Process/ProcessTile 顺序线性模拟（fp64）。"""
    rc = rope_cos.detach().cpu().double().numpy()
    rs = rope_sin.detach().cpu().double().numpy()
    bt = bt.detach().cpu().numpy().astype(np.int64)
    rope_rows, rope_dim = rc.shape
    stride = bt.shape[1]

    prefix = [0] * (actual + 1)
    for i in range(actual):
        seq = int(cu[i + 1] - cu[i])
        cnt = 0
        if sp[i] >= 0 and seq > 0:
            cnt = int((sp[i] + seq) // cmp_ratio) - int(sp[i] // cmp_ratio)
        prefix[i + 1] = prefix[i] + cnt
    valid_rows = min(prefix[actual], num_rows)

    cos = np.ones((num_rows, rope_dim), dtype=np.float64)
    sin = np.zeros((num_rows, rope_dim), dtype=np.float64)
    slot = np.full((num_rows, 2), -1, dtype=np.int64)

    def write_invalid(o, rows):
        for k in range(rows):
            cos[o + k] = 1.0
            sin[o + k] = 0.0
            slot[o + k] = (-1, kv_block_size - 1) if slot_format == 2 else (-1, -1)

    r = 0
    req = 0
    while r < valid_rows and req < actual:
        rb, re = prefix[req], prefix[req + 1]
        if r >= re:
            req += 1
            continue
        rows_in_req = min(valid_rows - r, re - r)
        comp0 = int(sp[req] // cmp_ratio)
        cpos = comp0 + (r - rb)
        out_row = r
        remaining = rows_in_req
        while remaining > 0:
            blk_off = cpos % kv_block_size
            rows_to_end = kv_block_size - blk_off
            cur = min(remaining, tile_rows, rows_to_end)
            blk_idx = cpos // kv_block_size
            ok = True
            bid = -1
            if blk_idx >= stride:
                ok = False
            else:
                bid = int(bt[req, blk_idx])
                if bid < 0:
                    ok = False
                elif slot_format == 1 and bid * kv_block_size + blk_off + cur - 1 > INT32_MAX:
                    ok = False
            if ok and (cpos + cur - 1) * cmp_ratio >= rope_rows:
                ok = False
            if ok:
                for k in range(cur):
                    cos[out_row + k] = rc[(cpos + k) * cmp_ratio]
                    sin[out_row + k] = rs[(cpos + k) * cmp_ratio]
                    if slot_format == 1:
                        slot[out_row + k] = (bid * kv_block_size + blk_off + k, 0)
                    else:
                        slot[out_row + k] = (bid, blk_off + k)
            else:
                write_invalid(out_row, cur)
            out_row += cur
            cpos += cur
            remaining -= cur
        r += rows_in_req
        req += 1
    if valid_rows < num_rows:
        write_invalid(valid_rows, num_rows - valid_rows)
    return cos, sin, slot


def make_rope(rope_rows, rope_dim, dtype):
    p = np.arange(rope_rows, dtype=np.float64)[:, None]
    d = np.arange(rope_dim, dtype=np.float64)[None, :]
    c = ((p * 37.0 + d * 3.0) % 97.0) / 97.0 - 0.5
    s = ((p * 11.0 + d * 5.0) % 89.0) / 89.0 - 0.5
    return (torch.tensor(c, dtype=torch.float64).to(dtype).to(DEV),
            torch.tensor(s, dtype=torch.float64).to(dtype).to(DEV))


def run_case(name, seq_lens, start_positions, num_rows, rope_dim, dtype,
             kv_block_size=8, cmp_ratio=4, slot_format=SLOT_MAPPING_FLAT,
             actual=None, kv_stride=None, bad_blocks=(), tile_rows=512,
             extra_rope_rows=64, rope_rows_override=None):
    cu, sp = build_case(seq_lens, start_positions, cmp_ratio)
    nreq = len(seq_lens)
    if actual is None:
        actual = nreq
    # 估算需要的 rope 行数：max((compressedPos+cnt)*cmpRatio)
    need = 1
    for i in range(actual):
        seq = int(cu[i + 1] - cu[i])
        if sp[i] >= 0 and seq > 0:
            need = max(need, int((sp[i] + seq) // cmp_ratio) * cmp_ratio + 1)
    rope_rows = max(need, int((sp[:actual].max() if actual else 0) // cmp_ratio) + 1) + extra_rope_rows
    if rope_rows_override is not None:
        rope_rows = rope_rows_override
    if kv_stride is None:
        kv_stride = max(1, (rope_rows // cmp_ratio) // kv_block_size + 1)

    gen = torch.Generator().manual_seed(1234)
    bt = torch.randint(0, 40, (nreq, kv_stride), generator=gen, dtype=torch.int32)
    for rr, cc in bad_blocks:
        bt[rr, cc] = -1
    bt = bt.to(DEV)

    rc, rs = make_rope(rope_rows, rope_dim, dtype)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=DEV)
    sp_t = torch.tensor(sp, dtype=torch.int32, device=DEV)

    cc, cs, sm = compressor_metadata(
        rc, rs, cu_t, sp_t, bt, num_rows, kv_block_size, slot_format,
        cmp_ratio, actual, tile_rows=tile_rows)
    torch.npu.synchronize()

    rcos, rsin, rslot = ref_impl(rc, rs, cu, sp, bt, num_rows, kv_block_size,
                                 slot_format, cmp_ratio, actual, tile_rows)

    e_cos = float(np.abs(cc.detach().cpu().double().numpy() - rcos).max())
    e_sin = float(np.abs(cs.detach().cpu().double().numpy() - rsin).max())
    got_slot = sm.detach().cpu().numpy().astype(np.int64).reshape(num_rows, -1)
    e_slot = float(np.abs(got_slot - rslot[:, : got_slot.shape[1]]).max())
    slot_ok = bool((got_slot == rslot[:, : got_slot.shape[1]]).all())

    floor = {torch.float32: 2e-6, torch.float16: 1e-3, torch.bfloat16: 4e-3}[dtype]
    n_invalid = int((rslot[:, 0] == -1).sum())
    ok = e_cos <= floor and e_sin <= floor and slot_ok
    emit(f"  {name:44s} cos_err={e_cos:.1e} sin_err={e_sin:.1e} "
         f"slot={'EQ' if slot_ok else 'NE(%g)' % e_slot}(<= {floor:.0e}) "
         f"invalid_rows={n_invalid:3d}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 100)
emit("compressor_metadata: Triton vs AscendC 语义（fp64 参考；slotMapping 必须完全相等）")
emit("=" * 100)

allok = True
# 1) flat / fp32 / 多请求 / 全部有效
allok &= run_case("flat f32 cmp=4 kv=8 全有效", [40, 17, 64, 8], [0, 12, 100, 260], 64, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4,
                  extra_rope_rows=128)
# 2) block_offset / bf16
allok &= run_case("block_offset bf16 cmp=4 kv=8", [40, 17, 64, 8], [0, 12, 100, 260], 64, 8,
                  torch.bfloat16, kv_block_size=8, cmp_ratio=4,
                  slot_format=SLOT_MAPPING_BLOCK_OFFSET, extra_rope_rows=128)
# 3) fp16 + 非 2 幂 ropeDim
allok &= run_case("flat f16 ropeDim=6 非2幂", [33, 21, 9], [3, 40, 77], 30, 6,
                  torch.float16, kv_block_size=8, cmp_ratio=4,
                  extra_rope_rows=128)
# 4) 部分请求 blockId=-1 -> 整块无效
allok &= run_case("flat f32 含 blockId=-1", [40, 17, 64, 8], [0, 12, 100, 260], 64, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4,
                  bad_blocks=[(1, 0), (2, 4)], extra_rope_rows=128)
# 5) blockIdOffset >= stride -> 无效
allok &= run_case("flat f32 blockIdx 越界(stride 小)", [40, 17, 64], [0, 12, 300], 40, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4, kv_stride=4,
                  extra_rope_rows=128)
# 6) numRows > validRows -> 尾部 padding 无效填充
allok &= run_case("flat f32 numRows>validRows(尾部填充)", [24, 9], [5, 61], 40, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4, extra_rope_rows=128)
# 7) actualNumReqs < numReqs
allok &= run_case("flat f32 actualNumReqs=2/4", [40, 17, 64, 8], [0, 12, 100, 260], 32, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4, actual=2, extra_rope_rows=128)
# 8) cmpRatio=2 (非整除起点) + startPos 非对齐
allok &= run_case("flat bf16 cmp=2 起点非对齐", [23, 41, 7], [1, 9, 128], 40, 8,
                  torch.bfloat16, kv_block_size=8, cmp_ratio=2, extra_rope_rows=128)
# 9) tileRows < kvBlockSize（UB 受限时的真实切块）
allok &= run_case("flat f32 tileRows=4<kvBlockSize", [30, 12], [0, 70], 24, 8,
                  torch.float32, kv_block_size=16, cmp_ratio=4, tile_rows=4,
                  extra_rope_rows=128)
# 10) 大 kv_block_size / cmp_ratio=1
allok &= run_case("flat f32 cmp=1 kv=128", [50, 33], [0, 128], 90, 8,
                  torch.float32, kv_block_size=128, cmp_ratio=1, extra_rope_rows=64)
# 11) startPos=-1 的请求（贡献 0 行）
allok &= run_case("flat f32 startPos=-1 请求", [16, 16], [-1, 8], 24, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4, extra_rope_rows=64)
# 12) rope 表偏短 -> 尾部 chunk 因 rope 越界整块无效
allok &= run_case("flat f32 rope 表偏短(越界无效)", [24, 16], [0, 12], 12, 8,
                  torch.float32, kv_block_size=8, cmp_ratio=4, rope_rows_override=21)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/compressor_metadata_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
