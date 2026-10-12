# -*- coding: utf-8 -*-
"""compressor(coff=2)：**AscendC vs Triton vs 两个 fp64 参考** 同输入三方对照。

判定规则（Lead 指定）：
  asc ≈ tri 但都 ≠ refA/refB   => 参考口径错（改参考，不动 kernel）
  asc ≈ refA/refB 而 tri 不同  => kernel 错（按 AscendC 修）
  三者互不相同                 => 缩小到最小可辨识例

AscendC 侧：`torch.ops._C_ascend.compressor`（需 VLLM_FL_TRITON_OPS=0 启动，走厂商实现）
Triton 侧 ：直接 import `triton_ops.compressor.compressor`（与本进程的调度无关）
参考侧    ：从 test_compressor_coff2.py 源码里抽出 ref_A / ref_B / _st_blk / _bit 原样执行

用法：MOE_DEV=npu:2 VLLM_FL_TRITON_OPS=0 python3 -W ignore trops/_probe_compressor_ab.py
"""
import math
import os
import re
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("MOE_DEV", "npu:2")
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
TEST = os.path.join(SRC, "triton_ops", "test_compressor_coff2.py")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from triton_ops.compressor import compressor as tri_compressor  # noqa: E402

print(f"[ab] VLLM_FL_TRITON_OPS={os.environ.get('VLLM_FL_TRITON_OPS', '<unset>')} "
      f"dev={DEV} has_asc_op={hasattr(torch.ops._C_ascend, 'compressor')}", flush=True)


def build(B, S, H, D, R, RD, dtype, seed=0):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    Sr = (S + R - 1) // R
    rs = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    return x, wkv, wgate, ape, nw, rs, rc


def load_refs():
    src = open(TEST, encoding="utf-8").read()
    ns = {"torch": torch, "math": math, "MIN_SCORE": -1.0e30, "NEG_INF": float("-inf")}
    for name in ("_st_blk", "_bit", "ref_A", "ref_B"):
        m = re.search(r"^def %s\(.*?(?=^def |^# ---|\Z)" % name, src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns


def case(name, B, S, H, D, R, RD, dtype, rotary_mode=2, seed=0, ns=None):
    x, wkv, wgate, ape, nw, rs, rc = build(B, S, H, D, R, RD, dtype, seed)
    Sr = rs.shape[1]
    nblk = (S + 15) // 16
    stride0 = 16 * 4 * D
    sc = torch.zeros(nblk, 16, 4 * D, dtype=torch.float32, device=DEV)
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV).reshape(1, nblk)
    sp = torch.zeros(1, dtype=torch.int32, device=DEV)
    su = torch.tensor([S], dtype=torch.int32, device=DEV)

    # ---- AscendC（本进程的 torch.ops._C_ascend.compressor，取决于 VLLM_FL_TRITON_OPS）----
    asc = None
    try:
        asc = torch.ops._C_ascend.compressor(
            x, wkv, wgate, sc.clone(), ape, nw, rs, rc,
            state_block_table=bt, cu_seqlens=None, seqused=su, start_pos=sp,
            rope_head_dim=RD, cmp_ratio=R, coff=2, norm_eps=1e-6,
            rotary_mode=rotary_mode, cache_mode=1)
        torch.npu.synchronize()
    except Exception as ex:  # noqa: BLE001
        print(f"  [asc] EXC {type(ex).__name__}: {str(ex)[:200]}", flush=True)

    # ---- Triton（直接 import）----
    tri, _ = tri_compressor(
        x, wkv, wgate, ape, nw, rs, rc, state_cache=sc.clone(),
        state_block_table=bt, cu_seqlens=None, seqused=su, start_pos=sp,
        rope_head_dim=RD, cmp_ratio=R, coff=2, norm_eps=1e-6,
        rotary_mode=rotary_mode, cache_mode=1,
        state_cache_stride_dim0=stride0, block_size=16)
    torch.npu.synchronize()

    # ---- 两个 fp64 参考（与 test 内 case() 对 BS 的传参完全一致）----
    sp_l, su_l = [0] * B, [S] * B
    x_base = [b * S for b in range(B)]
    c0_l, nc_l = [0] * B, [S // R] * B
    ob_l = [b * Sr for b in range(B)]
    common = (R, D, RD, 1e-6, rotary_mode == 1, None, [[0]], 1, 0, 1, False, (B, Sr, D), True)
    rA, _ = ns["ref_A"](x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)
    rB, _ = ns["ref_B"](x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)

    def dd(t):
        return t.detach().cpu().double().reshape(-1)

    tri_d, rA_d, rB_d = dd(tri), dd(rA), dd(rB)
    line = f"  {name:42s} |tri-refA|={float((tri_d - rA_d).abs().max()):.3e} " \
           f"|tri-refB|={float((tri_d - rB_d).abs().max()):.3e} |refA-refB|={float((rA_d - rB_d).abs().max()):.2e} "
    if asc is not None:
        asc_d = dd(asc)
        line += (f"\n      |asc-tri|={float((asc_d - tri_d).abs().max()):.3e} "
                 f"|asc-refA|={float((asc_d - rA_d).abs().max()):.3e} "
                 f"|asc-refB|={float((asc_d - rB_d).abs().max()):.3e} "
                 f"max|asc|={float(asc_d.abs().max()):.3f} max|tri|={float(tri_d.abs().max()):.3f} "
                 f"max|refA|={float(rA_d.abs().max()):.3f}")
    print(line, flush=True)


if __name__ == "__main__":
    ns = load_refs()
    print("\n== coff=2 三方对照（单次调用 + 零 state_cache + block_table）==", flush=True)
    case("D=128 H=64 R=4 RD=64 bf16 rm=2 (复测失败例)", 1, 24, 64, 128, 4, 64,
         torch.bfloat16, rotary_mode=2, ns=ns)
    case("D=128 H=64 R=4 RD=64 fp16 rm=1 (复测失败例)", 1, 24, 64, 128, 4, 64,
         torch.float16, rotary_mode=1, seed=1, ns=ns)
    case("D=512 H=256 R=4 RD=64 bf16 rm=2 (生产组合)", 1, 24, 256, 512, 4, 64,
         torch.bfloat16, rotary_mode=2, seed=7, ns=ns)
    print("\n[ab] done", flush=True)
