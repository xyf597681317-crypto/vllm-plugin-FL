# -*- coding: utf-8 -*-
"""compressor(coff=2) 失败定位：**输入退化 bisect**（kernel vs ref_A/ref_B，无 state）。

思路：把 score 链路的输入逐项归零，看 kernel 与参考的偏差在哪一步出现：
  ① wgate=0, ape=0  -> score≡0，softmax 均匀：只考「投影 + 2r 窗口 + 通道分半 + RMSNorm + RoPE」
  ② wgate=0, ape≠0  -> score=ape（与 x 无关）
  ③ wgate≠0, ape=0  -> score=x@wgate
  ④ 全随机         -> 完整链路
并分别打印「非 rope 区 / rope 区」误差，定位是 rope 前还是 rope 后。

用法：MOE_DEV=npu:3 python3 -W ignore triton_ops/_probe_compressor_bisect.py
"""
import math
import os
import re
import sys

import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("MOE_DEV", "npu:3")
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
TEST = os.path.join(SRC, "triton_ops", "test_compressor_coff2.py")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from triton_ops.compressor import compressor as tri_compressor  # noqa: E402


def load_refs():
    src = open(TEST, encoding="utf-8").read()
    ns = {"torch": torch, "math": math, "MIN_SCORE": -1.0e30, "NEG_INF": float("-inf")}
    for name in ("_st_blk", "_bit", "ref_A", "ref_B"):
        m = re.search(r"^def %s\(.*?(?=^def |^# ---|\Z)" % name, src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns


def run(tag, zero_wgate, zero_ape, B=1, S=24, H=64, D=128, R=4, RD=64,
        dtype=torch.bfloat16, rotary_mode=2, seed=0, ns=None):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    x = (torch.randn(B, S, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wkv = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    wgate = (torch.randn(2 * D, H, dtype=torch.float32, device=DEV) * scale).to(dtype)
    if zero_wgate:
        wgate = torch.zeros_like(wgate)
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=DEV) * 0.1
    if zero_ape:
        ape = torch.zeros_like(ape)
    nw = torch.rand(D, dtype=torch.float32, device=DEV) * 0.4 + 0.2
    Sr = (S + R - 1) // R
    rs = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1
    rc = torch.randn(B, Sr, RD, dtype=torch.float32, device=DEV) * 0.1

    out, _ = tri_compressor(x, wkv, wgate, ape, nw, rs, rc,
                            rope_head_dim=RD, cmp_ratio=R, coff=2,
                            rotary_mode=rotary_mode)
    torch.npu.synchronize()
    sp_l, su_l = [0] * B, [S] * B
    x_base = [b * S for b in range(B)]
    c0_l, nc_l = [0] * B, [S // R] * B
    ob_l = [b * Sr for b in range(B)]
    common = (R, D, RD, 1e-6, rotary_mode == 1, None, [[0]], 1, 0, 1, False, (B, Sr, D), True)
    rA, _ = ns["ref_A"](x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)
    rB, _ = ns["ref_B"](x, wkv, wgate, ape, nw, rs, rc, sp_l, su_l, x_base, c0_l, nc_l, ob_l, *common)

    o = out.detach().cpu().double().reshape(-1, D)
    a = rA.detach().cpu().double().reshape(-1, D)
    b = rB.detach().cpu().double().reshape(-1, D)
    eA = (o - a).abs()
    nr = float(eA[:, :D - RD].max()) if D > RD else float("nan")
    rp = float(eA[:, D - RD:].max())
    print(f"  {tag:34s} |tri-refA|={float(eA.max()):.3e} |tri-refB|={float((o-b).abs().max()):.3e} "
          f"|A-B|={float((a-b).abs().max()):.2e} [非rope={nr:.2e} rope={rp:.2e}] "
          f"max|ref|={float(a.abs().max()):.3f}", flush=True)
    if float(eA.max()) > 1e-6 and D > RD:
        # 首个偏差出现的位置（行/列）
        idx = (o - a).abs().argmax().item()
        r_, c_ = divmod(idx, D)
        print(f"      worst @ row={r_} col={c_} tri={float(o[r_, c_]):.6f} refA={float(a[r_, c_]):.6f}", flush=True)


if __name__ == "__main__":
    ns = load_refs()
    print(f"[bisect] dev={DEV} triton=import ok", flush=True)
    print("== coff=2 D=128 H=64 R=4 RD=64 bf16 rotary_mode=2 ==", flush=True)
    run("① wgate=0 ape=0 (uniform softmax)", True, True, ns=ns)
    run("② wgate=0 ape≠0", True, False, ns=ns)
    run("③ wgate≠0 ape=0", False, True, ns=ns)
    run("④ 全随机", False, False, ns=ns)
    print("== 同组但 rotary_mode=1 (HALF) ==", flush=True)
    run("④H 全随机 rm=1", False, False, rotary_mode=1, ns=ns)
    print("== 生产参数 D=512 H=256 R=4 RD=64 bf16 rm=2 ==", flush=True)
    run("④prod 全随机 D=512", False, False, H=256, D=512, ns=ns)
    print("\n[bisect] done", flush=True)
