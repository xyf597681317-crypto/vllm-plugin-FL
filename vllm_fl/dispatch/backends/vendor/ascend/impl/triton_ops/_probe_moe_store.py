# -*- coding: utf-8 -*-
"""诊断 C：_apply_quant_kernel 的 store 丢写（sentinel 检测）。

输出每个 d 的 33/64 列地图：
    '.' = 与参考一致   'X' = 仍是 sentinel（store 完全没写）   '#' = 写了但值错
用法（容器内）：python3 -W ignore _probe_moe_store.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_ops.moe_init_routing_custom import (  # noqa: E402
    _apply_quant_kernel, _rowscale_kernel,
)

DEV = os.environ.get("MOE_DEV", "npu:1")
SENT = -99


def ref_int8(x, total, k):
    rows = torch.arange(total) // k
    xv = x.detach().cpu().float()[rows]
    s = xv.abs().amax(dim=1) / 127.0
    q = xv / s[:, None]
    q16 = q.half().float()
    r = torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))
    return s, r.clamp(-128, 127).to(torch.int8)


def run(tag, cols, blk_c, blk_d, total=32, k=2, seed=0, full_map=False):
    torch.manual_seed(seed)
    n = total // k
    x = (torch.randn(n, cols, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    src = torch.arange(total, dtype=torch.int32, device=DEV)
    de = torch.zeros(total, dtype=torch.int32, device=DEV)
    smat = torch.zeros(1, dtype=torch.float32, device=DEV)
    esc = torch.zeros(total, dtype=torch.float32, device=DEV)
    xq = torch.full((total, cols), SENT, dtype=torch.int8, device=DEV)
    try:
        _rowscale_kernel[(triton.cdiv(total, blk_d),)](
            x, src, de, smat, esc, total, k, cols,
            BLK_D=blk_d, BLK_C=blk_c, HAS_SMOOTH=False)
        _apply_quant_kernel[(triton.cdiv(total, blk_d), triton.cdiv(cols, blk_c))](
            x, src, de, smat, esc, xq, total, k, cols,
            BLK_D=blk_d, BLK_C=blk_c, HAS_SMOOTH=False)
        torch.npu.synchronize()
    except Exception as ex:  # noqa: BLE001
        print(f"[{tag}] EXC {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        return
    s_r, rx = ref_int8(x, total, k)
    got = xq.detach().cpu()
    dropped = (got == SENT)
    wrong = (got != SENT) & (got.long() != rx.long())
    ok = (got.long() == rx.long())
    s_err = (esc.detach().cpu().double() - s_r.double()).abs().max().item()
    print(f"\n[{tag}] cols={cols} blk_c={blk_c} blk_d={blk_d} total={total} "
          f"grid=({triton.cdiv(total, blk_d)},{triton.cdiv(cols, blk_c)})", flush=True)
    print(f"  ok={ok.sum().item()}/{got.numel()} dropped={dropped.sum().item()} "
          f"wrong={wrong.sum().item()} scale_err={s_err:.3e}", flush=True)
    show = range(total) if full_map else range(total)
    for d in show:
        m = "".join("X" if dropped[d, c] else ("." if ok[d, c] else "#")
                    for c in range(cols))
        if "X" in m or "#" in m:
            print(f"  d={d:3d} {m}", flush=True)
    if dropped.sum().item() == 0 and wrong.sum().item() == 0:
        print("  (clean)", flush=True)


if __name__ == "__main__":
    print(f"[probe-store] dev={DEV} triton={triton.__version__}", flush=True)
    run("A pow2 cols=64 blk_c=64 (no col mask)", 64, 64, 8)
    run("B cols=32 blk_c=32 (no col mask)", 32, 32, 8)
    run("C cols=33 blk_c=64 (col mask)", 33, 64, 8)
    run("D cols=33 blk_c=64 blk_d=2", 33, 64, 2)
    run("E cols=33 blk_c=32 (col mask, 2 col blocks)", 33, 32, 8)
    run("F cols=17 blk_c=32 (col mask)", 17, 32, 8)
    run("G cols=33 blk_c=128 (col mask)", 33, 128, 2)
    run("H cols=48 blk_c=64 (col mask)", 48, 64, 8)
    print("\n[probe-store] done", flush=True)
