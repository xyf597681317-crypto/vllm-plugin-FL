# -*- coding: utf-8 -*-
"""诊断 A：动态量化(scale=None) 退化用例 x_err 的来源，以及 tile 敏感性。

打印：
  1. 公开入口结果 vs fp64 参考：逐行/逐列误差分布、失配元素明细
  2. 同一 shape 下手工指定 (BLK_C, BLK_D) 组合的误差 -> 判断是否 tiling 相关
用法（容器内）：python3 -W ignore _probe_moe_quant.py [N k cols E dtype seed]
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
    DYNAMIC_QUANT, GATHER, KEY_VALUE, NON_QUANT,
    _apply_quant_kernel, _rowscale_kernel, moe_init_routing_custom,
)

DEV = os.environ.get("MOE_DEV", "npu:1")
TAG = os.environ.get("MOE_TAG", "run")
print(f"[probe-quant] torch={torch.__version__} triton={triton.__version__} dev={DEV}", flush=True)


def setup(N, k, cols, E, dtype, seed):
    torch.manual_seed(seed)
    x = (torch.randn(N, cols, dtype=torch.float32, device=DEV) * 2.0).to(dtype)
    ei = torch.randint(0, E, (N, k), dtype=torch.int32, device=DEV)
    return x, ei


def ref_quant(x, ei, k):
    """fp32 复刻参考：返回 (order, s, xv, q16, int8)。"""
    e = ei.reshape(-1).long().cpu()
    order = torch.argsort(e, stable=True)
    row = order // k
    xv = x.detach().cpu().float()[row]
    s = xv.abs().amax(dim=1) / 127.0
    q = xv / s[:, None]
    q16 = q.half().float()
    r = torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))
    return order, s, xv, q16, r.clamp(-128, 127).to(torch.int8)


def diag(N, k, cols, E, dtype, seed):
    print(f"\n===== diag N={N} k={k} cols={cols} E={E} dtype={dtype} =====", flush=True)
    x, ei = setup(N, k, cols, E, dtype, seed)
    ex, ridx, cnt, esc = moe_init_routing_custom(
        x, ei, None, None, expert_num=E, expert_tokens_num_type=KEY_VALUE,
        expert_tokens_num_flag=True, quant_mode=DYNAMIC_QUANT, row_idx_type=GATHER)
    torch.npu.synchronize()
    order, s, xv, q16, rx = ref_quant(x, ei, k)
    total = N * k

    # 1) 路由映射是否一致
    dstof = torch.empty(total, dtype=torch.long)
    dstof[order] = torch.arange(total)
    print(f"rowidx(GATHER) map diff = {(ridx.detach().cpu().long() - dstof).abs().max().item()}",
          flush=True)

    # 2) scale 对比
    s_k = esc.detach().cpu().double()
    s_r = s.double()
    print(f"scale: max|diff|={ (s_k - s_r).abs().max().item():.3e} "
          f"nan_k={torch.isnan(s_k).sum().item()} nan_r={torch.isnan(s_r).sum().item()} "
          f"zero_k={(s_k == 0).sum().item()} zero_r={(s_r == 0).sum().item()}", flush=True)
    bad_s = (s_k - s_r).abs() > 1e-9
    if bad_s.any():
        idx = torch.nonzero(bad_s).reshape(-1).tolist()
        for d in idx[:8]:
            print(f"  scale mismatch d={d} kernel={s_k[d].item():.9g} ref={s_r[d].item():.9g} "
                  f"amax_k={s_k[d].item() * 127:.6g} amax_r={s_r[d].item() * 127:.6g}", flush=True)

    # 3) 量化值对比
    kern = ex.detach().cpu().long()
    rl = rx.long()
    diff = kern - rl
    print(f"x_err(max)={diff.abs().max().item()} n_mismatch={(diff != 0).sum().item()}/{diff.numel()}",
          flush=True)
    print(f"per-row max|diff| = {diff.abs().amax(dim=1).tolist()}", flush=True)
    print(f"per-row #mismatch = {(diff != 0).sum(dim=1).tolist()}", flush=True)

    # 4) 行内原始数据（最差行 / 有失配的行）
    worst_rows = torch.nonzero(diff.abs().amax(dim=1) > 0).reshape(-1).tolist()
    for d in worst_rows[:4]:
        print(f"  --- d={d} src={order[d].item()} row={order[d].item() // k} amax={s_r[d].item() * 127:.6g} s={s_r[d].item():.9g}", flush=True)
        print(f"      x_raw(fp32): {[round(v, 5) for v in xv[d].tolist()]}", flush=True)
        print(f"      ref  int8  : {rl[d].tolist()}", flush=True)
        print(f"      kern int8  : {kern[d].tolist()}", flush=True)
        print(f"      q16 (ref)  : {[round(v, 4) for v in q16[d].tolist()]}", flush=True)
    return x, ei, order, s, rx


def scan_tiles(N, k, cols, E, dtype, seed, tiles):
    print(f"\n===== tile scan N={N} k={k} cols={cols} dtype={dtype} =====", flush=True)
    x, ei = setup(N, k, cols, E, dtype, seed)
    order, s, xv, q16, rx = ref_quant(x, ei, k)
    total = N * k
    src = order.to(torch.int32).to(DEV)
    de = ei.reshape(-1).long()[order].to(torch.int32).to(DEV)
    smat = torch.zeros(1, dtype=torch.float32, device=DEV)
    for blk_c, blk_d in tiles:
        try:
            esc = torch.full((total,), -777.0, dtype=torch.float32, device=DEV)
            xq = torch.full((total, cols), -99, dtype=torch.int8, device=DEV)
            _rowscale_kernel[(triton.cdiv(total, blk_d),)](
                x, src, de, smat, esc, total, k, cols,
                BLK_D=blk_d, BLK_C=blk_c, HAS_SMOOTH=False)
            _apply_quant_kernel[(triton.cdiv(total, blk_d), triton.cdiv(cols, blk_c))](
                x, src, de, smat, esc, xq, total, k, cols,
                BLK_D=blk_d, BLK_C=blk_c, HAS_SMOOTH=False)
            torch.npu.synchronize()
            d = (xq.cpu().long() - rx.long()).abs().max().item()
            nz = (xq.cpu().long() != rx.long()).sum().item()
            ds = (esc.cpu().double() - s.double()).abs().max().item()
            print(f"  tile blk_c={blk_c:4d} blk_d={blk_d:3d}: x_err={d:4d} "
                  f"n_mismatch={nz:4d} scale_err={ds:.3e}", flush=True)
        except Exception as ex:  # noqa: BLE001
            msg = str(ex).replace("\n", " ")[:180]
            print(f"  tile blk_c={blk_c:4d} blk_d={blk_d:3d}: EXC {type(ex).__name__}: {msg}", flush=True)


if __name__ == "__main__":
    a = [int(v) for v in sys.argv[1:]]
    N, k, cols, E, seed = (a + [16, 2, 33, 4, 0])[:5]
    dtype = torch.float16
    diag(N, k, cols, E, dtype, seed)
    scan_tiles(N, k, cols, E, dtype, seed,
               [(32, 2), (32, 4), (64, 2), (64, 4), (64, 8), (64, 16), (128, 2), (128, 4)])
    print("\n[probe-quant] done", flush=True)
