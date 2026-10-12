# -*- coding: utf-8 -*-
"""定位探针：SB（内部 score 缓冲）逐 token 统计 + out 逐 token 统计。

回答三个问题：
  1) 哪些 (token, head) 行的 SB 是 nan / 全 NEG？
  2) 这些行的 out 是不是全 nan？
  3) out 的 nan 是 lse=0（除零）还是 SB 本身 nan（污染）？

用法（容器内 /workspace/FlagGems/src）：
    DSH_NPU=npu:3 python3 -W ignore triton_ops/_probe_saskv_loc.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_ops.sparse_attn_sharedkv as S  # noqa: E402
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = os.environ.get("SASKV_DEV", "npu:0")
D = 512
NEG = S._NEG


def make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size=16, seed=81):
    torch.manual_seed(seed)
    B = len(qlens)
    T = sum(qlens)
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc).to(dtype)
    geo = {}
    for tag, lens in (("ori", kvolens), ("cmp", kclens)):
        mb = max((l + block_size - 1) // block_size for l in lens)
        nb = B * mb
        kv = (torch.randn(nb, block_size, N2, D, device=DEV) * 0.6).to(dtype)
        bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
        for b in range(B):
            bt[b] = (torch.randperm(mb, device=DEV) + b * mb).to(torch.int32)
        geo[tag] = (kv, bt, mb)
    cidx = torch.empty(T, N2, K, dtype=torch.int32, device=DEV)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    for t in range(T):
        b = int((cu[1:] <= t).sum().item())
        for n2 in range(N2):
            cidx[t, n2] = torch.randperm(max(1, kclens[b]), device=DEV)[:K].to(torch.int32)
    return q, geo, cidx, cu, T


def one(tag, qlens, kvolens, kclens, N1, N2, K, dtype, block_size=16, sinks=True, cmp_ratio=1):
    print("=" * 100, flush=True)
    print(f"[{tag}] qlens={qlens} kvolens={kvolens} kclens={kclens} N1={N1} N2={N2} K={K} "
          f"{dtype} bs={block_size} cmp_ratio={cmp_ratio}", flush=True)
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    S._DBG.clear()
    os.environ["SASKV_DUMP_SCORES"] = "1"
    out, lmx, lsm = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
        sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
        cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    sb = S._DBG["sb"].detach().cpu().float()
    o = out.detach().cpu().float()
    n_ori = 128
    k_idx = K
    print(f"  sb shape={tuple(sb.shape)} out shape={tuple(o.shape)}", flush=True)
    print(f"  block_table ori={geo['ori'][1].cpu().tolist()} cmp={geo['cmp'][1].cpu().tolist()}", flush=True)
    print(f"  cidx[:3]={cidx[:3].cpu().tolist()}", flush=True)
    # 逐 token 汇总
    print("   t  b  s1 |  sb_nan  sb_ori_nan sb_cmp_nan  sb_ori_max  sb_cmp_max | out_nan  out_absmax", flush=True)
    for t in range(T):
        for n1 in (0,):
            row = t * N1 + n1
            sbr = sb[row]
            so = sbr[:n_ori]
            scmp = sbr[n_ori:n_ori + k_idx]
            or_ = o[t, n1]
            b = 0 if t < qlens[0] else 1
            s1 = t - (0 if b == 0 else qlens[0])
            print(f"  {t:2d}  {b}  {s1:2d} | {int(torch.isnan(sbr).sum()):7d} "
                  f"{int(torch.isnan(so).sum()):9d} {int(torch.isnan(scmp).sum()):9d}  "
                  f"{float(so[~torch.isnan(so)].max()) if not torch.isnan(so).all() else float('nan'):10.3f}  "
                  f"{float(scmp[~torch.isnan(scmp)].max()) if not torch.isnan(scmp).all() else float('nan'):10.3f} | "
                  f"{int(torch.isnan(or_).sum()):7d}  {float(or_[~torch.isnan(or_)].abs().max()) if not torch.isnan(or_).all() else float('nan'):.4e}",
                  flush=True)
    # row0/row4 的 ori 段前若干 lane
    for row in (0, 16):
        if row >= sb.shape[0]:
            continue
        v = sb[row]
        nz = torch.nonzero(v[:n_ori] != NEG).flatten()
        print(f"  row{row} ori: n_valid_lane={nz.numel()} first={int(nz[0]) if nz.numel() else -1} "
              f"last={int(nz[-1]) if nz.numel() else -1}", flush=True)
        seg = v[:20]
        print(f"    lanes0-19 = {[('NEG' if abs(float(x) - NEG) < 1 else round(float(x),4)) for x in seg]}", flush=True)
        print(f"    lanes60-75 = {[('NEG' if abs(float(x) - NEG) < 1 else round(float(x),4)) for x in v[60:76]]}",
              flush=True)
    del os.environ["SASKV_DUMP_SCORES"]


if __name__ == "__main__":
    which = os.environ.get("SASKV_LOC", "all")
    if which in ("all", "b2"):
        one("B2 [5,4] kvo[64,48] kvc[24,16] N1=4 fp16", [5, 4], [64, 48], [24, 16], 4, 1, 8,
            torch.float16)
    if which in ("all", "b1"):
        one("B1 [6] kvo[64] kvc[24] N1=4 fp16", [6], [64], [24], 4, 1, 8, torch.float16)
    if which in ("all", "b1s"):
        one("B1 [2] kvo[64] kvc[24] N1=4 fp16 (短 q)", [2], [64], [24], 4, 1, 8, torch.float16)
    print("LOCDONE", flush=True)
