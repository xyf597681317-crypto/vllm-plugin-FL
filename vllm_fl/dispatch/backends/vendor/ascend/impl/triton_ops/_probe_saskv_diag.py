# -*- coding: utf-8 -*-
"""Probe3: 精确诊断 TND+PA 的 (a) cmp 上界 (b) 全分支 NaN。

跑法：SASKV_DUMP_SCORES=1 DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_diag.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
os.environ["SASKV_DUMP_SCORES"] = "1"
import triton_ops.sparse_attn_sharedkv as M  # noqa: E402
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
D = 512
BLOCK = 16
FP16 = torch.float16


def make(qlens, kvolens, kclens, N1, N2, K, dtype, block_size=BLOCK, seed=81):
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


def dense_of(geo, tag, b):
    kv, bt, mb = geo[tag]
    KD = kv.detach().cpu().double()
    BT = bt.detach().cpu()
    bs = KD.shape[1]
    dd = torch.zeros(mb * bs, KD.shape[2], D, dtype=torch.float64)
    for lb in range(mb):
        dd[lb * bs:(lb + 1) * bs] = KD[int(BT[b, lb])]
    return dd


def ref_scores(q, geo, cidx, cu, qlens, kvolens, kclens, scale, win_left, win_right,
               cmp_ratio, with_ori, with_cmp):
    Q = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, _ = Q.shape
    gs = N1 // IDX.shape[1]
    n_ori = (win_left + win_right + 1) if with_ori else 0
    K = IDX.shape[-1] if with_cmp else 0
    total = n_ori + K
    sc = torch.full((T, N1, total), float("-inf"), dtype=torch.float64)
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        for n1 in range(N1):
            n2 = n1 // gs
            if with_ori:
                dd = dense_of(geo, "ori", b)
                kl = kvolens[b]
                s1_abs = s1 + (kl - qlens[b])
                for col, j in enumerate(range(s1_abs - win_left, s1_abs + win_right + 1)):
                    if 0 <= j < kl:
                        sc[t, n1, col] = float(Q[t, n1] @ dd[j, n2])
            if with_cmp:
                dd = dense_of(geo, "cmp", b)
                n_valid = min(kclens[b], (s1 + (kvolens[b] - qlens[b]) + 1) // cmp_ratio)
                for slot in range(K):
                    i0 = int(IDX[t, n2, slot])
                    if 0 <= i0 < n_valid:
                        sc[t, n1, n_ori + slot] = float(Q[t, n1] @ dd[i0, n2])
    return sc


def go(tag, qlens, kvolens, kclens, N1, N2, K, win_left=127, win_right=0, cmp_ratio=1,
       with_ori=True, with_cmp=True, sinks=True, show_rows=0, blk_k=None):
    if blk_k is not None:
        os.environ["SASKV_BLK_K"] = str(blk_k)
    else:
        os.environ.pop("SASKV_BLK_K", None)
    q, geo, cidx, cu, T = make(qlens, kvolens, kclens, N1, N2, K, FP16)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    M._DBG.clear()
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0] if with_ori else None,
        cmp_kv=geo["cmp"][0] if with_cmp else None,
        cmp_sparse_indices=cidx if with_cmp else None,
        sinks=sk, softmax_scale=1.0, ori_win_left=win_left, ori_win_right=win_right,
        cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1] if with_ori else None,
        cmp_block_table=geo["cmp"][1] if with_cmp else None,
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    sb = M._DBG.get("sb")
    o = out.detach().cpu().double()
    print("=" * 110, flush=True)
    print(f"## {tag} | win_left={win_left} cmp_ratio={cmp_ratio} ori={with_ori} cmp={with_cmp}"
          f" blk_k={os.environ.get('SASKV_BLK_K', '8')}", flush=True)
    print(f"  out nan={int(torch.isnan(o).sum())} inf={int(torch.isinf(o).sum())}"
          f" |out|max={o[torch.isfinite(o)].abs().max().item():.4f}", flush=True)
    if sb is None or not with_ori:
        return
    rsc = ref_scores(q, geo, cidx, cu, qlens, kvolens, kclens, 1.0, win_left, win_right,
                     cmp_ratio, with_ori, with_cmp)
    S = sb.detach().cpu().double().reshape(T, N1, -1)
    if S.shape[-1] != rsc.shape[-1]:
        print(f"  !! width mismatch kernel={S.shape[-1]} ref={rsc.shape[-1]}", flush=True)
        return
    nan_mask = torch.isnan(S)
    print(f"  score nan={int(nan_mask.sum())} at {nan_mask.nonzero()[:5].tolist()}"
          f"  isinf={int(torch.isinf(S).sum())}", flush=True)
    valid = torch.isfinite(rsc)
    nvalid_tri = (S > -1e29).sum(-1)
    nvalid_ref = valid.sum(-1)
    diff = (nvalid_tri != nvalid_ref)
    print(f"  rows w/ count mismatch: {int(diff.sum())}/{T * N1}", flush=True)
    # 合法列上的数值误差
    fin = valid & torch.isfinite(S)
    sd = (S - rsc)[fin]
    print(f"  score err on ref-valid cols: |max|={sd.abs().max().item():.3e} (n={sd.numel()})",
          flush=True)
    # 只在 cmp 列上比较（若 ref 认为无效但值有限 → 说明 kernel 多算了 cmp 列）
    n_ori = win_left + win_right + 1
    cmpcol = torch.zeros_like(valid)
    if with_cmp:
        cmpcol[..., n_ori:] = True
    extra = cmpcol & ~valid & torch.isfinite(S)
    print(f"  kernel 多算的 cmp 列数={int(extra.sum())}（ref 判无效但 kernel 给了有限值）",
          flush=True)
    missing = valid & ~(S > -1e29)
    print(f"  kernel 漏算的列数={int(missing.sum())}", flush=True)
    for (tt, n1i) in [(int(a), int(b)) for a, b in (diff.nonzero()[:show_rows]
                                                   if show_rows else [])]:
        srow = S[tt, n1i]
        rrow = rsc[tt, n1i]
        print(f"   row t={tt} n1={n1i}: kernel valid={int((srow > -1e29).sum())}"
              f" ref valid={int(torch.isfinite(rrow).sum())}"
              f" | first8 kernel={[round(float(x), 4) for x in srow[:8]]}"
              f" ref={[round(float(x), 4) for x in rrow[:8]]}"
              f" | cmpcols kernel={[round(float(x), 4) for x in srow[n_ori:n_ori + 8]]}"
              f" ref={[round(float(x), 4) for x in rrow[n_ori:n_ori + 8]]}", flush=True)


if __name__ == "__main__":
    cases = [
        ("D1 win127 全分支", dict(win_left=127), 2),
        ("D2 win127 仅cmp", dict(win_left=127, with_ori=False), 0),
        ("D3 win127 仅ori", dict(win_left=127, with_cmp=False), 2),
        ("D4 win3 全分支", dict(win_left=3), 2),
    ]
    for tag, kw, show in cases:
        try:
            go(tag, [5, 4], [64, 48], [24, 16], 4, 1, 8, show_rows=show, **kw)
        except Exception:
            print(f"## {tag} EXCEPTION", flush=True)
            traceback.print_exc()
    print("DIAGDONE", flush=True)
