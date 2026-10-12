# -*- coding: utf-8 -*-
"""Probe: TND+PA 下 ori 分支 / cmp 分支 分别单独跑，定位错在哪一支。

用法（容器内）：
    DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_ab.py
"""
import hashlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
print("import path:", sys.path[0], flush=True)
try:
    import triton_ops.sparse_attn_sharedkv as M
    print("module file:", M.__file__, flush=True)
    print("module md5:", hashlib.md5(open(M.__file__, "rb").read()).hexdigest(), flush=True)
except Exception as e:  # noqa: BLE001
    print("IMPORT FAIL:", e, flush=True)
    raise

from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
D = 512
FLOOR = {torch.float16: 1e-3, torch.bfloat16: 4e-3}


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


def ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, scale, win_left, cmp_ratio,
                 with_ori=True, with_cmp=True):
    Q = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, _ = Q.shape
    N2 = IDX.shape[1]
    gs = N1 // N2
    SK = sk.detach().cpu().double() if sk is not None else None
    dense = {}
    for tag in ("ori", "cmp"):
        kv, bt, mb = geo[tag]
        KD = kv.detach().cpu().double()
        BT = bt.detach().cpu()
        dense[tag] = (KD, BT, KD.shape[1], mb)
    out = torch.zeros(T, N1, D, dtype=torch.float64)
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        ctx = {}
        for tag, lens in (("ori", kvolens), ("cmp", kclens)):
            KD, BT, bs, mb = dense[tag]
            dd = torch.zeros(mb * bs, N2, D, dtype=torch.float64)
            for lb in range(mb):
                dd[lb * bs:(lb + 1) * bs] = KD[int(BT[b, lb])]
            ctx[tag] = (dd, lens[b])
        for n1 in range(N1):
            n2 = n1 // gs
            vecs = []
            if with_ori:
                dd, kl = ctx["ori"]
                s1_abs = s1 + (kl - qlens[b])
                js = [j for j in range(max(0, s1_abs - win_left), min(kl - 1, s1_abs) + 1)]
                if js:
                    jt = torch.tensor(js, dtype=torch.long)
                    vecs.append((Q[t, n1] @ dd[jt, n2].T, dd[jt, n2]))
            if with_cmp:
                dd, kl = ctx["cmp"]
                s1_abs_orig = s1 + (kvolens[b] - qlens[b])
                n_valid = min(kl, (s1_abs_orig + 1) // cmp_ratio)
                js = [int(i) for i in IDX[t, n2] if 0 <= int(i) < n_valid]
                if js:
                    jt = torch.tensor(js, dtype=torch.long)
                    vecs.append((Q[t, n1] @ dd[jt, n2].T, dd[jt, n2]))
            if not vecs:
                continue
            logits = torch.cat([v[0] for v in vecs]) * scale
            vals = torch.cat([v[1] for v in vecs], dim=0)
            if SK is not None:
                m = torch.maximum(logits.max(), SK[n1])
                p = torch.exp(logits - m)
                lse = p.sum() + torch.exp(SK[n1] - m)
            else:
                p = torch.exp(logits - logits.max())
                lse = p.sum()
            out[t, n1] = (p[:, None] * vals).sum(0) / lse
    return out


def case(tag, qlens, kvolens, kclens, N1, N2, K, dtype, with_ori=True, with_cmp=True,
         sinks=True, cmp_ratio=1, block_size=16):
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype,
                                        block_size)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    ori_kv = geo["ori"][0] if with_ori else None
    cmp_kv = geo["cmp"][0] if with_cmp else None
    obt = geo["ori"][1] if with_ori else None
    cbt = geo["cmp"][1] if with_cmp else None
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=ori_kv, cmp_kv=cmp_kv,
        cmp_sparse_indices=cidx if with_cmp else None,
        sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
        cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=obt, cmp_block_table=cbt,
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    r = ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, 1.0, 127, cmp_ratio,
                     with_ori=with_ori, with_cmp=with_cmp)
    o = out.detach().cpu().double()
    err = (o - r).abs()
    emax = err.max().item()
    good = emax <= FLOOR[dtype]
    # 误差最大的位置
    idx = err.reshape(-1).argmax().item() if err.numel() else 0
    ti = idx // (N1 * D)
    n1i = (idx % (N1 * D)) // D
    di = idx % D
    line = (f"{tag:44s} err={emax:.3e}(<= {FLOOR[dtype]:.0e}) {'PASS' if good else 'FAIL'}"
            f"  worst@t={ti} n1={n1i} d={di} got={o[ti, n1i, di]:.5f} ref={r[ti, n1i, di]:.5f}"
            f" |ref|max={r.abs().max().item():.3f} out_nan={bool(torch.isnan(o).any())}")
    print(line, flush=True)
    return good


print("=" * 120, flush=True)
allok = True
# A/B：ori / cmp 分支单独跑
allok &= case("AB1 TND+PA 仅 ori 滑窗", [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.float16,
              with_cmp=False)
allok &= case("AB2 TND+PA 仅 cmp 稀疏", [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.float16,
              with_ori=False)
allok &= case("AB3 TND+PA 全分支（对照）", [5, 4], [64, 48], [24, 16], 4, 1, 8,
              torch.float16)
print("=" * 120, flush=True)
print("AB_RESULT:", "全部 PASS" if allok else "存在 FAIL", flush=True)
print("ABDONE", flush=True)
