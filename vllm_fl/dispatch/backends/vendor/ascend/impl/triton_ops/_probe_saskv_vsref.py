# -*- coding: utf-8 -*-
"""Probe22：直接跑算子 + 测试自带参考函数，逐 (t,n1,d) 定位差异。

复用测试文件的 make_saskv_pa / ref_saskv_pa（同一份代码、同一 seed），
然后打印误差最大的位置、误差分布、以及该位置的 score 参考值。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_vsref.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tsaskv", "/workspace/FlagGems/src/triton_ops/test_sparse_attn_sharedkv.py")
# 测试文件顶层会跑全部用例；这里只取它的两个函数，所以手工 exec 会太重。
# 改为复制最小实现（与测试文件逐行一致）。
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
D = 512


def make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size=16, seed=81):
    """逐行抄自 test_sparse_attn_sharedkv.py"""
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


def ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, scale, win_left, cmp_ratio):
    """逐行抄自 test_sparse_attn_sharedkv.py"""
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
            dd, kl = ctx["ori"]
            s1_abs = s1 + (kl - qlens[b])
            js = [j for j in range(max(0, s1_abs - win_left), min(kl - 1, s1_abs) + 1)]
            if js:
                jt = torch.tensor(js, dtype=torch.long)
                vecs.append((Q[t, n1] @ dd[jt, n2].T, dd[jt, n2]))
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


def run(tag, qlens, kvolens, kclens, N1, N2, K, sinks=True, cmp_ratio=1):
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K,
                                        torch.float16)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
        sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
        cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    r = ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, 1.0, 127, cmp_ratio)
    o = out.detach().cpu().double()
    err = (o - r).abs()
    print(f"## {tag}", flush=True)
    print(f"  |ref|max={r.abs().max().item():.4f}  err|max|={err.max().item():.3e}"
          f"  nan={int(torch.isnan(o).sum())}  count(>1e-3)={int((err > 1e-3).sum())}"
          f" / {err.numel()}", flush=True)
    # 逐 token 汇总
    for t in range(T):
        e = err[t].max().item()
        n = int((err[t] > 1e-3).sum())
        print(f"    t={t:2d} err|max|={e:.3e} 超标元素={n}", flush=True)
    idx = err.reshape(-1).argmax().item()
    N1c = o.shape[1]
    tt, n1i, di = idx // (N1c * D), (idx % (N1c * D)) // D, idx % D
    print(f"  最差位置 t={tt} n1={n1i} d={di}: got={o[tt, n1i, di]:.6f}"
          f" ref={r[tt, n1i, di]:.6f}", flush=True)
    return err.max().item()


if __name__ == "__main__":
    run("TND+PA 全分支 fp16 (测试第一例)", [5, 4], [64, 48], [24, 16], 4, 1, 8)
    print("VSREFDONE", flush=True)
