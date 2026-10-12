# -*- coding: utf-8 -*-
"""Probe2: TND+PA 数值定位 —— 对比内部 score 缓冲 vs 参考 logits。

跑法（容器内）：
    SASKV_DUMP_SCORES=1 DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_dump.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/workspace/FlagGems/src")
import triton_ops.sparse_attn_sharedkv as M  # noqa: E402
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

os.environ.setdefault("SASKV_DUMP_SCORES", "1")

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
D = 512
FLOOR = torch.float16
BLOCK = 16


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


def ref_scores_and_out(q, geo, cidx, cu, qlens, kvolens, kclens, sk, scale,
                       win_left, win_right, cmp_ratio, with_ori, with_cmp):
    """返回 (score 矩阵 [T,N1,total], 参考 out [T,N1,D])。score 列 = ori lanes 后接 cmp slots。"""
    Q = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, _ = Q.shape
    N2 = IDX.shape[1]
    gs = N1 // N2
    SK = sk.detach().cpu().double() if sk is not None else None
    n_ori = (win_left + win_right + 1) if with_ori else 0
    K = IDX.shape[-1] if with_cmp else 0
    total = n_ori + K
    sc = torch.full((T, N1, total), float("-inf"), dtype=torch.float64)
    out = torch.zeros(T, N1, D, dtype=torch.float64)
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        for n1 in range(N1):
            n2 = n1 // gs
            vecs = []
            if with_ori:
                dd = dense_of(geo, "ori", b)
                kl = kvolens[b]
                s1_abs = s1 + (kl - qlens[b])
                for col, j in enumerate(range(s1_abs - win_left, s1_abs + win_right + 1)):
                    if 0 <= j < kl:
                        sc[t, n1, col] = float(Q[t, n1] @ dd[j, n2])
                        vecs.append(dd[j, n2])
            if with_cmp:
                dd = dense_of(geo, "cmp", b)
                s1_abs_orig = s1 + (kvolens[b] - qlens[b])
                n_valid = min(kclens[b], (s1_abs_orig + 1) // cmp_ratio)
                for slot in range(K):
                    i0 = int(IDX[t, n2, slot])
                    if 0 <= i0 < n_valid:
                        sc[t, n1, n_ori + slot] = float(Q[t, n1] @ dd[i0, n2])
                        vecs.append(dd[i0, n2])
            if not vecs:
                continue
            v = torch.stack(vecs)
            lg = sc[t, n1][torch.isfinite(sc[t, n1])] * scale
            m = torch.maximum(lg.max(), SK[n1]) if SK is not None else lg.max()
            p = torch.exp(lg - m)
            lse = p.sum() + (torch.exp(SK[n1] - m) if SK is not None else 0.0)
            out[t, n1] = (p[:, None] * v).sum(0) / lse
    return sc, out


def case(tag, qlens, kvolens, kclens, N1, N2, K, dtype=FLOOR, win_left=127, win_right=0,
         cmp_ratio=1, with_ori=True, with_cmp=True, sinks=True, seed=81):
    q, geo, cidx, cu, T = make(qlens, kvolens, kclens, N1, N2, K, dtype, seed=seed)
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
    rsc, rout = ref_scores_and_out(q, geo, cidx, cu, qlens, kvolens, kclens, sk, 1.0,
                                   win_left, win_right, cmp_ratio, with_ori, with_cmp)
    o = out.detach().cpu().double()
    print("=" * 120, flush=True)
    print(f"## {tag}  qlens={qlens} kvo={kvolens} kvc={kclens} N1={N1} N2={N2} K={K} "
          f"win_left={win_left} cmp_ratio={cmp_ratio} ori={with_ori} cmp={with_cmp}",
          flush=True)
    if sb is not None:
        S = sb.detach().cpu().double().reshape(T, N1, -1)
        valid = torch.isfinite(rsc)
        print(f"  score-buf shape={tuple(S.shape)} ref valid cols={int(valid.sum())}", flush=True)
        # 无效列（Triton 写成 NEG）应 ≈ -1e30
        inv = ~valid
        if inv.any():
            print(f"  invalid-lane max={S[inv].max().item():.3e} (期望 ~ -1e30)", flush=True)
        sd = (S - rsc)[valid]
        print(f"  score err |max|={(sd.abs().max().item() if sd.numel() else 0):.3e}"
              f"  mean={sd.mean().item():.3e}" if sd.numel() else "  score err: n/a", flush=True)
        if sd.numel():
            flat = (S - rsc).abs()
            flat[~valid] = -1
            i = flat.reshape(-1).argmax().item()
            tt = i // (N1 * S.shape[-1])
            n1i = (i % (N1 * S.shape[-1])) // S.shape[-1]
            ci = i % S.shape[-1]
            print(f"  worst score @t={tt} n1={n1i} col={ci}: got={S[tt, n1i, ci]:.6f} "
                  f"ref={rsc[tt, n1i, ci]:.6f}", flush=True)
            # 每行（t,n1）的有效列数与最大值对比
            nvalid_tri = (S > -1e29).sum(-1)
            nvalid_ref = valid.sum(-1)
            bad = (nvalid_tri != nvalid_ref).nonzero()
            print(f"  rows with different valid-col count: {bad.shape[0]}"
                  + (f" e.g. {bad[:6].tolist()}" if bad.shape[0] else ""), flush=True)
    else:
        print("  score-buf NOT captured (SASKV_DUMP_SCORES off?)", flush=True)
    err = (o - rout).abs()
    print(f"  OUT err|max|={err.max().item():.3e}  count(>1e-3)={int((err > 1e-3).sum())}"
          f"  |out|max={o.abs().max().item():.3f}", flush=True)
    return


if __name__ == "__main__":
    # C1：正常窗口（基线复现）
    case("C1 win127 全分支基线", [5, 4], [64, 48], [24, 16], 4, 1, 8, win_left=127)
    # C2：极小窗口
    case("C2 win0 仅ori", [5, 4], [64, 48], [24, 16], 4, 1, 8, win_left=0, with_cmp=False)
    # C3：win=3 仅 ori
    case("C3 win3 仅ori", [5, 4], [64, 48], [24, 16], 4, 1, 8, win_left=3, with_cmp=False)
    # C4：仅 cmp
    case("C4 仅cmp", [5, 4], [64, 48], [24, 16], 4, 1, 8, with_ori=False)
    # C5：单 token batch、短 kv（最简可复现）
    case("C5 B1 qlen1 kvo32", [1], [32], [16], 4, 1, 4, win_left=127)
    print("=" * 120, flush=True)
    print("DUMPDONE", flush=True)
