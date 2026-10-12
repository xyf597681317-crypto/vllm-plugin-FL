# -*- coding: utf-8 -*-
"""Probe4：TND+PA 的 (1) 块表是否真的被 kernel 使用 (2) 逐列定位 ori 分差。

跑法：SASKV_DUMP_SCORES=1 DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_map.py
"""
import os
import sys

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


def build(qlens, kvolens, kclens, N1, N2, K, block_size=BLOCK, seed=81, perm="random"):
    torch.manual_seed(seed)
    B = len(qlens)
    T = sum(qlens)
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc).to(FP16)
    geo = {}
    for tag, lens in (("ori", kvolens), ("cmp", kclens)):
        mb = max((l + block_size - 1) // block_size for l in lens)
        nb = B * mb
        kv = (torch.randn(nb, block_size, N2, D, device=DEV) * 0.6).to(FP16)
        bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
        for b in range(B):
            if perm == "identity":
                bt[b] = torch.arange(b * mb, (b + 1) * mb, dtype=torch.int32, device=DEV)
            else:
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


def ref_scores(q, geo, cidx, cu, qlens, kvolens, kclens, win_left, win_right, cmp_ratio,
               with_ori, with_cmp):
    Q = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, _ = Q.shape
    gs = N1 // IDX.shape[1]
    n_ori = (win_left + win_right + 1) if with_ori else 0
    K = IDX.shape[-1] if with_cmp else 0
    sc = torch.full((T, N1, n_ori + K), float("-inf"), dtype=torch.float64)
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


def densify_rows(geo, tag, b, pad_rows, block_size, n2count, D_):
    """把某 batch 的物理页拼成 [pad_rows, n2count, D]，行号 = 块表给出的物理页*bs+r。"""
    kv, bt, mb = geo[tag]
    KD = kv.detach().cpu().double()
    BT = bt.detach().cpu()
    out = torch.zeros(pad_rows, n2count, D_, dtype=torch.float64)
    for lb in range(mb):
        p = int(BT[b, lb])
        out[lb * block_size:(lb + 1) * block_size] = KD[p]
    return out


def go(tag, qlens, kvolens, kclens, N1, N2, K, win_left=127, win_right=0, cmp_ratio=1,
       with_ori=True, with_cmp=True, perm="random", detail_rows=2):
    q, geo, cidx, cu, T = build(qlens, kvolens, kclens, N1, N2, K, perm=perm)
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
    print("=" * 110, flush=True)
    print(f"## {tag} perm={perm}", flush=True)
    if with_ori:
        rsc = ref_scores(q, geo, cidx, cu, qlens, kvolens, kclens, win_left, win_right,
                         cmp_ratio, with_ori, with_cmp)
        S = M._DBG["sb"].detach().cpu().double().reshape(T, N1, -1)
        valid = torch.isfinite(rsc)
        fin = valid & torch.isfinite(S)
        d = (S - rsc).abs()
        print(f"  score |max| on ref-valid = {d[fin].max().item():.3e}  nan={int(torch.isnan(S).sum())}",
              flush=True)
        # 逐列错位诊断：对每行，找 kernel 行里 最大误差列
        for r in range(min(detail_rows, T * N1)):
            tt, n1i = r // N1, r % N1
            drow = d[tt, n1i].clone()
            drow[~fin[tt, n1i]] = -1
            order = torch.argsort(drow, descending=True)[:6]
            parts = []
            for ci in order.tolist():
                col = drow[ci].item()
                if col <= 0:
                    break
                parts.append(f"col{ci}(lane j={ci - win_left}): ker={S[tt, n1i, ci].item():.5f}"
                             f" ref={rsc[tt, n1i, ci].item():.5f} d={col:.3e}")
            print(f"  row t={tt} n1={n1i} worst: " + " | ".join(parts), flush=True)
        # 块表身份 vs 随机：若 kernel 忽略块表，两者应完全一样
        if perm == "random":
            rsc_id = ref_scores(q, geo, cidx, cu, qlens, kvolens, kclens, win_left, win_right,
                                cmp_ratio, with_ori, with_cmp)
            print(f"  (ref with block table used, sanity |ref|max="
                  f"{rsc[torch.isfinite(rsc)].abs().max().item():.3f})", flush=True)
    print("MAPDONE-CASE", flush=True)


if __name__ == "__main__":
    # E1: 身份块表 —— 若实现正确，应与参考一致
    go("E1 身份块表 全分支", [5, 4], [64, 48], [24, 16], 4, 1, 8, perm="identity",
       detail_rows=2)
    # E2: 随机块表 —— 与 E1 对比可判断块表是否生效
    go("E2 随机块表 全分支", [5, 4], [64, 48], [24, 16], 4, 1, 8, perm="random",
       detail_rows=2)
    # E3: 仅 ori、随机块表（NaN 只在这个组合出现过）
    go("E3 随机块表 仅ori", [5, 4], [64, 48], [24, 16], 4, 1, 8, perm="random",
       with_cmp=False, detail_rows=2)
    print("MAPDONE", flush=True)
