# -*- coding: utf-8 -*-
"""Probe23：算子内部 score 缓冲 vs 参考 logits，逐列定位（含是哪一支）。

用完整算子（含两个 kernel），SASKV_DUMP_SCORES=1 取内部 SB；
参考 logits 用测试同款 reference 逐列算（含 ori 与 cmp 两段）。
跑法：SASKV_DUMP_SCORES=1 DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_sb.py
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


def main():
    qlens, kvolens, kclens = [5, 4], [64, 48], [24, 16]
    N1, N2, K = 4, 1, 8
    win_left = 127
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K,
                                        torch.float16)
    torch.manual_seed(5)
    sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    M._DBG.clear()
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
        sinks=sk, softmax_scale=1.0, ori_win_left=win_left, ori_win_right=0,
        cmp_ratio=1, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    sb = M._DBG.get("sb")
    print("sb captured:", sb is not None, flush=True)
    S = sb.detach().cpu().double().reshape(T, N1, -1)
    print("sb shape per (t,n1):", S.shape[-1], "(N_ORI+K =", win_left + 1 + K, ")",
          flush=True)
    QD = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    CU = cu.detach().cpu()

    def dense(tag, b):
        kv, bt, mb = geo[tag]
        KD = kv.detach().cpu().double()
        BT = bt.detach().cpu()
        bs = KD.shape[1]
        dd = torch.zeros(mb * bs, N2, D, dtype=torch.float64)
        for lb in range(mb):
            dd[lb * bs:(lb + 1) * bs] = KD[int(BT[b, lb])]
        return dd

    for t in (0, 5):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        ddo = dense("ori", b)
        ddc = dense("cmp", b)
        s1_abs = s1 + (kvolens[b] - qlens[b])
        n_valid = min(kclens[b], (s1_abs + 1) // 1)
        n1 = 0
        n2 = n1 // (N1 // N2)
        print(f"=== t={t} b={b} s1={s1} kvolo={kvolens[b]} kvolc={kclens[b]}"
              f" n_valid={n_valid} ===", flush=True)
        # ori 段
        ori_bad, ori_nan = [], []
        for lane in range(win_left + 1):
            j = s1_abs - win_left + lane
            exp = float(QD[t, n1] @ ddo[j, n2]) if 0 <= j < kvolens[b] else None
            got = float(S[t, n1, lane])
            if got != got:
                ori_nan.append((lane, j))
            elif exp is not None and abs(got - exp) > 2e-3:
                ori_bad.append((lane, j, got, exp))
        # cmp 段
        cmp_bad, cmp_nan, cmp_info = [], [], []
        for slot in range(K):
            i0 = int(IDX[t, n2, slot])
            got = float(S[t, n1, win_left + slot])
            if not (0 <= i0 < n_valid):
                cmp_info.append((slot, i0, "invalid", got))
                continue
            exp = float(QD[t, n1] @ ddc[i0, n2])
            if got != got:
                cmp_nan.append((slot, i0))
            elif abs(got - exp) > 2e-3:
                cmp_bad.append((slot, i0, got, exp))
        print(f"  ori: nan={len(ori_nan)} 数值错={len(ori_bad)}"
              + (f"  首个nan lane={ori_nan[0][0]} j={ori_nan[0][1]}" if ori_nan else "")
              + (f"  首个错 lane={ori_bad[0][0]} j={ori_bad[0][1]}"
                 f" got={ori_bad[0][2]:.5f} exp={ori_bad[0][3]:.5f}"
                 if ori_bad else ""), flush=True)
        print(f"  cmp: nan={len(cmp_nan)} 数值错={len(cmp_bad)}", flush=True)
        for x in cmp_bad[:4]:
            print(f"    cmp slot={x[0]} i0={x[1]} got={x[2]:.6f} exp={x[3]:.6f}", flush=True)
        print(f"  cmp 原始 idx={IDX[t, n2].tolist()} n_valid={n_valid}", flush=True)
        print(f"  cmp 读到的值(SB)={[round(float(S[t, n1, win_left + s]), 4) for s in range(K)]}",
              flush=True)
        print(f"  cmp 期望值     ="
              f"{[round(float(QD[t, n1] @ ddc[int(IDX[t, n2, s]), n2]), 4) if 0 <= int(IDX[t, n2, s]) < n_valid else None for s in range(K)]}",
              flush=True)
    print("SBDONE", flush=True)


if __name__ == "__main__":
    main()
