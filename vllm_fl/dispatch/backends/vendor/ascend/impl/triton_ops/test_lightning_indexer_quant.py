"""验证 lightning_indexer_quant 的 Triton 实现 vs AscendC 语义。

公式（service_cube.h L542-543 reluPre=1 + service_vector.h L383/389-406）：
    score[s1,n2,s2] = key_dequant_scale[s2] * Σ_g relu( query[s1,n2*gs+g]·key[s2,n2] )
                                     * weights[s1,n2*gs+g] * query_dequant_scale[s1,n2*gs+g]
    下三角（sparse_mode=3）：s2 <= s1 + (kvLen - qLen)
    top-K 降序，不足补 -1（INVALID_IDX）
索引类零容差：sparse_indices 必须**逐元素完全相等**。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.lightning_indexer_quant import lightning_indexer_quant  # noqa: E402

DEV = "npu:0"
L = []
D = 128


def emit(s=""):
    L.append(str(s))


def ref_scores(query, key, w, qs, ks, bt, cu, qlens, kvlens, N1, N2, sparse_mode, block_size):
    """fp64 参考：返回 per (t, n2) 的 score 向量（dense，物化分页 KV）。"""
    Q = query.detach().cpu().double()
    KD = key.detach().cpu().double()
    W = w.detach().cpu().double()
    QS = qs.detach().cpu().double()
    KS = ks.detach().cpu().double()
    BT = bt.detach().cpu()
    CU = cu.detach().cpu()
    T, N1_, _ = Q.shape
    _, bs_k, N2_, _ = KD.shape
    mb = BT.shape[1]
    gs = N1 // N2
    out = []
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        dense = torch.zeros(mb * bs_k, N2, D, dtype=torch.float64)
        dks = torch.zeros(mb * bs_k, dtype=torch.float64)
        for lb in range(mb):
            pb = int(BT[b, lb])
            dense[lb * bs_k:(lb + 1) * bs_k] = KD[pb]
            dks[lb * bs_k:(lb + 1) * bs_k] = KS[pb]
        if sparse_mode == 3:
            n_valid = s1 + 1 + (kvlens[b] - qlens[b])
        else:
            n_valid = kvlens[b]
        n_valid = max(0, min(kvlens[b], n_valid))
        for n2 in range(N2):
            sc = torch.full((dense.shape[0],), -1e30, dtype=torch.float64)
            jt = torch.arange(n_valid, dtype=torch.long)
            if jt.numel():
                qk = Q[t, n2 * gs:(n2 + 1) * gs] @ dense[jt, n2].T
                wq = W[t, n2 * gs:(n2 + 1) * gs] * QS[t, n2 * gs:(n2 + 1) * gs]
                sc[:n_valid] = (torch.clamp(qk, min=0.0) * wq[:, None]).sum(0) * dks[jt]
            out.append(sc)
    return out


def make(qlens, kvlens, N1, N2, block_size=16, seed=17):
    torch.manual_seed(seed)
    B = len(qlens)
    T = sum(qlens)
    mb = max((kl + block_size - 1) // block_size for kl in kvlens)
    nb = B * mb
    q = torch.randint(-8, 9, (T, N1, D), device=DEV).to(torch.int8)
    key = torch.randint(-8, 9, (nb, block_size, N2, D), device=DEV).to(torch.int8)
    w = ((torch.rand(T, N1, device=DEV) * 2 - 1) * 0.0015).to(torch.float16)
    qs = (torch.rand(T, N1, device=DEV) + 0.5).to(torch.float16)
    ks = (torch.rand(nb, block_size, device=DEV) + 0.5).to(torch.float16)
    bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
    for b in range(B):
        bt[b] = (torch.randperm(mb, device=DEV) + b * mb).to(torch.int32)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    return q, key, w, qs, ks, bt, cu, T


def run_case(name, qlens, kvlens, N1, N2, K, sparse_mode=3, block_size=16, layout_query="TND"):
    q, key, w, qs, ks, bt, cu, T = make(qlens, kvlens, N1, N2, block_size)
    B = len(qlens)
    idx = lightning_indexer_quant(
        query=q, key=key, weights=w, query_dequant_scale=qs, key_dequant_scale=ks,
        actual_seq_lengths_query=cu,
        actual_seq_lengths_key=torch.tensor(kvlens, dtype=torch.int32, device=DEV),
        block_table=bt, query_quant_mode=0, key_quant_mode=0,
        layout_query=layout_query, layout_key="PA_BSND",
        sparse_count=K, sparse_mode=sparse_mode)
    torch.npu.synchronize()
    scs = ref_scores(q, key, w, qs, ks, bt, cu, qlens, kvlens, N1, N2, sparse_mode, block_size)
    bad = 0
    gaps = []
    gs = N1 // N2
    for t in range(T):
        for n2 in range(N2):
            sc = scs[t * N2 + n2]
            sv, si = torch.sort(sc, descending=True)
            top = si[:K]
            top = torch.where(sv[:K] <= -1e29, torch.full_like(top, -1), top)
            got = idx[t, n2].detach().cpu().long()
            bad += int((got != top).sum().item())
            if sc.numel() > K and sv[K - 1] > -1e29 and sv[K] > -1e29:
                gaps.append(float(sv[K - 1] - sv[K]))
    gap = min(gaps) if gaps else float("inf")
    ok = bad == 0
    line = (f"  {name:46s} shape={tuple(idx.shape)} idx_mismatch={bad} gapK={gap:.3e}  "
            f"{'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    return ok


emit("=" * 96)
emit("lightning_indexer_quant: Triton vs AscendC（索引零容差，逐元素完全相等）")
emit("=" * 96)

ok = True
ok &= run_case("TND+PA B2 qlen[4,3] kvlen[32,24] N1=8  N2=1 K=8", [4, 3], [32, 24], 8, 1, 8)
ok &= run_case("TND+PA B2 qlen[4,3] kvlen[32,24] N1=16 N2=2 K=8 gs=8", [4, 3], [32, 24], 16, 2, 8)
ok &= run_case("TND+PA B1 qlen[6]   kvlen[40]    N1=8  N2=1 K=16", [6], [40], 8, 1, 16)
ok &= run_case("TND+PA B2 qlen[5,5] kvlen[48,48] N1=8 N2=1 K=8 mode0", [5, 5], [48, 48], 8, 1, 8,
               sparse_mode=0)
ok &= run_case("TND+PA B2 qlen[8,4] kvlen[64,32] N1=16 N2=2 K=12", [8, 4], [64, 32], 16, 2, 12)

emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/lightning_indexer_quant_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
