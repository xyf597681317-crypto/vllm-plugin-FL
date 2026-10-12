"""验证 lightning_indexer（= npu_lightning_indexer）的 Triton 实现。

口径（见 lightning_indexer.py docstring）：非量化版 = 量化版在恒等 scale 下的特例，
    score[s1,n2,s2] = Σ_g relu( query[s1,n2*gs+g] · key[s2,n2] ) * weights[s1,n2*gs+g]
索引类零容差：sparse_indices 逐元素完全相等（不足 sparse_count 补 -1）。
调用点风格：device_op.py L632-640 的真实 kwargs（TND + PA_BSND + sparse_mode=3）。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.lightning_indexer import lightning_indexer, npu_lightning_indexer  # noqa: E402
from triton_ops.lightning_indexer_quant import lightning_indexer_quant  # noqa: E402

DEV = "npu:0"
L = []
D = 128


def emit(s=""):
    L.append(str(s))


def ref_scores(query, key, w, bt, cu, qlens, kvlens, N1, N2, sparse_mode):
    """fp64 参考（非量化）：score = Σ_g relu(q·k) * w（无 dequant scale）。"""
    Q = query.detach().cpu().double()
    KD = key.detach().cpu().double()
    W = w.detach().cpu().double()
    BT = bt.detach().cpu()
    CU = cu.detach().cpu()
    T, _, _ = Q.shape
    _, bs_k, _, _ = KD.shape
    mb = BT.shape[1]
    gs = N1 // N2
    out = []
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        dense = torch.zeros(mb * bs_k, N2, D, dtype=torch.float64)
        for lb in range(mb):
            dense[lb * bs_k:(lb + 1) * bs_k] = KD[int(BT[b, lb])]
        n_valid = (s1 + 1 + (kvlens[b] - qlens[b])) if sparse_mode == 3 else kvlens[b]
        n_valid = max(0, min(kvlens[b], n_valid))
        for n2 in range(N2):
            sc = torch.full((dense.shape[0],), -1e30, dtype=torch.float64)
            jt = torch.arange(n_valid, dtype=torch.long)
            if jt.numel():
                qk = Q[t, n2 * gs:(n2 + 1) * gs] @ dense[jt, n2].T
                sc[:n_valid] = (torch.clamp(qk, min=0.0) * W[t, n2 * gs:(n2 + 1) * gs, None]).sum(0)
            out.append(sc)
    return out


def make(qlens, kvlens, N1, N2, block_size=16, seed=23):
    torch.manual_seed(seed)
    B = len(qlens)
    T = sum(qlens)
    mb = max((kl + block_size - 1) // block_size for kl in kvlens)
    nb = B * mb
    q = torch.randint(-8, 9, (T, N1, D), device=DEV).to(torch.int8)
    key = torch.randint(-8, 9, (nb, block_size, N2, D), device=DEV).to(torch.int8)
    w = ((torch.rand(T, N1, device=DEV) * 2 - 1) * 0.0015).to(torch.float16)
    bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
    for b in range(B):
        bt[b] = (torch.randperm(mb, device=DEV) + b * mb).to(torch.int32)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    return q, key, w, bt, cu, T


def run_case(name, qlens, kvlens, N1, N2, K, sparse_mode=3, block_size=16, use_alias=False):
    q, key, w, bt, cu, T = make(qlens, kvlens, N1, N2, block_size)
    B = len(qlens)
    kvl = torch.tensor(kvlens, dtype=torch.int32, device=DEV)
    fn = npu_lightning_indexer if use_alias else lightning_indexer
    # 调用点风格：真实 kwargs（TND + PA_BSND + sparse_mode + sparse_count）
    idx = fn(query=q, key=key, weights=w,
             actual_seq_lengths_query=cu, actual_seq_lengths_key=kvl,
             block_table=bt, layout_query="TND", layout_key="PA_BSND",
             sparse_count=K, sparse_mode=sparse_mode)
    torch.npu.synchronize()

    # (a) vs fp64 参考（索引零容差）
    scs = ref_scores(q, key, w, bt, cu, qlens, kvlens, N1, N2, sparse_mode)
    bad = 0
    gaps = []
    for t in range(T):
        for n2 in range(N2):
            sv, si = torch.sort(scs[t * N2 + n2], descending=True)
            top = si[:K]
            top = torch.where(sv[:K] <= -1e29, torch.full_like(top, -1), top)
            got = idx[t, n2].detach().cpu().long()
            bad += int((got != top).sum().item())
            if sv.numel() > K and sv[K - 1] > -1e29 and sv[K] > -1e29:
                gaps.append(float(sv[K - 1] - sv[K]))
    gap = min(gaps) if gaps else float("inf")

    # (b) 恒等 scale 等价性：必须与量化版（scale=1）逐元素相同
    ones_q = torch.ones(T, N1, dtype=torch.float16, device=DEV)
    ones_k = torch.ones(key.shape[0], block_size, dtype=torch.float16, device=DEV)
    idx_q = lightning_indexer_quant(q, key, w, ones_q, ones_k,
                                    actual_seq_lengths_query=cu, actual_seq_lengths_key=kvl,
                                    block_table=bt, layout_query="TND", layout_key="PA_BSND",
                                    sparse_count=K, sparse_mode=sparse_mode)
    torch.npu.synchronize()
    ident_bad = int((idx.detach().cpu() != idx_q.detach().cpu()).sum().item())

    ok = bad == 0 and ident_bad == 0
    line = (f"  {name:44s} shape={tuple(idx.shape)} idx_mismatch={bad} "
            f"vs_quant_identity={ident_bad} gapK={gap:.3e}  {'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    return ok


emit("=" * 100)
emit("lightning_indexer (=npu_lightning_indexer): 索引零容差 + 恒等 scale 与量化版一致")
emit("=" * 100)

ok = True
ok &= run_case("TND+PA B2 qlen[4,3] kvlen[32,24] N1=8  N2=1 K=8", [4, 3], [32, 24], 8, 1, 8)
ok &= run_case("TND+PA B2 qlen[4,3] kvlen[32,24] N1=16 N2=2 K=8 gs=8", [4, 3], [32, 24], 16, 2, 8)
ok &= run_case("TND+PA B1 qlen[6]   kvlen[40]    N1=8  N2=1 K=16", [6], [40], 8, 1, 16)
ok &= run_case("TND+PA B2 qlen[5,5] kvlen[48,48] N1=8  N2=1 K=8 mode0", [5, 5], [48, 48], 8, 1, 8,
               sparse_mode=0)
ok &= run_case("别名 npu_lightning_indexer 入口", [4, 3], [32, 24], 8, 1, 8, use_alias=True)

emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/lightning_indexer_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
