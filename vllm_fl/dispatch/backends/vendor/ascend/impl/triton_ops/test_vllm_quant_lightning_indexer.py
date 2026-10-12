"""验证 vllm_quant_lightning_indexer 的 Triton 实现 vs AscendC 语义。

- sparse_indices (int32)：必须与 fp64 参考的 top-k **完全相等**
- sparse_values (fp32)：误差 <= 2e-6
- 同时打印"第 K 与第 K+1 名得分的间隔"，佐证 fp32 排序与 AscendC 的 bf16 排序一致
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.vllm_quant_lightning_indexer import vllm_quant_lightning_indexer  # noqa: E402

DEV = "npu:0"
L = []
FP32_FLOOR = 2e-6


def emit(s=""):
    L.append(str(s))


def ref(query, key, weights, qs, ks, K, sparse_mode):
    """fp64 参考，严格按 quant_lightning_indexer_vector1.h 的公式。"""
    q = query.detach().cpu().double()
    k = key.detach().cpu().double()
    w = weights.detach().cpu().double()
    qsd = qs.detach().cpu().double()
    ksd = ks.detach().cpu().double()
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    gs = N1 // N2
    qr = q.view(B, S1, N2, gs, D)
    kr = k.view(B, S2, N2, D)
    qk = torch.einsum("bsngd,btnd->bsngt", qr, kr)          # [B,S1,N2,gs,S2]
    relu = torch.clamp(qk, min=0.0)
    wq = (w.view(B, S1, N2, gs) * qsd.view(B, S1, N2, gs))
    score = (relu * wq[..., None]).sum(dim=3) * ksd.view(B, 1, 1, S2)
    if sparse_mode == 3:
        s1i = torch.arange(S1).view(1, S1, 1, 1)
        s2i = torch.arange(S2).view(1, 1, 1, S2)
        score = torch.where(s2i <= s1i + (S2 - S1), score, torch.full_like(score, -1e30))
    val, ind = torch.topk(score, K, dim=-1, largest=True, sorted=True)
    ind = torch.where(val <= -1e29, torch.full_like(ind, -1), ind)
    # 第 K / 第 K+1 名的间隔（只在有效项上算）
    sv, _ = torch.sort(score, dim=-1, descending=True)
    if sv.shape[-1] > K:
        gap = (sv[..., K - 1] - sv[..., K])
        gap = gap[sv[..., K - 1] > -1e29]
        gapv = gap.min().item() if gap.numel() else float("inf")
    else:
        gapv = float("inf")
    return ind, val, gapv


def run_case(name, B, S1, S2, N1, N2, K, dtype, sparse_mode=3, return_values=True,
             qmax=8, wscale=0.0015, seed=7):
    torch.manual_seed(seed)
    D = 128
    query = torch.randint(-qmax, qmax + 1, (B, S1, N1, D), device=DEV).to(torch.int8)
    key = torch.randint(-qmax, qmax + 1, (B, S2, N2, D), device=DEV).to(torch.int8)
    weights = ((torch.rand(B, S1, N1, device=DEV) * 2 - 1) * wscale).to(dtype)
    qs = (torch.rand(B, S1, N1, device=DEV) * 1.0 + 0.5).to(dtype)
    ks = (torch.rand(B, S2, device=DEV) * 1.0 + 0.5).to(dtype)

    idx, vals = vllm_quant_lightning_indexer(
        query, key, weights, qs, ks,
        sparse_count=K, sparse_mode=sparse_mode, return_values=return_values)
    torch.npu.synchronize()
    rind, rval, gap = ref(query, key, weights, qs, ks, K, sparse_mode)

    idx_np = idx.detach().cpu()
    bad = int((idx_np != rind).sum().item())
    # sparse_values：只在参考 index>=0 的有效槽位上比较（无效槽位的值 AscendC 未定义，本实现约定为 0.0）
    if return_values:
        vmask = rind >= 0
        d = (vals.detach().cpu().double() - rval).abs()
        err = d[vmask].max().item() if vmask.any() else 0.0
        pad_bad = int((vals.detach().cpu()[~vmask] != 0).sum().item()) if (~vmask).any() else 0
    else:
        err = 0.0
        pad_bad = 0
    vok = err <= FP32_FLOOR
    iok = bad == 0
    ok = iok and vok
    emit(f"  {name:44s} idx_mismatch={bad:4d} val_err={err:.3e}(<= {FP32_FLOOR:.0e}) "
         f"gapK={gap:.3e} pad_nonzero={pad_bad}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 100)
emit("vllm_quant_lightning_indexer: Triton vs AscendC 语义（fp64 参考）")
emit("  索引 int32 必须完全相等；sparse_values fp32 误差 <= 2e-6；gapK=第K与第K+1名得分间隔")
emit("=" * 100)

F16 = torch.float16
BF16 = torch.bfloat16
ok = True

ok &= run_case("B2 S1=4 S2=128 N1=8 N2=1 K=8 fp16 mode0", 2, 4, 128, 8, 1, 8, F16, sparse_mode=0)
ok &= run_case("B2 S1=4 S2=128 N1=8 N2=1 K=8 bf16 mode0", 2, 4, 128, 8, 1, 8, BF16, sparse_mode=0)
ok &= run_case("B2 S1=64 S2=64 N1=8 N2=1 K=8 fp16 下三角", 2, 64, 64, 8, 1, 8, F16, sparse_mode=3)
# gs<=4 时加权和项数太少，参考侧极易出现 K/K+1 精确平局（gapK=0）；平局取序在 top-k 里本质不确定。
# 这里用 gs=8 的两种 shape/dtype/mode 覆盖 GQA 分组（N2>1），保证边界无平局。
ok &= run_case("B2 S1=32 S2=64 N1=16 N2=2 K=8 gs=8 fp16 下三角", 2, 32, 64, 16, 2, 8, F16, sparse_mode=3)
ok &= run_case("B2 S1=16 S2=128 N1=16 N2=2 K=16 gs=8 bf16 mode0", 2, 16, 128, 16, 2, 16, BF16, sparse_mode=0)
ok &= run_case("B1 S1=5 S2=100 N1=4 N2=1 K=5 非整除 fp16", 1, 5, 100, 4, 1, 5, F16, sparse_mode=0)
ok &= run_case("B2 S1=8 S2=64 N1=8 N2=1 K=16 fp16 下三角", 2, 8, 64, 8, 1, 16, F16, sparse_mode=3)
ok &= run_case("B2 S1=16 S2=256 N1=8 N2=1 K=8 fp16 mode0(分两块)", 2, 16, 256, 8, 1, 8, F16, sparse_mode=0)
ok &= run_case("B2 S1=64 S2=64 N1=8 N2=1 K=8 bf16 下三角", 2, 64, 64, 8, 1, 8, BF16, sparse_mode=3)

# return_values=False → sparse_values 必须是空张量
torch.manual_seed(3)
q_ = torch.randint(-8, 9, (1, 4, 8, 128), device=DEV).to(torch.int8)
k_ = torch.randint(-8, 9, (1, 32, 1, 128), device=DEV).to(torch.int8)
w_ = ((torch.rand(1, 4, 8, device=DEV) * 2 - 1) * 0.02).to(F16)
s_ = (torch.rand(1, 4, 8, device=DEV) + 0.5).to(F16)
ks_ = (torch.rand(1, 32, device=DEV) + 0.5).to(F16)
i2, v2 = vllm_quant_lightning_indexer(q_, k_, w_, s_, ks_, sparse_count=4, sparse_mode=0,
                                      return_values=False)
empty_ok = (v2.numel() == 0)
emit(f"  {'return_values=False -> sparse_values 为空':46s} numel={v2.numel()}  "
     f"{'PASS' if empty_ok else 'FAIL'}")
ok &= empty_ok

emit()
emit(f"总体(BSND 等价性): {'全部 PASS' if ok else '存在 FAIL'}")

# ============ 真实 layout 组合：layout_query="TND" + layout_key="PA_BSND" ============
# 对齐 dsa_v1.py L2658-2678 的真实调用参数组合
def make_qli_pa(qlens, kvlens, N1, N2, dtype=torch.float16, block_size=16, seed=31):
    torch.manual_seed(seed)
    B = len(qlens)
    T = sum(qlens)
    D = 128
    max_blocks = max((kl + block_size - 1) // block_size for kl in kvlens)
    nb = B * max_blocks
    q = torch.randint(-8, 9, (T, N1, D), device=DEV).to(torch.int8)
    key = torch.randint(-8, 9, (nb, block_size, N2, D), device=DEV).to(torch.int8)
    weights = ((torch.rand(T, N1, device=DEV) * 2 - 1) * 0.0015).to(dtype)
    qs = (torch.rand(T, N1, device=DEV) + 0.5).to(dtype)
    ks = (torch.rand(nb, block_size, device=DEV) + 0.5).to(dtype)
    bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=DEV)
    for b in range(B):
        bt[b] = (torch.randperm(max_blocks, device=DEV) + b * max_blocks).to(torch.int32)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    return q, key, weights, qs, ks, bt, cu, T, block_size, max_blocks


def ref_qli_pa(q, key, w, qs, ks, bt, cu, qlens, kvlens, K, sparse_mode, cmp_ratio, bs_k):
    Q = q.detach().cpu().double()
    KD = key.detach().cpu().double()
    W = w.detach().cpu().double()
    QS = qs.detach().cpu().double()
    KS = ks.detach().cpu().double()
    BT = bt.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, D = Q.shape
    _, _, N2, _ = KD.shape
    max_blocks = BT.shape[1]
    gs = N1 // N2
    out = torch.full((T, N2, K), -1, dtype=torch.long)
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        dense = torch.zeros(max_blocks * bs_k, N2, D, dtype=torch.float64)
        dks = torch.zeros(max_blocks * bs_k, dtype=torch.float64)
        for lb in range(max_blocks):
            pb = int(BT[b, lb])
            dense[lb * bs_k:(lb + 1) * bs_k] = KD[pb]
            dks[lb * bs_k:(lb + 1) * bs_k] = KS[pb]
        if sparse_mode == 3:
            n_valid = (s1 + 1 + (kvlens[b] - qlens[b])) // cmp_ratio
        else:
            n_valid = kvlens[b]
        n_valid = max(0, min(kvlens[b], n_valid))
        for n2 in range(N2):
            sc = torch.full((dense.shape[0],), -1e30, dtype=torch.float64)
            jt = torch.arange(n_valid, dtype=torch.long)
            if jt.numel():
                qk = Q[t, n2 * gs:(n2 + 1) * gs] @ dense[jt, n2].T      # [gs, j]
                wq = W[t, n2 * gs:(n2 + 1) * gs] * QS[t, n2 * gs:(n2 + 1) * gs]
                sc[:n_valid] = (torch.clamp(qk, min=0.0) * wq[:, None]).sum(0) * dks[jt]
            val, ind = torch.topk(sc, K, largest=True, sorted=True)
            ind = torch.where(val <= -1e29, torch.full_like(ind, -1), ind)
            out[t, n2] = ind
    return out


def run_case_qli_pa(name, qlens, kvlens, N1, N2, K, dtype=torch.float16, sparse_mode=3,
                    cmp_ratio=1, block_size=16):
    q, key, w, qs, ks, bt, cu, T, bs_k, max_blocks = make_qli_pa(
        qlens, kvlens, N1, N2, dtype, block_size)
    idx, _ = vllm_quant_lightning_indexer(
        query=q, key=key, weights=w, query_dequant_scale=qs, key_dequant_scale=ks,
        actual_seq_lengths_query=cu,
        actual_seq_lengths_key=torch.tensor(kvlens, dtype=torch.int32, device=DEV),
        block_table=bt, query_quant_mode=0, key_quant_mode=0,
        layout_query="TND", layout_key="PA_BSND", sparse_count=K, sparse_mode=sparse_mode,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, return_value=False)
    torch.npu.synchronize()
    r = ref_qli_pa(q, key, w, qs, ks, bt, cu, qlens, kvlens, K, sparse_mode, cmp_ratio, bs_k)
    bad = int((idx.detach().cpu().long() != r).sum().item())
    good = bad == 0 and tuple(idx.shape) == (T, N2, K)
    line = f"  {name:46s} shape={tuple(idx.shape)} idx_mismatch={bad}  {'PASS' if good else 'FAIL'}"
    emit(line)
    print(line, flush=True)
    return good


emit()
emit("-- 真实 layout 组合：layout_query='TND' + layout_key='PA_BSND' --")
ok &= run_case_qli_pa("TND+PA B2 qlen[4,3] kvlen[32,24] N1=8 N2=1 K=8 下三角", [4, 3], [32, 24], 8, 1, 8)
ok &= run_case_qli_pa("TND+PA B2 qlen[4,3] kvlen[32,24] N1=16 N2=2 K=8 gs=8", [4, 3], [32, 24], 16, 2, 8)
ok &= run_case_qli_pa("TND+PA B2 qlen[4,4] kvlen[32,32] K=8 cmp_ratio=4", [4, 4], [32, 32], 8, 1, 8,
                      cmp_ratio=4)
ok &= run_case_qli_pa("TND+PA B1 qlen[6] kvlen[40] N1=8 N2=1 K=16 mode0", [6], [40], 8, 1, 16,
                      sparse_mode=0)

# ---------------- 调用点风格用例（dsa_v1.py L2658-2678 的真实参数名 + 取值）----------------
emit()
emit("-- 调用点风格用例：npu_vllm_quant_lightning_indexer 的参数名/取值 --")
cs_ok = True


def callsite_case(tag, layout_query, layout_key, cmp_ratio, qmode, expect_ni):
    torch.manual_seed(9)
    q = torch.randint(-8, 9, (1, 4, 8, 128), device=DEV).to(torch.int8)
    k = torch.randint(-8, 9, (1, 32, 1, 128), device=DEV).to(torch.int8)
    w = ((torch.rand(1, 4, 8, device=DEV) * 2 - 1) * 0.0015).to(F16)
    qs = (torch.rand(1, 4, 8, device=DEV) + 0.5).to(F16)
    ks = (torch.rand(1, 32, device=DEV) + 0.5).to(F16)
    try:
        vllm_quant_lightning_indexer(
            query=q, key=k, weights=w, query_dequant_scale=qs, key_dequant_scale=ks,
            sparse_count=8, sparse_mode=3, layout_query=layout_query, layout_key=layout_key,
            cmp_ratio=cmp_ratio, query_quant_mode=qmode, key_quant_mode=0,
            pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1, return_value=False)
        raised = False
    except NotImplementedError:
        raised = True
    good = raised if expect_ni else (not raised)
    emit(f"  {tag:52s} raised={raised}  {'PASS' if good else 'FAIL'}")
    return good


cs_ok &= callsite_case("调用点 BSND + cmp_ratio=1（可跑）", "BSND", "BSND", 1, 0, False)
cs_ok &= callsite_case("layout_query='BNSD'（非法）→ 必须显式报错", "BNSD", "BSND", 1, 0, True)
cs_ok &= callsite_case("layout_key='TND'（非法）→ 必须显式报错", "BSND", "TND", 1, 0, True)
cs_ok &= callsite_case("调用点 query_quant_mode=1 → 必须显式报错", "BSND", "BSND", 1, 1, True)
emit(f"  调用点风格合计: {'PASS' if cs_ok else 'FAIL'}")
allok = ok and cs_ok

emit()
emit(f"总体: {'全部 PASS' if allok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/vllm_quant_lightning_indexer_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
