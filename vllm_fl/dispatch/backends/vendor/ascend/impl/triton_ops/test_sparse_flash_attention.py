"""验证 sparse_flash_attention 的 Triton 实现 vs AscendC 语义（MLA 稀疏注意力）。

输出 dtype 地板：fp16 1e-3 / bf16 4e-3；softmax_max/sum 为 fp32 → 2e-6
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.sparse_flash_attention import sparse_flash_attention  # noqa: E402

DEV = os.environ.get("DSH_NPU", "npu:0")   # 允许换卡（多 agent 并发时避免抢 npu:0）
# ⚠️ 必须显式 set_device：否则张量分配在 npu:N 上、而 kernel 发到**当前设备**（默认 0），
#    指针跨设备 → 设备侧 `507035 / aivec error`（实测：不设时 npu:12/npu:13 在第一个用例就挂，
#    npu:0 因为恰好等于当前设备所以正常）。
torch.npu.set_device(DEV)
L = []
FLOOR = {torch.float16: 1e-3, torch.bfloat16: 4e-3}
D = 512
DR = 64


def emit(s=""):
    L.append(str(s))


def ref(q, k, v, qr, kr, ind, scale, sparse_mode, bs, return_lse):
    """fp64 参考：MLA 稀疏注意力（只在 sparse_indices 指定的位置上 softmax）。"""
    Q = q.detach().cpu().double()
    K = k.detach().cpu().double()
    V = v.detach().cpu().double()
    QR = qr.detach().cpu().double()
    KR = kr.detach().cpu().double()
    IDX = ind.detach().cpu()
    B, S1, N1, _ = Q.shape
    _, S2, N2, _ = K.shape
    gs = N1 // N2
    KK = IDX.shape[-1]
    out = torch.zeros(B, S1, N1, D, dtype=torch.float64)
    lmax = torch.zeros(B, N2, S1, gs, dtype=torch.float64)
    lsum = torch.zeros(B, N2, S1, gs, dtype=torch.float64)
    for b in range(B):
        for s1 in range(S1):
            for n1 in range(N1):
                n2 = n1 // gs
                js = []
                for slot in range(KK):
                    i0 = int(IDX[b, s1, n2, slot])
                    if i0 < 0:
                        continue
                    for sub in range(bs):
                        j = i0 * bs + sub
                        if j >= S2:
                            continue
                        if sparse_mode == 3 and j > s1 + (S2 - S1):
                            continue
                        js.append(j)
                if not js:
                    continue
                jt = torch.tensor(js, dtype=torch.long)
                sc = (Q[b, s1, n1] @ K[b, jt, n2].T + QR[b, s1, n1] @ KR[b, jt, n2].T) * scale
                m = sc.max()
                p = torch.exp(sc - m)
                lse = p.sum()
                out[b, s1, n1] = (p[:, None] * V[b, jt, n2]).sum(0) / lse
                if return_lse:
                    lmax[b, n2, s1, n1 % gs] = m
                    lsum[b, n2, s1, n1 % gs] = lse
    return out, lmax, lsum


def make(B, S1, S2, N1, N2, K, dtype, seed=11):
    torch.manual_seed(seed)
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(B, S1, N1, D, device=DEV) * sc).to(dtype)
    qr = (torch.randn(B, S1, N1, DR, device=DEV) * 0.2).to(dtype)
    k = (torch.randn(B, S2, N2, D, device=DEV) * sc).to(dtype)
    kr = (torch.randn(B, S2, N2, DR, device=DEV) * 0.2).to(dtype)
    # V 幅度取 0.3：保证 |out| <= 1，bf16/fp16 的量化地板（4e-3/1e-3）才是有效判据
    v = (torch.randn(B, S2, N2, D, device=DEV) * 0.3).to(dtype)
    # 每个 (b,s1,n2) 随机选 K 个不重复的 KV 位置
    idx = torch.empty(B, S1, N2, K, dtype=torch.int32, device=DEV)
    for b in range(B):
        for s1 in range(S1):
            for n2 in range(N2):
                idx[b, s1, n2] = torch.randperm(S2, device=DEV)[:K].to(torch.int32)
    return q, k, v, qr, kr, idx


def run_case(name, B, S1, S2, N1, N2, K, dtype, scale=1.0, sparse_mode=0, bs=1,
             invalid=0, return_lse=False, seed=11):
    q, k, v, qr, kr, idx = make(B, S1, S2, N1, N2, K, dtype, seed)
    if invalid:
        torch.manual_seed(999)
        m = torch.rand(idx.shape, device=DEV) < 0.25
        idx = torch.where(m, torch.full_like(idx, -1), idx)
    out, lmax, lsum = sparse_flash_attention(
        q, k, v, idx, qr, kr, scale_value=scale, sparse_block_size=bs,
        sparse_mode=sparse_mode, return_softmax_lse=return_lse)
    torch.npu.synchronize()
    rout, rmax, rsum = ref(q, k, v, qr, kr, idx, scale, sparse_mode, bs, return_lse)
    err = (out.detach().cpu().double() - rout).abs().max().item()
    floor = FLOOR[dtype]
    ok = err <= floor
    extra = ""
    if return_lse:
        em = (lmax.detach().cpu().double() - rmax).abs().max().item()
        es = (lsum.detach().cpu().double() - rsum).abs().max().item()
        # softmax_max O(1) → 绝对地板 2e-6；softmax_sum ~Σexp（量级 ~K）→ 按其自身量级换算
        sc_s = max(1.0, rsum.abs().max().item())
        es_rel = es / sc_s
        ok = ok and em <= 2e-6 and es_rel <= 2e-6
        extra = (f" lse_max_err={em:.2e} lse_sum_err={es:.2e}"
                 f"(rel={es_rel:.2e}, scale={sc_s:.0f})")
    line = (f"  {name:50s} err={err:.3e}(<= {floor:.0e}){extra}  {'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    return ok


emit("=" * 104)
emit("sparse_flash_attention: Triton vs AscendC 语义（MLA 稀疏注意力, fp64 参考）")
emit(f"  D={D} D_rope={DR}; 地板 fp16 1e-3 / bf16 4e-3; softmax_max/sum(fp32) 2e-6")
emit("=" * 104)

F16 = torch.float16
BF = torch.bfloat16
ok = True

ok &= run_case("B1 S1=8 S2=64 N1=4 N2=1 K=16 fp16 mode0", 1, 8, 64, 4, 1, 16, F16)
ok &= run_case("B1 S1=8 S2=64 N1=4 N2=1 K=16 bf16 mode0", 1, 8, 64, 4, 1, 16, BF)
ok &= run_case("B1 S1=32 S2=32 N1=4 N2=1 K=16 fp16 下三角(mode3)", 1, 32, 32, 4, 1, 16, F16, sparse_mode=3)
ok &= run_case("B1 S1=8 S2=64 N1=8 N2=2 K=24 fp16 含-1无效项", 1, 8, 64, 8, 2, 24, F16, invalid=1)
ok &= run_case("B2 S1=4 S2=48 N1=4 N2=1 K=8 fp16 block_size=2", 2, 4, 48, 4, 1, 8, F16, bs=2)
ok &= run_case("B1 S1=16 S2=96 N1=4 N2=1 K=32 fp16 非整除", 1, 16, 96, 4, 1, 32, F16)
ok &= run_case("B1 S1=8 S2=64 N1=4 N2=1 K=16 fp16 return_lse", 1, 8, 64, 4, 1, 16, F16, return_lse=True)
ok &= run_case("B1 S1=32 S2=32 N1=4 N2=1 K=16 bf16 下三角+无效", 1, 32, 32, 4, 1, 16, BF, sparse_mode=3, invalid=1)

emit()
emit(f"总体(BSND 等价性): {'全部 PASS' if ok else '存在 FAIL'}")

# ================= TND + PA_BSND（真实调用点的 layout 组合）=================
# op_host def.cpp: layout_query="TND" → query [T,N1,D]；layout_kv="PA_BSND" →
# key/value [numBlocks, blockSize, N2, D] + block_table[b, logicalBlock] → physicalBlock
def make_pa(B, qlens, kvlens, N1, N2, K, dtype, block_size=16, seed=71, invalid=0):
    """构造 TND query + PA_BSND 分页 KV cache。"""
    torch.manual_seed(seed)
    T = sum(qlens)
    max_blocks = max((kl + block_size - 1) // block_size for kl in kvlens)
    num_blocks = B * max_blocks
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc).to(dtype)
    qr = (torch.randn(T, N1, DR, device=DEV) * 0.2).to(dtype)
    key = (torch.randn(num_blocks, block_size, N2, D, device=DEV) * sc).to(dtype)
    kr = (torch.randn(num_blocks, block_size, N2, DR, device=DEV) * 0.2).to(dtype)
    v = (torch.randn(num_blocks, block_size, N2, D, device=DEV) * 0.3).to(dtype)
    # 每个 batch 用一段连续物理块（但打乱顺序，确保寻址真的走 block_table）
    bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=DEV)
    for b in range(B):
        perm = torch.randperm(max_blocks, device=DEV) + b * max_blocks
        bt[b] = perm.to(torch.int32)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    idx = torch.empty(T, N2, K, dtype=torch.int32, device=DEV)
    for t in range(T):
        for n2 in range(N2):
            # 只从该 token 所属 batch 的 kv 长度内选位置（并排序，便于复现）
            b = int((cu[1:] <= t).sum().item())
            idx[t, n2] = torch.randperm(kvlens[b], device=DEV)[:K].to(torch.int32)
    if invalid:
        torch.manual_seed(999)
        m = torch.rand(idx.shape, device=DEV) < 0.25
        idx = torch.where(m, torch.full_like(idx, -1), idx)
    return q, key, v, qr, kr, idx, bt, cu


def ref_pa(q, key, v, qr, kr, idx, bt, cu, qlens, kvlens, scale, sparse_mode,
           block_size, return_lse):
    Q = q.detach().cpu().double()
    QR = qr.detach().cpu().double()
    K = key.detach().cpu().double()
    KR = kr.detach().cpu().double()
    V = v.detach().cpu().double()
    IDX = idx.detach().cpu()
    BT = bt.detach().cpu()
    CU = cu.detach().cpu()
    T, N1, _ = Q.shape
    _, block_size_k, N2, _ = K.shape
    max_blocks = BT.shape[1]
    S2 = block_size_k * max_blocks
    outs, lmaxs, lsums = [], [], []
    for t in range(T):
        b = int((CU[1:] <= t).sum().item())
        s1 = t - int(CU[b])
        dense = torch.zeros(S2, N2, D, dtype=torch.float64)
        for lb in range(max_blocks):
            pb = int(BT[b, lb])
            dense[lb * block_size_k:(lb + 1) * block_size_k] = K[pb]
        for n1 in range(N1):
            n2 = n1 // (N1 // N2)
            js = []
            for i0 in IDX[t, n2].tolist():
                if i0 < 0 or i0 >= kvlens[b]:
                    continue
                if sparse_mode == 3 and i0 > s1 + (kvlens[b] - qlens[b]):
                    continue
                js.append(int(i0))
            if not js:
                outs.append(torch.zeros(D, dtype=torch.float64))
                lmaxs.append(0.0)
                lsums.append(0.0)
                continue
            jt = torch.tensor(js, dtype=torch.long)
            kd = K[BT[b, jt // block_size_k].long(), jt % block_size_k, n2]
            krd = KR[BT[b, jt // block_size_k].long(), jt % block_size_k, n2]
            vd = V[BT[b, jt // block_size_k].long(), jt % block_size_k, n2]
            sc = (Q[t, n1] @ kd.T + QR[t, n1] @ krd.T) * scale
            m = sc.max()
            p = torch.exp(sc - m)
            lse = p.sum()
            outs.append((p[:, None] * vd).sum(0) / lse)
            lmaxs.append(m.item())
            lsums.append(lse.item())
    return torch.stack(outs), lmaxs, lsums


def run_case_pa(name, qlens, kvlens, N1, N2, K, dtype, scale=1.0, sparse_mode=3,
                invalid=0, return_lse=False, block_size=16, seed=71):
    q, key, v, qr, kr, idx, bt, cu = make_pa(len(qlens), qlens, kvlens,
                                             N1, N2, K, dtype, block_size, seed, invalid)
    B = len(qlens)
    dev = DEV
    # 真实调用点参数组合：layout_query="TND", layout_kv="PA_BSND", attention_mode=2
    out, lmax, lsum = sparse_flash_attention(
        query=q, key=key, value=v, sparse_indices=idx,
        query_rope=qr, key_rope=kr, scale_value=scale, sparse_block_size=1,
        block_table=bt, actual_seq_lengths_query=cu,
        actual_seq_lengths_kv=torch.tensor(kvlens, dtype=torch.int32, device=dev),
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=sparse_mode,
        attention_mode=2, return_softmax_lse=return_lse)
    torch.npu.synchronize()
    rout_flat, rmaxs, rsums = ref_pa(q, key, v, qr, kr, idx, bt, cu, qlens, kvlens,
                                     scale, sparse_mode, block_size, return_lse)
    # ref_pa 按 (t, n1) 顺序产出 → reshape 成 TND 的 [T, N1, D]
    rout = rout_flat.reshape(sum(qlens), N1, D)
    err = (out.detach().cpu().double() - rout).abs().max().item()
    floor = FLOOR[dtype]
    good = err <= floor
    extra = ""
    if return_lse:
        T = sum(qlens)
        gs = N1 // N2
        rm = torch.zeros(N2, T, gs, dtype=torch.float64)
        rs = torch.zeros(N2, T, gs, dtype=torch.float64)
        for t in range(T):
            for n1 in range(N1):
                rm[n1 // gs, t, n1 % gs] = rmaxs[t * N1 + n1]
                rs[n1 // gs, t, n1 % gs] = rsums[t * N1 + n1]
        em = (lmax.detach().cpu().double() - rm).abs().max().item()
        es = (lsum.detach().cpu().double() - rs).abs().max().item()
        sc_s = max(1.0, rs.abs().max().item())
        good = good and em <= 2e-6 and es / sc_s <= 2e-6
        extra = f" lse_max_err={em:.2e} lse_sum_err={es:.2e}(rel={es / sc_s:.2e})"
    line = f"  {name:50s} err={err:.3e}(<= {floor:.0e}){extra}  {'PASS' if good else 'FAIL'}"
    emit(line)
    print(line, flush=True)
    return good


emit()
emit("-- 真实 layout 组合：layout_query='TND' + layout_kv='PA_BSND' + attention_mode=2 --")
ok &= run_case_pa("TND+PA B2 qlen[6,4] kvlen[24,16] fp16 下三角", [6, 4], [24, 16], 4, 1, 8, F16)
ok &= run_case_pa("TND+PA B2 qlen[6,4] kvlen[24,16] bf16 下三角", [6, 4], [24, 16], 4, 1, 8, BF)
ok &= run_case_pa("TND+PA B1 qlen[8] kvlen[40] fp16 mode0", [8], [40], 4, 1, 12, F16, sparse_mode=0)
ok &= run_case_pa("TND+PA B2 qlen[5,7] kvlen[20,32] N2=2 fp16 含-1", [5, 7], [20, 32], 8, 2, 10, F16, invalid=1)

# ---------------- 调用点风格用例（device_op.py L682-698 的真实参数名 + 取值）----------------
emit()
emit("-- 调用点风格用例：npu_sparse_flash_attention 的参数名/取值 --")
cs_ok = True


def callsite_case(tag, layout_query, layout_kv, attention_mode, expect_ni):
    """按调用点风格构造调用；不支持的 layout 必须显式 NotImplementedError。"""
    q, k, v, qr, kr, idx = make(1, 8, 64, 4, 1, 16, torch.float16, seed=41)
    try:
        sparse_flash_attention(
            query=q, key=k, value=v, sparse_indices=idx,
            query_rope=qr, key_rope=kr, scale_value=0.5, sparse_block_size=1,
            layout_query=layout_query, layout_kv=layout_kv,
            sparse_mode=3, attention_mode=attention_mode)
        raised = False
    except NotImplementedError as e:
        raised = True
        msg = str(e)[:60]
    good = raised if expect_ni else (not raised)
    emit(f"  {tag:52s} raised={raised}  {'PASS' if good else 'FAIL'}")
    return good


cs_ok &= callsite_case("调用点 BSND + attention_mode=2（可跑）", "BSND", "BSND", 2, False)
cs_ok &= callsite_case("layout_query='BNSD'（非法）→ 必须显式报错", "BNSD", "BSND", 2, True)
cs_ok &= callsite_case("调用点 attention_mode=0 → 必须显式报错", "BSND", "BSND", 0, True)
emit(f"  调用点风格合计: {'PASS' if cs_ok else 'FAIL'}")

emit()
emit(f"总体: {'全部 PASS' if (ok and cs_ok) else '存在 FAIL'}")
allok = ok and cs_ok
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/sparse_flash_attention_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
