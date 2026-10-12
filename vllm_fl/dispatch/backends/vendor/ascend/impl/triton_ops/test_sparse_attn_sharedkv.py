"""验证 sparse_attn_sharedkv 的 Triton 实现 vs AscendC 语义。

语义：ori_kv 滑窗 [s1_abs-win_left, s1_abs+win_right] 与 cmp_kv 稀疏挑选位置的联合 softmax，
       外加逐 q head 的 attention sink（只进分母）。
输出 dtype 地板：fp16 1e-3 / bf16 4e-3；softmax_max/sum 为 fp32 → 2e-6
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float16: 1e-3, torch.bfloat16: 4e-3}
D = 512


def emit(s=""):
    L.append(str(s))


def ref(q, okv, ckv, cidx, sinks, scale, win_left, win_right, return_lse, cmp_ratio=1):
    Q = q.detach().cpu().double()
    B, S1, N1, _ = Q.shape
    OKV = okv.detach().cpu().double() if okv is not None else None
    CKV = ckv.detach().cpu().double() if ckv is not None else None
    N2 = (CKV if CKV is not None else OKV).shape[2]
    S2O = OKV.shape[1] if OKV is not None else 0
    S2C = CKV.shape[1] if CKV is not None else 0
    gs = N1 // N2
    IDX = cidx.detach().cpu() if cidx is not None else None
    SK = sinks.detach().cpu().double() if sinks is not None else None
    out = torch.zeros(B, S1, N1, D, dtype=torch.float64)
    lmx = torch.zeros(B, N2, S1, gs, dtype=torch.float64)
    lsm = torch.zeros(B, N2, S1, gs, dtype=torch.float64)
    for b in range(B):
        for s1 in range(S1):
            for n1 in range(N1):
                n2 = n1 // gs
                vecs, srcs = [], []
                if OKV is not None:
                    s1_abs = s1 + (S2O - S1)
                    lo = s1_abs - win_left
                    hi = s1_abs + win_right
                    js = [j for j in range(max(0, lo), min(S2O - 1, hi) + 1)]
                    if js:
                        jt = torch.tensor(js, dtype=torch.long)
                        vecs.append(Q[b, s1, n1] @ OKV[b, jt, n2].T)
                        srcs.append(OKV[b, jt, n2])
                if CKV is not None and IDX is not None:
                    K = IDX.shape[-1]
                    # cmp_mask_mode=3：在**原始序列位置**折叠 cmp_ratio
                    # （swa_kernel.h L405: thresHold = (cmpMaskRight + s1EndIdx + 1)/cmpRatio）
                    # 无 ori 分支时用 cmp 长度×cmp_ratio 还原原始长度（与 kernel 一致）
                    if OKV is not None:
                        ref_len = S2O
                    else:
                        ref_len = S2C * cmp_ratio
                    n_valid = (s1 + 1 + (ref_len - S1)) // cmp_ratio
                    js = [int(t) for t in IDX[b, s1, n2] if 0 <= int(t) < min(S2C, n_valid)]
                    if js:
                        jt = torch.tensor(js, dtype=torch.long)
                        vecs.append(Q[b, s1, n1] @ CKV[b, jt, n2].T)
                        srcs.append(CKV[b, jt, n2])
                if not vecs:
                    continue
                logits = torch.cat(vecs) * scale
                vals = torch.cat(srcs, dim=0)
                if SK is not None:
                    m = torch.maximum(logits.max(), SK[n1])
                    p = torch.exp(logits - m)
                    lse = p.sum() + torch.exp(SK[n1] - m)
                else:
                    m = logits.max()
                    p = torch.exp(logits - m)
                    lse = p.sum()
                out[b, s1, n1] = (p[:, None] * vals).sum(0) / lse
                if return_lse:
                    lmx[b, n2, s1, n1 % gs] = m
                    lsm[b, n2, s1, n1 % gs] = lse
    return out, lmx, lsm


def make(B, S1, S2O, S2C, N1, N2, K, dtype, seed=21, with_ori=True, with_cmp=True, invalid=0):
    torch.manual_seed(seed)
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(B, S1, N1, D, device=DEV) * sc).to(dtype)
    okv = (torch.randn(B, S2O, N2, D, device=DEV) * 0.6).to(dtype) if with_ori else None
    ckv = (torch.randn(B, S2C, N2, D, device=DEV) * 0.6).to(dtype) if with_cmp else None
    cidx = None
    if with_cmp:
        cidx = torch.empty(B, S1, N2, K, dtype=torch.int32, device=DEV)
        for b in range(B):
            for s1 in range(S1):
                for n2 in range(N2):
                    cidx[b, s1, n2] = torch.randperm(S2C, device=DEV)[:K].to(torch.int32)
        if invalid:
            torch.manual_seed(777)
            m = torch.rand(cidx.shape, device=DEV) < 0.25
            cidx = torch.where(m, torch.full_like(cidx, -1), cidx)
    return q, okv, ckv, cidx


def run_case(name, B, S1, S2O, S2C, N1, N2, K, dtype, scale=1.0, with_ori=True, with_cmp=True,
             sinks=True, invalid=0, return_lse=False, seed=21):
    q, okv, ckv, cidx = make(B, S1, S2O, S2C, N1, N2, K, dtype, seed, with_ori, with_cmp, invalid)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    out, lmx, lsm = sparse_attn_sharedkv(
        q, ori_kv=okv, cmp_kv=ckv, cmp_sparse_indices=cidx, sinks=sk,
        softmax_scale=scale, ori_win_left=127, ori_win_right=0,
        return_softmax_lse=return_lse)
    torch.npu.synchronize()
    rout, rmx, rsm = ref(q, okv, ckv, cidx, sk, scale, 127, 0, return_lse)
    err = (out.detach().cpu().double() - rout).abs().max().item()
    floor = FLOOR[dtype]
    ok = err <= floor
    extra = ""
    if return_lse:
        em = (lmx.detach().cpu().double() - rmx).abs().max().item()
        es = (lsm.detach().cpu().double() - rsm).abs().max().item()
        # softmax_max 是 O(1) 量级 → 绝对地板 2e-6；
        # softmax_sum 是 Σexp（量级 ~ 参与项数，这里最多 144）→ 按自身量级换算 fp32 地板
        sc_s = max(1.0, rsm.abs().max().item())
        es_rel = es / sc_s
        ok = ok and em <= 2e-6 and es_rel <= 2e-6
        extra = (f" lse_max_err={em:.2e} lse_sum_err={es:.2e}"
                 f"(rel={es_rel:.2e}, scale={sc_s:.0f})")
    emit(f"  {name:52s} err={err:.3e}(<= {floor:.0e}){extra}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 106)
emit("sparse_attn_sharedkv: Triton vs AscendC 语义（滑窗 ori_kv + 稀疏 cmp_kv + attention sink）")
emit(f"  D={D}; 滑窗=128(win_left=127,win_right=0); 地板 fp16 1e-3 / bf16 4e-3; lse(fp32) 2e-6")
emit("=" * 106)

F16 = torch.float16
BF = torch.bfloat16
ok = True

ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 全分支", 1, 8, 192, 48, 4, 1, 16, F16)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 bf16 全分支", 1, 8, 192, 48, 4, 1, 16, BF)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 无 sink", 1, 8, 192, 48, 4, 1, 16, F16, sinks=False)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=8 N2=2 K=24 fp16 gs=4", 1, 8, 192, 48, 8, 2, 24, F16)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 含-1无效", 1, 8, 192, 48, 4, 1, 16, F16, invalid=1)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 仅cmp分支", 1, 8, 192, 48, 4, 1, 16, F16, with_ori=False)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 仅ori滑窗", 1, 8, 192, 48, 4, 1, 16, F16, with_cmp=False)
ok &= run_case("B1 S1=8 S2o=192 S2c=48 N1=4 N2=1 K=16 fp16 return_lse", 1, 8, 192, 48, 4, 1, 16, F16, return_lse=True)

emit()
emit(f"总体(BSND 等价性): {'全部 PASS' if ok else '存在 FAIL'}")

# ============ 真实 layout：layout_q="TND" + layout_kv="PA_ND"（ori/cmp 双分页）============
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


def ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, scale, win_left, cmp_ratio):
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
            # cmp_mask_mode=3：在**原始序列位置**上折叠 cmp_ratio
            # （swa_kernel.h L405 / scfa_kernel.h L373；这里的 s1_abs 用 ori 长度对齐）
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


def run_case_saskv_pa(name, qlens, kvolens, kclens, N1, N2, K, dtype, sinks=True,
                      cmp_ratio=1, block_size=16):
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size)
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
    err = (out.detach().cpu().double() - r).abs().max().item()
    floor = FLOOR[dtype]
    good = err <= floor
    line = f"  {name:50s} err={err:.3e}(<= {floor:.0e})  {'PASS' if good else 'FAIL'}"
    emit(line)
    print(line, flush=True)
    return good


emit()
emit("-- 真实 layout：layout_q='TND' + layout_kv='PA_ND'（ori/cmp 双分页 cache）--")
ok &= run_case_saskv_pa("TND+PA B2 qlen[5,4] kvo[64,48] kvc[24,16] fp16",
                        [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.float16)
ok &= run_case_saskv_pa("TND+PA B2 qlen[5,4] kvo[64,48] kvc[24,16] bf16",
                        [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.bfloat16)
ok &= run_case_saskv_pa("TND+PA B2 N1=8 N2=2 K=12 fp16 无 sink",
                        [5, 4], [64, 48], [24, 16], 8, 2, 12, torch.float16, sinks=False)
ok &= run_case_saskv_pa("TND+PA B2 cmp_ratio=4 fp16（生产只允许 4/128）",
                        [5, 4], [64, 48], [32, 24], 4, 1, 8, torch.float16, cmp_ratio=4)

# ---------------- 调用点风格用例：按 op_host def.cpp 的参数名 + 默认取值 ----------------
emit()
emit("-- 调用点风格用例：SparseAttnSharedkv 的参数名/默认取值 --")
cs_ok = True
q_, okv_, ckv_, cidx_ = make(1, 8, 192, 48, 4, 1, 16, torch.float16, seed=61)


def cs_case(tag, expect_ni, **kw):
    base = dict(q=q_, ori_kv=okv_, cmp_kv=ckv_, cmp_sparse_indices=cidx_,
                softmax_scale=0.5, ori_win_left=127, ori_win_right=0, cmp_ratio=1,
                ori_mask_mode=4, cmp_mask_mode=3, ori_kv_stride=0, cmp_kv_stride=0,
                layout_q="BSND", layout_kv="BSND", return_softmax_lse=False)
    base.update(kw)
    try:
        sparse_attn_sharedkv(**base)
        raised = False
    except NotImplementedError:
        raised = True
    good = raised if expect_ni else (not raised)
    emit(f"  {tag:52s} raised={raised}  {'PASS' if good else 'FAIL'}")
    return good


cs_ok &= cs_case("调用点 BSND 全默认（可跑）", False)
cs_ok &= cs_case("layout_q='BNSD'（非法）→ 必须显式报错", True, layout_q="BNSD")
cs_ok &= cs_case("ori_mask_mode=0 → 必须显式报错", True, ori_mask_mode=0)
cs_ok &= cs_case("cmp_mask_mode=0 → 必须显式报错", True, cmp_mask_mode=0)
cs_ok &= cs_case("ori_kv_stride=512 → 必须显式报错", True, ori_kv_stride=512)
cs_ok &= cs_case("ori_sparse_indices 非 None → 必须显式报错", True,
                 ori_sparse_indices=torch.zeros(1, 1, 1, 1, dtype=torch.int32, device=DEV))
cs_ok &= cs_case("seqused_kv（已支持）→ 不应抛错", False,
                 seqused_kv=torch.full((1,), 192, dtype=torch.int32, device=DEV))
emit(f"  调用点风格合计: {'PASS' if cs_ok else 'FAIL'}")
allok = ok and cs_ok

emit()
emit(f"总体: {'全部 PASS' if allok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/sparse_attn_sharedkv_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
