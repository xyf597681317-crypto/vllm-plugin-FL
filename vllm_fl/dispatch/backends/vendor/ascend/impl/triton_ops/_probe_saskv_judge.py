# -*- coding: utf-8 -*-
"""AscendC 同输入裁判 —— sparse_attn_sharedkv (TND + PA_ND)

对同一份输入分别算三路输出并逐元素对比：
  A) Triton  实现 (`triton_ops.sparse_attn_sharedkv`)
  B) AscendC 实现 (`torch.ops._C_ascend.npu_sparse_attn_sharedkv`)
  C) 测试自带 CPU 参考 (`ref_saskv_pa`)

TND 布局由 `cu_seqlens_q` 构造，A/B 在**同一个进程**里跑（同一份设备张量），
因此不存在"两次随机不同的输入"的问题。

用法（容器内 /workspace/FlagGems/src）：
    DSH_NPU=npu:2 python3 -W ignore triton_ops/_probe_saskv_judge.py
"""
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = os.environ.get("SASKV_DEV", "npu:0")
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


def stat(tag, x, ref=None):
    xf = x.detach().cpu().double()
    nan = int(torch.isnan(xf).sum().item())
    mx = float("nan")
    try:
        mx = float(xf.abs().max().item())
    except Exception:
        pass
    line = f"  [{tag:8s}] shape={tuple(x.shape)} dtype={x.dtype} nan={nan} absmax={mx:.4e}"
    if ref is not None:
        d = (xf - ref).abs()
        dn = int(torch.isnan(d).sum().item())
        line += f" | vs_ref: bad={dn} max|d|={(d[~torch.isnan(d)].max().item() if dn < d.numel() else float('nan')):.4e}"
    print(line, flush=True)
    return xf


def call_native(q, geo, cidx, sk, cu, kvolens, kclens, cmp_ratio=1.0):
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        q,
        geo["ori"][0], geo["cmp"][0], None, cidx,
        geo["ori"][1], geo["cmp"][1],
        cu,
        torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        torch.tensor(kclens, dtype=torch.int32, device=DEV),
        None, None, sk, None,
        1.0, int(cmp_ratio), 4, 3, 127, 0, "TND", "PA_ND", False,
    )


def run(name, qlens, kvolens, kclens, N1, N2, K, dtype, sinks=True, cmp_ratio=1, block_size=16):
    print("=" * 96, flush=True)
    print(f"CASE {name}  qlens={qlens} kvolens={kvolens} kclens={kclens} N1={N1} N2={N2} K={K} "
          f"dtype={dtype} cmp_ratio={cmp_ratio} bs={block_size}", flush=True)
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)

    r = ref_saskv_pa(q, geo, cidx, cu, qlens, kvolens, kclens, sk, 1.0, 127, cmp_ratio)
    print(f"  [ref cpu] T={T} kvolens={kvolens} qlens={qlens}", flush=True)
    stat("REF", r)

    tri = None
    try:
        tri, _, _ = sparse_attn_sharedkv(
            q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
            sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
            cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
            cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
            cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
            ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
            layout_q="TND", layout_kv="PA_ND")
        torch.npu.synchronize()
        stat("TRITON", tri, r)
    except Exception as e:
        print(f"  [TRITON] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    nat = None
    try:
        o = call_native(q, geo, cidx, sk, cu, kvolens, kclens, cmp_ratio)
        nat = o[0] if isinstance(o, (tuple, list)) else o
        torch.npu.synchronize()
        stat("ASCENDC", nat, r)
    except Exception as e:
        print(f"  [ASCENDC] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    if tri is not None and nat is not None:
        a = tri.detach().cpu().double()
        b = nat.detach().cpu().double()
        d = (a - b).abs()
        dn = int(torch.isnan(d).sum().item())
        both_nan = int((torch.isnan(a) & torch.isnan(b)).sum().item())
        v = d[~torch.isnan(d)]
        print(f"  [TRI vs ASC] nan={dn} (both_nan={both_nan}) max|d|="
              f"{(v.max().item() if v.numel() else float('nan')):.4e}", flush=True)
        # 逐 token 摘要
        n_n1 = a.shape[1]
        print("  per-token:", flush=True)
        for t in range(a.shape[0]):
            ta = a[t]
            tb = b[t]
            print(f"    t={t:2d} tri_nan={int(torch.isnan(ta).sum().item()):5d} "
                  f"asc_nan={int(torch.isnan(tb).sum().item()):5d} "
                  f"tri|max|={float(ta[~torch.isnan(ta)].abs().max().item()) if not torch.isnan(ta).all() else float('nan'):.3e} "
                  f"asc|max|={float(tb[~torch.isnan(tb)].abs().max().item()) if not torch.isnan(tb).all() else float('nan'):.3e}",
                  flush=True)
    return tri, nat, r


if __name__ == "__main__":
    which = os.environ.get("SASKV_CASES", "all")
    if which in ("all", "base"):
        run("TND+PA B2 qlen[5,4] kvo[64,48] kvc[24,16] fp16",
            [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.float16)
    if which in ("all", "bf16"):
        run("TND+PA same bf16", [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.bfloat16)
    if which in ("all", "n1"):
        run("TND+PA N1=8 N2=2 K=12 fp16 no-sink",
            [5, 4], [64, 48], [24, 16], 8, 2, 12, torch.float16, sinks=False)
    if which in ("all", "cr4"):
        run("TND+PA cmp_ratio=4 fp16", [5, 4], [64, 48], [32, 24], 4, 1, 8,
            torch.float16, cmp_ratio=4)
    print("JUDGEDONE", flush=True)
