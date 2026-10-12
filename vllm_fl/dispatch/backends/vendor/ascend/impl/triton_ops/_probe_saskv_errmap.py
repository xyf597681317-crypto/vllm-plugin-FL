# -*- coding: utf-8 -*-
"""精确误差地图 + SB ground truth（设备端逐 lane 复算 q·kv）。

输出：
  A) 每个 (token, n1) 的 err / 最大误差出现的 d 位置
  B) 对 row 选定的若干行：SB 的 ori 段逐 lane vs 设备端独立复算的 q·kv
     （独立复算 = 用 CPU 从**设备张量**按块表还原 dense 后算，非测试参考）
  C) out 里 max 误差的 d 索引 → 判断是"整行错"还是"少数 lane 错"
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_ops.sparse_attn_sharedkv as S  # noqa: E402
from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv  # noqa: E402

DEV = os.environ.get("SASKV_DEV", "npu:0")
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


def dense_of(geo, tag, b, qlens, kvolens, kclens):
    """按块表还原 batch b 的 dense KV（cpu float64）。"""
    kv, bt, mb = geo[tag]
    KD = kv.detach().cpu().double()
    BT = bt.detach().cpu()
    return torch.stack([KD[int(BT[b, lb])] for lb in range(mb)]).reshape(mb * KD.shape[1], KD.shape[2], D)


def ref_logits_rows(q, geo, cidx, qlens, kvolens, kclens, t, n1, win_left=127):
    """一个 (t,n1) 的参考 logits（ori 段顺序 + cmp 段顺序），fp64。"""
    Q = q.detach().cpu().double()
    IDX = cidx.detach().cpu()
    b = 0 if t < qlens[0] else 1
    s1 = t - (0 if b == 0 else qlens[0])
    N2 = IDX.shape[1]
    gs = Q.shape[1] // N2
    n2 = n1 // gs
    dd_o = dense_of(geo, "ori", b, qlens, kvolens, kclens)
    dd_c = dense_of(geo, "cmp", b, qlens, kvolens, kclens)
    kl_o, kl_c = kvolens[b], kclens[b]
    s1_abs = s1 + (kl_o - qlens[b])
    js = [j for j in range(max(0, s1_abs - win_left), min(kl_o - 1, s1_abs) + 1)]
    ori = Q[t, n1] @ dd_o[js, n2].T if js else torch.zeros(0, dtype=torch.float64)
    n_valid = min(kl_c, (s1_abs + 1) // 1)
    jc = [int(i) for i in IDX[t, n2] if 0 <= int(i) < n_valid]
    cmp_ = Q[t, n1] @ dd_c[jc, n2].T if jc else torch.zeros(0, dtype=torch.float64)
    return js, ori, jc, cmp_


def run(tag, qlens, kvolens, kclens, N1, N2, K, dtype, block_size=16, sinks=True, cmp_ratio=1,
        probe_rows=(0, 4, 16, 20, 32)):
    print("=" * 100, flush=True)
    print(f"[{tag}] qlens={qlens} kvolens={kvolens} kclens={kclens} N1={N1} N2={N2} K={K} "
          f"{dtype} cmp_ratio={cmp_ratio}", flush=True)
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype, block_size)
    sk = None
    if sinks:
        torch.manual_seed(5)
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    os.environ["SASKV_DUMP_SCORES"] = "1"
    S._DBG.clear()
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
        sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
        cmp_ratio=cmp_ratio, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    sb = S._DBG["sb"].detach().cpu().double()
    o = out.detach().cpu().double()
    # 参考 out（fp64）
    R = torch.zeros(T, N1, D, dtype=torch.float64)
    SK = sk.detach().cpu().double() if sk is not None else None
    for t in range(T):
        for n1 in range(N1):
            js, ori, jc, cmpr = ref_logits_rows(q, geo, cidx, qlens, kvolens, kclens, t, n1)
            b = 0 if t < qlens[0] else 1
            dd_o = dense_of(geo, "ori", b, qlens, kvolens, kclens)
            dd_c = dense_of(geo, "cmp", b, qlens, kvolens, kclens)
            N2 = cidx.shape[1]
            gs = N1 // N2
            n2 = n1 // gs
            vals = []
            if len(js):
                vals.append(dd_o[js, n2])
            if len(jc):
                vals.append(dd_c[jc, n2])
            if not vals:
                continue
            logits = torch.cat([ori, cmpr])
            V = torch.cat(vals, 0)
            if SK is not None:
                m = torch.maximum(logits.max(), SK[n1])
                p = torch.exp(logits - m)
                lse = p.sum() + torch.exp(SK[n1] - m)
            else:
                p = torch.exp(logits - logits.max())
                lse = p.sum()
            R[t, n1] = (p[:, None] * V).sum(0) / lse

    # A) 逐 (t,n1) 误差
    print("   t n1 |      err      out_nan  ref_absmax  d_of_max", flush=True)
    for t in range(T):
        for n1 in range(min(N1, 4)):
            d = (o[t, n1] - R[t, n1]).abs()
            nn = int(torch.isnan(d).sum())
            am = float(R[t, n1].abs().max())
            dd_ = int(torch.nan_to_num(d, nan=-1).argmax()) if d.numel() else -1
            print(f"  {t:2d} {n1:2d} | {float(torch.nan_to_num(d, nan=float('nan')).max()):10.4e}  {nn:7d}  {am:10.4e}  {dd_:4d}",
                  flush=True)
        break_flag = False

    # B) SB ground truth：逐 lane 比较（只用 ori 段前 128 列 + cmp 段）
    print("  -- SB vs 设备张量独立复算（fp64） --", flush=True)
    for t, n1 in [(0, 0), (4, 0), (5, 0), (8, 0)]:
        row = t * N1 + n1
        js, ori, jc, cmpr = ref_logits_rows(q, geo, cidx, qlens, kvolens, kclens, t, n1)
        sbrow = sb[row]
        NEGV = S._NEG
        got_o = sbrow[:128]
        got_c = sbrow[128:128 + cidx.shape[-1]]
        # ori：SB 的 lane i 对应 j = base_o + i，base 由参考 js[0] 反推
        b = 0 if t < qlens[0] else 1
        s1 = t - (0 if b == 0 else qlens[0])
        base_o = s1 + (kvolens[b] - qlens[b]) - 127
        mism = []
        for i in range(128):
            j = base_o + i
            if 0 <= j < kvolens[b]:
                idx = j - js[0]
                if 0 <= idx < len(ori):
                    g = float(got_o[i])
                    e = float(ori[idx])
                    if abs(g - e) > 1e-5 * max(1.0, abs(e)):
                        mism.append((i, j, g, e))
        print(f"  t={t} n1={n1}: ori base_o={base_o} js0={js[0] if js else None} n_js={len(js)} "
              f"ori_mismatch={len(mism)}", flush=True)
        for m in mism[:8]:
            print(f"      lane={m[0]} j={m[1]} got={m[2]:.6f} exp={m[3]:.6f} ratio={m[2]/m[3] if m[3] else float('inf'):.4f}",
                  flush=True)
        mism_c = 0
        for k in range(len(jc)):
            g = float(got_c[k])
            e = float(cmpr[k])
            if abs(g - e) > 1e-5 * max(1.0, abs(e)):
                mism_c += 1
                if mism_c <= 8:
                    print(f"      cmp slot={k} idx={jc[k]} got={g:.6f} exp={e:.6f}", flush=True)
        print(f"      cmp_mismatch={mism_c}/{len(jc)} cidx={cidx[t,0].cpu().tolist()}", flush=True)
    del os.environ["SASKV_DUMP_SCORES"]


if __name__ == "__main__":
    run("B2 [5,4] kvo[64,48] kvc[24,16] N1=4 fp16 (测试用例)",
        [5, 4], [64, 48], [24, 16], 4, 1, 8, torch.float16)
    print("ERRMAPDONE", flush=True)
