# -*- coding: utf-8 -*-
"""罪证定位：坏 lane 到底读到了哪一行？

已知（设备实测）：t=0 的 ori 段 lane 68..71（对应逻辑行 j=0..3）数值错，
lane 72+（j>=4）精确；cmp 段 8/8 全对。
本探针把 SB 里这些 lane 的值与**所有候选 (kv 张量, 页, 块内行, head)** 的 q·kv 逐一比对，
直接指认 kernel 实际读到的地址。
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


def main():
    qlens, kvolens, kclens = [5, 4], [64, 48], [24, 16]
    N1, N2, K = 4, 1, 8
    dtype = torch.float16
    q, geo, cidx, cu, T = make_saskv_pa(qlens, kvolens, kclens, N1, N2, K, dtype)
    torch.manual_seed(5)
    sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
    os.environ["SASKV_DUMP_SCORES"] = "1"
    S._DBG.clear()
    out, _, _ = sparse_attn_sharedkv(
        q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0], cmp_sparse_indices=cidx,
        sinks=sk, softmax_scale=1.0, ori_win_left=127, ori_win_right=0,
        cmp_ratio=1, cu_seqlens_q=cu,
        cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
        cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
        ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
        layout_q="TND", layout_kv="PA_ND")
    torch.npu.synchronize()
    sb = S._DBG["sb"].detach().cpu().double()
    out_c = out.detach().cpu().double()

    Q = q.detach().cpu().double()
    BT_o = geo["ori"][1].detach().cpu()
    BT_c = geo["cmp"][1].detach().cpu()
    KD_o = geo["ori"][0].detach().cpu().double()
    KD_c = geo["cmp"][0].detach().cpu().double()
    mb_o, mb_c = geo["ori"][2], geo["cmp"][2]
    bs = 16

    print(f"  ori BT={BT_o.tolist()}  cmp BT={BT_c.tolist()}  blockSize={bs}", flush=True)

    for (t, n1) in [(0, 0), (0, 1), (5, 0)]:
        b = 0 if t < qlens[0] else 1
        s1 = t - (0 if b == 0 else qlens[0])
        base_o = s1 + (kvolens[b] - qlens[b]) - 127
        row = t * N1 + n1
        qv = Q[t, n1]
        print(f"\n  === t={t} n1={n1} b={b} s1={s1} base_o={base_o} ===", flush=True)
        for lane in range(0, 128):
            j = base_o + lane
            if not (0 <= j < kvolens[b]):
                continue
            if lane > abs(base_o) + 5:      # 只看坏 lane 附近
                break
            got = float(sb[row, lane])
            # 候选：所有 (页 in 0..nb-1, 块内行, head) → 与 q 的点积
            best = []
            for page in range(KD_o.shape[0]):
                for r in range(bs):
                    for n2 in range(N2):
                        e = float(qv @ KD_o[page, r, n2])
                        if abs(e - got) < 1e-4 * max(1.0, abs(got)):
                            best.append(("ori_raw", page, r, n2, e))
            for page in range(KD_c.shape[0]):
                for r in range(bs):
                    for n2 in range(N2):
                        e = float(qv @ KD_c[page, r, n2])
                        if abs(e - got) < 1e-4 * max(1.0, abs(got)):
                            best.append(("cmp_raw", page, r, n2, e))
            # 期望那条（按块表）
            exp_page_o = int(BT_o[b, j // bs])
            exp = float(qv @ KD_o[exp_page_o, j % bs, n1 // 4])
            print(f"    lane={lane:3d} j={j:3d} got={got:12.6f} exp={exp:10.6f} "
                  f"hits={best[:3]}", flush=True)
    del os.environ["SASKV_DUMP_SCORES"]
    print("WHICHDONE", flush=True)


if __name__ == "__main__":
    main()
