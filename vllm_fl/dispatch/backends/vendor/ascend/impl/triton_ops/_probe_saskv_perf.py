# -*- coding: utf-8 -*-
"""性能对照：ROW_MODE=1（逐行 1-D）vs ROW_MODE=0（2-D tile）在 TND+PA 生产形状下的耗时。

ROW_MODE 是 constexpr，两种模式各编译一次；计时前先 warmup 一次。
"""
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("SASKV_DEV", "npu:0")
D = 512


def build(B, S1, S2O, S2C, N1, N2, K, dtype, block_size=16, seed=81):
    torch.manual_seed(seed)
    T = B * S1
    sc = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc).to(dtype)
    cu = torch.arange(0, (B + 1) * S1, S1, dtype=torch.int32, device=DEV)
    geo = {}
    for tag, S2 in (("ori", S2O), ("cmp", S2C)):
        mb = (S2 + block_size - 1) // block_size
        nb = B * mb
        kv = (torch.randn(nb, block_size, N2, D, device=DEV) * 0.6).to(dtype)
        bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
        for b in range(B):
            bt[b] = (torch.randperm(mb, device=DEV) + b * mb).to(torch.int32)
        geo[tag] = (kv, bt)
    cidx = torch.empty(T, N2, K, dtype=torch.int32, device=DEV)
    for t in range(T):
        b = min(t // S1, B - 1)
        for n2 in range(N2):
            cidx[t, n2] = torch.randperm(S2C, device=DEV)[:K].to(torch.int32)
    return q, geo, cidx, cu


def main():
    import importlib
    import triton_ops.sparse_attn_sharedkv as S
    from triton_ops.sparse_attn_sharedkv import sparse_attn_sharedkv

    cases = [
        ("测试形状 B2 S1=5/4 N1=4 D=512", 2, 5, 64, 24, 4, 1, 8, torch.float16),
        ("放大 B8 S1=64 N1=4 D=512", 8, 64, 512, 128, 4, 1, 64, torch.float16),
    ]
    for name, B, S1, S2O, S2C, N1, N2, K, dt in cases:
        q, geo, cidx, cu = build(B, S1, S2O, S2C, N1, N2, K, dt)
        qlens = [S1] * B
        kvolens = [S2O] * B
        kclens = [S2C] * B
        sk = (torch.randn(N1, device=DEV) * 2.0).to(torch.float32)
        print(f"\n== {name} ==", flush=True)
        for mode in ("1", "0"):
            os.environ["SASKV_ROW_MODE"] = mode
            kw = dict(q=q, ori_kv=geo["ori"][0], cmp_kv=geo["cmp"][0],
                      cmp_sparse_indices=cidx, sinks=sk, softmax_scale=1.0,
                      ori_win_left=127, ori_win_right=0, cmp_ratio=1, cu_seqlens_q=cu,
                      cu_seqlens_ori_kv=torch.tensor(kvolens, dtype=torch.int32, device=DEV),
                      cu_seqlens_cmp_kv=torch.tensor(kclens, dtype=torch.int32, device=DEV),
                      ori_block_table=geo["ori"][1], cmp_block_table=geo["cmp"][1],
                      layout_q="TND", layout_kv="PA_ND")
            try:
                sparse_attn_sharedkv(**kw)
                torch.npu.synchronize()
                t0 = time.time()
                n = 3
                for _ in range(n):
                    sparse_attn_sharedkv(**kw)
                torch.npu.synchronize()
                el = (time.time() - t0) / n
                print(f"  ROW_MODE={mode}: {el*1e3:9.2f} ms/call", flush=True)
            except Exception as e:
                print(f"  ROW_MODE={mode}: EXCEPTION {type(e).__name__}: {str(e)[:200]}", flush=True)
    os.environ.pop("SASKV_ROW_MODE", None)
    print("PERFDONE", flush=True)


if __name__ == "__main__":
    main()
