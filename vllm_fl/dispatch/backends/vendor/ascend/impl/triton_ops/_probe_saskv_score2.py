# -*- coding: utf-8 -*-
"""Probe11：干净复刻 score 的 ori 分支（修掉上一版探针的参考错误）。

D=512（不切块），逐 lane 对比 kernel 的 sc 与 CPU 参考，并 dump 首个 D 值。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_score2.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/FlagGems/src")
from triton_ops.sparse_attn_sharedkv import _kv_elem_off  # noqa: E402

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
BS = 16
MB = 4
NROWS = 2 * MB * BS
N2 = 1
D = 512
N1 = 4
GS = 4
N_ORI = 128
BLK_K = 8
WIN_LEFT = 127


@triton.jit
def _k_s(Q, OKV, BTO, TOK_B, TOK_S1, KVLEN_O, QLEN, DBG_SC, DBG_KV,
         N1C, N2C, GS_C, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
         DC: tl.constexpr, S2C: tl.constexpr,
         N_ORIC: tl.constexpr, BLK_KC: tl.constexpr, WINL: tl.constexpr):
    row = tl.program_id(0)
    n1 = row % N1C
    t = row // N1C
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS_C
    d = tl.arange(0, DC)
    q = tl.load(Q + row * DC + d).to(tl.float32)
    kvlo = tl.load(KVLEN_O + b)
    base_o = s1 + (kvlo - tl.load(QLEN + b)) - WINL
    for c in range(0, tl.cdiv(N_ORIC, BLK_KC)):
        lane = c * BLK_KC + tl.arange(0, BLK_KC)
        window = lane < N_ORIC
        shift_w = tl.where(base_o < 0, -base_o, 0)
        row0_o = tl.maximum(base_o, 0)
        cnt = tl.minimum(N_ORIC, kvlo - row0_o)
        rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
        vld = (lane >= shift_w) & (lane < shift_w + cnt)
        off = _kv_elem_off(OKV, BTO, b, row0_o + rw, n2, BLOCK_SIZE, MAX_BLOCKS, N2C, DC,
                           S2C, True)
        kv = tl.load(OKV + off[:, None] + d[None, :])
        sc = tl.sum(kv * q[None, :], axis=1)
        tl.store(DBG_SC + row * N_ORIC + lane, tl.where(vld, sc, -1.0e30), mask=window)


def dense_of(KD, BTD, b, mb, bs):
    dd = torch.zeros(mb * bs, KD.shape[1], KD.shape[2], dtype=torch.float64)
    for lb in range(mb):
        dd[lb * bs:(lb + 1) * bs] = KD[int(BTD[b, lb])]
    return dd


def main():
    torch.manual_seed(3)
    q = (torch.randn(9, N1, D, device=DEV) * 0.1).to(torch.float16)
    okv = (torch.randn(NROWS, N2, D, device=DEV) * 0.6).to(torch.float16)
    # 身份块表 + 随机块表都测
    for tag, bt in (("identity", torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=DEV)),
            ("random", torch.tensor(
            [[1, 2, 3, 0], [7, 5, 6, 4]], dtype=torch.int32, device=DEV))):
        tok_b = torch.zeros(9, dtype=torch.int32, device=DEV)
        tok_b[5:] = 1
        tok_s1 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=torch.int32, device=DEV)
        kvlen_o = torch.tensor([64, 48], dtype=torch.int32, device=DEV)
        qlen = torch.tensor([5, 4], dtype=torch.int32, device=DEV)
        dbg_sc = torch.zeros((9 * N1, N_ORI), dtype=torch.float32, device=DEV)
        dbg_kv = torch.zeros((N_ORI, 8), dtype=torch.float16, device=DEV)
        _k_s[(9 * N1,)](q, okv, bt, tok_b, tok_s1, kvlen_o, qlen, dbg_sc, dbg_kv,
                        N1, N2, GS, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=D, S2C=MB * BS,
                        N_ORIC=N_ORI, BLK_KC=BLK_K, WINL=WIN_LEFT)
        torch.npu.synchronize()
        S = dbg_sc.detach().cpu()
        K = dbg_kv.detach().cpu()
        QD = q.detach().cpu().double()
        KD = okv.detach().cpu().double()
        BTD = bt.detach().cpu()
        for t in (0, 1, 5):
            b = int(tok_b[t].item())
            s1 = int(tok_s1[t].item())
            kvlo = int(kvlen_o[b].item())
            base = s1 + (kvlo - int(qlen[b].item())) - WIN_LEFT
            dd = dense_of(KD, BTD, b, MB, BS)
            for n1 in (0, 1, 3):
                row = t * N1 + n1
                n2 = n1 // GS
                got = S[row]
                bad = []
                for lane in range(N_ORI):
                    j = base + lane
                    if not (0 <= j < kvlo):
                        continue
                    exp = float(QD[t, n1] @ dd[j, n2])
                    g = float(got[lane])
                    if abs(g - exp) > 2e-3:
                        bad.append((lane, j, g, exp))
                print(f"[{tag}] t={t} b={b} n1={n1} base={base} 不符={len(bad)}"
                      + (f"  首例 lane={bad[0][0]} j={bad[0][1]} got={bad[0][2]:.5f}"
                         f" exp={bad[0][3]:.5f}" if bad else ""), flush=True)
        # dump 第一行：valid 区间前 12 个 lane 的 kernel/exp 值
        t, n1 = 0, 0
        b = int(tok_b[t].item())
        s1 = int(tok_s1[t].item())
        kvlo = int(kvlen_o[b].item())
        base = s1 + (kvlo - int(qlen[b].item())) - WIN_LEFT
        dd0 = dense_of(KD, BTD, b, MB, BS)
        print(f"[{tag}] row(0,0) base={base} valid lane 从 {max(0, -base)} 开始；逐列：",
              flush=True)
        got0 = S[0]
        for lane in range(max(0, -base), max(0, -base) + 12):
            j = base + lane
            exp = float(QD[t, n1] @ dd0[j, 0]) if 0 <= j < kvlo else float("nan")
            print(f"    lane={lane:3d} j={j:3d} got={float(got0[lane]):+.6f}"
                  f" exp={exp:+.6f} d={float(got0[lane]) - exp:+.3e}", flush=True)
    print("SCORE2DONE", flush=True)


if __name__ == "__main__":
    main()
