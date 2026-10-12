# -*- coding: utf-8 -*-
"""Probe10：1:1 复刻 _saskv_score_kernel 的 ori 分支，dump 逐 lane 的 kv/off/sc。

只跑 1 行（row 0）且 D 不切块（BLK_D=D），与 CPU 参考逐列对比。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_score.py
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
SCALE = 1.0 / (D ** 0.5)     # 测试里 run_case_saskv_pa 用 1.0，这里两种都试


@triton.jit
def _k_score_repro(Q, OKV, BTO, TOK_B, TOK_S1, KVLEN_O, QLEN, DBG_OFF, DBG_SC,
                   N1C, N2C, GS_C, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
                   DC: tl.constexpr, S2C: tl.constexpr, SCALEC,
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
        sc = tl.sum(kv * q[None, :], axis=1) * SCALEC
        tl.store(DBG_OFF + row * N_ORIC + lane, off, mask=window)
        tl.store(DBG_SC + row * N_ORIC + lane, tl.where(vld, sc, -1.0e30), mask=window)


def main():
    torch.manual_seed(3)
    q = (torch.randn(9, N1, D, device=DEV) * (1.0 / D ** 0.5)).to(torch.float16)
    okv = (torch.randn(NROWS, N2, D, device=DEV) * 0.6).to(torch.float16)
    bt = torch.tensor([[1, 2, 3, 0], [7, 5, 6, 4]], dtype=torch.int32, device=DEV)
    tok_b = torch.zeros(9, dtype=torch.int32, device=DEV)
    tok_b[5:] = 1
    tok_s1 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=torch.int32, device=DEV)
    kvlen_o = torch.tensor([64, 48], dtype=torch.int32, device=DEV)
    qlen = torch.tensor([5, 4], dtype=torch.int32, device=DEV)
    dbg_off = torch.full((9 * N1, N_ORI), -1, dtype=torch.int32, device=DEV)
    dbg_sc = torch.zeros((9 * N1, N_ORI), dtype=torch.float32, device=DEV)

    _k_score_repro[(9 * N1,)](q, okv, bt, tok_b, tok_s1, kvlen_o, qlen, dbg_off, dbg_sc,
                              N1, N2, GS, BLOCK_SIZE=BS, MAX_BLOCKS=MB, DC=D, S2C=MB * BS,
                              SCALEC=1.0, N_ORIC=N_ORI, BLK_KC=BLK_K, WINL=WIN_LEFT)
    torch.npu.synchronize()
    O = dbg_off.detach().cpu()
    S = dbg_sc.detach().cpu()
    QD = q.detach().cpu().double()
    KD = okv.detach().cpu().double()
    BTD = bt.detach().cpu()

    def dense(b):
        dd = torch.zeros(MB * BS, N2, D, dtype=torch.float64)
        for lb in range(MB):
            dd[lb * BS:(lb + 1) * BS] = KD[int(BTD[b, lb])]
        return dd

    for t in (0, 5):
        b = int(tok_b[t].item())
        s1 = int(tok_s1[t].item())
        kvlo = int(kvlen_o[b].item())
        base = s1 + (kvlo - int(qlen[b].item())) - WIN_LEFT
        shift = max(0, -base)
        dd = dense(b)
        for n1 in (0, 4):
            row = t * N1 + n1
            n2 = n1 // GS
            o = O[row]
            s = S[row]
            bad = []
            for lane in range(N_ORI):
                j = base + lane
                if not (0 <= j < kvlo):
                    continue
                exp_sc = float(QD[t, n1] @ dd[j, n2])
                got_sc = float(s[lane])
                # 该 lane 实际读到的行（由 off 反推）
                off = int(o[lane])
                got_row = off // (N2 * D)
                exp_off = ((dense_row_of(BTD, b, j, BS, MB)) * N2 + n2) * D
                if abs(got_sc - exp_sc) > 1e-3 or off != exp_off:
                    bad.append((lane, j, off, exp_off, got_sc, exp_sc))
            print(f"t={t} n1={n1} base={base} shift={shift} 有效lane={kvlo - max(0,-base)}"
                  f" 不符={len(bad)}", flush=True)
            for x in bad[:6]:
                print(f"    lane={x[0]} j={x[1]} off_got={x[2]} off_exp={x[3]}"
                      f" sc_got={x[4]:.6f} sc_exp={x[5]:.6f} d={x[4] - x[5]:.3e}", flush=True)
    print("SCOREDONE", flush=True)


def dense_row_of(BTD, b, j, BS, MB):
    lb = min(j // BS, MB - 1)
    return int(BTD[b, lb]) * BS + j % BS


if __name__ == "__main__":
    main()
