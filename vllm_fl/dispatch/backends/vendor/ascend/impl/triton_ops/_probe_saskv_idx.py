# -*- coding: utf-8 -*-
"""Probe24：把 score kernel 里读到的 cmp 索引/cmp 段 score 原样 dump 出来。

在同一 kernel 内额外写两个调试缓冲：
  D_IDX[t*N1+n1, slot] = i0（kernel 视角）
  D_CMP[t*N1+n1, slot] = cmp 段的 sc
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_idx.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, "/workspace/FlagGems/src")

DEV = "npu:" + os.environ.get("DSH_NPU", "npu:1").split(":")[-1]
D = 512
N1 = 4
N2 = 1
GS = 4
K = 8
BLOCK = 16
N_ORI = 128
BLK_K = 8
WINL = 127
CMP_RATIO = 1


@triton.jit
def _k(Q, OKV, CKV, CIDX, BTO, BTC, TOK_B, TOK_S1, KVLEN_O, KVLEN_C, QLEN,
       D_IDX, D_CMP, D_ORI,
       N1C, N2C, GS_C, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
       DC: tl.constexpr, S2C: tl.constexpr,
       N_ORIC: tl.constexpr, KIDXC: tl.constexpr, BLK_KC: tl.constexpr,
       WINL: tl.constexpr, RATIO: tl.constexpr):
    row = tl.program_id(0)
    n1 = row % N1C
    t = row // N1C
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS_C
    d = tl.arange(0, DC)
    q = tl.load(Q + row * DC + d).to(tl.float32)
    qlen = tl.load(QLEN + b)
    kvlo = tl.load(KVLEN_O + b)
    kvlc = tl.load(KVLEN_C + b)
    base_o = s1 + (kvlo - qlen) - WINL
    # ---- ori ----
    for c in range(0, tl.cdiv(N_ORIC, BLK_KC)):
        lane = c * BLK_KC + tl.arange(0, BLK_KC)
        window = lane < N_ORIC
        j = base_o + lane
        vld = (j >= 0) & (j < kvlo)
        js = tl.where(vld, j, 0)
        off = (tl.maximum(tl.load(BTO + b * MAX_BLOCKS +
                                  tl.minimum(tl.maximum(js // BLOCK_SIZE, 0),
                                             MAX_BLOCKS - 1)), 0) * BLOCK_SIZE
               + (js % BLOCK_SIZE)) * DC
        kv = tl.load(OKV + off[:, None] + d[None, :])
        sc = tl.sum(kv * q[None, :], axis=1)
        tl.store(D_ORI + row * N_ORIC + lane, tl.where(vld, sc, -1.0e30), mask=window)
    # ---- cmp ----
    n_valid = (s1 + 1 + (kvlo - qlen)) // RATIO
    lim = tl.minimum(kvlc, n_valid)
    for c in range(0, tl.cdiv(KIDXC, BLK_KC)):
        slot = c * BLK_KC + tl.arange(0, BLK_KC)
        window = slot < KIDXC
        i0 = tl.load(CIDX + (t * N2C + n2) * KIDXC + slot, mask=window, other=-1)
        vld = (i0 >= 0) & (i0 < lim)
        r = tl.where(vld, i0, 0)
        off = (tl.maximum(tl.load(BTC + b * MAX_BLOCKS +
                                  tl.minimum(tl.maximum(r // BLOCK_SIZE, 0),
                                             MAX_BLOCKS - 1)), 0) * BLOCK_SIZE
               + (r % BLOCK_SIZE)) * DC
        kv = tl.load(CKV + off[:, None] + d[None, :])
        sc = tl.sum(kv * q[None, :], axis=1)
        tl.store(D_IDX + row * KIDXC + slot, i0, mask=window)
        tl.store(D_CMP + row * KIDXC + slot, tl.where(vld, sc, -1.0e30), mask=window)


def main():
    torch.manual_seed(81)
    qlens, kvolens, kclens = [5, 4], [64, 48], [24, 16]
    B = 2
    T = sum(qlens)
    sc_ = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc_).to(torch.float16)
    geo = {}
    for tag, lens in (("ori", kvolens), ("cmp", kclens)):
        mb = max((l + BLOCK - 1) // BLOCK for l in lens)
        nb = B * mb
        kv = (torch.randn(nb, BLOCK, N2, D, device=DEV) * 0.6).to(torch.float16)
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
    tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32),
                                    torch.tensor(qlens, dtype=torch.int32)).to(DEV)
    tok_s1 = (torch.arange(T, dtype=torch.int64) - cu.cpu()[:-1].repeat_interleave(
        torch.tensor(qlens))).to(torch.int32).to(DEV)
    kvlen_o = torch.tensor(kvolens, dtype=torch.int32, device=DEV)
    kvlen_c = torch.tensor(kclens, dtype=torch.int32, device=DEV)
    qlen = torch.tensor(qlens, dtype=torch.int32, device=DEV)
    nrows = T * N1
    d_idx = torch.full((nrows, K), -9, dtype=torch.int32, device=DEV)
    d_cmp = torch.full((nrows, K), -9.0, dtype=torch.float32, device=DEV)
    d_ori = torch.full((nrows, N_ORI), -9.0, dtype=torch.float32, device=DEV)
    _k[(nrows,)](q, geo["ori"][0], geo["cmp"][0], cidx, geo["ori"][1], geo["cmp"][1],
                 tok_b, tok_s1, kvlen_o, kvlen_c, qlen, d_idx, d_cmp, d_ori,
                 N1, N2, GS, BLOCK_SIZE=BLOCK, MAX_BLOCKS=geo["ori"][2], DC=D,
                 S2C=max(kvolens), N_ORIC=N_ORI, KIDXC=K, BLK_KC=BLK_K, WINL=WINL,
                 RATIO=CMP_RATIO)
    torch.npu.synchronize()
    IDX = d_idx.detach().cpu()
    CMP = d_cmp.detach().cpu()
    CIDX = cidx.detach().cpu()
    QD = q.detach().cpu().double()

    def dense(tag, b):
        kv, bt, mb = geo[tag]
        KD = kv.detach().cpu().double()
        BT = bt.detach().cpu()
        bs = KD.shape[1]
        dd = torch.zeros(mb * bs, N2, D, dtype=torch.float64)
        for lb in range(mb):
            dd[lb * bs:(lb + 1) * bs] = KD[int(BT[b, lb])]
        return dd

    for t in (0, 5):
        b = int(tok_b[t].item())
        s1 = int(tok_s1[t].item())
        n1 = 0
        row = t * N1 + n1
        n2 = 0
        ddc = dense("cmp", b)
        n_valid = min(kclens[b], (s1 + 1 + (kvolens[b] - qlens[b])) // CMP_RATIO)
        print(f"=== t={t} b={b} row={row} n_valid={n_valid} ===", flush=True)
        print(f"  CIDX 张量  ={CIDX[t, n2].tolist()}", flush=True)
        print(f"  kernel 读到={IDX[row].tolist()}", flush=True)
        print(f"  kernel sc  ={[round(float(x), 4) for x in CMP[row]]}", flush=True)
        print(f"  期望 sc    ="
              f"{[round(float(QD[t, n1] @ ddc[int(CIDX[t, n2, s]), n2]), 4) for s in range(K)]}",
              flush=True)
    print("IDXDONE", flush=True)


if __name__ == "__main__":
    main()
