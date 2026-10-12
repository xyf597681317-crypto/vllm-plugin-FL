# -*- coding: utf-8 -*-
"""Probe21：区分 "SB 写入错位" vs "sc 计算错"。

同一个 kernel 内：
  DBG_ECHO[lane] = 原样写回 lane 的浮点值（走与 SB 完全相同的 store 路径/mask）
  DBG_SC[lane]   = 真正的 score
  DBG_LANE[lane] = 显式写入 lane 值（再走一次 store）
如果 ECHO 正确而 SC 错 → 计算问题；如果 ECHO 也错 → store 路径问题。

数据用测试夹具同款：seed=81, dtype=fp16, qlens=[5,4], kvolens=[64,48], bt 随机置换。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_store.py
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


@triton.jit
def _k(Q, OKV, BTO, TOK_B, TOK_S1, KVLEN_O, QLEN, ECHO, SC, LANE, CIDX, DBG_IDX,
       N1C, N2C, GS_C, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
       DC: tl.constexpr, S2C: tl.constexpr, N_ORIC: tl.constexpr, BLK_KC: tl.constexpr,
       WINL: tl.constexpr, KIDX: tl.constexpr):
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
        j = base_o + lane
        vld = (j >= 0) & (j < kvlo)
        js = tl.where(vld, j, 0)
        bidx = tl.minimum(tl.maximum(js // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
        page = tl.maximum(tl.load(BTO + b * MAX_BLOCKS + bidx), 0)
        off = (page * BLOCK_SIZE + (js % BLOCK_SIZE)) * DC
        kv = tl.load(OKV + off[:, None] + d[None, :])
        sc = tl.sum(kv * q[None, :], axis=1) * 1.0
        # 三路写出，全部使用与 SB 相同的 mask/形状
        tl.store(ECHO + row * N_ORIC + lane, js.to(tl.float32), mask=window)
        tl.store(SC + row * N_ORIC + lane, tl.where(vld, sc, -1.0e30), mask=window)
        tl.store(LANE + row * N_ORIC + lane, lane.to(tl.float32), mask=window)


def main():
    torch.manual_seed(81)
    B = 2
    qlens = [5, 4]
    kvolens = [64, 48]
    kclens = [24, 16]
    T = sum(qlens)
    sc_ = 1.0 / (D ** 0.5)
    q = (torch.randn(T, N1, D, device=DEV) * sc_).to(torch.float16)
    mb = max((l + BLOCK - 1) // BLOCK for l in kvolens)
    nb = B * mb
    okv = (torch.randn(nb, BLOCK, N2, D, device=DEV) * 0.6).to(torch.float16)
    bt = torch.zeros(B, mb, dtype=torch.int32, device=DEV)
    for b in range(B):
        bt[b] = (torch.randperm(mb, device=DEV) + b * mb).to(torch.int32)
    cidx = torch.zeros(T, N2, K, dtype=torch.int32, device=DEV)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(qlens, dtype=torch.int32, device=DEV).cumsum(0)
    for t in range(T):
        b = int((cu[1:] <= t).sum().item())
        for n2 in range(N2):
            cidx[t, n2] = torch.randperm(max(1, kclens[b]), device=DEV)[:K].to(torch.int32)
    tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32),
                                    torch.tensor(qlens, dtype=torch.int32)).to(DEV)
    tok_s1 = (torch.arange(T, dtype=torch.int64)
              - cu.cpu()[:-1].repeat_interleave(torch.tensor(qlens))).to(torch.int32).to(DEV)
    kvlen_o = torch.tensor(kvolens, dtype=torch.int32, device=DEV)
    qlen = torch.tensor(qlens, dtype=torch.int32, device=DEV)
    nrows = T * N1
    echo = torch.full((nrows, N_ORI), -7.0, dtype=torch.float32, device=DEV)
    scb = torch.full((nrows, N_ORI), -7.0, dtype=torch.float32, device=DEV)
    laneb = torch.full((nrows, N_ORI), -7.0, dtype=torch.float32, device=DEV)
    _k[(nrows,)](q, okv, bt, tok_b, tok_s1, kvlen_o, qlen, echo, scb, laneb, cidx, cidx,
                 N1, N2, GS, BLOCK_SIZE=BLOCK, MAX_BLOCKS=mb, DC=D, S2C=mb * BLOCK,
                 N_ORIC=N_ORI, BLK_KC=BLK_K, WINL=127, KIDX=K)
    torch.npu.synchronize()
    E = echo.detach().cpu()
    S = scb.detach().cpu()
    L = laneb.detach().cpu()
    KD = okv.detach().cpu().double()
    BTD = bt.detach().cpu()
    QD = q.detach().cpu().double()
    print("bt =", BTD.tolist(), flush=True)
    for t in (0, 5):
        b = int(tok_b[t].item())
        s1 = int(tok_s1[t].item())
        kvlo = int(kvlen_o[b].item())
        base = s1 + (kvlo - int(qlen[b].item())) - 127
        n1 = 0
        row = t * N1 + n1
        dd = torch.zeros(mb * BLOCK, N2, D, dtype=torch.float64)
        for lb in range(mb):
            dd[lb * BLOCK:(lb + 1) * BLOCK] = KD[int(BTD[b, lb])]
        # 1) lane 数组是否正确
        lane_bad = [i for i in range(N_ORI)
                    if abs(float(L[row, i]) - i) > 1e-6]
        # 2) echo(js) 是否 = max(base+lane,0)（vld 为真时就是 base+lane）
        echo_bad = []
        for i in range(N_ORI):
            j = base + i
            if 0 <= j < kvlo:
                if abs(float(E[row, i]) - j) > 1e-6:
                    echo_bad.append((i, float(E[row, i]), j))
        # 3) score vs 参考
        sc_bad = []
        for i in range(N_ORI):
            j = base + i
            if 0 <= j < kvlo:
                exp = float(QD[t, n1] @ dd[j, 0])
                if abs(float(S[row, i]) - exp) > 2e-3:
                    sc_bad.append((i, j, float(S[row, i]), exp))
        print(f"t={t} b={b} base={base} row={row} 有效={kvlo - max(0, -base)}", flush=True)
        print(f"  LANE 错={len(lane_bad)}   ECHO(js) 错={len(echo_bad)}   SC 错={len(sc_bad)}",
              flush=True)
        for x in echo_bad[:4]:
            print(f"    ECHO lane={x[0]} got={x[1]} exp={x[2]}", flush=True)
        for x in sc_bad[:4]:
            print(f"    SC   lane={x[0]} j={x[1]} got={x[2]:.6f} exp={x[3]:.6f}"
                  f" d={x[2] - x[3]:.3e}", flush=True)
    print("STOREDONE", flush=True)


if __name__ == "__main__":
    main()
