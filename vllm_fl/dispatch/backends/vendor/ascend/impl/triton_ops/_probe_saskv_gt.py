# -*- coding: utf-8 -*-
"""Probe12：设备端 ground truth。

K1: 直接读 dd[j]（j 为标量行号），算 q_row·dd[j]  —— 标量索引，无任何块表逻辑
K2: 走完整窗口/块表路径（与算子一致）
两者逐 lane 对比，就能判定"块表寻址"到底对不对，不依赖任何 CPU 参考。
跑法：DSH_NPU=npu:1 python3 -W ignore /tmp/_probe_saskv_gt.py
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
N_ORI = 128
BLK_K = 8
WIN_LEFT = 127


@triton.jit
def _k_gt(Q, OKV, DBG, JLO, NLANE: tl.constexpr, DC: tl.constexpr, N2C: tl.constexpr,
          DSTRIDE: tl.constexpr):
    """DBG[lane] = q · OKV[(JLO+lane), 0, :]（标量行号 → 绝对地址）"""
    lane = tl.arange(0, NLANE)
    d = tl.arange(0, DC)
    q = tl.load(Q + d).to(tl.float32)
    row = JLO + lane
    ptr = (row * N2C + 0) * DSTRIDE
    kv = tl.load(OKV + ptr[:, None] + d[None, :])
    tl.store(DBG + lane, tl.sum(kv * q[None, :], axis=1))


@triton.jit
def _k_win(Q, OKV, BTO, DBG, BASE, BIDX, BLOCK_SIZE: tl.constexpr,
           MAX_BLOCKS: tl.constexpr, NLANE: tl.constexpr, DC: tl.constexpr,
           N2C: tl.constexpr, S2C: tl.constexpr):
    """走 `_kv_elem_off` + 窗口相对量的完整路径。"""
    lane = tl.arange(0, NLANE)
    d = tl.arange(0, DC)
    q = tl.load(Q + d).to(tl.float32)
    shift_w = tl.where(BASE < 0, -BASE, 0)
    row0_o = tl.maximum(BASE, 0)
    cnt = tl.minimum(NLANE, S2C - row0_o)
    rw = tl.minimum(tl.maximum(lane - shift_w, 0), cnt - 1)
    off = _kv_elem_off(OKV, BTO, BIDX, row0_o + rw, 0, BLOCK_SIZE, MAX_BLOCKS, N2C, DC,
                       S2C, True)
    kv = tl.load(OKV + off[:, None] + d[None, :])
    tl.store(DBG + lane, tl.sum(kv * q[None, :], axis=1))


def main():
    torch.manual_seed(3)
    q0 = (torch.randn(D, device=DEV) * 0.1).to(torch.float16)
    okv = (torch.randn(NROWS, N2, D, device=DEV) * 0.6).to(torch.float16)
    bt = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device=DEV)
    gt = torch.zeros(N_ORI, dtype=torch.float32, device=DEV)
    win = torch.zeros(N_ORI, dtype=torch.float32, device=DEV)

    for base in (-68, -67, 0, 5):
        # ground truth：逻辑行 j = base+lane 的**物理**行（用块表换算后直接读）
        jlo = max(0, -base)
        phys = []
        for lane in range(N_ORI):
            j = base + lane
            if j < 0 or j >= MB * BS:
                phys.append(-1)
            else:
                phys.append(int(bt[j // BS].item()) * BS + j % BS)
        # 只取连续的非负段做 GT
        start = next((i for i, p in enumerate(phys) if p >= 0), 0)
        end = start
        while end < N_ORI and phys[end] >= 0 and (end == start or phys[end] == phys[end - 1] + 1):
            end += 1
        n = end - start
        if n <= 0:
            continue
        gt.fill_(0)
        win.fill_(0)
        _k_gt[(1,)](q0, okv, gt, phys[start], NLANE=n, DC=D, N2C=N2, DSTRIDE=D)
        _k_win[(1,)](q0, okv, bt, win, base, 0, BLOCK_SIZE=BS, MAX_BLOCKS=MB, NLANE=n,
                     DC=D, N2C=N2, S2C=MB * BS)
        torch.npu.synchronize()
        g = gt[:n].detach().cpu()
        w = win[:n].detach().cpu()
        dif = (g - w).abs()
        print(f"base={base} 段=[lane{start}..lane{end-1}] j=[{base + start}..{base + end - 1}]"
              f" 物理行=[{phys[start]}..{phys[end - 1]}] (连续)  最大差={dif.max().item():.3e}",
              flush=True)
        if dif.max().item() > 1e-3:
            bad = (dif > 1e-3).nonzero().flatten().tolist()
            for i in bad[:6]:
                print(f"    lane{start + i}: gt={g[i].item():+.6f} win={w[i].item():+.6f}"
                      f" 物理行={phys[start + i]}", flush=True)
    print("GTDONE", flush=True)


if __name__ == "__main__":
    main()
