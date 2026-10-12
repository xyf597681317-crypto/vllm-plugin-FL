# -*- coding: utf-8 -*-
"""task-3 机制探针：2-D 掩码 store 在非 2 幂形状下的静默丢写（形态矩阵）。

背景：fix-moe 实测确认「[BLK_D, BLK_C] 2-D block + 2-D 掩码 store + 较长值链」
在 cols 非 2 的幂时会静默丢掉部分 lane 的写入（lane 保持输出缓冲初值）。

本探针把「形态」当变量，逐条二分：同一份值链（rmsnorm 化 + 除法 + fp16 往返 +
就近偶数取整 + int8 饱和），只改 store 的形态：
  2D_CHAIN      : [BLK_R, BLK_C] tile + 2-D 掩码 store + 长值链      ← 怀疑触发点
  2D_CHAIN_MC   : 同上，但掩码只用列比较（1-D 广播到 2-D）
  2D_CONST      : 2-D 掩码 store，值 = 常数（无值链）
  2D_LOAD       : 2-D 掩码 store，值 = load 直通（无值链）
  2D_RESHAPE    : 2-D 计算 → tl.reshape 成 1-D → 1-D 掩码 store
  1D_CHAIN      : 标量行号 + 1-D 列 tile + 1-D 掩码 store + 同值链   ← 期望安全

判定用「双填充」：用两个不同初值各跑一次，两次都没被写过的 lane = 丢写（dropped）；
被写过但与 fp64 参考不一致 = wrong。逐元素统计，不看均值。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_form.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)

F1, F2 = -99, -77
EPS = 1e-6
QMAX = 127.0


# --------------------------------------------------------------------------
# 2-D 形态：[BLK_R, BLK_C] tile，offset = rows[:,None]*COLS + d[None,:]
# --------------------------------------------------------------------------
@triton.jit
def k_2d(X, G, Y, ROWS, COLS, EPS, INV, QMAX,
         BLK_R: tl.constexpr, BLK_C: tl.constexpr,
         VALUEMODE: tl.constexpr,   # 0=常数 1=load直通 2=长值链
         FP16RT: tl.constexpr,      # 长值链里是否插入 fp16 往返
         MASKCOL: tl.constexpr,     # 掩码只用列比较（1-D 广播）还是 2-D 与
         RESHAPE: tl.constexpr):    # 是否 reshape 成 1-D 后再 store
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS
    dr = tl.arange(0, BLK_C)
    nloop = tl.cdiv(COLS, BLK_C)

    # 趟 1：行内平方和
    ss = tl.zeros([BLK_R], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_C + dr
        m2 = rmask[:, None] & (d < COLS)[None, :]
        x = tl.load(X + rows[:, None] * COLS + d[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    rstd = 1.0 / tl.sqrt(ss * INV + EPS)

    # 趟 2：行内最大绝对值
    m1 = tl.zeros([BLK_R], dtype=tl.float32)
    if VALUEMODE == 2:
        for i in range(nloop):
            d = i * BLK_C + dr
            m2 = rmask[:, None] & (d < COLS)[None, :]
            x = tl.load(X + rows[:, None] * COLS + d[None, :], mask=m2, other=0.0).to(tl.float32)
            g = tl.load(G + d, mask=d < COLS, other=0.0).to(tl.float32)
            v = x * rstd[:, None] * g[None, :]
            m1 = tl.maximum(m1, tl.max(tl.where(m2, tl.abs(v), 0.0), axis=1))
    st = QMAX / tl.where(m1 > 0.0, m1, 1.0)

    # 趟 3：计算 + store
    for i in range(nloop):
        d = i * BLK_C + dr
        dmm = d < COLS
        if MASKCOL:
            m2 = dmm[None, :]
            store_mask = dmm
        else:
            m2 = rmask[:, None] & dmm[None, :]
            store_mask = m2
        x = tl.load(X + rows[:, None] * COLS + d[None, :], mask=m2, other=0.0).to(tl.float32)
        if VALUEMODE == 0:
            qi = tl.zeros([BLK_R, BLK_C], dtype=tl.int8)
        elif VALUEMODE == 1:
            qi = x.to(tl.int8)
        else:
            g = tl.load(G + d, mask=dmm, other=0.0).to(tl.float32)
            v = x * rstd[:, None] * g[None, :]
            z = v * st[:, None]
            if FP16RT:
                z = z.to(tl.float16).to(tl.float32)
            f = tl.floor(z)
            fr = z - f
            od = (f * 0.5) != tl.floor(f * 0.5)
            q = tl.where((fr > 0.5) | ((fr == 0.5) & od), f + 1.0, f)
            qi = q.to(tl.int8)
        if RESHAPE:
            off = tl.reshape(rows[:, None] * COLS + d[None, :], [BLK_R * BLK_C])
            qf = tl.reshape(qi, [BLK_R * BLK_C])
            mf = tl.reshape(
                (rmask[:, None] & dmm[None, :]) if not MASKCOL else dmm[None, :].broadcast_to([BLK_R, BLK_C]),
                [BLK_R * BLK_C])
            tl.store(Y + off, qf, mask=mf)
        else:
            tl.store(Y + rows[:, None] * COLS + d[None, :], qi, mask=store_mask)


# --------------------------------------------------------------------------
# 1-D 形态：标量行号 + [BLK_C] 列 tile，offset = row*COLS + d
# --------------------------------------------------------------------------
@triton.jit
def k_1d(X, G, Y, ROWS, COLS, EPS, INV, QMAX,
         BLK_C: tl.constexpr, FP16RT: tl.constexpr):
    row = tl.program_id(0)
    dr = tl.arange(0, BLK_C)
    nloop = tl.cdiv(COLS, BLK_C)
    ssv = tl.zeros([BLK_C], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_C + dr
        x = tl.load(X + row * COLS + d, mask=d < COLS, other=0.0).to(tl.float32)
        ssv += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(ssv) * INV + EPS)
    mv = tl.zeros([BLK_C], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_C + dr
        dm = d < COLS
        x = tl.load(X + row * COLS + d, mask=dm, other=0.0).to(tl.float32)
        g = tl.load(G + d, mask=dm, other=0.0).to(tl.float32)
        v = x * rstd * g
        mv = tl.maximum(mv, tl.where(dm, tl.abs(v), 0.0))
    st = QMAX / tl.where(tl.max(mv, axis=0) > 0.0, tl.max(mv, axis=0), 1.0)
    for i in range(nloop):
        d = i * BLK_C + dr
        dm = d < COLS
        x = tl.load(X + row * COLS + d, mask=dm, other=0.0).to(tl.float32)
        g = tl.load(G + d, mask=dm, other=0.0).to(tl.float32)
        z = (x * rstd * g) * st
        if FP16RT:
            z = z.to(tl.float16).to(tl.float32)
        f = tl.floor(z)
        fr = z - f
        od = (f * 0.5) != tl.floor(f * 0.5)
        q = tl.where((fr > 0.5) | ((fr == 0.5) & od), f + 1.0, f)
        tl.store(Y + row * COLS + d, q.to(tl.int8), mask=dm)


def ref_chain(x, g, eps=EPS, qmax=QMAX, fp16rt=True):
    xd = x.detach().cpu().double()
    gd = g.detach().cpu().double()
    R, C = xd.shape
    ss = (xd * xd).sum(dim=1)
    rstd = 1.0 / torch.sqrt(ss / C + eps)
    v = xd * rstd[:, None] * gd[None, :]
    m = v.abs().amax(dim=1)
    st = qmax / torch.where(m > 0, m, torch.ones_like(m))
    z = v * st[:, None]
    if fp16rt:
        z = z.to(torch.float16).double()
    return torch.round(z).clamp(-128, 127)


def two_fill(launch, rows, cols):
    outs = []
    for f in (F1, F2):
        o = torch.full((rows, cols), f, dtype=torch.int8, device=DEV)
        launch(o)
        torch.npu.synchronize()
        outs.append(o.detach().cpu())
    dropped = (outs[0] == F1) & (outs[1] == F2)
    return outs[0], dropped


def run(tag, rows, cols, blk_r, blk_c, launch, fp16rt, refv):
    try:
        got, dropped = two_fill(launch, rows, cols)
    except Exception as ex:  # noqa: BLE001
        print(f"[{tag:28s}] cols={cols:5d} EXC {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        return
    wrong = (~dropped) & (got.long() != refv.long())
    nd, nw = int(dropped.sum()), int(wrong.sum())
    flag = "DROP!" if nd else ("wrong" if nw else "clean")
    print(f"[{tag:28s}] cols={cols:5d} blk=({blk_r},{blk_c}) rows={rows} "
          f"dropped={nd:5d} wrong={nw:4d}  {flag}", flush=True)
    if nd:
        cols_with_drop = [c for c in range(cols) if dropped[:, c].any()]
        rows_with_drop = [r for r in range(rows) if dropped[r].any()]
        print(f"      rows_with_drop={rows_with_drop[:8]} (n={len(rows_with_drop)}) "
              f"cols_with_drop={cols_with_drop[:20]} (n={len(cols_with_drop)})", flush=True)
        r0 = rows_with_drop[0]
        print("      map r=%d: %s" % (r0, "".join("X" if dropped[r0, c] else "." for c in range(cols))), flush=True)


def main():
    print(f"[form] dev={DEV} triton={triton.__version__}", flush=True)
    cols_list = [17, 24, 33, 48, 100, 127, 129, 1000, 7168, 32, 64, 128, 1024]
    blk_r = int(os.environ.get("FORM_BLK_R", "1"))     # >1 = 每 program 多行（fix-moe 的复现配置）
    reps = int(os.environ.get("FORM_REPEAT", "1"))     # 同一配置重复跑，防「单次全 0」误判
    rows = 8 * blk_r
    print(f"[form] blk_r={blk_r} rows={rows} reps={reps}", flush=True)
    for rep in range(reps):
        if reps > 1:
            print(f"########## REPEAT {rep} ##########", flush=True)
        for cols in cols_list:
            blk_c = 128
            torch.manual_seed(11)
            x = (torch.randn(rows, cols, dtype=torch.float32, device=DEV) * 2.0).to(torch.float32)
            g = (torch.randn(cols, dtype=torch.float32, device=DEV) * 0.5 + 1.0)
            refv = ref_chain(x, g, fp16rt=True)

            def make_2d(mode, fp16rt, maskcol, reshape):
                def fn(o):
                    k_2d[(triton.cdiv(rows, blk_r),)](
                        x, g, o, rows, cols, EPS, 1.0 / cols, QMAX,
                        BLK_R=blk_r, BLK_C=blk_c, VALUEMODE=mode, FP16RT=fp16rt,
                        MASKCOL=maskcol, RESHAPE=reshape)
                return fn

            def make_1d(fp16rt):
                def fn(o):
                    k_1d[(rows,)](x, g, o, rows, cols, EPS, 1.0 / cols, QMAX,
                                  BLK_C=blk_c, FP16RT=fp16rt)
                return fn

            print(f"--- cols={cols} (blk_c={blk_c}, rows={rows}, blk_r={blk_r}) ---", flush=True)
            run("2D_CHAIN", rows, cols, blk_r, blk_c, make_2d(2, True, False, False), True, refv)
            run("2D_CHAIN_MASKCOL", rows, cols, blk_r, blk_c, make_2d(2, True, True, False), True, refv)
            run("1D_CHAIN", rows, cols, blk_r, blk_c, make_1d(True), True, refv)
            print("", flush=True)
    print("[form] done", flush=True)


if __name__ == "__main__":
    main()
