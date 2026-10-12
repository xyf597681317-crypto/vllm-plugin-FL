# -*- coding: utf-8 -*-
"""task-3 探针：rms_norm_dynamic_quant 的 2-D 掩码 store（L148/L160）。

两块证据：
  A) 形态对照：本文件内置「生产 kernel 的 2-D 原形」与「1-D block 改写形」两份拷贝，
     同一份数据 + 双填充 sentinel 检测，逐元素统计 dropped / wrong。
  B) 生产算子端到端：非 2 幂 D 下与 fp64 参考逐元素比对（不看均值），
     并用「同一输入跑两次、比较两次输出」检测「未写入 lane 保持 uninitialized」。

用法（容器内）：DSH_NPU=npu:3 python3 -W ignore _probe_2dstore_rmsdq.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:3")
torch.npu.set_device(DEV)
F1, F2 = -99, -77
QMAX = 127.0
NEG = -1.0e30


# ==========================================================================
# 2-D 原形（与 triton_ops/rms_norm_dynamic_quant.py 的 kernel 逐字一致）
# ==========================================================================
@triton.jit
def _rmsdq_2d_orig(
    X, GAMMA, SMOOTH1, SMOOTH2, BETA,
    Y1, Y2, SCALE1, SCALE2,
    ROWS, D,
    EPS, INV_D, QMAX,
    BLK_R: tl.constexpr, BLK_D: tl.constexpr,
    HAS_SMOOTH1: tl.constexpr, HAS_SMOOTH2: tl.constexpr, HAS_BETA: tl.constexpr,
    DO_Q1: tl.constexpr, DO_Q2: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS
    dr = tl.arange(0, BLK_D)
    nloop = tl.cdiv(D, BLK_D)
    ss = tl.zeros([BLK_R], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_D + dr
        m2 = rmask[:, None] & (d < D)[None, :]
        x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    rstd = 1.0 / tl.sqrt(ss * INV_D + EPS)
    m1 = tl.zeros([BLK_R], dtype=tl.float32)
    mv2 = tl.zeros([BLK_R], dtype=tl.float32)
    if DO_Q1:
        for i in range(nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
            t = x * rstd[:, None] * g[None, :]
            if HAS_BETA:
                bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
                t = t + bt[None, :]
            if HAS_SMOOTH1:
                s1 = tl.load(SMOOTH1 + d, mask=d < D, other=1.0).to(tl.float32)
                z1 = t * s1[None, :]
            else:
                z1 = t
            m1 = tl.maximum(m1, tl.max(tl.where(m2, tl.abs(z1), 0.0), axis=1))
    if DO_Q2:
        for i in range(nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
            t = x * rstd[:, None] * g[None, :]
            if HAS_BETA:
                bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
                t = t + bt[None, :]
            if HAS_SMOOTH2:
                s2 = tl.load(SMOOTH2 + d, mask=d < D, other=1.0).to(tl.float32)
                z2 = t * s2[None, :]
            else:
                z2 = t
            mv2 = tl.maximum(mv2, tl.max(tl.where(m2, tl.abs(z2), 0.0), axis=1))
    st1 = QMAX / tl.where(m1 > 0.0, m1, 1.0)
    st2 = QMAX / tl.where(mv2 > 0.0, mv2, 1.0)
    for i in range(nloop):
        d = i * BLK_D + dr
        m2 = rmask[:, None] & (d < D)[None, :]
        x = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + d, mask=d < D, other=0.0).to(tl.float32)
        t = x * rstd[:, None] * g[None, :]
        if HAS_BETA:
            bt = tl.load(BETA + d, mask=d < D, other=0.0).to(tl.float32)
            t = t + bt[None, :]
        if DO_Q1:
            if HAS_SMOOTH1:
                s1 = tl.load(SMOOTH1 + d, mask=d < D, other=1.0).to(tl.float32)
                z1 = t * s1[None, :]
            else:
                z1 = t
            v1 = z1 * st1[:, None]
            f1 = tl.floor(v1)
            fr1 = v1 - f1
            od1 = (f1 * 0.5) != tl.floor(f1 * 0.5)
            q1 = tl.where((fr1 > 0.5) | ((fr1 == 0.5) & od1), f1 + 1.0, f1)
            tl.store(Y1 + rows[:, None] * D + d[None, :], q1.to(tl.int8), mask=m2)
        if DO_Q2:
            if HAS_SMOOTH2:
                s2 = tl.load(SMOOTH2 + d, mask=d < D, other=1.0).to(tl.float32)
                z2 = t * s2[None, :]
            else:
                z2 = t
            v2 = z2 * st2[:, None]
            f2 = tl.floor(v2)
            fr2 = v2 - f2
            od2 = (f2 * 0.5) != tl.floor(f2 * 0.5)
            q2 = tl.where((fr2 > 0.5) | ((fr2 == 0.5) & od2), f2 + 1.0, f2)
            tl.store(Y2 + rows[:, None] * D + d[None, :], q2.to(tl.int8), mask=m2)
    if DO_Q1:
        tl.store(SCALE1 + rows, tl.where(m1 > 0.0, m1 / QMAX, 0.0), mask=rmask)
    if DO_Q2:
        tl.store(SCALE2 + rows, tl.where(mv2 > 0.0, mv2 / QMAX, 0.0), mask=rmask)


# ==========================================================================
# 1-D 改写形：标量行号 + [BLK_D] 列 tile + 1-D 掩码 store
# ==========================================================================
@triton.jit
def _rmsdq_1d_fixed(
    X, GAMMA, SMOOTH1, SMOOTH2, BETA,
    Y1, Y2, SCALE1, SCALE2,
    ROWS, D,
    EPS, INV_D, QMAX,
    BLK_D: tl.constexpr,
    HAS_SMOOTH1: tl.constexpr, HAS_SMOOTH2: tl.constexpr, HAS_BETA: tl.constexpr,
    DO_Q1: tl.constexpr, DO_Q2: tl.constexpr,
):
    row = tl.program_id(0)
    dr = tl.arange(0, BLK_D)
    nloop = tl.cdiv(D, BLK_D)
    ssv = tl.zeros([BLK_D], dtype=tl.float32)
    for i in range(nloop):
        d = i * BLK_D + dr
        x = tl.load(X + row * D + d, mask=d < D, other=0.0).to(tl.float32)
        ssv += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(ssv, axis=0) * INV_D + EPS)
    m1 = tl.zeros([], dtype=tl.float32)
    mv2 = tl.zeros([], dtype=tl.float32)
    if DO_Q1:
        for i in range(nloop):
            d = i * BLK_D + dr
            dm = d < D
            x = tl.load(X + row * D + d, mask=dm, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=dm, other=0.0).to(tl.float32)
            t = x * rstd * g
            if HAS_BETA:
                t = t + tl.load(BETA + d, mask=dm, other=0.0).to(tl.float32)
            if HAS_SMOOTH1:
                t = t * tl.load(SMOOTH1 + d, mask=dm, other=1.0).to(tl.float32)
            m1 = tl.maximum(m1, tl.max(tl.where(dm, tl.abs(t), 0.0), axis=0))
    if DO_Q2:
        for i in range(nloop):
            d = i * BLK_D + dr
            dm = d < D
            x = tl.load(X + row * D + d, mask=dm, other=0.0).to(tl.float32)
            g = tl.load(GAMMA + d, mask=dm, other=0.0).to(tl.float32)
            t = x * rstd * g
            if HAS_BETA:
                t = t + tl.load(BETA + d, mask=dm, other=0.0).to(tl.float32)
            if HAS_SMOOTH2:
                t = t * tl.load(SMOOTH2 + d, mask=dm, other=1.0).to(tl.float32)
            mv2 = tl.maximum(mv2, tl.max(tl.where(dm, tl.abs(t), 0.0), axis=0))
    st1 = QMAX / tl.where(m1 > 0.0, m1, 1.0)
    st2 = QMAX / tl.where(mv2 > 0.0, mv2, 1.0)
    for i in range(nloop):
        d = i * BLK_D + dr
        dm = d < D
        x = tl.load(X + row * D + d, mask=dm, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + d, mask=dm, other=0.0).to(tl.float32)
        t = x * rstd * g
        if HAS_BETA:
            t = t + tl.load(BETA + d, mask=dm, other=0.0).to(tl.float32)
        if DO_Q1:
            if HAS_SMOOTH1:
                z1 = t * tl.load(SMOOTH1 + d, mask=dm, other=1.0).to(tl.float32)
            else:
                z1 = t
            v1 = z1 * st1
            f1 = tl.floor(v1)
            fr1 = v1 - f1
            od1 = (f1 * 0.5) != tl.floor(f1 * 0.5)
            q1 = tl.where((fr1 > 0.5) | ((fr1 == 0.5) & od1), f1 + 1.0, f1)
            tl.store(Y1 + row * D + d, q1.to(tl.int8), mask=dm)
        if DO_Q2:
            if HAS_SMOOTH2:
                z2 = t * tl.load(SMOOTH2 + d, mask=dm, other=1.0).to(tl.float32)
            else:
                z2 = t
            v2 = z2 * st2
            f2 = tl.floor(v2)
            fr2 = v2 - f2
            od2 = (f2 * 0.5) != tl.floor(f2 * 0.5)
            q2 = tl.where((fr2 > 0.5) | ((fr2 == 0.5) & od2), f2 + 1.0, f2)
            tl.store(Y2 + row * D + d, q2.to(tl.int8), mask=dm)
    if DO_Q1:
        tl.store(SCALE1 + row, tl.where(m1 > 0.0, m1 / QMAX, 0.0))
    if DO_Q2:
        tl.store(SCALE2 + row, tl.where(mv2 > 0.0, mv2 / QMAX, 0.0))


def ref_fp64(x, gamma, s1, s2, beta, eps, do1, do2):
    xd = x.detach().cpu().double()
    g = gamma.detach().cpu().double()
    R, D = xd.shape
    ss = (xd * xd).sum(dim=1)
    rstd = 1.0 / torch.sqrt(ss * (1.0 / D) + eps)
    t = xd * rstd[:, None] * g[None, :]
    if beta is not None:
        t = t + beta.detach().cpu().double()[None, :]

    def q(z):
        m = z.abs().amax(dim=1)
        safe = m > 0
        st = torch.where(safe, QMAX / m, torch.zeros_like(m))
        return (torch.round(z * st[:, None]).clamp(-128, 127),
                torch.where(safe, m / QMAX, torch.zeros_like(m)))

    o1 = q(t * (s1.detach().cpu().double()[None, :] if s1 is not None else 1.0)) if do1 else (None, None)
    o2 = q(t * (s2.detach().cpu().double()[None, :] if s2 is not None else 1.0)) if do2 else (None, None)
    return o1, o2


def launch_2d(x, gm, s1, s2, y1, sc1, R, D, blk, smooth):
    _rmsdq_2d_orig[(R,)](x, gm, s1, s2, x, y1, y1, sc1, sc1, R, D,
                         1e-6, 1.0 / D, QMAX, BLK_R=1, BLK_D=blk,
                         HAS_SMOOTH1=smooth, HAS_SMOOTH2=smooth, HAS_BETA=False,
                         DO_Q1=True, DO_Q2=False)


def launch_1d(x, gm, s1, s2, y1, sc1, R, D, blk, smooth):
    _rmsdq_1d_fixed[(R,)](x, gm, s1, s2, x, y1, y1, sc1, sc1, R, D,
                          1e-6, 1.0 / D, QMAX, BLK_D=blk,
                          HAS_SMOOTH1=smooth, HAS_SMOOTH2=smooth, HAS_BETA=False,
                          DO_Q1=True, DO_Q2=False)


def kernel_cmp(tag, launch, R, D, blk, smooth=True, seed=3):
    torch.manual_seed(seed)
    x = (torch.randn(R, D, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gm = (torch.randn(D, dtype=torch.float32, device=DEV) * 0.5 + 1.0).to(torch.float16)
    s1 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5)).to(torch.float16) if smooth else x
    s2 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5)).to(torch.float16) if smooth else x
    (rq1, rs1), _ = ref_fp64(x, gm, s1 if smooth else None, s2 if smooth else None,
                             None, 1e-6, True, False)
    outs = []
    try:
        for f in (F1, F2):
            y1 = torch.full((R, D), f, dtype=torch.int8, device=DEV)
            sc1 = torch.full((R,), NEG, dtype=torch.float32, device=DEV)
            launch(x, gm, s1, s2, y1, sc1, R, D, blk, smooth)
            torch.npu.synchronize()
            outs.append((y1.detach().cpu(), sc1.detach().cpu()))
    except Exception as ex:  # noqa: BLE001
        print(f"  [{tag:10s}] D={D:5d} blk={blk:4d} EXC {type(ex).__name__}: {str(ex)[:110]}", flush=True)
        return
    dropped = (outs[0][0] == F1) & (outs[1][0] == F2)
    wrong = (~dropped) & (outs[0][0].long() != rq1.long())
    nd, nw = int(dropped.sum()), int(wrong.sum())
    se = float((outs[0][1].double() - rs1).abs().max())
    flag = "DROP!" if nd else ("wrong" if nw else "clean")
    print(f"  [{tag:10s}] D={D:5d} blk={blk:4d} dropped={nd:5d} wrong={nw:5d}/{R * D} "
          f"scale_err={se:.2e}  {flag}", flush=True)
    if nd:
        r0 = next(r for r in range(R) if dropped[r].any())
        print("      r=%d: %s" % (r0, "".join("X" if dropped[r0, c] else "." for c in range(D))), flush=True)


def e2e(tag, R, D, smooth=True, seed=5):
    """生产算子端到端：逐元素比对 + 「跑两次比较」检测 uninitialized lane。"""
    from triton_ops.rms_norm_dynamic_quant import rms_norm_dynamic_quant
    torch.manual_seed(seed)
    x = (torch.randn(R, D, dtype=torch.float32, device=DEV) * 2.0).to(torch.float16)
    gm = (torch.randn(D, dtype=torch.float32, device=DEV) * 0.5 + 1.0).to(torch.float16)
    s1 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5)).to(torch.float16) if smooth else None
    s2 = ((torch.rand(D, dtype=torch.float32, device=DEV) + 0.5)).to(torch.float16) if smooth else None
    (rq1, rs1), _ = ref_fp64(x, gm, s1, s2, None, 1e-6, True, False)
    res = []
    try:
        for pat in (-99, 123):
            junk1 = torch.full((R, D), pat, dtype=torch.int8, device=DEV)
            junk2 = torch.full((R,), 0.0, dtype=torch.float32, device=DEV)
            del junk1, junk2
            y1, y2, sc1, sc2 = rms_norm_dynamic_quant(x, gm, s1, s2, None, 1e-6, [True, False])
            torch.npu.synchronize()
            res.append((y1.detach().cpu(), sc1.detach().cpu()))
    except Exception as ex:  # noqa: BLE001
        print(f"  [{tag:14s}] D={D:5d} EXC {type(ex).__name__}: {str(ex)[:110]}", flush=True)
        return
    a, b = res[0][0], res[1][0]
    nondet = (a != b)
    wrong = (a.long() != rq1.long())
    se = float((res[0][1].double() - rs1).abs().max())
    print(f"  [{tag:14s}] D={D:5d} wrong={int(wrong.sum()):5d}/{R * D} "
          f"run2diff={int(nondet.sum()):5d} scale_err={se:.2e}  "
          f"{'BAD' if (wrong.any() or nondet.any()) else 'OK'}", flush=True)
    if wrong.any():
        r0 = int(wrong.any(dim=1).nonzero()[0])
        print("      r=%d: %s" % (r0, "".join("X" if wrong[r0, c] else "." for c in range(D))), flush=True)


def main():
    print(f"[rmsdq] dev={DEV} triton={triton.__version__}", flush=True)
    from triton_ops.common import next_pow2
    R = 16
    print("== A) 形态对照：2-D 原形 vs 1-D 改写形（双填充 sentinel） ==", flush=True)
    for D in (17, 24, 33, 48, 100, 127, 128, 129, 255, 256, 1000, 1024, 7168):
        blk = 128 if D > 128 else next_pow2(D)
        kernel_cmp("2D_orig", launch_2d, R, D, blk)
        kernel_cmp("1D_fixed", launch_1d, R, D, blk)
    print("== B) 生产算子端到端（fp64 逐元素 + 跑两次一致性） ==", flush=True)
    for D in (17, 24, 33, 48, 100, 127, 128, 129, 255, 256, 1000, 1024, 2048, 7168):
        e2e("e2e smooth", R, D)
    for D in (33, 100, 1000):
        e2e("e2e nosmooth", R, D, smooth=False)
    print("[rmsdq] done", flush=True)


if __name__ == "__main__":
    main()
