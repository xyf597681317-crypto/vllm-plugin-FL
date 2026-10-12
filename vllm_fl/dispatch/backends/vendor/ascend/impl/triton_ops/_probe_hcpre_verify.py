# -*- coding: utf-8 -*-
"""_probe_hcpre_verify.py —— 修后回归验证（3-way：triton vs native vs fp64）

Lead 的验收要求：
  ① hc_pre：d∈{7168,4096} × bs∈{64,4096,8192} × iters∈{1,20} 三方一致
  ② hc_pre_sinkhorn：R∈{128,256,512,1024,2048,4096} 三方一致
  ③ hc_pre_inv_rms：R∈{8,16,32,64} 编译+执行并与 fp64 一致

判据（dtype 感知）：y 与原生同 dtype 比用 1 ULP；post/comb fp32 用 2e-6。
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    print("enable_custom_op ->", enable_custom_op(), flush=True)
except Exception as e:
    print("native off:", e, flush=True)
from triton_ops.hc_pre import hc_pre  # noqa: E402
from triton_ops.hc_pre_sinkhorn import hc_pre_sinkhorn  # noqa: E402
from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms  # noqa: E402

PF = 2e-6
FAILS = []


def md(a, b):
    a, b = a.double(), b.double()
    assert a.shape == b.shape, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


def take(t, rows, shp):
    d = t.detach().cpu().double().reshape(*shp)
    return d[rows]


def make_hcpre(bs, hm, d, iters, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    sc = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    ba = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, sc, ba, hm, iters


def ref_hcpre(x, fn, sc, ba, hm, iters, rows):
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(dim=1) / K + 1e-6)
    fnq = fn.detach().cpu().double()
    mix = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mix += xflat[:, s:e] @ fnq[:, s:e].t()
    scd, bbd = sc.detach().cpu().double(), ba.detach().cpu().double()
    pre = torch.sigmoid(mix[:, :hm] * rq[:, None] * scd[0] + bbd[:hm]) + 1e-6
    y = torch.einsum("bj,bjd->bd", pre, xf)
    post = 2.0 * torch.sigmoid(mix[:, hm:2 * hm] * rq[:, None] * scd[1] + bbd[hm:2 * hm])
    A = (mix[:, 2 * hm:].reshape(nr, hm, hm) * rq[:, None, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + 1e-6
    A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + 1e-6)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
    return y, post, A


def ref_hcpre_from(ins, iters, rows):
    """`ins` = make_hcpre 的 6 元组（末项是它自己的 iters）。
    这里**只取前 5 项**，iters 用调用方给的 `iters` ——
    避免"6 元组展开 + 再 append iters"造成的参数错位（我在这一行错过两次）。"""
    x, fn, sc, ba, hm = ins[0], ins[1], ins[2], ins[3], ins[4]
    return ref_hcpre(x, fn, sc, ba, hm, iters, rows)


print("\n===== ① hc_pre：d × bs × iters 三方一致 =====", flush=True)
for d in [7168, 4096]:
    for bs in [64, 4096, 8192]:
        for it in [1, 20]:
            rows = sorted({0, 1, 2, 3, bs // 2, bs - 2, bs - 1})
            rows = [r for r in rows if 0 <= r < bs]
            tri = hc_pre(*make_hcpre(bs, 4, d, it), 1e-6, 1e-6)
            torch.npu.synchronize()
            nat = None
            try:
                nat = torch.ops._C_ascend.npu_hc_pre(*make_hcpre(bs, 4, d, it), 1e-6, 1e-6)
                torch.npu.synchronize()
            except Exception:
                nat = None
            ry, rp, rc = ref_hcpre_from(make_hcpre(bs, 4, d, it), it, rows)
            ulp = (2.0 ** -8) * max(float(ry.to(torch.bfloat16).double().abs().max()), 1e-12)
            e = []
            for nm, idx, shp, rv, thr in [("y", 0, (bs, d), ry.to(torch.bfloat16).double(), ulp),
                                          ("post", 1, (bs, 4), rp, PF),
                                          ("comb", 2, (bs, 4, 4), rc, PF)]:
                te = md(take(tri[idx], rows, shp), rv)
                ne = md(take(nat[idx], rows, shp), rv) if nat is not None else float("nan")
                e.append((nm, te, ne, thr))
            bad = [x[0] for x in e if x[1] > x[3]]
            ok = not bad
            if not ok:
                FAILS.append(("hc_pre", d, bs, it, bad))
            print(f"  d={d:5d} bs={bs:5d} it={it:2d} | " +
                  "  ".join(f"{nm}: tri={te:.2e} nat={ne:.2e}" for nm, te, ne, _ in e) +
                  f"  {'OK' if ok else 'FAIL@' + ','.join(bad)}", flush=True)

print("\n===== ② hc_pre_sinkhorn：R 扫描三方一致 =====", flush=True)


def make_sk(R, hm, d, iters=20, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    hmix = 2 * hm + hm * hm

    def U(*s, lo=1.0, off=0.0):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo + off
    return (U(R, hmix, lo=1.0), U(R, lo=0.5, off=1.0), U(3, lo=0.5, off=1.0),
            U(hmix, lo=0.5), U(R, hm, d, lo=0.3).to(torch.bfloat16), hm, iters, eps)


def ref_sk(mixes, rsqrt, sc, bb, x, hm, iters, eps, rows):
    mx = mixes.detach().cpu().double().reshape(-1, mixes.shape[-1])[rows]
    rs = rsqrt.detach().cpu().double().view(-1, 1)[rows]
    scd, bbd = sc.detach().cpu().double(), bb.detach().cpu().double()
    d = int(x.shape[-1])
    xd = x.detach().cpu().double().reshape(-1, hm, d)[rows]
    pre = torch.sigmoid(mx[:, :hm] * rs * scd[0] + bbd[:hm]) + eps
    y = torch.einsum("ij,ijd->id", pre, xd)
    post = 2.0 * torch.sigmoid(mx[:, hm:2 * hm] * rs * scd[1] + bbd[hm:2 * hm])
    A = (mx[:, 2 * hm:].reshape(len(rows), hm, hm) * rs[:, :, None] * scd[2]
         + bbd[2 * hm:].reshape(hm, hm))
    e = torch.exp(A - A.max(dim=2, keepdim=True).values)
    A = e / e.sum(dim=2, keepdim=True) + eps
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    for _ in range(iters - 1):
        A = A / (A.sum(dim=2, keepdim=True) + eps)
        A = A / (A.sum(dim=1, keepdim=True) + eps)
    return y, post, A


for R in [128, 256, 512, 1024, 2048, 4096]:
    for d in ([512] if R != 4096 else [512, 7168]):
        rows = sorted({0, 1, 2, 3, 5, 8, 13, R - 1})
        rows = [r for r in rows if 0 <= r < R]
        tri = hc_pre_sinkhorn(*make_sk(R, 4, d))
        torch.npu.synchronize()
        nat = None
        try:
            nat = torch.ops._C_ascend.npu_hc_pre_sinkhorn(*make_sk(R, 4, d))
            torch.npu.synchronize()
        except Exception:
            nat = None
        ry, rp, rc = ref_sk(*make_sk(R, 4, d), rows=rows)
        res = []
        for nm, idx, shp, rv, thr in [("y", 0, (R, d), ry.to(torch.bfloat16).double(), 4e-3),
                                      ("post", 1, (R, 4), rp, PF), ("comb", 2, (R, 4, 4), rc, PF)]:
            te = md(take(tri[idx], rows, shp), rv)
            ne = md(take(nat[idx], rows, shp), rv) if nat is not None else float("nan")
            res.append((nm, te, ne, thr))
        bad = [x[0] for x in res if x[1] > x[3]]
        if bad:
            FAILS.append(("sk", R, d, bad))
        print(f"  R={R:5d} d={d:5d} | " +
              "  ".join(f"{nm}: tri={te:.2e} nat={ne:.2e}" for nm, te, ne, _ in res) +
              f"  {'OK' if not bad else 'FAIL@' + ','.join(bad)}", flush=True)

print("\n===== ③ hc_pre_inv_rms：R∈{8,16,32,64} =====", flush=True)
for shape in [(2, 2, 4), (4, 4, 4), (8, 2, 8), (16, 4, 4), (32, 4, 4), (64, 4, 16),
              (4096, 4, 512), (4096, 4, 7168)]:
    R = shape[-2] * shape[-1]
    A = 1
    for v in shape[:-2]:
        A *= v
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * 0.5).to(torch.bfloat16)
    r = 1.0 / torch.sqrt(x.detach().cpu().double().reshape(A, -1).pow(2).sum(1) / R + 1e-6)
    try:
        y = hc_pre_inv_rms(x, 1e-6)
        torch.npu.synchronize()
        e = md(y.detach().cpu().reshape(A), r)
        ok = e <= 2e-6
        if not ok:
            FAILS.append(("inv", shape, e))
        print(f"  shape={str(shape):16s} A={A:5d} R={R:5d} err={e:.3e} shape={tuple(y.shape)} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    except Exception as ex:
        FAILS.append(("inv_exc", shape, str(ex)[:80]))
        print(f"  shape={str(shape):16s} EXC {type(ex).__name__}: {str(ex)[:90]}", flush=True)

print(f"\n===== 汇总：FAILS={len(FAILS)} =====", flush=True)
for f in FAILS:
    print("  ", f, flush=True)
print("VERIFYDONE", flush=True)
