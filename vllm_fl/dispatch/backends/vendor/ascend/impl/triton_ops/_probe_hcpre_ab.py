# -*- coding: utf-8 -*-
"""_probe_hcpre_ab.py —— 修法 A/B（在探针内定义变体，不改生产文件）

③ hc_pre_inv_rms：
   现行 = `while r < R`（runtime bound）+ [BLK_A=4, BLK_R] 二维 tile + 二维掩码
   变体 = 逐行 1-D 累加（标量行号 + 1-D 列 tile），块内按行循环
① hc_pre 的 sinkhorn 部分：
   现行 = `while dcur < D` / `while j < HM`（**runtime bound**）+ [BLK_R, BLK_D] 二维 tile
   变体 = 静态范围 + 1-D
本探针只求"A/B 哪条能修好"，确认后再改生产文件。
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

try:
    from vllm_fl.ascend_custom_ops import enable_custom_op
    enable_custom_op()
except Exception:
    pass


def md(a, b):
    a = a.double() if a.is_floating_point() else a.double()
    b = b.double() if b.is_floating_point() else b.double()
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


def cdiv(a, b):
    return (a + b - 1) // b


# ============================================== ③ inv_rms 变体
@triton.jit
def _inv_rowwise(X, Y, A, R, EPS, BLK_A: tl.constexpr, BLK_R: tl.constexpr,
                 NCH: tl.constexpr):
    """逐行 1-D 累加：标量行号 + 1-D 列 tile，块内按行循环。"""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLK_R)
    for i in range(BLK_A):
        row = pid * BLK_A + i
        if row < A:
            acc = tl.zeros((BLK_R,), dtype=tl.float32)
            for c in range(NCH):
                col = c * BLK_R + cols
                x = tl.load(X + row * R + col, mask=col < R, other=0.0).to(tl.float32)
                acc += x * x
            s = tl.sum(acc, axis=0)
            tl.store(Y + row, 1.0 / tl.sqrt(s / R + EPS))


def ref_inv(x, eps):
    xd = x.detach().cpu().double()
    R = xd.shape[-2] * xd.shape[-1]
    A = xd.numel() // R
    return 1.0 / torch.sqrt(xd.reshape(A, -1).pow(2).sum(1) / R + eps)


print("=" * 100, flush=True)
print("③ hc_pre_inv_rms：现行 vs 逐行变体", flush=True)
print("=" * 100, flush=True)
from triton_ops.hc_pre_inv_rms import hc_pre_inv_rms as cur  # noqa: E402

print(f"  {'shape':>16} {'A':>5} {'R':>6} | {'现行':>26} | {'逐行变体':>14}", flush=True)
for shape in [(2, 2, 4), (8, 1, 8), (4, 4, 4), (16, 4, 4), (17, 4, 16), (64, 4, 16),
              (4096, 4, 512), (4096, 4, 7168), (1, 4, 7168), (33, 4, 100)]:
    R = shape[-2] * shape[-1]
    A = 1
    for v in shape[:-2]:
        A *= v
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * 0.5).to(torch.bfloat16)
    r = ref_inv(x, 1e-6)
    col1 = ""
    try:
        y = cur(x, 1e-6)
        torch.npu.synchronize()
        col1 = f"err={md(y.detach().cpu().reshape(A), r):.2e} shape={tuple(y.shape)}"
    except Exception as e:
        col1 = f"EXC {type(e).__name__}: {str(e)[:70]}"
    col2 = ""
    try:
        blk_r = max(2, min(512, 1 << (R - 1).bit_length()))
        nch = cdiv(R, blk_r)
        blk_a = 4
        y2 = torch.empty((A,), dtype=torch.float32, device=DEV)
        _inv_rowwise[(cdiv(A, blk_a),)](x.contiguous().view(A, R), y2, A, R, 1e-6,
                                        BLK_A=blk_a, BLK_R=blk_r, NCH=nch)
        torch.npu.synchronize()
        col2 = f"err={md(y2, r):.2e}"
    except Exception as e:
        col2 = f"EXC {type(e).__name__}: {str(e)[:70]}"
    print(f"  {str(shape):>16} {A:>5} {R:>6} | {col1:>26} | {col2:>14}", flush=True)

# ============================================== ① hc_pre sinkhorn 变体
print("\n" + "=" * 100, flush=True)
print("① hc_pre：rsqrt / mixes 是否随 d 变化（定位错在哪个子步骤）", flush=True)
print("=" * 100, flush=True)
from triton_ops.hc_pre import hc_pre  # noqa: E402


def make(bs, hm, d, seed=13):
    torch.manual_seed(seed)
    hmix, K = 2 * hm + hm * hm, hm * d
    x = (torch.randn(bs, hm, d, dtype=torch.float32, device=DEV) * 1.2).to(torch.bfloat16)
    fn = torch.randn(hmix, K, dtype=torch.float32, device=DEV) * (1.0 / (K ** 0.5))
    sc = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32, device=DEV)
    ba = torch.randn(hmix, dtype=torch.float32, device=DEV) * 0.1
    return x, fn, sc, ba, hm


def ref_mixes(x, fn, rows):
    """fp64 复算 mixes 与 rsqrt（只算采样行）。"""
    hm = x.shape[1]
    xf = x.detach().cpu().double().reshape(-1, hm, x.shape[-1])[rows]
    nr, _, d = xf.shape
    K = hm * d
    xflat = xf.reshape(nr, K)
    rq = 1.0 / torch.sqrt((xflat * xflat).sum(1) / K + 1e-6)
    fnq = fn.detach().cpu().double()
    mix = torch.zeros(nr, fnq.shape[0], dtype=torch.float64)
    for s in range(0, K, 8192):
        e = min(K, s + 8192)
        mix += xflat[:, s:e] @ fnq[:, s:e].t()
    return mix, rq


from triton_ops.hc_pre import _hc_pre_mixes_kernel, _hc_pre_rsqrt_kernel  # noqa: E402

print(f"  {'d':>6} {'K':>7} {'blk_k':>7} {'kch':>5} | {'rsqrt_err':>11} {'mixes_err':>11} "
      f"{'post_err(from mixes)':>20}", flush=True)
for d in [4096, 5120, 6144, 7168, 8192]:
    bs, hm = 64, 4
    x, fn, sc, ba, hmv = make(bs, hm, d)
    hmix, K = 2 * hm + hm * hm, hm * d
    rows = [0, 1, 2, 3, 5, 8, 13, 63]
    # 复刻 host 的 blk_k 选择
    blk_k = 64
    while cdiv(K, blk_k) > 32:
        blk_k *= 2
    kch = cdiv(K, blk_k)
    blk_m = 16
    bs_pad = cdiv(bs, blk_m) * blk_m
    k_pad = kch * blk_k
    xpad = torch.zeros((bs_pad, k_pad), dtype=x.dtype, device=DEV)
    xpad[:bs, :K] = x.contiguous().view(bs, K)
    fnpad = torch.zeros((hmix, k_pad), dtype=torch.float32, device=DEV)
    fnpad[:, :K] = fn
    rsqrt_dev = torch.empty((bs,), dtype=torch.float32, device=DEV)
    mixes_dev = torch.empty((bs, hmix), dtype=torch.float32, device=DEV)
    _hc_pre_rsqrt_kernel[(bs_pad // blk_m,)](xpad, rsqrt_dev, bs, K, k_pad, 1e-6,
                                             BLK_M=blk_m, BLK_K=blk_k, KCH=kch)
    _hc_pre_mixes_kernel[(bs_pad // blk_m, hmix)](xpad, fnpad, mixes_dev, bs, K, k_pad, hmix,
                                                 BLK_M=blk_m, BLK_K=blk_k, KCH=kch)
    torch.npu.synchronize()
    mix_ref, rq_ref = ref_mixes(x, fn, rows)
    e_rq = md(rsqrt_dev.detach().cpu()[rows], rq_ref)
    e_mx = md(mixes_dev.detach().cpu()[rows], mix_ref)
    # post 由 mixes/rsqrt 推出，看误差是否与端到端一致
    bb = ba.detach().cpu().double()
    post_ref = 2.0 * torch.sigmoid(mix_ref[:, hm:2 * hm]
                                   * rq_ref[:, None] * float(sc[1]) + bb[hm:2 * hm])
    post_got = 2.0 * torch.sigmoid(mixes_dev.detach().cpu().double()[rows][:, hm:2 * hm]
                                   * rsqrt_dev.detach().cpu().double()[rows][:, None]
                                   * float(sc[1]) + bb[hm:2 * hm])
    e_post = md(post_got, post_ref)
    print(f"  {d:>6} {K:>7} {blk_k:>7} {kch:>5} | {e_rq:>11.3e} {e_mx:>11.3e} {e_post:>20.3e}",
          flush=True)
print("ABDONE", flush=True)
