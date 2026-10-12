# -*- coding: utf-8 -*-
"""_probe_cov_j3_gmm4.py —— glist_type=1 复核（修正版参考），确认"FAIL 是我的参考错"

背景（重要，避免留错结论）：
  `_probe_cov_j3_gmm.py` 曾报 `v2 M=64 K=2048 I=512 glist_type=1` FAIL（max|Δ|=127、scale_rel=1.12e30）。
  但 `_probe_cov_j3_gmm3.py` 用**独立的手工 fp64 参考**对照后显示：
      kernel 的 scale 与"按计数语义分组后的正确参考"**逐位一致**
      （E2：row=10 .. 19，kernel_sc == ref(组1)_sc；E3 对照累计偏移语义也一致）
  ⇒ **kernel 是对的；报 FAIL 的是我那份 `ref_rows_v2` 在 glist_type=1 下的行切片路径。**
  本探针用**重写的、无共享状态的**参考再跑一遍，把结论钉死。

参考（本文件内自包含，故意不 import 上一份）：
    counts 语义（glist_type=1）：counts = gl；组 g 的行区间 = [Σ_{<g} counts, Σ_{≤g} counts)
    cumulative 语义（glist_type=0）：ends = gl；组 g 的行区间 = [ends[g-1], ends[g])
    两端都由 counts 推出，**不依赖 gl 的原始含义**，避免"同一份错误被复制两次"。
"""
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

from triton_ops.grouped_matmul_swiglu_quant_v2 import grouped_matmul_swiglu_quant_v2 as gv2  # noqa: E402

sys.path.insert(0, "/workspace/FlagGems/src/triton_ops")
import importlib.util as _iu  # noqa: E402

_s = _iu.spec_from_file_location("cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = _iu.module_from_spec(_s)
_s.loader.exec_module(cov)


def _rint_fp16(t):
    q16 = t.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref_selfcontained(x, w, ws, xs, counts, smooth, rows):
    """counts: list[int]，各组行数（两种语义都先归一到 counts）。"""
    G = len(counts)
    I = w[0].shape[1] // 2
    bounds = []
    acc = 0
    for c in counts:
        bounds.append((acc, acc + c))
        acc += c
    xd = x.detach().cpu().double()
    XS = xs.detach().cpu().double()
    smt = smooth.detach().cpu().double()
    ys = torch.zeros(len(rows), I, dtype=torch.int64)
    sc = torch.zeros(len(rows), dtype=torch.float64)
    for g in range(G):
        s, e = bounds[g]
        sel = [i for i, r in enumerate(rows) if s <= r < e]
        if not sel:
            continue
        rr = [rows[i] for i in sel]
        C = (xd[rr] * XS[rr, None]) @ (w[g].detach().cpu().double()
                                       * ws[g].detach().cpu().double()[None, :])
        gate, up = C[:, :I], C[:, I:]
        sw = (gate * torch.sigmoid(gate)) * up
        if smt.dim() == 2:
            sw = sw * smt[g][None, :]
        else:
            sw = sw * smt[g]
        mx = sw.abs().amax(dim=1)
        sc[sel] = mx / 127.0
        ys[sel] = _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64)
    return ys, sc


def build(M, K, I, G, counts, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = [torch.randint(-40, 41, (K, 2 * I), dtype=torch.int8, device=DEV) for _ in range(G)]
    ws = [torch.rand(2 * I, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(G)]
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    smooth = torch.rand(G, I, dtype=torch.float32, device=DEV) * 0.8 + 0.6
    return x, w, ws, xs, smooth


def call(ins, gl):
    x, w, ws, xs, smooth = ins
    return gv2(x, weight=w, weight_scale=ws, x_scale=xs, group_list=gl, smooth_scale=smooth,
               weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0,
               quant_mode=0, quant_dtype=0, transpose_weight=False,
               group_list_type=int(ins[5]) if len(ins) > 5 else 0, tuning_config=[0],
               swiglu_limit=0.0)


def run(name, M, K, I, counts, glist_type, seed=0, n_sample=16):
    G = len(counts)
    ins = build(M, K, I, G, counts, seed)
    x, w, ws, xs, smooth = ins
    if glist_type == 1:
        gl = torch.tensor(counts, dtype=torch.int64, device=DEV)
    else:
        acc, ends = 0, []
        for c in counts:
            acc += c
            ends.append(acc)
        gl = torch.tensor(ends, dtype=torch.int64, device=DEV)
    try:
        y, ys = gv2(x, weight=w, weight_scale=ws, x_scale=xs, group_list=gl,
                    smooth_scale=smooth, weight_assist_matrix=None, bias=None,
                    dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                    transpose_weight=False, group_list_type=glist_type,
                    tuning_config=[0], swiglu_limit=0.0)
        torch.npu.synchronize()
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {str(e)[:160]}", flush=True)
        return False
    rows = cov.probe_rows(M, n_sample)
    ry, rsc = ref_selfcontained(x, w, ws, xs, counts, smooth, rows)
    yc = y.detach().cpu()[rows]
    sc = ys.detach().cpu().double()[rows]
    d = (yc.double() - ry.double()).abs()
    md = float(d.max())
    ex = float((d == 0).double().mean())
    over1 = int((d > 1).sum().item())
    erel = float(((sc - rsc).abs() / rsc.abs().clamp_min(1e-30)).max())
    bad = [(rows[i], int(d[i].max())) for i in range(len(rows)) if float(d[i].max()) > 0]
    ok = md <= 0 and ex >= 1.0 and erel <= 2e-6
    print(f"[{'PASS' if ok else 'FAIL'}] {name:46s} 采样{len(rows)}行 max|Δ|={md:.0f} "
          f"完全一致={ex * 100:.3f}% >1LSB={over1} scale_rel={erel:.2e} 差异行={bad}",
          flush=True)
    return ok


print(f"J3-GMM4 | DEV={DEV}", flush=True)
allok = True
allok &= run("v2 M=20 K=32 I=16 计数[10,10] glist=1", 20, 32, 16, [10, 10], 1)
allok &= run("v2 M=64 K=2048 I=512 计数[32,32] glist=1", 64, 2048, 512, [32, 32], 1)
allok &= run("v2 M=64 K=2048 I=512 计数[20,16,8,20] glist=1", 64, 2048, 512, [20, 16, 8, 20], 1)
allok &= run("v2 M=64 K=2048 I=512 累计[32,64] glist=0", 64, 2048, 512, [32, 32], 0)
allok &= run("v2 M=4096 K=2048 I=512 计数[2048,2048] glist=1", 4096, 2048, 512, [2048, 2048], 1)
print(f"\nALL_PASS={allok}", flush=True)
print("J3GMM4DONE", flush=True)
