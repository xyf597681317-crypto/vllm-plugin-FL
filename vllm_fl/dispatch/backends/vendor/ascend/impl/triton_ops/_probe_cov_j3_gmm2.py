# -*- coding: utf-8 -*-
"""_probe_cov_j3_gmm2.py —— 追查 `glist_type=1`（计数语义）的数值错误

上一轮（`_probe_cov_j3_gmm.py`）：`v2 M=64 K=2048 I=512 glist_type=1` FAIL
    max|Δ|=127  完全一致=66.479%  >1LSB=985  scale_rel=1.12e+30  边界行=63
而同形状 `glist_type=0` **PASS（100% 一致）** ⇒ 差异只在计数语义的解释。

本探针回答三件事：
  A) 是不是"只在生产宽度暴露"？用 M=20 K=32 I=16（**单测里 PASS 的那个形状**）同参数对照
  B) 哪些行错？（16 行采样 + 逐行明细，而不是只报聚合）
  C) 错的方向是什么？（把 got / exp 的 y 与 scale 逐行并排打出来，便于判"整行错"还是"缩放错"）
"""
import importlib.util
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

_s = importlib.util.spec_from_file_location(
    "cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cov)

sys.path.insert(0, "/workspace/FlagGems/src/triton_ops")
import importlib.util as _iu  # noqa: E402
_g = _iu.spec_from_file_location(
    "g", "/workspace/FlagGems/src/triton_ops/_probe_cov_j3_gmm.py")
gmod = _iu.module_from_spec(_g)
_g.loader.exec_module(gmod)
# 复用已验证正确的参考与输入构造，避免再造第二份
make_v2_inputs = gmod.make_v2_inputs
fn_v2 = gmod.fn_v2
ref_rows_v2 = gmod.ref_rows_v2
J3 = []


def detail(name, M, K, I, G=2, glist_type=1, n_sample=16):
    ins = make_v2_inputs(M, K, I, G, "col", glist_type)
    x, w, ws, xs, gl, smooth, glt = ins
    print(f"\n===== {name} =====", flush=True)
    print(f"  group_list({glt})={gl.detach().cpu().tolist()}", flush=True)
    try:
        y, ys = fn_v2(*ins)
        torch.npu.synchronize()
    except Exception as e:
        print(f"  EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    rows = cov.probe_rows(M, n_sample)
    ry = ref_rows_v2(*ins, rows=rows)
    rsc = ref_rows_v2(*ins, rows=rows, want_scale=True)
    yc = y.detach().cpu()[rows]
    sc = ys.detach().cpu().double()[rows]
    d = (yc.double() - ry.double()).abs()
    print(f"  采样 {len(rows)} 行: max|Δ|={float(d.max()):.0f} "
          f"完全一致={float((d == 0).double().mean()) * 100:.3f}%", flush=True)
    bad = [i for i in range(len(rows)) if float(d[i].max()) > 0]
    print(f"  有差异的采样行: {[(rows[i], int(d[i].max())) for i in bad]}", flush=True)
    for i in bad[:6]:
        r = rows[i]
        print(f"    row={r:5d} got_max|y|={float(yc[i].abs().max()):.0f} "
              f"exp_max|y|={float(ry[i].abs().max()):.0f} "
              f"got_sc={float(sc[i]):.6e} exp_sc={float(rsc[i]):.6e} "
              f"sc_ratio={float(sc[i] / rsc[i]) if float(rsc[i]) != 0 else float('nan'):.4f}", flush=True)
        print(f"        got[:6]={[int(v) for v in yc[i][:6].tolist()]}  "
              f"exp[:6]={[int(v) for v in ry[i][:6].tolist()]}", flush=True)
    # 该行的组归属（按 ref 的口径）
    cnt = gl.detach().cpu().tolist()
    ends = [sum(cnt[: g + 1]) for g in range(len(cnt))]
    print(f"  ref 口径: glist_type={glt} -> 边界(cumsum)={ends} "
          f"（0 = 累计偏移；1 = 每组计数）", flush=True)


if __name__ == "__main__":
    print(f"J3-GMM2 | DEV={DEV}", flush=True)
    # A) 单测里 PASS 的形状（小），同参数
    detail("A 单测形状 M=20 K=32 I=16 glist_type=1", 20, 32, 16, 2, 1, n_sample=16)
    # A2) 同形状但 glist_type=0 作对照
    detail("A2 同形状 glist_type=0（对照）", 20, 32, 16, 2, 0, n_sample=16)
    # B) 生产宽度（原本 FAIL 的形状），16 行采样
    detail("B 生产宽度 M=64 K=2048 I=512 glist_type=1", 64, 2048, 512, 2, 1, n_sample=16)
    # C) 中间宽度，看是否渐进
    detail("C M=64 K=256 I=64 glist_type=1", 64, 256, 64, 2, 1, n_sample=16)
    # D) 不均匀分组（更接近真实 MoE 的 group_list）
    def detail_uneven():
        M, K, I, G = 64, 2048, 512, 4
        torch.manual_seed(0)
        x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
        w = [torch.randint(-40, 41, (K, 2 * I), dtype=torch.int8, device=DEV) for _ in range(G)]
        ws = [torch.rand(2 * I, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(G)]
        xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
        cnts = [20, 16, 8, 20]                     # 非均匀
        gl = torch.tensor(cnts, dtype=torch.int64, device=DEV)   # glist_type=1：计数
        smooth = torch.rand(G, I, dtype=torch.float32, device=DEV) * 0.8 + 0.6
        ins = (x, w, ws, xs, gl, smooth, 1)
        print("\n===== D 非均匀计数 [20,16,8,20] glist_type=1 =====", flush=True)
        y, ys = fn_v2(*ins)
        torch.npu.synchronize()
        rows = cov.probe_rows(M, 16)
        ry = ref_rows_v2(*ins, rows=rows)
        d = (y.detach().cpu()[rows].double() - ry.double()).abs()
        print(f"  max|Δ|={float(d.max()):.0f} 完全一致={float((d == 0).double().mean()) * 100:.3f}%",
              flush=True)
        bad = [(rows[i], int(d[i].max())) for i in range(len(rows)) if float(d[i].max()) > 0]
        print(f"  有差异行: {bad}", flush=True)
    detail_uneven()
    print("J3GMM2DONE", flush=True)
