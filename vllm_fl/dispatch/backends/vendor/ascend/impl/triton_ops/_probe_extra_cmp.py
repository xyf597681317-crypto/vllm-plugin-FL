# -*- coding: utf-8 -*-
"""比对 `_probe_extra_oracle.py` 两个进程（AscendC vs Triton）的输出：逐元素 max|Δ| / exact。"""
import numpy as np

A = np.load("/tmp/extra_oracle_ascendc.npz")
B = np.load("/tmp/extra_oracle_triton.npz")
ka, kb = set(A.files), set(B.files)
print(f"[cmp] ascendc keys={len(ka)} triton keys={len(kb)}")
print(f"[cmp] 只在 ascendc: {sorted(ka - kb)}   只在 triton: {sorted(kb - ka)}")
worst = 0.0
for k in sorted(ka & kb):
    a, b = A[k], B[k]
    if a.shape != b.shape:
        print(f"  {k:22s} SHAPE MISMATCH {a.shape} vs {b.shape}")
        continue
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    md = float(d.max()) if d.size else 0.0
    exact = float((d == 0).mean()) * 100 if d.size else 100.0
    worst = max(worst, md)
    print(f"  {k:22s} max|Δ|={md:.3e}  exact={exact:.4f}%  n={d.size}")
print(f"[cmp] 全部 key 的 max|Δ| 上界 = {worst:.3e}")
print("[cmp] done")
