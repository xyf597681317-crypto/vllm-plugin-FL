# -*- coding: utf-8 -*-
"""生产形状探针：moe_gating_top_k 的 UB 需求（Qwen3.6-35B-A3B: E=256, k=8, g=8, kg=4）。

用法（容器内）：
    MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_mgating_ub.py deployed
    MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_mgating_ub.py tuned
  deployed = 复刻容器里那份旧文件的旋钮（BLK_R_CAP=4, ROW_BUDGET=4096）
  tuned    = 本仓库当前文件默认（BLK_R_CAP=1, ROW_BUDGET=1024）
"""
import os
import re
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "tuned"
# 显式设置（不依赖部署的那份文件的默认值）：deployed 复刻旧旋钮，tuned 是当前仓库默认，
# env 表示完全用外部环境变量控制（用于扫 BLK_R_CAP / ROW_BUDGET）
if MODE == "deployed":
    os.environ["MOE_GATING_BLK_R_CAP"] = "4"
    os.environ["MOE_GATING_ROW_BUDGET"] = "4096"
elif MODE == "tuned":
    os.environ["MOE_GATING_BLK_R_CAP"] = "1"
    os.environ["MOE_GATING_ROW_BUDGET"] = "1024"
SETS = sys.argv[2] if len(sys.argv) > 2 else "prod"

import torch  # noqa: E402
import torch_npu  # noqa: F401, E402
import triton  # noqa: E402

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_ops.moe_gating_top_k import moe_gating_top_k  # noqa: E402
import triton_ops.moe_gating_top_k as M  # noqa: E402

DEV = os.environ.get("MOE_DEV", "npu:0")
print(f"[probe-mgating] mode={MODE} BLK_R_CAP={M.BLK_R_CAP} ROW_BUDGET={M.ROW_BUDGET} dev={DEV}",
      flush=True)


def try_case(tag, E, k, dtype=torch.bfloat16, R=16, bias=True, **kw):
    torch.manual_seed(0)
    x = torch.randn(R, E, dtype=torch.float32, device=DEV).to(dtype)
    b = (torch.randn(E, dtype=torch.float32, device=DEV) * 0.5).to(dtype) if bias else None
    blk_e = 1 << (E - 1).bit_length()
    blk_r = max(1, min(M.BLK_R_CAP, M.ROW_BUDGET // blk_e))
    try:
        y, idx, out = moe_gating_top_k(x, k, bias_opt=b, **kw)
        torch.npu.synchronize()
        print(f"  [{tag}] OK   (blk_r={blk_r} blk_e={blk_e} rows={R})", flush=True)
        return None
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        m = re.search(r"requires\s+([0-9]+)\s*bits", msg)
        bits = int(m.group(1)) if m else None
        ratio = f" ({bits / 1572864:.2f}x)" if bits else ""
        print(f"  [{tag}] FAIL bits={bits}{ratio} (blk_r={blk_r} blk_e={blk_e} rows={R})",
              flush=True)
        return bits


PROD = dict(k_group=4, group_count=8, group_select_mode=1, renorm=1, norm_type=1,
            out_flag=False, routed_scaling_factor=1.0, eps=1e-20)

if __name__ == "__main__":
    print(f"\n== SETS={SETS} ==", flush=True)
    if SETS == "prod":
        print("== 生产组合（Qwen3.6-35B-A3B: E=256 k=8 g=8 kg=4 SUM sigmoid +bias renorm=1）==",
              flush=True)
        try_case("prod E=256 k8 g8 kg4 bf16 +bias", 256, 8, **PROD)
        try_case("prod E=256 k8 g8 kg4 bf16 no-bias", 256, 8, bias=False, **PROD)
        print("\n== 对照：其它生产可能组合 ==", flush=True)
        try_case("E=256 k8 g1 (no grouping)", 256, 8,
                 k_group=1, group_count=1, group_select_mode=1, renorm=1, norm_type=1)
        try_case("E=256 k8 g8 kg4 softmax renorm1", 256, 8,
                 k_group=4, group_count=8, group_select_mode=1, renorm=1, norm_type=0)
        try_case("E=256 k8 g8 kg4 SUM fp32", 256, 8, dtype=torch.float32, **PROD)
        try_case("E=128 k8 g8 kg4 SUM bf16", 128, 8, **PROD)
        try_case("E=512 k8 g8 kg4 SUM bf16", 512, 8, **PROD)
        try_case("E=1024 k8 g8 kg4 SUM bf16", 1024, 8, **PROD)
    else:
        print(f"== BLK_E 边界扫描（BLK_R_CAP={M.BLK_R_CAP} ROW_BUDGET={M.ROW_BUDGET}）==",
              flush=True)
        for E in (2048, 4096, 8192):
            try_case(f"E={E} k8 g8 kg4 SUM bf16", E, 8, **PROD)
    print("\n[probe-mgating] done", flush=True)
