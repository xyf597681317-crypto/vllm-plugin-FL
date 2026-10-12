# -*- coding: utf-8 -*-
"""生产形状正确性探针：Qwen3.6-35B-A3B 的真实调用组合（E=256, k=8, g=8, kg=4, SUM）。

调用形态照抄 `impl/moe/experts_selector.py::_select_experts_with_fusion_ops`：
    moe_gating_top_k(router_logits, k=top_k, k_group=topk_group,
                     group_count=num_expert_group, group_select_mode=1,
                     renorm=renorm, norm_type=1(sigmoid)/0(softmax),
                     out_flag=False, routed_scaling_factor=..., eps=1e-20,
                     bias_opt=e_score_correction_bias)
覆盖 rows ∈ {1, 7, 64, 4096}（rows=1 是 decode 形态）。
用法：MOE_DEV=npu:0 python3 -W ignore triton_ops/_probe_mgating_prod.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_ops.moe_gating_top_k import moe_gating_top_k  # noqa: E402

DEV = os.environ.get("MOE_DEV", "npu:0")


def ref_impl(x, bias, k, k_group, group_count, gsm, renorm, norm_type, rsf, eps):
    xd = x.detach().cpu().double()
    rows, E = xd.shape
    if norm_type == 1:
        s = torch.sigmoid(xd)
    else:
        e = torch.exp(xd - xd.max(dim=1, keepdim=True).values)
        s = e / e.sum(dim=1, keepdim=True)
    routing = s + (bias.detach().cpu().double()[None, :] if bias is not None else 0.0)
    per_group = E // group_count
    r = routing.clone()
    if group_count > 1 and k_group < group_count:
        rg = r.reshape(rows, group_count, per_group)
        gs = rg.topk(2, dim=2).values.sum(dim=2) if gsm == 1 else rg.max(dim=2).values
        sel = torch.sort(-gs, dim=1, stable=True).indices[:, :k_group]
        m = torch.zeros(rows, group_count, dtype=torch.bool)
        m.scatter_(1, sel, True)
        keep = m.unsqueeze(2).expand(rows, group_count, per_group).reshape(rows, E)
        r = torch.where(keep, r, torch.full_like(r, float("-inf")))
    idx = torch.sort(-r, dim=1, stable=True).indices[:, :k]
    y = torch.gather(s, 1, idx)
    if (norm_type == 1) or (norm_type == 0 and renorm == 1):
        y = y / (y.sum(dim=1, keepdim=True) + eps)
    return y * rsf, idx


def run(tag, rows, E=256, k=8, k_group=4, group_count=8, gsm=1, renorm=1,
        norm_type=1, dtype=torch.bfloat16, bias=True, rsf=1.0, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(rows, E, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
    b = (torch.randn(E, dtype=torch.float32, device=DEV) * 0.5).to(dtype) if bias else None
    y, idx, out = moe_gating_top_k(
        x, k, k_group=k_group, group_count=group_count, group_select_mode=gsm,
        renorm=renorm, norm_type=norm_type, out_flag=False,
        routed_scaling_factor=rsf, eps=1e-20, bias_opt=b)
    torch.npu.synchronize()
    ry, ridx = ref_impl(x, b, k, k_group, group_count, gsm, renorm, norm_type, rsf, 1e-20)
    floor = {torch.float32: 2e-6, torch.float16: 1e-3, torch.bfloat16: 4e-3}[dtype]
    ey = float((y.detach().cpu().double() - ry).abs().max())
    idx_ok = bool((idx.detach().cpu().long() == ridx).all())
    ok = ey <= floor and idx_ok and out is None
    print(f"  {tag:52s} rows={rows:5d} y_err={ey:.2e}(<={floor:.0e}) "
          f"idx={'EQ' if idx_ok else 'NE'}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"[probe-mgating-prod] dev={DEV}（生产组合 E=256 k=8 g=8 kg=4 gsm=SUM）", flush=True)
    allok = True
    allok &= run("prod sigmoid+bias renorm=1 bf16 rows=1 (decode)", 1)
    allok &= run("prod sigmoid+bias renorm=1 bf16 rows=7", 7)
    allok &= run("prod sigmoid+bias renorm=1 bf16 rows=64", 64)
    allok &= run("prod sigmoid+bias renorm=1 bf16 rows=4096 (prefill)", 4096)
    allok &= run("prod softmax+bias renorm=1 bf16 rows=64", 64, norm_type=0)
    allok &= run("prod softmax+bias renorm=0 bf16 rows=64", 64, norm_type=0, renorm=0)
    allok &= run("prod sigmoid no-bias bf16 rows=64", 64, bias=False)
    allok &= run("prod sigmoid+bias renorm=1 fp16 rows=64", 64, dtype=torch.float16)
    allok &= run("prod sigmoid+bias rsf=2.5 bf16 rows=64", 64, rsf=2.5)
    print(f"\n[probe-mgating-prod] {'ALL PASS' if allok else 'FAIL'}", flush=True)
