"""验证 moe_gating_top_k 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")   # 容器内可用独立包名隔离
moe_gating_top_k = importlib.import_module(_PKG + ".moe_gating_top_k").moe_gating_top_k

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))


def ref_impl(x, bias, k, k_group, group_count, gsm, renorm, norm_type, out_flag, rsf, eps):
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
        if gsm == 1:
            top2 = rg.topk(2, dim=2).values
            gscore = top2.sum(dim=2)
        else:
            gscore = rg.max(dim=2).values
        sel = torch.sort(-gscore, dim=1, stable=True).indices[:, :k_group]   # 稳定排序：平局取小号
        selmask = torch.zeros(rows, group_count, dtype=torch.bool)
        selmask.scatter_(1, sel, True)
        keep = selmask.unsqueeze(2).expand(rows, group_count, per_group).reshape(rows, E)
        r = torch.where(keep, r, torch.full_like(r, float("-inf")))
    # 稳定降序：平局时保留原（小）索引，与本 Triton 实现的取号规则一致
    idx = torch.sort(-r, dim=1, stable=True).indices[:, :k]
    y = torch.gather(s, 1, idx)
    need_renorm = (norm_type == 1) or (norm_type == 0 and renorm == 1)
    if need_renorm:
        y = y / (y.sum(dim=1, keepdim=True) + eps)
    y = y * rsf
    # 顺序校验：top-k 与第 k+1 名的间隔（仅用于提示平局风险，不参与判定）
    margin = float("inf")
    if k < E:
        part = torch.sort(-r, dim=1, stable=True).values
        diff = part[:, k - 1] - part[:, k]
        finite = torch.isfinite(diff)
        margin = float(diff[finite].min()) if bool(finite.any()) else float("inf")
    return y, idx, s, margin


def run_case(name, R, E, k, dtype, k_group=1, group_count=1, gsm=0, renorm=0,
             norm_type=1, out_flag=False, rsf=1.0, bias=True, eps=1e-20, callsite=False):
    torch.manual_seed(11)
    x = (torch.randn(R, E, dtype=torch.float32, device=DEV)).to(dtype)
    b = (torch.randn(E, dtype=torch.float32, device=DEV) * 0.5).to(dtype) if bias else None

    if callsite:
        # ← 调用点风格：用 bias_opt（npu_moe_gating_top_k 的 `Tensor? bias_opt`）
        y, idx, out = moe_gating_top_k(x, k, k_group, group_count, gsm, renorm,
                                       norm_type, out_flag, rsf, eps, bias_opt=b)
    else:
        y, idx, out = moe_gating_top_k(x, k, k_group, group_count, gsm, renorm,
                                       norm_type, out_flag, rsf, eps, b)
    torch.npu.synchronize()
    ry, ridx, rs, margin = ref_impl(x, b, k, k_group, group_count, gsm, renorm,
                                    norm_type, out_flag, rsf, eps)

    floor = {torch.float32: 2e-6, torch.float16: 1e-3, torch.bfloat16: 4e-3}[dtype]
    ey = float((y.detach().cpu().double() - ry).abs().max())
    idx_ok = bool((idx.detach().cpu().long() == ridx).all())
    ok = ey <= floor and idx_ok
    msg = f"y_err={ey:.2e}(<= {floor:.0e}) idx={'EQ' if idx_ok else 'NE'} margin={margin:.3f}"
    if out_flag:
        eo = float((out.detach().cpu().double() - rs).abs().max())
        ok &= eo <= 2e-6
        msg += f" out_err={eo:.2e}"
    else:
        ok &= out is None
    emit(f"  {name:46s} {msg}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 118)
emit("moe_gating_top_k: Triton vs AscendC 语义（fp64 参考）；expert_idx 必须完全相等")
emit("=" * 118)

allok = True
allok &= run_case("sig e256 g1 k8 bf16 +bias", 32, 256, 8, torch.bfloat16)
allok &= run_case("sig e256 g1 k8 fp32 +bias", 16, 256, 8, torch.float32)
allok &= run_case("softmax e128 g1 k6 fp16 renorm=0", 24, 128, 6, torch.float16, norm_type=0)
allok &= run_case("softmax e128 g1 k6 fp16 renorm=1", 24, 128, 6, torch.float16, norm_type=0, renorm=1)
allok &= run_case("DSV3 sig e256 g8 kg4 k8 bf16 SUM", 32, 256, 8, torch.bfloat16,
                  k_group=4, group_count=8, gsm=1)
allok &= run_case("DSV3 sig e256 g8 kg4 k8 fp32 SUM", 32, 256, 8, torch.float32,
                  k_group=4, group_count=8, gsm=1)
allok &= run_case("MAX e64 g8 kg2 k4 fp16", 24, 64, 4, torch.float16,
                  k_group=2, group_count=8, gsm=0)
allok &= run_case("MAX e96 g6 kg3 k5 bf16 no-bias", 20, 96, 5, torch.bfloat16,
                  k_group=3, group_count=6, gsm=0, bias=False)
allok &= run_case("SUM e192 g8 kg4 k8 bf16 out_flag", 16, 192, 8, torch.bfloat16,
                  k_group=4, group_count=8, gsm=1, out_flag=True)
allok &= run_case("DSV3 rsf=2.5 e256 g8 kg4", 32, 256, 8, torch.bfloat16,
                  k_group=4, group_count=8, gsm=1, rsf=2.5)
allok &= run_case("gc==E pe=1 e16 k4 sigmoid", 16, 16, 4, torch.float16,
                  k_group=8, group_count=16, gsm=0)
allok &= run_case("kg==gc e64 g8 kg8 k4 softmax", 16, 64, 4, torch.float32,
                  k_group=8, group_count=8, gsm=0, norm_type=0)
allok &= run_case("softmax e256 g8 kg4 k6 fp16 renorm1", 16, 256, 6, torch.float16,
                  k_group=4, group_count=8, gsm=1, norm_type=0, renorm=1)
allok &= run_case("调用点风格 bias_opt e256 g8 kg4 bf16", 32, 256, 8, torch.bfloat16,
                  k_group=4, group_count=8, gsm=1, callsite=True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/moe_gating_top_k_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
