"""验证 dequant_swiglu_quant 的 Triton 实现 vs AscendC 语义（fp64 参考）。

覆盖 task-6 要求的三组参数组合：
  1) dynamic + group_index=None + activate_left=True + swiglu_mode=0
  2) dynamic + group_index=None + activate_left=True + swiglu_mode=1（clamp/alpha/bias）
  3) dynamic + group_index=<每组 token 数> + activate_left=True + swiglu_mode=1  ← moe_mlp.py 真实路径
"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
dequant_swiglu_quant = importlib.import_module(_PKG + ".dequant_swiglu_quant").dequant_swiglu_quant

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))


def ref_impl(x, ws, asc, bias, qs, gidx, activate_left, swiglu_mode, c, alpha, gbias):
    """按 dequant_swiglu_quant.h + _cut_group.h 的语义（fp64）。"""
    M, two_h = x.shape
    H = two_h // 2
    xd = x.detach().cpu().double()
    wsd = ws.detach().cpu().double()
    if gidx is not None:
        gi = gidx.detach().cpu().long()
        counts = gi[:, 1] if gi.dim() == 2 else gi
        rows_g = gi[:, 0] if gi.dim() == 2 else torch.arange(gi.numel())
        gid = torch.zeros(M, dtype=torch.long)
        off = 0
        for g in range(counts.numel()):
            gid[off:off + int(counts[g])] = rows_g[g]
            off += int(counts[g])
        ws_row = wsd[gid]                                    # [M, 2H]
        bias_row = bias.detach().cpu().double()[gid] if bias is not None else None
    else:
        ws_row = wsd[None, :].expand(M, -1)
        bias_row = bias.detach().cpu().double()[None, :] if bias is not None else None
    xf = xd * ws_row
    if asc is not None:
        xf = xf * asc.detach().cpu().double()[:, None]
    if bias_row is not None:
        xf = xf + bias_row
    if activate_left:
        act, gate = xf[:, :H], xf[:, H:]
    else:
        act, gate = xf[:, H:], xf[:, :H]
    if swiglu_mode == 1:
        if c > 0:
            gate = torch.clamp(gate, -c, c)
        gate = gate + gbias
        if c > 0:
            act = torch.clamp(act, max=c)
        y = gate * (act / (1.0 + torch.exp(-alpha * act)))
    else:
        y = gate * (act / (1.0 + torch.exp(-act)))
    if qs is not None:
        y = y * qs.detach().cpu().double()[None, :]
    mx = y.abs().amax(dim=1)
    scale = mx / 127.0
    yq = torch.round(y / scale[:, None])
    return yq, scale


def run_case(name, M, H, gidx=None, n_groups=None, activate_left=True, swiglu_mode=0,
             c=0.0, alpha=1.0, gbias=0.0, asc=True, qs=False, g2d=False):
    torch.manual_seed(9)
    x = (torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8)
    if gidx is None:
        ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        gi = None
    else:
        counts = gidx
        n_groups = len(counts)
        assert sum(counts) == M, f"组计数和 {sum(counts)} != M={M}"
        ws = torch.rand(n_groups, 2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
        if g2d:
            gi = torch.stack([torch.arange(n_groups, device=DEV), torch.tensor(counts, device=DEV)], 1).to(torch.int64)
        else:
            gi = torch.tensor(counts, dtype=torch.int64, device=DEV)
    a = (torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25) if asc else None
    q = (torch.rand(H, dtype=torch.float32, device=DEV) + 0.5) if qs else None

    y, sc = dequant_swiglu_quant(x, ws, a, None, q, None, gi, activate_left,
                                 "dynamic", torch.int8, "rint", -1, swiglu_mode,
                                 c if swiglu_mode else None,
                                 alpha if swiglu_mode else None,
                                 gbias if swiglu_mode else None)
    torch.npu.synchronize()
    ry, rsc = ref_impl(x, ws, a, None, q, gi, activate_left, swiglu_mode, c, alpha, gbias)

    d = (y.detach().cpu().double() - ry).abs()
    md = float(d.max())
    exact = float((d == 0).double().mean())
    scd = sc.detach().cpu().double()
    erel = float(((scd - rsc).abs() / rsc.abs().clamp_min(1e-30)).max())
    ok = md <= 1.0 and exact >= 0.999 and erel <= 2e-6
    emit(f"  {name:44s} y max|d|={md:.0f} exact={exact * 100:.3f}% scale_rel={erel:.2e}  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 112)
emit("dequant_swiglu_quant: Triton vs AscendC 语义（fp64 参考；int32 + dynamic + int8）")
emit("y(int8)：max|Δ|<=1 且逐元素相等率>=99.9%；scale(fp32)：相对误差 <=2e-6")
emit("=" * 112)

allok = True
emit("-- 组合 1: group_index=None, activate_left=True, swiglu_mode=0 --")
allok &= run_case("M=33 H=100 组合1", 33, 100)
allok &= run_case("M=48 H=512 组合1+qscale", 48, 512, qs=True)
allok &= run_case("M=32 H=128 组合1+activate_left=F", 32, 128, activate_left=False)

emit("-- 组合 2: group_index=None, activate_left=True, swiglu_mode=1 --")
allok &= run_case("M=33 H=100 组合2 clamp=10 a=1.702 b=1", 33, 100,
                  swiglu_mode=1, c=10.0, alpha=1.702, gbias=1.0)
allok &= run_case("M=24 H=256 组合2 clamp=0(仅alpha/bias)", 24, 256,
                  swiglu_mode=1, c=0.0, alpha=1.702, gbias=1.0)
allok &= run_case("M=24 H=256 组合2 clamp=2 强截断", 24, 256,
                  swiglu_mode=1, c=2.0, alpha=1.702, gbias=1.0)
allok &= run_case("M=16 H=64 组合2 activate_left=F", 16, 64,
                  swiglu_mode=1, c=5.0, alpha=1.0, gbias=0.5, activate_left=False)

emit("-- 组合 3（moe_mlp.py 真实路径）: group_index=<每组token数>, activate_left=True, swiglu_mode=1 --")
allok &= run_case("M=64 H=512 G=4 组计数[20,16,8,20]", 64, 512, gidx=[20, 16, 8, 20],
                  swiglu_mode=1, c=10.0, alpha=1.702, gbias=1.0)
allok &= run_case("M=48 H=256 G=8 组计数[6]*8", 48, 256, gidx=[6] * 8,
                  swiglu_mode=1, c=10.0, alpha=1.702, gbias=1.0)
allok &= run_case("M=37 H=100 G=5 非均匀[5,9,3,12,8]", 37, 100, gidx=[5, 9, 3, 12, 8],
                  swiglu_mode=1, c=4.0, alpha=1.702, gbias=0.5)
allok &= run_case("M=64 H=128 G=4 [G,2] 组号交错", 64, 128, gidx=[16, 16, 16, 16],
                  swiglu_mode=1, c=10.0, alpha=1.702, gbias=1.0, g2d=True)
allok &= run_case("M=64 H=128 G=4 组合3 swiglu_mode=0", 64, 128, gidx=[16, 16, 16, 16])
allok &= run_case("M=64 H=128 G=4 组计数含 0 的组", 64, 128, gidx=[32, 0, 16, 16],
                  swiglu_mode=1, c=10.0, alpha=1.702, gbias=1.0)

emit("-- 组合 4（回归 D1）: grid 第 0 维上限 65535 —— M=65535 过 / 65536 挂 / 65537 过（非单调）--")


def probe_row_ids(M, n=16):
    if M <= n:
        return list(range(M))
    return sorted({int(round(i * (M - 1) / (n - 1))) for i in range(n)})


def run_case_bigm(name, M, H=64, n_sample=16):
    """大 M 回归：**抽样**跑 fp64 参考（与 J3 同一判据），成本 O(采样行)。

    背景（实测）：`_dsq_swiglu_kernel` 原用 `grid=(M,)`，M=65536 触发
        Invalid_Argument(EE1003): KernelLaunch failed because value 65536 for
        parameter coreDim is invalid. Expected value: less than or equal to 65535.
    而 M=65537 又能过 ⇒ **非单调**，因此三个点都必须覆盖，不能只测一边。
    """
    torch.manual_seed(11)
    x = (torch.randint(-1000, 1000, (M, 2 * H), device=DEV, dtype=torch.int32) * 8)
    ws = torch.rand(2 * H, dtype=torch.float32, device=DEV) * 0.02 + 0.001
    a = torch.rand(M, dtype=torch.float32, device=DEV) * 0.5 + 0.25
    try:
        y, sc = dequant_swiglu_quant(x, ws, a, None, None, None, None, True, "dynamic",
                                     torch.int8, "rint", -1, 0, None, None, None)
        torch.npu.synchronize()
    except Exception as e:  # noqa: BLE001
        emit(f"  {name:44s} 抛异常: {type(e).__name__}: {str(e)[:90]}  FAIL")
        return False
    rows = probe_row_ids(M, n_sample)
    ry, rsc = ref_impl(x[rows], ws, a[rows], None, None, None, True, 0, 0.0, 1.0, 0.0)
    d = (y.detach().cpu()[rows].double() - ry).abs()
    md = float(d.max())
    exact = float((d == 0).double().mean())
    erel = float(((sc.detach().cpu().double()[rows] - rsc).abs()
                  / rsc.abs().clamp_min(1e-30)).max())
    ok = md <= 1.0 and exact >= 0.999 and erel <= 2e-6
    emit(f"  {name:44s} M={M} 采样{len(rows)}行 max|d|={md:.0f} exact={exact * 100:.3f}% "
         f"scale_rel={erel:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


allok &= run_case_bigm("D1 回归 M=65535（上限内）", 65535)
allok &= run_case_bigm("D1 回归 M=65536（原本必挂点）", 65536)
allok &= run_case_bigm("D1 回归 M=65537（上限外，非单调对照）", 65537)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/dequant_swiglu_quant_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
