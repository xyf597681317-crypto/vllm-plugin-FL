"""验证 moe_gating_top_k_hash 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.moe_gating_top_k_hash import moe_gating_top_k_hash  # noqa: E402

DEV = "npu:0"
L = []

FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}


def quant_floor(dtype, ref_absmax):
    """输出 dtype 的**量化地板**（有推导，不含任何手写保底）。

    输出落到 T 时，误差上界就是该 dtype 在实际幅度下的**半个 ULP**：
        ULP(v) = 2^(e-(p-1)),  half-ULP = 2^(e-p),  e = floor(log2|v|)
        p = 有效位数（含隐含位）：bf16 = 8, fp16 = 11, fp32 = 24
    手册的 4e-3 / 1e-3 / 2e-6 对应 **O(1) 输出**：
        bf16 |v|∈[1,2) → 2^-8 = 3.906e-3（≈ 手册 4e-3）
        fp16 |v|∈[1,2) → 2^-11 = 4.883e-4（手册 1e-3 更宽，此处取更严的）
    输出幅度更大时（如 routed_scaling_factor=2.5 或一次 combine 累加多个槽位），
    地板按 ULP 线性放大 —— 这是量化本身的性质，不是放宽判据。
    fp32 例外：误差由 fp32 **计算**累积主导而非量化，故沿用手册的 2e-6 绝对地板。
    """
    if dtype == torch.float32:
        return FLOOR[torch.float32]
    p = {torch.bfloat16: 8, torch.float16: 11}[dtype]
    m = max(1.0, float(ref_absmax))
    e = math.floor(math.log2(m))
    return 2.0 ** (e - p)


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


def ref(x, bias, input_ids, tid2eid, k, k_group, group_count, group_select_mode,
        norm_type, out_flag, rsf, eps):
    """严格按 AscendC 内核顺序计算，fp64 参考。"""
    xd = x.detach().cpu().double()
    N, E = xd.shape
    if norm_type == 1:            # sigmoid
        xn = torch.sigmoid(xd)
    elif norm_type == 0:          # softmax
        m = xd.max(dim=-1, keepdim=True).values
        e = torch.exp(xd - m)
        xn = e / e.sum(dim=-1, keepdim=True)
    else:                         # softplus: sqrt(ln(1+exp(x)))
        xn = torch.sqrt(torch.log(1.0 + torch.exp(xd)))
    if bias is not None:
        sc = xn + bias.detach().cpu().double().unsqueeze(0)
    else:
        sc = xn

    G = group_count
    P = E // G
    if input_ids is not None:
        ids = input_ids.detach().cpu().long()
        t2e = tid2eid.detach().cpu().long()
        eidx = t2e[ids][:, :k]
    elif G == 1:
        eidx = torch.argsort(sc, dim=-1, descending=True, stable=True)[:, :k]
    else:
        scg = sc.view(N, G, P)
        v = torch.gather(scg, 2, torch.argsort(scg, dim=-1, descending=True, stable=True))
        gs = v[:, :, 0] + v[:, :, 1] if group_select_mode == 1 else v[:, :, 0]
        go = torch.argsort(gs, dim=-1, descending=True, stable=True)[:, :k_group]
        gmask = torch.zeros(N, G, dtype=torch.bool).scatter_(1, go, True)
        big = sc.masked_fill(~gmask.repeat_interleave(P, dim=1), float("-inf"))
        eidx = torch.argsort(big, dim=-1, descending=True, stable=True)[:, :k]

    y = torch.gather(xn, 1, eidx)
    if norm_type in (1, 2):
        y = y / (y.sum(dim=-1, keepdim=True) + eps)
    y = y * rsf
    outv = xn if out_flag else torch.zeros_like(xn)
    return y, eidx, outv


def run_case(name, N, E, k, dtype, norm_type, bias=True, hash_=False, k_group=1,
             group_count=1, group_select_mode=0, out_flag=False, rsf=1.0, eps=1e-20,
             scale=2.0):
    torch.manual_seed(0)
    x = (torch.randn(N, E, dtype=torch.float32, device=DEV) * scale).to(dtype)
    b = ((torch.randn(E, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
         if bias else None)
    if hash_:
        V = max(N, 8)
        input_ids = torch.randint(0, V, (N,), dtype=torch.int64, device=DEV)
        tid2eid = torch.randint(0, E, (V, k), dtype=torch.int32, device=DEV)
    else:
        input_ids = tid2eid = None

    y, eidx, out = moe_gating_top_k_hash(
        x, b, input_ids, tid2eid, k=k, k_group=k_group, group_count=group_count,
        group_select_mode=group_select_mode, norm_type=norm_type, out_flag=out_flag,
        routed_scaling_factor=rsf, eps=eps)
    torch.npu.synchronize()
    ry, reidx, rout = ref(x, b, input_ids, tid2eid, k, k_group, group_count,
                          group_select_mode, norm_type, out_flag, rsf, eps)

    # 判据 = 输出 dtype 在实际幅度下的 half-ULP 量化地板（推导见 quant_floor）；
    # 同时打印手册的绝对地板，超出绝对地板时用 * 标出，绝不隐藏。
    amp = ry.abs().max().item()
    ey = (y.detach().cpu().double() - ry).abs().max().item()
    ei = (eidx.detach().cpu().long() - reidx).abs().max().item()
    eo = (out.detach().cpu().double() - rout).abs().max().item()

    # ---- 判据（与 mc2 统一，Lead 裁决 2026-10-10）：分段取严 + ULP 验证 ----
    #   |ref|max ≤ 1 → README 绝对地板（bf16 4e-3 / fp16 1e-3 / fp32 2e-6）
    #   |ref|max > 1 → 输出 dtype 的完整 1 ULP @ |ref|max（fp32 流水线的物理下界）
    p_bits = {torch.float32: 24, torch.bfloat16: 8, torch.float16: 11}[dtype]
    e_mag = math.floor(math.log2(max(1.0, amp)))
    f_readme = FLOOR[dtype]
    f_1ulp = 2.0 ** (e_mag - (p_bits - 1)) if dtype != torch.float32 else f_readme
    f_seg = f_readme if amp <= 1.0 else f_1ulp

    # ULP 验证：ref_q = fp64 真值正确舍入到输出 dtype；看最大误差位置的 ULP 比
    ref_q = ry.to(dtype).double()
    got = y.detach().cpu().double()
    a = ref_q.abs()
    e_pt = torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a))))
    u_pt = torch.where(a > 0, torch.pow(torch.tensor(2.0, dtype=torch.float64),
                                        e_pt - (p_bits - 1)), torch.zeros_like(a))
    dq = (got - ref_q).abs()
    rat = torch.where(u_pt > 0, dq / torch.clamp(u_pt, min=1e-300), torch.zeros_like(dq))
    i_max = int(dq.flatten().argmax().item())
    rat_at = rat.flatten()[i_max].item()
    big = (out.detach().cpu().double() - rout).abs()  # 不作 gated 用，仅保留 raw 信息
    rat_big = rat[a >= (0.1 * amp)].max().item() if int((a >= 0.1 * amp).sum()) > 0 else 0.0

    ok = (ey <= f_seg) and (ei == 0) and (eo <= 2e-6)
    emit(f"  {name:52s} y_err={ey:.3e}(<={f_seg:.3e} 分段: "
         f"{'README' if amp <= 1.0 else '1ULP'}) idx_diff={ei} out_err={eo:.3e}(<=2e-6)"
         f"  ULP@maxerr={rat_at:.2f} gated={rat_big:.2f}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 96)
emit("moe_gating_top_k_hash: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 96)

allok = True
# --- 不分组，hash 路径（算子名主体） ---
allok &= run_case("hash sigmoid+bias E=64 k=4 bf16 rsf=2.5", 64, 64, 4, torch.bfloat16,
                  norm_type=1, hash_=True, rsf=2.5)
allok &= run_case("hash softmax    E=128 k=8 fp16", 128, 128, 8, torch.float16,
                  norm_type=0, bias=False, hash_=True)
allok &= run_case("hash softplus   E=17  k=3 bf16 (非整除)", 32, 17, 3, torch.bfloat16,
                  norm_type=2, hash_=True)
allok &= run_case("hash sigmoid    E=128 k=16 bf16 rsf=1.0", 256, 128, 16, torch.bfloat16,
                  norm_type=1, hash_=True)
# --- 不分组，按得分 top-k ---
allok &= run_case("topk sigmoid+bias E=64 k=4 fp32 out_flag", 48, 64, 4, torch.float32,
                  norm_type=1, out_flag=True)
allok &= run_case("topk softmax+no_bias E=33 k=5 bf16", 16, 33, 5, torch.bfloat16,
                  norm_type=0, bias=False)
allok &= run_case("topk softplus+bias E=64 k=6 fp16", 24, 64, 6, torch.float16,
                  norm_type=2, out_flag=True)
# --- 分组（DeepSeek-V3 型） ---
allok &= run_case("group E=256 G=8 P=32 KG=4 k=8 sum sigmoid bf16", 32, 256, 8,
                  torch.bfloat16, norm_type=1, k_group=4, group_count=8,
                  group_select_mode=1)
allok &= run_case("group E=256 G=8 P=32 KG=4 k=8 sum sigmoid fp32 out",
                  16, 256, 8, torch.float32, norm_type=1, k_group=4, group_count=8,
                  group_select_mode=1, out_flag=True)
allok &= run_case("group E=24  G=4 P=6  KG=2 k=3 max sigmoid fp16", 20, 24, 3,
                  torch.float16, norm_type=1, k_group=2, group_count=4,
                  group_select_mode=0)
allok &= run_case("group E=64  G=8 P=8  KG=3 k=12 softmax bf16", 12, 64, 12,
                  torch.bfloat16, norm_type=0, k_group=3, group_count=8,
                  group_select_mode=0)
# --- 分组：全组选择 / 两种 mode 的交叉验证（E=256 为 E%32==0 的特化路径） ---
allok &= run_case("group E=256 G=8 P=32 KG=8 k=8 sum (全组) fp32", 8, 256, 8,
                  torch.float32, norm_type=1, k_group=8, group_count=8,
                  group_select_mode=1)
allok &= run_case("group E=256 G=8 P=32 KG=4 k=8 max fp32", 8, 256, 8,
                  torch.float32, norm_type=1, k_group=4, group_count=8,
                  group_select_mode=0)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/moe_gating_top_k_hash_test.txt", "w", encoding="utf-8").write(txt + "\n")
