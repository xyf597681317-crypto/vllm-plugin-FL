"""验证 moe_init_routing_custom 的 Triton 实现 vs AscendC 语义。

参考实现要点（与 AscendC 对齐）：
  * 稳定升序排序：key = 专家号，平局按展平源下标升序（AscendC 用硬件 Sort，
    文档未保证相等键稳定性；本实现取确定性约定，两边一致）。
  * expanded_row_idx: row_idx_type=0(GATHER) ⇒ out[src]=dst；=1(SCATTER) ⇒ out[dst]=src。
  * expert_tokens_count_or_cumsum:
      COUNT    = 每专家计数
      CUMSUM   = 含端点前缀和 cumsum[e] = #{token : expert <= e}
      KEY_VALUE= (expert_id, count) 对，按专家号升序，只保留 count!=0，其余补 0
判据：
  * expanded_row_idx / expert_tokens_count_or_cumsum 为索引/整数类 ⇒ **必须完全相等**
  * expanded_x 非量化 ⇒ 精确相等（纯 gather）；动态量化 int8 ⇒ ±1 LSB
    （AscendC 走 CAST_ODD(fp32→fp16)+CAST_RINT(fp16→int8)，与 Triton RN 存在至多 1 个量化台阶差）
  * expanded_scale (fp32) ⇒ ≤ 2e-6 × max(1, |ref|)
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.moe_init_routing_custom import (  # noqa: E402
    CUMSUM, COUNT, KEY_VALUE, GATHER, SCATTER, DYNAMIC_QUANT, NON_QUANT,
    moe_init_routing_custom,
)

DEV = os.environ.get("MOE_DEV", "npu:0")   # 可用 MOE_DEV=npu:2 指定卡，避免与他人并行时抢卡
L = []


def emit(s=""):
    L.append(str(s))
    print(str(s), flush=True)


def ref(x, expert_idx, scale, expert_num, expert_start, expert_end,
        row_idx_type, quant_mode, num_type, num_flag):
    """fp64 参考（量化部分按 fp32 复刻 AscendC 的量化链路）。"""
    N, cols = x.shape
    k = expert_idx.shape[1]
    total = N * k
    e = expert_idx.reshape(-1).long().cpu()

    order = torch.argsort(e, stable=True)           # dst -> src（专家升序、稳定）
    dstof = torch.empty(total, dtype=torch.long)
    dstof[order] = torch.arange(total)

    row = order // k
    if quant_mode == DYNAMIC_QUANT:
        e_dst = e[order]
        xv = x.detach().cpu().float()[row]                      # [total, cols]
        # scale=None → 生产调用点的「纯动态量化」分支（AscendC
        # moe_v2_fullload_dynamic_quant.h L238-247）：没有 smooth 因子
        sv = xv if scale is None else xv * scale.detach().cpu().float()[e_dst]
        s = sv.abs().amax(dim=1) / 127.0                        # [total]
        q = sv / s[:, None]
        q16 = q.half().float()                                  # CAST_ODD 近似
        r = torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))
        ex = r.clamp(-128, 127).to(torch.int8)
        escale = s
    else:
        ex = x.detach().cpu()[row]
        escale = torch.zeros(total, dtype=torch.float32)

    if row_idx_type == SCATTER:
        ridx = order
    else:
        ridx = dstof

    hist = torch.bincount(torch.clamp(e, 0, expert_num - 1), minlength=expert_num)
    hist = hist[expert_start:expert_end].to(torch.int64)
    if not num_flag:
        cnt = torch.zeros(expert_end - expert_start, dtype=torch.int64)
    elif num_type == COUNT:
        cnt = hist.clone()
    elif num_type == CUMSUM:
        cnt = hist.cumsum(0)
    else:
        cnt = torch.zeros((expert_num, 2), dtype=torch.int64)
        nz = torch.nonzero(hist, as_tuple=False).reshape(-1)
        for j, idx in enumerate(nz.tolist()):
            cnt[j, 0] = expert_start + idx
            cnt[j, 1] = hist[idx]
    return ex, ridx, cnt, escale


def run_case(name, N, k, cols, E, dtype, row_idx_type=GATHER, num_type=CUMSUM,
             num_flag=True, quant_mode=NON_QUANT, erange=None, xscale=2.0,
             smooth=True, zero_row=False):
    torch.manual_seed(0)
    x = (torch.randn(N, cols, dtype=torch.float32, device=DEV) * xscale).to(dtype)
    if zero_row:
        # 整零行：amax=0 ⇒ s=0 ⇒ 参考实现 q=0/0=nan，`clamp().to(int8)` 得 0；
        # 内核用 1.0 占位（0/1e-30=0）与之一致（不加占位会得到 -128）。
        x[0] = 0
    lo, hi = (0, E) if erange is None else erange
    expert_idx = torch.randint(lo, hi, (N, k), dtype=torch.int32, device=DEV)
    scale = None
    if quant_mode == DYNAMIC_QUANT and smooth:
        scale = (torch.rand(E, cols, dtype=torch.float32, device=DEV) * 0.5 + 0.75)

    ex, ridx, cnt, escale = moe_init_routing_custom(
        x, expert_idx, scale, None,
        expert_num=E, expert_tokens_num_type=num_type,
        expert_tokens_num_flag=num_flag, quant_mode=quant_mode,
        active_expert_range=(list(erange) if erange else []),
        row_idx_type=row_idx_type)
    torch.npu.synchronize()
    rx, rr, rc, rs = ref(x, expert_idx, scale, E, lo, hi, row_idx_type,
                         quant_mode, num_type, num_flag)

    e_ridx = (ridx.detach().cpu().long() - rr).abs().max().item()
    e_cnt = (cnt.detach().cpu().long() - rc).abs().max().item()
    if quant_mode == DYNAMIC_QUANT:
        e_x = (ex.detach().cpu().long() - rx.long()).abs().max().item()
        f_x = 1
    else:
        e_x = (ex.detach().cpu().double() - rx.double()).abs().max().item()
        f_x = 0
    e_s = (escale.detach().cpu().double() - rs.double()).abs().max().item()
    f_s = 2e-6 * max(1.0, rs.abs().max().item())
    ok = e_x <= f_x and e_ridx == 0 and e_cnt == 0 and e_s <= f_s
    emit(f"  {name:56s} x_err={e_x}(<={f_x}) rowidx_diff={e_ridx} cnt_diff={e_cnt} "
         f"scale_err={e_s:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 112)
emit("moe_init_routing_custom: Triton vs AscendC 语义")
emit("=" * 112)

allok = True
allok &= run_case("非量化 fp32 N=64 k=4 cols=32 E=8 CUMSUM GATHER", 64, 4, 32, 8, torch.float32)
allok &= run_case("非量化 bf16 N=64 k=8 cols=64 E=16 COUNT SCATTER", 64, 8, 64, 16,
                  torch.bfloat16, row_idx_type=SCATTER, num_type=COUNT)
allok &= run_case("非量化 fp16 N=32 k=2 cols=48 E=6 KEY_VALUE GATHER", 32, 2, 48, 6,
                  torch.float16, num_type=KEY_VALUE)
allok &= run_case("非量化 bf16 N=7 k=3 cols=17 E=5 (非整除)", 7, 3, 17, 5,
                  torch.bfloat16, num_type=COUNT)
allok &= run_case("非量化 fp32 N=16 k=4 cols=32 E=8 num_flag=False", 16, 4, 32, 8,
                  torch.float32, num_flag=False)
allok &= run_case("非量化 fp32 N=32 k=4 cols=24 E=8 range=[2,6)", 32, 4, 24, 8,
                  torch.float32, erange=(2, 6))
allok &= run_case("动态量化 bf16 N=32 k=4 cols=64 E=8 CUMSUM GATHER", 32, 4, 64, 8,
                  torch.bfloat16, quant_mode=DYNAMIC_QUANT)
allok &= run_case("动态量化 fp16 N=16 k=2 cols=33 E=4 SCATTER KEY_VALUE", 16, 2, 33, 4,
                  torch.float16, row_idx_type=SCATTER, num_type=KEY_VALUE,
                  quant_mode=DYNAMIC_QUANT)
allok &= run_case("动态量化 fp32 N=8 k=4 cols=128 E=16 COUNT GATHER", 8, 4, 128, 16,
                  torch.float32, num_type=COUNT, quant_mode=DYNAMIC_QUANT)
# 生产真实形态：token_dispatcher.py L401 传 quant_mode=1 + scale=None
allok &= run_case("动态量化(scale=None) bf16 N=32 k=4 cols=64 E=8 GATHER", 32, 4, 64, 8,
                  torch.bfloat16, quant_mode=DYNAMIC_QUANT, smooth=False)
allok &= run_case("动态量化(scale=None) fp16 N=16 k=2 cols=33 E=4 KEY_VALUE", 16, 2, 33, 4,
                  torch.float16, num_type=KEY_VALUE, quant_mode=DYNAMIC_QUANT, smooth=False)
# 生产列宽：cols=7168（token_dispatcher.py L401 的真实调用形态），同时验证
# 「cols 非 2 幂 / 大列宽」下 1-D 写法不丢写（旧 2-D 掩码 store 写法会静默丢写）
allok &= run_case("动态量化(scale=None) fp16 N=8 k=2 cols=7168 E=8 (生产列宽)", 8, 2, 7168, 8,
                  torch.float16, quant_mode=DYNAMIC_QUANT, smooth=False)
allok &= run_case("非量化 bf16 N=8 k=2 cols=7168 E=8 (生产列宽)", 8, 2, 7168, 8,
                  torch.bfloat16, quant_mode=NON_QUANT)
allok &= run_case("动态量化(scale=None) fp16 N=16 k=2 cols=33 E=4 整零行", 16, 2, 33, 4,
                  torch.float16, quant_mode=DYNAMIC_QUANT, smooth=False, zero_row=True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
os.makedirs("/root/dsh_work", exist_ok=True)
open("/root/dsh_work/moe_init_routing_custom_test.txt", "w", encoding="utf-8").write(txt + "\n")
