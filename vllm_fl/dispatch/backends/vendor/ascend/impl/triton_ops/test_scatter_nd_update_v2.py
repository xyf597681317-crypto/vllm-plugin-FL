"""验证 scatter_nd_update_v2 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.scatter_nd_update_v2 import scatter_nd_update_v2  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3,
         torch.int32: 0.0, torch.int64: 0.0}


def emit(s=""):
    L.append(str(s))


def dense_strides(shape, K):
    """var 连续布局下前 K 维的 strides（元素单位，含 tail 维）。"""
    full = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        full[k] = full[k + 1] * int(shape[k + 1])
    return full[:K]


def ref(var, indices, updates, strides):
    """fp64 参考：按扁平序号 n 递增写入，同一地址 last-write-wins。"""
    K = int(indices.shape[-1])
    S = 1
    for d in var.shape[K:]:
        S *= int(d)
    N = 1
    for d in indices.shape[:-1]:
        N *= int(d)
    y = var.detach().cpu().double().clone().view(-1)
    idx = indices.detach().cpu().long().view(N, K)
    upd = updates.detach().cpu().double().view(N, S)
    for n in range(N):
        off = 0
        for k in range(K):
            off += int(idx[n, k].item()) * int(strides[k])
        y[off:off + S] = upd[n]
    return y.view(tuple(var.shape))


def run_case(name, var_shape, idx_shape, strides, dtype, dup=False, idx_dtype=torch.int32):
    torch.manual_seed(0)
    K = int(idx_shape[-1])
    if strides is None:
        strides = dense_strides(var_shape, K)
    total = 1
    for d in var_shape[:K]:
        total *= int(d)
    N = 1
    for d in idx_shape[:-1]:
        N *= int(d)

    var = (torch.randn(*var_shape, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
    if dup:
        # 故意制造重复：N 行只覆盖 total//2 个地址（K=1）
        base = (torch.arange(N, dtype=torch.int64, device=DEV) % max(1, total // 2))
        idxs = base.view(-1, 1).repeat(1, K).to(idx_dtype)
    else:
        # 每一维分别取 [0, var.shape[k]) 内的随机下标（K>1 时不能按展平总数取）
        cols = [torch.randint(0, int(var_shape[k]), (N,), dtype=torch.int64, device=DEV)
                for k in range(K)]
        idxs = torch.stack(cols, dim=1).to(idx_dtype)
    updates = (torch.randn(*idx_shape[:-1], *var_shape[K:], dtype=torch.float32, device=DEV) * 1.5).to(dtype)

    y = scatter_nd_update_v2(var, idxs, updates, strides)
    torch.npu.synchronize()
    r = ref(var, idxs, updates, strides)

    got = y.detach().cpu().double()
    err = (got - r).abs().max().item()
    floor = FLOOR[dtype]
    exact = bool(torch.equal(y.detach().cpu(), r.to(dtype).cpu()))
    ok = err <= floor and exact and tuple(y.shape) == tuple(var.shape)
    emit(f"  {name:48s} err={err:.3e} (<= {floor:.0e}) exact={exact}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 86)
emit("scatter_nd_update_v2: Triton vs AscendC 语义（fp64 参考，last-write-wins）")
emit("=" * 86)

allok = True
# K=1：MoE 场景 var[E, D] ← updates[N, D]
allok &= run_case("K=1 var[64,128] idx[257,1] bf16 (N 非整除)", (64, 128), (257, 1), [128], torch.bfloat16)
allok &= run_case("K=1 var[256,512] idx[1024,1] fp16", (256, 512), (1024, 1), [512], torch.float16)
allok &= run_case("K=1 var[512,64] idx[500,1] fp32", (512, 64), (500, 1), [64], torch.float32)
allok &= run_case("K=1 var[128,32] idx[200,1] int32", (128, 32), (200, 1), [32], torch.int32)
allok &= run_case("K=1 var[128,32] idx[200,1] int64-indices", (128, 32), (200, 1), [32], torch.float32,
                  idx_dtype=torch.int64)
# K=2：indices[N,2]，var[A,B,*tail]
allok &= run_case("K=2 var[8,16,32] idx[300,2] bf16", (8, 16, 32), (300, 2), None, torch.bfloat16)
allok &= run_case("K=2 var[32,32,16] idx[129,2] fp32", (32, 32, 16), (129, 2), None, torch.float32)
# K=3
allok &= run_case("K=3 var[8,8,8,16] idx[77,3] fp16", (8, 8, 8, 16), (77, 3), None, torch.float16)
# 重复索引：last-write-wins
allok &= run_case("K=1 dup var[16,64] idx[128,1] fp32", (16, 64), (128, 1), [64], torch.float32, dup=True)
allok &= run_case("K=1 dup var[64,32] idx[300,1] bf16", (64, 32), (300, 1), [32], torch.bfloat16, dup=True)
# 非稠密 strides：var[8,16,16] 用 strides=[256,16]，每行更新区间 [i*256+j*16, +16) 互不重叠
allok &= run_case("K=2 非稠密 strides var[8,16,16] idx[64,2] fp32",
                  (8, 16, 16), (64, 2), [256, 16], torch.float32)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/scatter_nd_update_v2_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
