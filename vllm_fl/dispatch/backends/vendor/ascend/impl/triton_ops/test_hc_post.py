"""验证 hc_post 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.hc_post import hc_post  # noqa: E402

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}


def emit(s=""):
    L.append(str(s))


def ref(x, residual, post, comb):
    """fp64 参考：y[b,h,d] = post[b,h]*x[b,d] + sum_k comb[b,k,h]*residual[b,k,d]（支持任意前导维）。"""
    lead = tuple(int(v) for v in x.shape[:-1])
    D = int(x.shape[-1])
    H = int(residual.shape[-2])
    B = 1
    for v in lead:
        B *= v
    xd = x.detach().cpu().double().reshape(B, D)
    rd = residual.detach().cpu().double().reshape(B, H, D)
    pd = post.detach().cpu().double().reshape(B, H)
    cd = comb.detach().cpu().double().reshape(B, H, H)
    y = pd[:, :, None] * xd[:, None, :] + torch.einsum("bkh,bkd->bhd", cd, rd)
    return y.reshape(*lead, H, D)


def run_case(name, B, H, D, t1, t2, scale=0.4):
    """输入取 [-scale, scale] 均匀分布：保证 |y| = O(1)，
    这样「绝对误差 <= dtype 量化地板」才是可判定的（bf16 地板 4e-3 对应 |y|~1）。"""
    torch.manual_seed(0)
    x = (torch.rand(B, D, dtype=torch.float32, device=DEV) * 2 - 1).mul(scale).to(t1)
    residual = (torch.rand(B, H, D, dtype=torch.float32, device=DEV) * 2 - 1).mul(scale).to(t1)
    post = ((torch.rand(B, H, dtype=torch.float32, device=DEV) * 2 - 1).mul(scale)).to(t2)
    comb = ((torch.rand(B, H, H, dtype=torch.float32, device=DEV) * 2 - 1).mul(scale)).to(t2)

    y = hc_post(x, residual, post, comb)
    torch.npu.synchronize()
    r = ref(x, residual, post, comb)

    got = y.detach().cpu().double()
    err = (got - r).abs().max().item()
    amax = r.abs().max().item()
    floor = FLOOR[t1]
    ok = err <= floor and tuple(y.shape) == tuple(r.shape)
    emit(f"  {name:42s} err={err:.3e} (<= {floor:.0e})  rel={err / amax:.2e} "
         f"|y|max={amax:.2f}  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 80)
emit("hc_post: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 80)

allok = True
allok &= run_case("B=8   H=4 D=512  bf16/bf16", 8, 4, 512, torch.bfloat16, torch.bfloat16)
allok &= run_case("B=4   H=4 D=1024 fp16/fp16", 4, 4, 1024, torch.float16, torch.float16)
allok &= run_case("B=8   H=4 D=512  fp16/bf16 (混合)", 8, 4, 512, torch.float16, torch.bfloat16)
allok &= run_case("B=8   H=4 D=512  bf16/fp32 (混合)", 8, 4, 512, torch.bfloat16, torch.float32)
allok &= run_case("B=3   H=4 D=100  fp32/fp32 (D 非对齐)", 3, 4, 100, torch.float32, torch.float32)
allok &= run_case("B=5   H=3 D=96   bf16/bf16 (H=3)", 5, 3, 96, torch.bfloat16, torch.bfloat16)
allok &= run_case("B=2   H=2 D=64   fp16/fp16 (H=2)", 2, 2, 64, torch.float16, torch.float16)
allok &= run_case("B=7   H=8 D=128  bf16/bf16 (H=8)", 7, 8, 128, torch.bfloat16, torch.bfloat16)
allok &= run_case("B=1   H=4 D=33   fp32/fp32 (D=33)", 1, 4, 33, torch.float32, torch.float32)

# =====================================================================================
# 调用点风格：vllm_ascend / vllm_fl 的真实传参
#   npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
#   → x/post 3D、residual/comb 4D（binding check_hc_post_shape_and_dtype L1404-1428）
# =====================================================================================
emit()
emit("调用点风格测试（x.unsqueeze(0) 形态，x/post 3D、residual/comb 4D）：")
emit("-" * 84)


def run_callsite(name, n, hc, d, t1, t2, use_dim=None):
    torch.manual_seed(0)

    def U(*shape, lo=0.4):
        return ((torch.rand(*shape, dtype=torch.float32, device=DEV) * 2 - 1) * lo)

    x = U(n, d).to(t1)
    residual = U(n, hc, d).to(t1)
    post = U(n, hc).to(t2)
    comb = U(n, hc, hc).to(t2)

    kwargs = {} if use_dim is None else {"dim": use_dim}
    y = hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0), **kwargs)
    torch.npu.synchronize()
    r = ref(x, residual, post, comb)
    err = (y.detach().cpu().double() - r).abs().max().item()
    shape_ok = tuple(y.shape) == (1, n, hc, d)
    floor = FLOOR[t1]
    ok = err <= floor and shape_ok
    emit(f"  {name:44s} err={err:.3e} (<= {floor:.0e}) y.shape={tuple(y.shape)}  "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


allok &= run_callsite("callsite n=128 hc=4 d=512 bf16/bf16", 128, 4, 512,
                      torch.bfloat16, torch.bfloat16)
allok &= run_callsite("callsite n=96  hc=4 d=256 bf16/fp32", 96, 4, 256,
                      torch.bfloat16, torch.float32)
allok &= run_callsite("callsite n=64  hc=4 d=128 fp16/fp16", 64, 4, 128,
                      torch.float16, torch.float16)
allok &= run_callsite("callsite n=32  hc=4 d=64  fp32/fp32 dim=0", 32, 4, 64,
                      torch.float32, torch.float32, use_dim=0)


def dim_rejected():
    """dim=1 必须显式报错（hc_post 没有 dim 参数）。"""
    global allok
    x = torch.zeros(1, 8, 16, dtype=torch.bfloat16, device=DEV)
    residual = torch.zeros(1, 8, 4, 16, dtype=torch.bfloat16, device=DEV)
    post = torch.zeros(1, 8, 4, dtype=torch.bfloat16, device=DEV)
    comb = torch.zeros(1, 8, 4, 4, dtype=torch.bfloat16, device=DEV)
    try:
        hc_post(x, residual, post, comb, dim=1)
    except AssertionError as e:
        emit(f"  {'dim=1 显式拒绝（无 dim 参数）':44s} 正确抛出 AssertionError  PASS")
        return True
    except Exception as e:  # noqa: BLE001
        emit(f"  {'dim=1 显式拒绝（无 dim 参数）':44s} 抛出 {type(e).__name__}（期望 AssertionError）  FAIL")
        allok = False
        return False
    emit(f"  {'dim=1 显式拒绝（无 dim 参数）':44s} 未抛异常  FAIL")
    allok = False
    return False


allok &= dim_rejected()

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/hc_post_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
