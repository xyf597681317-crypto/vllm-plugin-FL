"""验证 inplace_partial_rotary_mul 的 Triton 实现 vs AscendC 语义。

对拍用 fp64 参考（.detach().cpu().double()），误差地板按输出 dtype：
    fp32 2e-6 / bf16 4e-3 / fp16 1e-3
输入幅度取 |x|<=0.5、cos/sin∈[-1,1]，保证 |out|<=1，量化地板才有意义。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.inplace_partial_rotary_mul import (  # noqa: E402
    MODE_DEEPSEEK_INTERLEAVE, MODE_HALF, MODE_INTERLEAVE, MODE_QUARTER,
    inplace_partial_rotary_mul,
)

DEV = "npu:0"
L = []
FLOOR = {torch.float32: 2e-6, torch.bfloat16: 4e-3, torch.float16: 1e-3}
MODE_NAME = {0: "HALF", 1: "INTERLEAVE", 2: "QUARTER", 3: "DEEPSEEK_ILV"}


def emit(s=""):
    L.append(str(s))


def ref(x, cos, sin, mode, start, end):
    """严格按 AscendC 语义的 fp64 参考（只在最后一维 [start,end) 上旋转）。"""
    xd = x.detach().cpu().double()
    cd = cos.detach().cpu().double()
    sd = sin.detach().cpu().double()
    b, s, n, _ = xd.shape
    hd = end - start
    C = cd.expand(b, s, n, cd.shape[3])
    S = sd.expand(b, s, n, sd.shape[3])
    y = xd.clone()
    xs = xd[..., start:end]
    cs = C[..., :hd]
    ss = S[..., :hd]
    if mode == MODE_INTERLEAVE:
        a, bb = xs[..., 0::2], xs[..., 1::2]          # 相邻对 (2k, 2k+1)
        y[..., start:end:2] = a * cs[..., 0::2] - bb * ss[..., 0::2]
        y[..., start + 1:end:2] = bb * cs[..., 1::2] + a * ss[..., 1::2]
    elif mode == MODE_DEEPSEEK_INTERLEAVE:
        p = hd // 2
        a, bb = xs[..., 0::2], xs[..., 1::2]          # 读交错，写前后半
        y[..., start:start + p] = a * cs[..., :p] - bb * ss[..., :p]
        y[..., start + p:end] = bb * cs[..., p:] + a * ss[..., p:]
    elif mode == MODE_HALF:
        p = hd // 2
        x1, x2 = xs[..., :p], xs[..., p:]
        y[..., start:start + p] = x1 * cs[..., :p] - x2 * ss[..., :p]
        y[..., start + p:end] = x2 * cs[..., p:] + x1 * ss[..., p:]
    else:  # MODE_QUARTER：前后两半各做一次 HALF
        p = hd // 4
        q = [xs[..., i * p:(i + 1) * p] for i in range(4)]
        y[..., start:start + p] = q[0] * cs[..., 0:p] - q[1] * ss[..., 0:p]
        y[..., start + p:start + 2 * p] = q[1] * cs[..., p:2 * p] + q[0] * ss[..., p:2 * p]
        y[..., start + 2 * p:start + 3 * p] = q[2] * cs[..., 2 * p:3 * p] - q[3] * ss[..., 2 * p:3 * p]
        y[..., start + 3 * p:end] = q[3] * cs[..., 3 * p:4 * p] + q[2] * ss[..., 3 * p:4 * p]
    return y


def run_case(name, xshape, cshape, dtype, mode=MODE_INTERLEAVE, slc=(0, 0)):
    torch.manual_seed(1234)
    dev = DEV
    xf = (torch.rand(*xshape, device=dev) * 2.0 - 1.0) * 0.5           # U(-0.5,0.5)
    theta = torch.rand(*cshape, device=dev) * (2.0 * 3.141592653589793)
    cosf = torch.cos(theta)
    sinf = torch.sin(theta)
    x = xf.to(dtype)
    cos = cosf.to(dtype)
    sin = sinf.to(dtype)

    y = inplace_partial_rotary_mul(x, cos, sin, mode=mode, partial_slice=slc)
    torch.npu.synchronize()
    # (0,0) 是 AscendC 的 "不显式指定切片" 约定 → 参考里要解析成整段（否则 hd=0 变成空操作）
    s0, s1 = int(slc[0]), int(slc[1])
    if s0 == 0 and s1 == 0:
        s1 = xshape[-1]
    r = ref(x, cos, sin, mode, s0, s1)

    err = (y.detach().cpu().double() - r).abs().max().item()
    scale = r.abs().max().item()
    floor = FLOOR[dtype]
    ok = err <= floor
    line = (f"  {name:52s} err={err:.3e} (<= {floor:.0e})  |ref|max={scale:.2f}  "
            f"{'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)          # 逐例 flush：NPU 出错时也能看到进度
    return ok


emit("=" * 96)
emit("inplace_partial_rotary_mul: Triton vs AscendC 语义（fp64 参考）")
emit("  地板: fp32 2e-6 / bf16 4e-3 / fp16 1e-3 ; 输入 |x|<=0.5, cos/sin∈[-1,1]")
emit("=" * 96)

# ---------------------------------------------------------------- 调用点风格用例
# 复刻 vllm_fl/attention/ascend/dsa_v1.py L1762-1768 的真实调用：
#     inplace_partial_rotary_mul(x, cos, -sin, rotary_mode="interleave",
#                                partial_slice=[nope_head_dim, head_dim])
# DeepSeek MLA: head_dim=576, nope_head_dim=512 → 只旋转最后 64 维
def run_callsite(name, B, N, nope, head_dim, dtype, rope="interleave", seed=31):
    torch.manual_seed(seed)
    dev = DEV
    seq = 1
    xf = (torch.rand(B, seq, N, head_dim, device=dev) * 2 - 1) * 0.5
    theta = torch.rand(B, seq, 1, head_dim - nope, device=dev) * (2.0 * 3.141592653589793)
    cosf = torch.cos(theta)
    sinf = torch.sin(theta)
    x = xf.to(dtype)
    cos = cosf.to(dtype)
    sin = sinf.to(dtype)
    # 完全按调用点风格：位置参数 + rotary_mode 字符串 + partial_slice 列表
    y = inplace_partial_rotary_mul(x, cos, -sin,
                                   rotary_mode=rope,
                                   partial_slice=[nope, head_dim])
    torch.npu.synchronize()
    # 参考：kernel 里 r1=cos、r2=-sin（公式对 r1/r2 通用，不假设 r2 是 sin）
    r = ref(x, cos, -sin, MODE_NAME_INV[rope], nope, head_dim)
    err = (y.detach().cpu().double() - r).abs().max().item()
    floor = FLOOR[dtype]
    ok = err <= floor
    line = (f"  {name:52s} err={err:.3e} (<= {floor:.0e})  "
            f"{'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    return ok


MODE_NAME_INV = {"half": 0, "interleave": 1, "quarter": 2}

emit("-- 调用点风格用例（dsa_v1.py L1762: x, cos, -sin, rotary_mode='interleave', partial_slice=[512,576]) --")
ok = True
ok &= run_callsite("调用点 B1 N4 ds=576 rot=64 bf16", 1, 4, 512, 576, torch.bfloat16)
ok &= run_callsite("调用点 B2 N8 ds=576 rot=64 fp16", 2, 8, 512, 576, torch.float16)
ok &= run_callsite("调用点 B1 N4 ds=128 rot=64 bf16", 1, 4, 64, 128, torch.bfloat16)
emit("")

B = torch.bfloat16
H = torch.float16
F = torch.float32
emit("-- 覆盖各 mode / 广播 / 部分切片 / dtype 的等价性用例 --")

ok &= run_case("mode1 bf16 [2,8,4,64] cos[2,8,1,64] slice[0,64]", (2, 8, 4, 64), (2, 8, 1, 64), B)
ok &= run_case("mode1 bf16 [2,16,8,128] cos[2,16,1,64] slice[32,96] 部分", (2, 16, 8, 128), (2, 16, 1, 64), B, slc=(32, 96))
ok &= run_case("mode1 fp16 [2,16,8,128] cos[1,1,1,64] slice[0,64] 全播", (2, 16, 8, 128), (1, 1, 1, 64), H, slc=(0, 64))
ok &= run_case("mode1 bf16 [2,8,4,64] cos[1,8,1,64] 只播 dim0(BR=F,SR=T)", (2, 8, 4, 64), (1, 8, 1, 64), B)
ok &= run_case("mode1 bf16 [3,17,5,96] cos[3,17,1,48] slice[16,64] 非整除", (3, 17, 5, 96), (3, 17, 1, 48), B, slc=(16, 64))
ok &= run_case("mode1 fp32 [2,8,4,64] cos[2,8,1,64] slice[0,64]", (2, 8, 4, 64), (2, 8, 1, 64), F)
ok &= run_case("mode0 HALF bf16 [2,8,4,64] slice[0,64]", (2, 8, 4, 64), (2, 8, 1, 64), B, mode=MODE_HALF)
ok &= run_case("mode2 QUARTER bf16 [2,8,4,128] slice[32,96] 部分", (2, 8, 4, 128), (2, 8, 1, 64), B, mode=MODE_QUARTER, slc=(32, 96))
ok &= run_case("mode3 DEEPSEEK fp16 [2,8,4,128] slice[0,64] 部分", (2, 8, 4, 128), (2, 8, 1, 64), H, mode=MODE_DEEPSEEK_INTERLEAVE, slc=(0, 64))

emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/inplace_partial_rotary_mul_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
