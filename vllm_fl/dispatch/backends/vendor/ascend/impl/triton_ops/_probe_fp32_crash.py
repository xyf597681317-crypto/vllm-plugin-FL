"""系统排查：Ascend bishengir 在「fp32 state 进出」特化上的 ConvertLinalgRToBinary 崩溃。

每个变体一个最小 kernel，逐个 try/except，打印 CRASH/OK。
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

N, M = 64, 128
dev = "npu:0"


def mk(dtype):
    return (torch.randn(N, M, dtype=torch.float32, device=dev) * 0.3).to(dtype)


def check(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  {name:46s} OK")
        return "OK"
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:46s} {tag}: {msg[:70]}")
        return tag


# ---------------- v1: 2D tile, fp32 in -> fp32 out（疑似触发点） ----------------
@triton.jit
def k1(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0)
    tl.store(Y + r[:, None] * M + c[None, :], x + 1.0, mask=m)


# ---------------- v2: fp32 in -> bf16 out ----------------
@triton.jit
def k2(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0)
    tl.store(Y + r[:, None] * M + c[None, :], (x + 1.0).to(tl.bfloat16), mask=m)


# ---------------- v3: bf16 in -> fp32 out ----------------
@triton.jit
def k3(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(Y + r[:, None] * M + c[None, :], x + 1.0, mask=m)


# ---------------- v4: fp32 in/out，1D 展平（无 2D 广播） ----------------
@triton.jit
def k4(X, Y, n, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    m = o < n
    x = tl.load(X + o, mask=m, other=0.0)
    tl.store(Y + o, x + 1.0, mask=m)


# ---------------- v5: fp32 in/out，无 mask ----------------
@triton.jit
def k5(X, Y, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    x = tl.load(X + r[:, None] * BN + c[None, :])
    tl.store(Y + r[:, None] * BN + c[None, :], x + 1.0)


# ---------------- v6: fp32 in/out，store 在动态循环内 ----------------
@triton.jit
def k6(X, Y, N, M, ITER, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0)
    i = 0
    while i < ITER:
        tl.store(Y + r[:, None] * M + c[None, :], x + 1.0, mask=m)
        i += 1


# ---------------- v7: fp32 in/out，显式 .to(tl.float32) 往返 ----------------
@triton.jit
def k7(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(Y + r[:, None] * M + c[None, :], (x + 1.0).to(tl.float32), mask=m)


# ---------------- v8: fp32 in -> 中间转 bf16 -> 出口转回 fp32 ----------------
@triton.jit
def k8(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0)
    xb = (x + 1.0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + r[:, None] * M + c[None, :], xb, mask=m)


# ---------------- v9: fp32 out 但 input 也是 fp32，逐行 1D store ----------------
@triton.jit
def k9(X, Y, N, M, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, BN)
    m = (r[:, None] < N) & (c[None, :] < M)
    x = tl.load(X + r[:, None] * M + c[None, :], mask=m, other=0.0)
    y = x + 1.0
    for j in tl.static_range(4):
        cc = j * (BN // 4) + tl.arange(0, BN // 4)
        mm = (r[:, None] < N) & (cc[None, :] < M)
        tl.store(Y + r[:, None] * M + cc[None, :], tl.sum(tl.where(c[None, :] == cc[:, None], y[:, None, :], 0.0), axis=2), mask=mm)


def run(name, kern, x, y, **kw):
    grid = (triton.cdiv(N, 64),)
    return check(name, lambda: kern[grid](x, y, N, M, **kw))


print("=" * 84)
print("Ascend bishengir fp32-state 崩溃排查（N=64 M=128）")
print("=" * 84)

xf, yf = mk(torch.float32), mk(torch.float32)
xb, yb = mk(torch.bfloat16), mk(torch.bfloat16)

check("v1 2D tile fp32->fp32 (baseline)", lambda: k1[(1,)](xf, yf, N, M, BM=64, BN=128))
check("v2 2D tile fp32->bf16", lambda: k2[(1,)](xf, yb, N, M, BM=64, BN=128))
check("v3 2D tile bf16->fp32", lambda: k3[(1,)](xb, yf, N, M, BM=64, BN=128))
check("v4 1D flat fp32->fp32", lambda: k4[(1,)](xf.view(-1), yf.view(-1), N * M, BLK=1024))
check("v5 2D no-mask fp32->fp32", lambda: k5[(1,)](xf, yf, BM=64, BN=128))
check("v6 2D fp32->fp32 store in loop(2)", lambda: k6[(1,)](xf, yf, N, M, 2, BM=64, BN=128))
check("v7 2D fp32->fp32 显式 .to(f32)", lambda: k7[(1,)](xf, yf, N, M, BM=64, BN=128))
check("v8 fp32->bf16->fp32 中间往返", lambda: k8[(1,)](xf, yf, N, M, BM=64, BN=128))
check("v9 fp32->fp32 静态分片 store", lambda: k9[(1,)](xf, yf, N, M, BM=64, BN=128))
check("v1 num_warps=1", lambda: k1[(1,)](xf, yf, N, M, BM=64, BN=128, num_warps=1))
check("v1 num_warps=2", lambda: k1[(1,)](xf, yf, N, M, BM=64, BN=128, num_warps=2))
check("v1 num_stages=1", lambda: k1[(1,)](xf, yf, N, M, BM=64, BN=128, num_stages=1))
check("v1 num_stages=3", lambda: k1[(1,)](xf, yf, N, M, BM=64, BN=128, num_stages=3))
check("v1 BM=16 BN=32", lambda: k1[(4,)](xf, yf, N, M, BM=16, BN=32))
check("v1 BM=32 BN=64", lambda: k1[(2,)](xf, yf, N, M, BM=32, BN=64))

# 两个 kernel 拆分：k2 算并写 bf16，k3 读 bf16 写 fp32
print("-" * 84)
check("拆分: k2(fp32->bf16) 然后 k3(bf16->fp32)",
      lambda: (k2[(1,)](xf, yb, N, M, BM=64, BN=128), k3[(1,)](yb, yf, N, M, BM=64, BN=128)))
print("done")
