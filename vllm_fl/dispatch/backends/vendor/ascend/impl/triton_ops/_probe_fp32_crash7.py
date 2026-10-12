"""第六轮：reshape 语义 + for/while 差异 + 1-D store 是否能作为正确 workaround。"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

dev = "npu:0"


def check(name, fn, verify=None):
    try:
        fn()
        torch.npu.synchronize()
        extra = ""
        if verify is not None:
            extra = "  -> " + verify()
        print(f"  {name:52s} OK{extra}", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:52s} {tag}: {msg[:56]}", flush=True)


# T1: reshape 是否保序
@triton.jit
def t1(X, Y, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    x = tl.load(X + r[:, None] * BN + c[None, :])
    y = tl.reshape(x, [BM * BN])
    tl.store(Y + tl.arange(0, BM * BN), y)


# T2: while 循环内 2-D store（fp32）
@triton.jit
def t2(STATE, SSM, FINAL, T, NV, DK, DV, BLK_V: tl.constexpr, BLK_K: tl.constexpr, USE_FOR: tl.constexpr):
    pid_h = tl.program_id(0)
    v_off = tl.arange(0, BLK_V)
    k_off = tl.arange(0, BLK_K)
    m = (v_off[:, None] < DV) & (k_off[None, :] < DK)
    sm = tl.load(STATE + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :], mask=m, other=0.0)
    if USE_FOR:
        for t in range(0, T):
            s2 = tl.load(SSM + t).to(tl.int32)
            tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :], sm, mask=m)
    else:
        t = 0
        while t < T:
            s2 = tl.load(SSM + t).to(tl.int32)
            tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :], sm, mask=m)
            t += 1


# T3: 循环内 1-D store（reshape 取值 + 算术偏移），fp32
@triton.jit
def t3(STATE, SSM, FINAL, T, NV, DK, DV, BLK_V: tl.constexpr, BLK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    v_off = tl.arange(0, BLK_V)
    k_off = tl.arange(0, BLK_K)
    m = (v_off[:, None] < DV) & (k_off[None, :] < DK)
    sm = tl.load(STATE + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :], mask=m, other=0.0)
    i1 = tl.arange(0, BLK_V * BLK_K)
    vv = i1 // BLK_K
    kk = i1 % BLK_K
    off1 = vv * DK + kk
    m1 = (vv < DV - pid_h * 0) & (kk < DK)
    m1 = (vv < BLK_V) & (kk < DK)
    t = 0
    while t < T:
        s2 = tl.load(SSM + t).to(tl.int32)
        val = tl.reshape(sm, [BLK_V * BLK_K])
        tl.store(FINAL + (s2 * NV + pid_h) * DV * DK + off1, val, mask=m1)
        t += 1


T, NV, DK, DV, BLK_V, BLK_K = 4, 2, 8, 8, 8, 8
torch.manual_seed(0)
st = torch.arange(NV * DV * DK, dtype=torch.float32, device=dev).reshape(NV, DV, DK)
ssm = torch.arange(T, dtype=torch.int32, device=dev)

print("=" * 84, flush=True)
print("第六轮：reshape 语义 / for-vs-while / 1-D store workaround", flush=True)
print("=" * 84, flush=True)

# T1
BM, BN = 4, 8
X = torch.arange(BM * BN, dtype=torch.float32, device=dev).reshape(BM, BN)
Y = torch.full((BM * BN,), -1.0, device=dev)
check(f"T1 tl.reshape 2-D[{BM},{BN}]->1-D 是否保序", lambda: t1[(1,)](X, Y, BM=BM, BN=BN),
      verify=lambda: ("保序 OK" if torch.equal(Y.cpu(), torch.arange(BM * BN, dtype=torch.float32))
                       else f"乱序!! got[:8]={Y[:8].tolist()}"))

# T2 while vs for（fp32，已知 while+2D store 崩）
f_while = torch.zeros(T, NV, DV, DK, device=dev)
f_for = torch.zeros(T, NV, DV, DK, device=dev)
check("T2a while 循环内 2-D store fp32", lambda: t2[(NV,)](st, ssm, f_while, T, NV, DK, DV,
                                                          BLK_V=BLK_V, BLK_K=BLK_K, USE_FOR=False))
check("T2b for 循环内 2-D store fp32", lambda: t2[(NV,)](st, ssm, f_for, T, NV, DK, DV,
                                                         BLK_V=BLK_V, BLK_K=BLK_K, USE_FOR=True))

# T3 1-D store
f_1d = torch.zeros(T, NV, DV, DK, device=dev)
check("T3 while 循环内 1-D store（reshape+算术偏移）fp32",
      lambda: t3[(NV,)](st, ssm, f_1d, T, NV, DK, DV, BLK_V=BLK_V, BLK_K=BLK_K))
if True:
    try:
        torch.npu.synchronize()
        exp = st.unsqueeze(0).expand(T, NV, DV, DK).clone()
        same = torch.equal(f_1d.cpu(), exp.cpu())
        print(f"      T3 数值: {'正确' if same else '错误'}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"      T3 数值检查失败 {e}", flush=True)
print("done", flush=True)
