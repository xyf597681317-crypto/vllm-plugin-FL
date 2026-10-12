"""第三轮：验证 workaround —— 把循环内的 2-D store 展平成 1-D。"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

dev = "npu:0"


def check(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  {name:58s} OK", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:58s} {tag}: {msg[:60]}", flush=True)


# A1 复现：循环内 2-D store
@triton.jit
def a1(STATE, SSM, FINAL, T, NV, DK, DV, BLK_V: tl.constexpr, BLK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    v_off = tl.arange(0, BLK_V)
    k_off = tl.arange(0, BLK_K)
    m = (v_off[:, None] < DV) & (k_off[None, :] < DK)
    sm = tl.load(STATE + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :], mask=m, other=0.0)
    t = 0
    while t < T:
        s2 = tl.load(SSM + t).to(tl.int32)
        tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                 sm.to(FINAL.dtype.element_ty), mask=m)
        t += 1


# A4 workaround：循环内 1-D 展平 store
@triton.jit
def a4(STATE, SSM, FINAL, T, NV, DK, DV, BLK_V: tl.constexpr, BLK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    v_off = tl.arange(0, BLK_V)
    k_off = tl.arange(0, BLK_K)
    m = (v_off[:, None] < DV) & (k_off[None, :] < DK)
    sm = tl.load(STATE + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :], mask=m, other=0.0)
    off2 = v_off[:, None] * DK + k_off[None, :]
    off1 = tl.reshape(off2, [BLK_V * BLK_K])
    ms1 = tl.reshape(m, [BLK_V * BLK_K])
    t = 0
    while t < T:
        s2 = tl.load(SSM + t).to(tl.int32)
        base = (s2 * NV + pid_h) * DV * DK
        val = tl.reshape(sm.to(FINAL.dtype.element_ty), [BLK_V * BLK_K])
        tl.store(FINAL + base + off1, val, mask=ms1)
        t += 1


# A5 workaround：循环内 2-D store，但地址用 1-D 展平后 reshape 回来（保持 2-D store，仅改掩码构造）
@triton.jit
def a5(STATE, SSM, FINAL, T, NV, DK, DV, BLK_V: tl.constexpr, BLK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    v_off = tl.arange(0, BLK_V)
    k_off = tl.arange(0, BLK_K)
    m = (v_off[:, None] < DV) & (k_off[None, :] < DK)
    sm = tl.load(STATE + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :], mask=m, other=0.0)
    ptr2 = FINAL + (pid_h * DV + v_off[:, None]) * DK + k_off[None, :]
    t = 0
    while t < T:
        s2 = tl.load(SSM + t).to(tl.int32)
        tl.store(ptr2 + (s2 * NV) * DV * DK, sm.to(FINAL.dtype.element_ty), mask=m)
        t += 1


# B 组：3D + 动态循环内 masked reduce（sinkhorn 原始形态）
@triton.jit
def rc(MIX, OUT, R, HM: tl.constexpr, BLK_HM: tl.constexpr, DYN_LOOP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * 4 + tl.arange(0, 4)
    jj = tl.arange(0, BLK_HM)
    jmask = jj < HM
    m3 = (rows[:, None, None] < R) & jmask[None, :, None] & jmask[None, None, :]
    a = tl.load(MIX + rows[:, None, None] * (BLK_HM * BLK_HM) + jj[None, :, None] * BLK_HM + jj[None, None, :],
                mask=m3, other=0.0)
    a = tl.where(m3, a, -1.0e30)
    mx = tl.max(a, axis=2)
    A = tl.exp(a - mx[:, :, None]) + 1e-6
    if DYN_LOOP:
        it = 0
        while it < 2:
            A = A / (tl.sum(tl.where(jmask[None, None, :], A, 0.0), axis=2)[:, :, None] + 1e-6)
            A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + 1e-6)
            it += 1
    else:
        for _ in tl.static_range(2):
            A = A / (tl.sum(tl.where(jmask[None, None, :], A, 0.0), axis=2)[:, :, None] + 1e-6)
            A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + 1e-6)
    tl.store(OUT + rows[:, None, None] * (HM * HM) + jj[None, :, None] * HM + jj[None, None, :], A, mask=m3)


print("=" * 88, flush=True)
print("第三轮：workaround 验证（循环内 store 展平为 1-D）", flush=True)
print("=" * 88, flush=True)

T, NV, DK, DV = 8, 2, 32, 64
ssm = torch.arange(T, dtype=torch.int32, device=dev)

for dt, tag in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
    st = (torch.randn(NV, DV, DK, device=dev) * 0.1).to(dt)
    fn = torch.zeros(8, NV, DV, DK, device=dev).to(dt)
    check(f"A1 循环内 2-D store（原始，应崩）{tag}",
          lambda st=st, fn=fn: a1[(NV,)](st, ssm, fn, T, NV, DK, DV, BLK_V=64, BLK_K=32))
    check(f"A4 循环内 1-D 展平 store（workaround）{tag}",
          lambda st=st, fn=fn: a4[(NV,)](st, ssm, fn, T, NV, DK, DV, BLK_V=64, BLK_K=32))
    check(f"A5 循环内 2-D store + 常量 base 偏移 {tag}",
          lambda st=st, fn=fn: a5[(NV,)](st, ssm, fn, T, NV, DK, DV, BLK_V=64, BLK_K=32))

    # 数值正确性：A4 结果必须等于 sm 反复写入（最后一次）
    st = (torch.randn(NV, DV, DK, device=dev) * 0.1).to(dt)
    fn = torch.zeros(8, NV, DV, DK, device=dev).to(dt)
    a4[(NV,)](st, ssm, fn, T, NV, DK, DV, BLK_V=64, BLK_K=32)
    torch.npu.synchronize()
    ref = st.expand(8, NV, DV, DK).clone()   # 每次写同样的 sm
    ok = torch.equal(fn.to(torch.float32), ref.to(torch.float32))
    print(f"  A4 数值正确性 {tag}: {'OK' if ok else 'WRONG'}", flush=True)

print("-" * 88, flush=True)
mix = (torch.randn(64, 4, 4, device=dev) * 0.5)
o4 = torch.zeros(64, 4, 4, device=dev)
check("B1 3D + 动态 while 内 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=True))
check("B2 3D + 静态展开 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=False))
print("done", flush=True)
