"""精简版：recurrent 结构的 fp32/fp16 store 二分（-u 无缓冲）。

A 组：循环内写回 2-D state（地址来自 tl.load 的 slot）—— 真实内核的关键结构
B 组：3D + 动态循环内 masked reduce —— hc_pre_sinkhorn 原始崩溃形态
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

dev = "npu:0"


def check(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  {name:56s} OK", flush=True)
        return "OK"
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:56s} {tag}: {msg[:60]}", flush=True)
        return tag


@triton.jit
def ra(STATE, SSM, OUT, FINAL, T, NV, DK, DV,
       BLK_V: tl.constexpr, BLK_K: tl.constexpr,
       STORE_IN_LOOP: tl.constexpr, SLOT_FROM_LOAD: tl.constexpr,
       OUT_F32: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    if SLOT_FROM_LOAD:
        slot = tl.load(SSM).to(tl.int32)
    else:
        slot = 0
    k_off = tl.arange(0, BLK_K)
    kmask = k_off < DK
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < DV
    m = vmask[:, None] & kmask[None, :]
    sm = tl.load(STATE + ((slot * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                 mask=m, other=0.0)
    t = 0
    while t < T:
        kk = tl.load(STATE + t * DK + k_off, mask=kmask, other=0.0)
        sm = sm + kk[None, :]
        if STORE_IN_LOOP:
            s2 = tl.load(SSM + t).to(tl.int32)
            if OUT_F32:
                tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                         sm, mask=m)
            else:
                tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                         sm.to(FINAL.dtype.element_ty), mask=m)
        t += 1
    tl.store(OUT + (pid_h * DV) + v_off, tl.sum(sm, axis=1).to(OUT.dtype.element_ty), mask=vmask)


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
print("A 组：循环内写回 2-D state（真实内核结构）", flush=True)
print("=" * 88, flush=True)

T, NV, DK, DV = 8, 2, 32, 64
ssm = torch.arange(T, dtype=torch.int32, device=dev)
o_b = torch.zeros(NV, DV, dtype=torch.bfloat16, device=dev)
o_f = torch.zeros(NV, DV, dtype=torch.float32, device=dev)

for dt, tag in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
    st = (torch.randn(8, NV, DV, DK, device=dev) * 0.1).to(dt)
    fn = torch.zeros(8, NV, DV, DK, device=dev).to(dt)
    ob = o_b if dt == torch.bfloat16 else o_f
    use_f32 = dt == torch.float32
    check(f"A1 loop-store(slot=load) {tag}",
          lambda st=st, fn=fn, ob=ob, u=use_f32: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                            BLK_V=64, BLK_K=32, STORE_IN_LOOP=True,
                                                            SLOT_FROM_LOAD=True, OUT_F32=u))
    check(f"A2 只在循环外 store {tag}",
          lambda st=st, fn=fn, ob=ob, u=use_f32: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                            BLK_V=64, BLK_K=32, STORE_IN_LOOP=False,
                                                            SLOT_FROM_LOAD=True, OUT_F32=u))
    check(f"A3 loop-store(slot=常量) {tag}",
          lambda st=st, fn=fn, ob=ob, u=use_f32: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                            BLK_V=64, BLK_K=32, STORE_IN_LOOP=True,
                                                            SLOT_FROM_LOAD=False, OUT_F32=u))

print("-" * 88, flush=True)
print("B 组：3D + reduce（hc_pre_sinkhorn 原始崩溃形态）", flush=True)
mix = (torch.randn(64, 4, 4, device=dev) * 0.5)
o4 = torch.zeros(64, 4, 4, device=dev)
check("B1 3D + 动态 while 内 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=True))
check("B2 3D + 静态展开 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=False))
print("done", flush=True)
