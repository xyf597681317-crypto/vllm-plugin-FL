"""第二轮：定位真实结构（循环内 fp32 store / 3D 规约 / 初始状态加载）的崩溃触发点。"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

dev = "npu:0"


def check(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  {name:58s} OK")
        return "OK"
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:58s} {tag}: {msg[:60]}")
        return tag


# ============ A 组：recurrent 结构（循环内 store + 由 load 出来的 slot 决定地址） ============
@triton.jit
def ra(STATE, SSM, OUT, FINAL, T, NV, DK, DV,
       BLK_V: tl.constexpr, BLK_K: tl.constexpr, STORE_IN_LOOP: tl.constexpr, SLOT_FROM_LOAD: tl.constexpr):
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
            tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                     sm.to(FINAL.dtype.element_ty), mask=m)
        t += 1
    tl.store(OUT + (pid_h * DV) + v_off, tl.sum(sm, axis=1).to(OUT.dtype.element_ty), mask=vmask)


# ============ B 组：chunk 结构（fp32 初始状态 tile 载入 + final tile 落盘） ============
@triton.jit
def rb(INIT, OUT, HOUT_F32, K_V: tl.constexpr, BLK: tl.constexpr, STORE_F32: tl.constexpr):
    k_off = tl.arange(0, BLK)
    v_off = tl.arange(0, BLK)
    m = (k_off[:, None] < K_V) & (v_off[None, :] < K_V)
    H = tl.load(INIT + k_off[:, None] * BLK + v_off[None, :], mask=m, other=0.0).to(tl.float32)
    H = H * 1.5 + 1.0
    tl.store(OUT + k_off[:, None] * BLK + v_off[None, :], H.to(OUT.dtype.element_ty), mask=m)
    if STORE_F32:
        tl.store(HOUT_F32 + k_off[:, None] * BLK + v_off[None, :], H, mask=m)


# ============ C 组：hc_pre_sinkhorn 原始崩溃形态（3D + 动态循环内 reduce） ============
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


print("=" * 90)
print("第二轮：结构级触发点定位")
print("=" * 90)

T, NV, DK, DV = 8, 2, 32, 64
ssm = torch.arange(T, dtype=torch.int32, device=dev)
out_b = torch.zeros(NV, DV, dtype=torch.bfloat16, device=dev)
out_f = torch.zeros(NV, DV, dtype=torch.float32, device=dev)

for dt, tag in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
    st = (torch.randn(8, NV, DV, DK, device=dev) * 0.1).to(dt)
    fn = torch.zeros(8, NV, DV, DK, device=dev).to(dt)
    ob = out_b if dt == torch.bfloat16 else out_f
    check(f"A1 recurrent结构 loop-store slot=load {tag}",
          lambda st=st, fn=fn, ob=ob: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                  BLK_V=64, BLK_K=32, STORE_IN_LOOP=True,
                                                  SLOT_FROM_LOAD=True))
    check(f"A2 recurrent结构 只在循环外 store {tag}",
          lambda st=st, fn=fn, ob=ob: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                  BLK_V=64, BLK_K=32, STORE_IN_LOOP=False,
                                                  SLOT_FROM_LOAD=True))
    check(f"A3 地址用常量 slot（不 load）{tag}",
          lambda st=st, fn=fn, ob=ob: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                  BLK_V=64, BLK_K=32, STORE_IN_LOOP=True,
                                                  SLOT_FROM_LOAD=False))
    # 变体：中间转 bf16 再转回
    check(f"A4 loop-store {tag}（内部 bf16 往返）",
          lambda st=st, fn=fn, ob=ob: ra[(NV, 1)](st, ssm, ob, fn, T, NV, DK, DV,
                                                  BLK_V=64, BLK_K=32, STORE_IN_LOOP=True,
                                                  SLOT_FROM_LOAD=True))
    # 一次 store 到 bf16 中转缓冲，再用第二个 kernel 转 fp32
    tmp = torch.zeros(8, NV, DV, DK, dtype=torch.bfloat16, device=dev)
    check(f"A5 拆分：先写 bf16 中转再转 fp32 {tag}",
          lambda st=st, tmp=tmp, ob=ob, fn=fn: (
              ra[(NV, 1)](st, ssm, ob, tmp, T, NV, DK, DV, BLK_V=64, BLK_K=32,
                          STORE_IN_LOOP=True, SLOT_FROM_LOAD=True),
              ra[(NV, 1)](tmp.to(torch.float32), ssm, ob,
                          torch.zeros(8, NV, DV, DK, device=dev), T, NV, DK, DV,
                          BLK_V=64, BLK_K=32, STORE_IN_LOOP=True, SLOT_FROM_LOAD=True)))

print("-" * 90)
K_V, BLK = 64, 64
init_f = (torch.randn(K_V, K_V, device=dev) * 0.1)
init_b = init_f.to(torch.bfloat16)
o_b = torch.zeros(K_V, K_V, dtype=torch.bfloat16, device=dev)
o_f = torch.zeros(K_V, K_V, dtype=torch.float32, device=dev)
f_f = torch.zeros(K_V, K_V, dtype=torch.float32, device=dev)
check("B1 chunk结构 fp32 init -> bf16 out + fp32 final",
      lambda: rb[(1,)](init_f, o_b, f_f, K_V, BLK, STORE_F32=True))
check("B2 chunk结构 bf16 init -> bf16 out + fp32 final",
      lambda: rb[(1,)](init_b, o_b, f_f, K_V, BLK, STORE_F32=True))
check("B3 chunk结构 bf16 init -> bf16 out（无 fp32 final）",
      lambda: rb[(1,)](init_b, o_b, f_f, K_V, BLK, STORE_F32=False))
check("B4 chunk结构 fp32 init -> fp32 out（无 fp32 final）",
      lambda: rb[(1,)](init_f, o_f, f_f, K_V, BLK, STORE_F32=False))
check("B5 chunk结构 fp32 init -> fp32 out + fp32 final",
      lambda: rb[(1,)](init_f, o_f, f_f, K_V, BLK, STORE_F32=True))

print("-" * 90)
mix = (torch.randn(64, 4, 4, device=dev) * 0.5)
o4 = torch.zeros(64, 4, 4, device=dev)
check("C1 3D + 动态 while 循环内 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=True))
check("C2 3D + 静态展开 masked reduce", lambda: rc[(16,)](mix, o4, 64, HM=4, BLK_HM=4, DYN_LOOP=False))
print("done")
