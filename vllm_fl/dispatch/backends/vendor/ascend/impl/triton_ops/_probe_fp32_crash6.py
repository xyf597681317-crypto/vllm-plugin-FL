"""第五轮：从真实 recurrent 内核（fp32 state，**确定性** 0.5s 崩溃）逐项删减，定位触发点。

内核与 _analysis/triton_ops/recurrent_gated_delta_rule.py 的 _rgdr_kernel 同构，
用 FEAT_* 开关逐项关闭，看哪一项关掉后不再崩。
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
        print(f"  {name:60s} OK", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:60s} {tag}: {msg[:58]}", flush=True)


@triton.jit
def rk(Q, KEY, VAL, BETA, GAMA, GAMAK, STATE, CU, SSM, OUT, FINAL, NACC,
       T, NK, NV, DK, DV, SCALE,
       USE_G: tl.constexpr, USE_GK: tl.constexpr,
       FEAT_LOAD_QKV: tl.constexpr, FEAT_SUM_K: tl.constexpr, FEAT_OUTER: tl.constexpr,
       FEAT_OUT_STORE: tl.constexpr, FEAT_FINAL_STORE: tl.constexpr,
       FEAT_FINAL_AFTER: tl.constexpr,
       BLK_K: tl.constexpr, BLK_V: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = tl.program_id(2)
    seq0 = tl.load(CU + pid_b).to(tl.int32)
    seq1 = tl.load(CU + pid_b + 1).to(tl.int32)
    if seq1 <= seq0:
        return
    slot = tl.load(SSM + seq0).to(tl.int32)
    k_off = tl.arange(0, BLK_K)
    kmask = k_off < DK
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < DV
    hk = pid_h // (NV // NK)
    sm = tl.load(STATE + ((slot * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                 mask=vmask[:, None] & kmask[None, :], other=0.0)
    t = seq0
    while t < seq1:
        if FEAT_LOAD_QKV:
            q = tl.load(Q + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32) * SCALE
            kk = tl.load(KEY + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32)
            vv = tl.load(VAL + (t * NV + pid_h) * DV + v_off, mask=vmask, other=0.0).to(tl.float32)
            bb = tl.load(BETA + t * NV + pid_h).to(tl.float32)
        else:
            q = tl.zeros((BLK_K,), dtype=tl.float32)
            kk = tl.zeros((BLK_K,), dtype=tl.float32)
            vv = tl.zeros((BLK_V,), dtype=tl.float32)
            bb = 1.0
        if USE_G:
            gm = tl.exp(tl.load(GAMA + t * NV + pid_h).to(tl.float32))
            sm = sm * gm
        if USE_GK:
            gk = tl.exp(tl.load(GAMAK + (t * NV + pid_h) * DK + k_off, mask=kmask, other=0.0).to(tl.float32))
            sm = sm * gk[None, :]
        if FEAT_SUM_K:
            d = vv - tl.sum(sm * kk[None, :], axis=1)
        else:
            d = vv
        d = d * bb
        if FEAT_OUTER:
            sm = sm + d[:, None] * kk[None, :]
        if FEAT_OUT_STORE:
            o = tl.sum(sm * q[None, :], axis=1)
            tl.store(OUT + (t * NV + pid_h) * DV + v_off, o.to(OUT.dtype.element_ty), mask=vmask)
        if FEAT_FINAL_STORE:
            s2 = tl.load(SSM + t).to(tl.int32)
            tl.store(FINAL + ((s2 * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :],
                     sm.to(FINAL.dtype.element_ty), mask=vmask[:, None] & kmask[None, :])
        t += 1
    if FEAT_FINAL_AFTER:
        tl.store(FINAL + ((pid_h * DV + v_off[:, None])) * DK + k_off[None, :],
                 sm.to(FINAL.dtype.element_ty), mask=vmask[:, None] & kmask[None, :])


T, NK, NV, DK, DV, B = 32, 2, 2, 32, 32, 2
torch.manual_seed(0)
U = lambda *s, lo=0.2: (torch.rand(*s, device=dev) * 2 - 1) * lo
q = U(T, NK, DK).to(torch.bfloat16)
k = U(T, NK, DK).to(torch.bfloat16)
v = U(T, NV, DV).to(torch.bfloat16)
b = U(T, NV).to(torch.bfloat16)
g = U(T, NV, lo=0.1) - 0.05
gk = U(T, NV, DK, lo=0.1) - 0.05
cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=dev)
ssm = torch.arange(T, dtype=torch.int32, device=dev) * NV
nacc = cu
STATE_F = U(64, NV, DV, DK).to(torch.float32)
STATE_B = STATE_F.to(torch.bfloat16)
OUT_B = torch.zeros(T, NV, DV, dtype=torch.bfloat16, device=dev)
OUT_F = torch.zeros(T, NV, DV, dtype=torch.float32, device=dev)
FIN_F = torch.zeros(64, NV, DV, DK, dtype=torch.float32, device=dev)
FIN_B = torch.zeros(64, NV, DV, DK, dtype=torch.bfloat16, device=dev)


def run(state, out, fin, **feat):
    kw = dict(USE_G=True, USE_GK=True, FEAT_LOAD_QKV=True, FEAT_SUM_K=True, FEAT_OUTER=True,
              FEAT_OUT_STORE=True, FEAT_FINAL_STORE=True, FEAT_FINAL_AFTER=False,
              BLK_K=32, BLK_V=32)
    kw.update(feat)
    rk[(NV, 1, B)](q, k, v, b, g, gk, state, cu, ssm, out, fin, nacc,
                   T, NK, NV, DK, DV, 1.0, **kw)


print("=" * 96, flush=True)
print("第五轮：真实 recurrent 内核逐项删减（fp32 state = 已知崩）", flush=True)
print("=" * 96, flush=True)
check("R0 全功能 fp32 state（基线，应崩）", lambda: run(STATE_F, OUT_F, FIN_F))
check("R1 全功能 bf16 state（对照，应 OK）", lambda: run(STATE_B, OUT_B, FIN_B))
check("R2 去掉循环内 FINAL 2-D store", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_FINAL_STORE=False))
check("R3 去掉循环内 OUT 1-D store", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_OUT_STORE=False))
check("R4 去掉外积累加 FEAT_OUTER", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_OUTER=False))
check("R5 去掉 tl.sum(sm*kk) 归约", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_SUM_K=False))
check("R6 去掉 g/gk 门控", lambda: run(STATE_F, OUT_F, FIN_F, USE_G=False, USE_GK=False))
check("R7 只留 FINAL store（无 QKV load）", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_LOAD_QKV=False))
check("R8 FINAL 改到循环外 store", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_FINAL_STORE=False, FEAT_FINAL_AFTER=True))
check("R9 去掉 OUT store + 去 FINAL 循环 store", lambda: run(STATE_F, OUT_F, FIN_F, FEAT_OUT_STORE=False,
                                                              FEAT_FINAL_STORE=False, FEAT_FINAL_AFTER=True))
print("done", flush=True)
