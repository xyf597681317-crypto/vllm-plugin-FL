"""验证 sparse_attention_score 的 Triton 实现 vs AscendC 语义（fp64 参考）。"""
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_ops.sparse_attention_score as _sasa_mod  # noqa: E402
from triton_ops.sparse_attention_score import sparse_attention_score  # noqa: E402

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))
    print(s, flush=True)


def ref_sasa(q, k, v, sel, bt, selnum, actq, actkv, BS, topK, scale, KVH, T, Nh, D):
    """按 arch35 内核 L112-289 的任务循环写的 fp64 参考。"""
    qq = q.detach().cpu().double()
    kk = k.detach().cpu().double()
    vv = v.detach().cpu().double()
    sl = sel.detach().cpu().to(torch.int64)
    btl = bt.detach().cpu().to(torch.int64)
    snl = selnum.detach().cpu().to(torch.int64) if selnum is not None else None
    aq = actq.detach().cpu().to(torch.int64).tolist()
    akv = actkv.detach().cpu().to(torch.int64).tolist()
    G = Nh // KVH

    out = torch.zeros(T, Nh, D, dtype=torch.float64)
    lse = torch.zeros(T, Nh, dtype=torch.float64)
    for t in range(T):
        b = 0
        acc = 0
        for bb in range(len(aq)):
            if t < acc + aq[bb]:
                b = bb
                break
            acc += aq[bb]
        qIn = t - acc
        kvSeq, qSeq = akv[b], aq[b]
        hist = kvSeq - qSeq
        lastLogical = (hist + qIn) // BS
        lastTile = (hist + qIn) % BS + 1
        for h in range(KVH):
            vtopk = int(snl[h, t]) if snl is not None else topK
            ks, vs = [], []
            for i in range(min(vtopk, topK)):
                lid = int(sl[h, t, i])
                if lid < 0:
                    continue
                phys = int(btl[b, lid])
                tile = BS if lid != lastLogical else lastTile
                ks.append(kk[phys, :tile, h, :])
                vs.append(vv[phys, :tile, h, :])
            Qg = qq[t, h * G:(h + 1) * G, :]
            if len(ks) == 0:
                out[t, h * G:(h + 1) * G, :] = 0.0
                lse[t, h * G:(h + 1) * G] = 0.0
                continue
            K = torch.cat(ks, 0)
            V = torch.cat(vs, 0)
            S = (Qg @ K.t()) * scale
            m = S.max(dim=-1).values
            e = torch.exp(S - m.unsqueeze(-1))
            l = e.sum(dim=-1)
            P = e / l.unsqueeze(-1)
            out[t, h * G:(h + 1) * G, :] = P @ V
            lse[t, h * G:(h + 1) * G] = m + torch.log(l)
    return out, lse


def run_case(name, T, Nh, KVH, D, BS, topK, B, MQB, actq, actkv, dt, use_selnum,
             n_neg=2, seed=0):
    torch.manual_seed(seed)
    dev = DEV
    nb = B * MQB
    q = torch.randn(T, Nh, D, dtype=torch.float32, device=dev).to(dt)
    k = torch.randn(nb, BS, KVH, D, dtype=torch.float32, device=dev).to(dt)
    v = (torch.randn(nb, BS, KVH, D, dtype=torch.float32, device=dev) * 0.7).to(dt)
    sel = torch.randint(0, MQB, (KVH, T, topK), dtype=torch.int32, device=dev)
    # 制造若干无效项（负值）
    for h in range(KVH):
        for t in range(T):
            for i in range(n_neg):
                sel[h, t, (h + t + i) % topK] = -1
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(B, MQB)
    if use_selnum:
        sn = torch.full((KVH, T), topK, dtype=torch.int32, device=dev)
        sn[:, 0] = 0                      # 这一列完全没有块 → 输出 0
        sn[:, 1] = 1
    else:
        sn = None
    aq = torch.tensor(actq, dtype=torch.int32, device=dev)
    akv = torch.tensor(actkv, dtype=torch.int32, device=dev)

    scale = 1.0 / math.sqrt(D)
    out, lse = sparse_attention_score(q, k, v, sel, bt, sn, aq, akv,
                                      num_key_value_heads=KVH, scale_value=scale,
                                      block_size=BS, top_k=topK)
    torch.npu.synchronize()

    rout, rlse = ref_sasa(q, k, v, sel, bt, sn, aq, akv, BS, topK, scale, KVH, T, Nh, D)
    floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dt]
    eo = (out.detach().cpu().double() - rout).abs().max().item()
    el = (lse.detach().cpu().double().reshape(T, Nh) - rlse).abs().max().item()
    ok = eo <= floor and el <= 2e-6
    emit(f"  {name:52s} out_err={eo:.3e}(<= {floor:.0e}) "
         f"lse_err={el:.3e}(<= 2e-06)  {'PASS' if ok else 'FAIL'}")
    return ok


def ref_sasa_fp8(q, k, v, sel, bt, selnum, actq, actkv, qsc, ksc, vsc,
                 BS, topK, scale, KVH, T, Nh, D, p_fp8):
    """fp8 FullQuant 语义参考（arch35 full_quant 内核 L146-344）。"""
    qq = q.detach().cpu().double()
    kk = k.detach().cpu().double()
    vv = v.detach().cpu().double()
    sl = sel.detach().cpu().to(torch.int64)
    btl = bt.detach().cpu().to(torch.int64)
    snl = selnum.detach().cpu().to(torch.int64) if selnum is not None else None
    aq = actq.detach().cpu().to(torch.int64).tolist()
    akv = actkv.detach().cpu().to(torch.int64).tolist()
    maxQ = sel.shape[1]
    MQB = bt.shape[1]
    maxqblk = (maxQ + BS - 1) // BS
    q1 = qsc.detach().cpu().double() if qsc is not None else None
    k1 = ksc.detach().cpu().double() if ksc is not None else None
    v1 = vsc.detach().cpu().double() if vsc is not None else None
    G = Nh // KVH

    out = torch.zeros(T, Nh, D, dtype=torch.float64)
    for t in range(T):
        b = 0
        acc = 0
        for bb in range(len(aq)):
            if t < acc + aq[bb]:
                b = bb
                break
            acc += aq[bb]
        qIn = t - acc
        kvSeq, qSeq = akv[b], aq[b]
        hist = kvSeq - qSeq
        lastLogical = (hist + qIn) // BS
        lastTile = (hist + qIn) % BS + 1
        lqb = (hist + qIn) // BS
        for h in range(KVH):
            vtopk = int(snl[h, t]) if snl is not None else topK
            vtopk = min(max(vtopk, 0), topK)
            O = torch.zeros(G, D, dtype=torch.float64)
            for gi in range(G):
                head = h * G + gi
                sq = q1[b, head, lqb] if q1 is not None else 1.0
                ks, vs = [], []
                for i in range(vtopk):
                    lid = int(sl[h, t, i])
                    if lid < 0:
                        continue
                    phys = int(btl[b, lid])
                    tile = BS if lid != lastLogical else lastTile
                    sk = k1[b, h, lid] if k1 is not None else 1.0
                    sv = v1[b, h, lid] if v1 is not None else 1.0
                    ks.append(kk[phys, :tile, h, :] * sk)
                    vs.append(vv[phys, :tile, h, :] * sv)
                if len(ks) == 0:
                    continue
                K = torch.cat(ks, 0)
                V = torch.cat(vs, 0)
                S = (qq[t, head:head + 1, :] @ K.t()) * sq * scale
                m = S.max(dim=-1, keepdim=True).values
                e = torch.exp(S - m)
                P = e / e.sum(dim=-1, keepdim=True)
                if p_fp8:
                    P = P.to(torch.float8_e4m3fn).double()
                O[gi] = (P @ V)[0]
            out[t, h * G:(h + 1) * G, :] = O
    return out


def mk_fp8(shape, scale, dev, seed):
    """构造 fp8_e4m3fn 输入。

    ⚠️ 本机限制：Ascend 910C 的 BiShengHIR 直接拒绝 fp8 类型
    （`'hivm.hir.vbrc' op Current hardware doesn't support fp8 type`），
    因此**无法在本机跑通「真实 fp8 张量」的存储路径**。
    这里退化为「已按 e4m3 量化、但以 bf16 存放的张量」——e4m3 ⊂ bf16，
    二者数值逐位一致，配合 fp8_mode=True 走 fp8 FullQuant 通路，
    验证的是**反量化公式与后续计算**（assumed 语义），而不是 fp8 存储本身。
    返回 (tensor, used_real_fp8)。
    """
    torch.manual_seed(seed)
    x = torch.randn(*shape) * scale
    y = x.to(torch.float8_e4m3fn).to(torch.float32).to(torch.bfloat16).to(dev)
    return y, False


def run_case_fp8(name, T, Nh, KVH, D, BS, topK, B, MQB, actq, actkv, use_scales, use_selnum,
                 n_neg=2, seed=0):
    dev = DEV
    nb = B * MQB
    maxQ = T
    # qDequantScale 的 dim2 必须覆盖内核用的「绝对位置块号」上界
    # (historyLen + qTokenInBatch)//blockSize + 1（full_quant L210-213）
    maxqblk = max((max(actkv[b] - actq[b] + actq[b] - 1, 0)) // BS for b in range(B)) + 1
    q, _ = mk_fp8((T, Nh, D), 0.5, dev, seed + 100)
    k, _ = mk_fp8((nb, BS, KVH, D), 0.5, dev, seed + 200)
    v, _ = mk_fp8((nb, BS, KVH, D), 0.5, dev, seed + 300)
    real_fp8 = False
    torch.manual_seed(seed)
    sel = torch.randint(0, MQB, (KVH, T, topK), dtype=torch.int32, device=dev)
    for h in range(KVH):
        for t in range(T):
            for i in range(n_neg):
                sel[h, t, (h + t + i) % topK] = -1
    bt = torch.arange(nb, dtype=torch.int32, device=dev).reshape(B, MQB)
    if use_selnum:
        sn = torch.full((KVH, T), topK, dtype=torch.int32, device=dev)
        sn[:, 0] = 0
        sn[:, 1] = 1
    else:
        sn = None
    aq = torch.tensor(actq, dtype=torch.int32, device=dev)
    akv = torch.tensor(actkv, dtype=torch.int32, device=dev)
    if use_scales:
        qsc = (torch.rand(B, Nh, maxqblk, dtype=torch.float32, device=dev) * 0.5 + 0.5)
        ksc = (torch.rand(B, KVH, MQB, dtype=torch.float32, device=dev) * 0.5 + 0.5)
        vsc = (torch.rand(B, KVH, MQB, dtype=torch.float32, device=dev) * 0.5 + 0.5)
    else:
        qsc = ksc = vsc = None

    scale = 1.0 / math.sqrt(D)
    out, lse = sparse_attention_score(q, k, v, sel, bt, sn, aq, akv, qsc, ksc, vsc,
                                      num_key_value_heads=KVH, scale_value=scale,
                                      block_size=BS, top_k=topK,
                                      fp8_mode=(None if real_fp8 else True))
    torch.npu.synchronize()
    used_p_fp8 = bool(_sasa_mod._P_FP8_OK)
    rout = ref_sasa_fp8(q, k, v, sel, bt, sn, aq, akv, qsc, ksc, vsc,
                        BS, topK, scale, KVH, T, Nh, D, used_p_fp8)
    floor = 1e-3  # fp8 输入的输出 dtype 是 fp16
    eo = (out.detach().cpu().double() - rout).abs().max().item()
    ok = eo <= floor
    emit(f"  {name:52s} out_err={eo:.3e}(<= {floor:.0e}) "
         f"P@fp8={used_p_fp8} 真fp8张量={real_fp8}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_case_callsite(name, dt, fp8):
    """完全模仿真实调用点：只传 AscendC 原名（驼峰）参数，不传任何 snake_case。"""
    torch.manual_seed(21)
    dev = DEV
    T, Nh, KVH, D, BS, topK = 8, 16, 4, 128, 128, 4
    B, MQB = 2, 8
    actq, actkv = [4, 4], [200, 200]
    nb = B * MQB
    maxqblk = max((actkv[b] - 1) // BS for b in range(B)) + 1
    if fp8:
        q, _ = mk_fp8((T, Nh, D), 0.5, dev, 101)
        k, _ = mk_fp8((nb, BS, KVH, D), 0.5, dev, 102)
        v, _ = mk_fp8((nb, BS, KVH, D), 0.5, dev, 103)
        real_fp8 = False
    else:
        real_fp8 = True
        q = torch.randn(T, Nh, D, dtype=torch.float32, device=dev).to(dt)
        k = torch.randn(nb, BS, KVH, D, dtype=torch.float32, device=dev).to(dt)
        v = (torch.randn(nb, BS, KVH, D, dtype=torch.float32, device=dev) * 0.7).to(dt)
    selectIdx = torch.randint(0, MQB, (KVH, T, topK), dtype=torch.int32, device=dev)
    selectIdx[:, :, -1] = -1
    blockTable = torch.arange(nb, dtype=torch.int32, device=dev).reshape(B, MQB)
    selectNumIdx = torch.full((KVH, T), topK, dtype=torch.int32, device=dev)
    selectNumIdx[:, 0] = 0
    aq = torch.tensor(actq, dtype=torch.int32, device=dev)
    akv = torch.tensor(actkv, dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    if fp8:
        qs = torch.rand(B, Nh, maxqblk, dtype=torch.float32, device=dev) * 0.5 + 0.5
        ks = torch.rand(B, KVH, MQB, dtype=torch.float32, device=dev) * 0.5 + 0.5
        vs = torch.rand(B, KVH, MQB, dtype=torch.float32, device=dev) * 0.5 + 0.5
    else:
        qs = ks = vs = None

    out, lse = sparse_attention_score(          # noqa: E501
        q, k, v,
        selectIdx=selectIdx, blockTable=blockTable, selectNumIdx=selectNumIdx,
        actualSeqLengths=aq, actualSeqLengthsKv=akv,
        qDequantScale=qs, kDequantScale=ks, vDequantScale=vs,
        numKeyValueHeads=KVH, scaleValue=scale, blockSize=BS, topK=topK, innerPrecise=4,
        **({} if (not fp8 or real_fp8) else {"fp8_mode": True}))
    torch.npu.synchronize()

    sel_np = selectIdx.detach().cpu().clone()
    sel_np[sel_np < 0] = -1
    if fp8:
        rout = ref_sasa_fp8(q, k, v, sel_np, blockTable, selectNumIdx, aq, akv, qs, ks, vs,
                            BS, topK, scale, KVH, T, Nh, D, bool(_sasa_mod._P_FP8_OK))
        floor = 1e-3
    else:
        rout, _ = ref_sasa(q, k, v, sel_np, blockTable, selectNumIdx, aq, akv,
                           BS, topK, scale, KVH, T, Nh, D)
        floor = {torch.bfloat16: 4e-3, torch.float16: 1e-3}[dt]
    eo = (out.detach().cpu().double() - rout).abs().max().item()
    ok = eo <= floor
    emit(f"  {name:52s} out_err={eo:.3e}(<= {floor:.0e})  {'PASS' if ok else 'FAIL'}")
    return ok


emit("=" * 108)
emit("sparse_attention_score: Triton vs AscendC 语义（fp64 参考）")
emit("=" * 108)
allok = True
allok &= run_case("T=8  Nh=16 KVH=4 D=128 BS=128 topK=4 bf16 无 selnum",
                  8, 16, 4, 128, 128, 4, 2, 8, [4, 4], [200, 200], torch.bfloat16, False)
allok &= run_case("T=8  Nh=16 KVH=4 D=128 BS=128 topK=4 bf16 带 selnum",
                  8, 16, 4, 128, 128, 4, 2, 8, [4, 4], [200, 200], torch.bfloat16, True)
allok &= run_case("T=8  Nh=16 KVH=4 D=128 BS=128 topK=4 fp16 带 selnum",
                  8, 16, 4, 128, 128, 4, 2, 8, [4, 4], [200, 200], torch.float16, True)
allok &= run_case("T=32 Nh=128 KVH=8 D=128 BS=128 topK=8 bf16 (GQA16)",
                  32, 128, 8, 128, 128, 8, 2, 16, [16, 16], [300, 300], torch.bfloat16, True,
                  n_neg=3, seed=7)
allok &= run_case("T=6  Nh=16 KVH=16 D=128 BS=128 topK=2 bf16 (MHA G=1)",
                  6, 16, 16, 128, 128, 2, 2, 8, [3, 3], [100, 100], torch.bfloat16, True,
                  n_neg=1, seed=11)

emit()
emit("--- fp8_e4m3fn FullQuant（反量化公式：full_quant L210-213 / L251-254 / L315-318）---")
allok &= run_case_fp8("fp8 T=8  Nh=16 KVH=4 D=128 BS=128 topK=4 带 dequant scale+selnum",
                      8, 16, 4, 128, 128, 4, 2, 8, [4, 4], [200, 200], True, True)
allok &= run_case_fp8("fp8 T=8  Nh=16 KVH=4 D=128 BS=128 topK=4 带 dequant scale 无 selnum",
                      8, 16, 4, 128, 128, 4, 2, 8, [4, 4], [200, 200], True, False)
allok &= run_case_fp8("fp8 T=6  Nh=16 KVH=8 D=128 BS=128 topK=3 无 dequant scale (默认 1.0)",
                      6, 16, 8, 128, 128, 3, 2, 8, [3, 3], [150, 150], False, True,
                      n_neg=1, seed=5)
allok &= run_case_fp8("fp8 T=4  Nh=32 KVH=4 D=128 BS=128 topK=2 GQA8 带 scale",
                      4, 32, 4, 128, 128, 2, 2, 8, [2, 2], [64, 64], True, True, seed=9)

emit()
emit("--- 调用点风格（AscendC 原名参数）---")
allok &= run_case_callsite("callsite bf16 参数全用驼峰 (selectIdx/blockTable/...)", torch.bfloat16, False)
allok &= run_case_callsite("callsite fp8 参数全用驼峰 (+qDequantScale/...)", torch.float8_e4m3fn, True)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
try:
    open("/root/dsh_work/sparse_attention_score_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
