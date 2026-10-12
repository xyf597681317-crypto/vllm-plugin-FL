"""验证 causal_conv1d 的 Triton 实现 vs AscendC 语义（fp64 参考）。

两部分：
  A) fn 路径（run_mode=0）多 shape/dtype/状态语义
  B) **调用点风格**（npu_causal_conv1d_custom 的参数名 + run_mode=1 的 3D/2D/投机解码窗口）
"""
import importlib
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PKG = os.environ.get("DSH_OPS_PKG", "triton_ops")
causal_conv1d_fn = importlib.import_module(_PKG + ".causal_conv1d").causal_conv1d_fn

DEV = os.environ.get("DSH_NPU", "npu:2")
torch.npu.set_device(DEV)
L = []


def emit(s=""):
    L.append(str(s))


def ref_impl(x, w, bias, states, qsl, cidx, imode, activation, pad_slot,
             mode, seq_len=1, nat=None):
    """严格按 AscendC 语义（fp64）：mode 0=varlen / 1=batch3d / 2=decode2d。"""
    xd = x.detach().cpu().double()
    wd = w.detach().cpu().double()
    bd = bias.detach().cpu().double() if bias is not None else None
    st = states.detach().cpu().double().clone()
    width = wd.shape[0]
    dim = xd.shape[-1]
    if imode is not None:
        imode = imode.detach().cpu() if torch.is_tensor(imode) else imode
    if nat is not None:
        nat = nat.detach().cpu().tolist() if torch.is_tensor(nat) else list(nat)

    if mode == 1:
        batch, seq_len = xd.shape[0], xd.shape[1]
        xflat = xd.reshape(batch * seq_len, dim)
    else:
        xflat = xd
        batch = (qsl.numel() - 1) if mode == 0 else xd.shape[0]
    y = torch.zeros_like(xflat)
    for b in range(batch):
        if mode == 0:
            s, e = int(qsl[b]), int(qsl[b + 1])
            start, length = s, e - s
        elif mode == 1:
            start, length = b * seq_len, seq_len
        else:
            start, length = b, 1
        ci = int(cidx[b]) if cidx is not None else b
        hi = bool(imode[b]) if imode is not None else False
        sto = 0
        if nat is not None:
            sto = min(max(int(nat[b]) - 1, 0), st.shape[1] - (width - 1))
        hist = (st[ci, sto:sto + width - 1].clone() if hi
                else torch.zeros(width - 1, dim, dtype=torch.float64))
        seq = xflat[start:start + length]
        ext = torch.cat([hist, seq], dim=0)
        for t in range(length):
            acc = (bd.clone() if bd is not None else torch.zeros(dim, dtype=torch.float64))
            for k in range(width):
                acc = acc + wd[k] * ext[t + k]
            if activation:
                acc = acc / (1.0 + torch.exp(-acc))
            y[start + t] = acc
        if ci != pad_slot and length > 0:
            st[ci, 0:width - 1] = ext[length:length + width - 1]
    return (y.reshape(xd.shape) if mode == 1 else y), st


def check(name, y, ry, st, rst, dtype):
    ulp = (2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -11) * max(float(ry.abs().max()), 1e-12)
    ey = float((y.detach().cpu().double() - ry.to(dtype).double()).abs().max())
    es = float((st.detach().cpu().double() - rst.to(dtype).double()).abs().max())
    ok = ey <= ulp and es <= ulp
    emit(f"  {name:46s} y_err={ey:.2e}(<= {ulp:.2e}) state_err={es:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_case(name, lens, dim, width, dtype, cache_idx=None, init_mode=None,
             activation=1, pad_slot=-1, num_cache=None, bias_flag=True):
    torch.manual_seed(3)
    n_batch = len(lens)
    T = sum(lens)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                       dtype=torch.int32, device=DEV)
    x = (torch.randn(T, dim, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
    w = (torch.randn(width, dim, dtype=torch.float32, device=DEV) * 0.7).to(dtype)
    b = (torch.randn(dim, dtype=torch.float32, device=DEV) * 0.3).to(dtype) if bias_flag else None
    nc = num_cache or (max(cache_idx) + 1 if cache_idx is not None else n_batch)
    st = (torch.randn(nc, width - 1, dim, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
    st0 = st.clone()
    ci = torch.tensor(cache_idx, dtype=torch.int32, device=DEV) if cache_idx is not None else None
    im = torch.tensor(init_mode, dtype=torch.bool, device=DEV) if init_mode is not None else None
    y = causal_conv1d_fn(x, w, b, st, qsl, ci, im, activation, pad_slot, 0)
    torch.npu.synchronize()
    ry, rst = ref_impl(x, w, b, st0, qsl.cpu(), ci.cpu() if ci is not None else None,
                       im.cpu() if im is not None else None, activation, pad_slot, 0)
    return check(name, y, ry, st, rst, dtype)


def run_callsite(name, mode, dim, width, dtype, n_batch=4, seq_len=1, lens=None,
                 cache_idx=None, init_mode=None, activation=1, nat=None, pad_slot=-1,
                 bias_on=True, num_cache=None):
    """完全按调用点参数名（*_opt / conv_state / activation_mode）调用。"""
    torch.manual_seed(21)
    if mode == 1:
        x = (torch.randn(n_batch, seq_len, dim, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
        qsl, qsl_dev = None, None
        total = n_batch * seq_len
    elif mode == 0:
        lens = lens or [3] * n_batch
        total = sum(lens)
        qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()), dtype=torch.int32)
        qsl_dev = qsl.to(DEV)
        x = (torch.randn(total, dim, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
    else:
        x = (torch.randn(n_batch, dim, dtype=torch.float32, device=DEV) * 1.5).to(dtype)
        qsl, qsl_dev = None, None
        total = n_batch
    w = (torch.randn(width, dim, dtype=torch.float32, device=DEV) * 0.7).to(dtype)
    b = (torch.randn(dim, dtype=torch.float32, device=DEV) * 0.3).to(dtype) if bias_on else None
    nc = num_cache or (max(cache_idx) + 1 if cache_idx is not None else n_batch)
    st = (torch.randn(nc, max(width - 1, 3), dim, dtype=torch.float32, device=DEV) * 0.5).to(dtype)
    st0 = st.clone()
    ci = torch.tensor(cache_idx, dtype=torch.int32, device=DEV) if cache_idx is not None else None
    im = torch.tensor(init_mode, dtype=torch.bool, device=DEV) if init_mode is not None else None
    nt = torch.tensor(nat, dtype=torch.int32, device=DEV) if nat is not None else None

    # ← 调用点风格：npu_causal_conv1d_custom(output, x, weight, conv_state, bias_opt,
    #   query_start_loc_opt, cache_indices_opt, initial_state_mode_opt,
    #   num_accepted_tokens_opt, activation_mode, pad_slot_id, run_mode)
    y = causal_conv1d_fn(
        x=x, weight=w, conv_state=st, bias_opt=b,
        query_start_loc_opt=qsl_dev, cache_indices_opt=ci,
        initial_state_mode_opt=im, num_accepted_tokens_opt=nt,
        activation_mode=activation, pad_slot_id=pad_slot, run_mode=1,
    )
    torch.npu.synchronize()
    ry, rst = ref_impl(x, w, b, st0, qsl, ci.cpu() if ci is not None else None,
                       im.cpu() if im is not None else None, activation, pad_slot,
                       mode, seq_len, nat)
    return check(name, y, ry, st, rst, dtype)


emit("=" * 108)
emit("causal_conv1d: Triton vs AscendC 语义（fp64 参考；含 convStates 状态更新对拍）")
emit("=" * 108)
emit("A) fn 路径（run_mode=0）")
allok = True
allok &= run_case("w=4 dim=64 bf16 silu 3:5,8,3", [5, 8, 3], 64, 4, torch.bfloat16)
allok &= run_case("w=4 dim=64 fp16 silu 3:8,1,4", [8, 1, 4], 64, 4, torch.float16)
allok &= run_case("w=4 dim=100 fp16 非2幂 silu", [6, 7], 100, 4, torch.float16)
allok &= run_case("w=3 dim=128 bf16 silu", [9, 4, 12], 128, 3, torch.bfloat16)
allok &= run_case("w=2 dim=64 fp16 silu", [5, 6], 64, 2, torch.float16)
allok &= run_case("w=4 dim=640 fp16 多通道块", [4, 5], 640, 4, torch.float16)
allok &= run_case("w=4 dim=64 fp16 无 bias", [5, 6], 64, 4, torch.float16, bias_flag=False)
allok &= run_case("w=4 dim=64 fp16 activation=0", [5, 6], 64, 4, torch.float16, activation=0)
allok &= run_case("w=4 dim=64 bf16 cacheIdx[2,0,1]", [5, 6, 4], 64, 4, torch.bfloat16,
                  cache_idx=[2, 0, 1], num_cache=3)
allok &= run_case("w=4 dim=64 bf16 initMode 混合", [5, 6, 4], 64, 4, torch.bfloat16,
                  init_mode=[True, False, True])
allok &= run_case("w=4 dim=64 bf16 padSlot 命中", [5, 6], 64, 4, torch.bfloat16,
                  cache_idx=[0, 1], pad_slot=1, num_cache=2)
allok &= run_case("w=4 dim=64 bf16 短序列 len<w-1", [1, 2, 3], 64, 4, torch.bfloat16)
allok &= run_case("w=4 dim=32 fp16 initMode 全 False", [5, 5], 32, 4, torch.float16,
                  init_mode=[False, False])

emit("B) 调用点风格（run_mode=1：3D decode / 2D decode / varlen / 投机解码）")
allok &= run_callsite("3D [4,3,64] fp16 w=4 silu", 1, 64, 4, torch.float16, n_batch=4, seq_len=3)
allok &= run_callsite("3D [4,1,128] bf16 w=4 initMode", 1, 128, 4, torch.bfloat16, n_batch=4,
                      seq_len=1, init_mode=[True, False, True, False])
allok &= run_callsite("2D [4,64] fp16 w=4 decode", 2, 64, 4, torch.float16, n_batch=4)
allok &= run_callsite("2D [4,64] bf16 w=3 cacheIdx 独立槽", 2, 64, 3, torch.bfloat16, n_batch=4,
                      cache_idx=[1, 0, 2, 3], num_cache=4)
allok &= run_callsite("varlen [3,5,2] fp16 w=4", 0, 64, 4, torch.float16, n_batch=3, lens=[3, 5, 2])
allok &= run_callsite("投机解码 3D [4,2,64] nat=[2,3,1,2]", 1, 64, 4, torch.float16, n_batch=4,
                      seq_len=2, nat=[2, 3, 1, 2], init_mode=[True] * 4)
allok &= run_callsite("投机解码 2D [4,64] nat=[1,4,2,3]", 2, 64, 4, torch.float16, n_batch=4,
                      nat=[1, 4, 2, 3], init_mode=[True] * 4)
allok &= run_callsite("运行点风格 padSlot 命中", 2, 64, 4, torch.float16, n_batch=4,
                      cache_idx=[0, -1, 1, -1], pad_slot=-1, num_cache=2)

emit()
emit(f"总体: {'全部 PASS ✅' if allok else '存在 FAIL ❌'}")

txt = "\n".join(L)
open("/root/dsh_work/causal_conv1d_test.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
