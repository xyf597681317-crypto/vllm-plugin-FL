# -*- coding: utf-8 -*-
"""`_C_ascend_extra` 验证：adapter 契约（参数搬运 / 原地写 / 返回类型）。

裁判问题（先说结论）：**本容器里 AscendC 当不了裁判**。加载真实
`vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so` 后 schema 都在
（`hasattr(torch.ops._C_ascend, 'npu_hc_post') == True`），但一调用就报
    RuntimeError: aclnnHcPost or aclnnHcPostGetWorkspaceSize not in libopapi.so,
                  or libopapi.so not found.
即这些 `_C_ascend` 算子只是 aclnn 的薄封装，而本容器没装对应 aclnn 算子。
`VLLM_FL_TRITON_OPS=0/1` 两进程对照因此拿不到 AscendC 侧数据（见 exo_ascendc.log）。

替代 oracle（等价性更强地针对 adapter 的职责）：**同一个进程内**
  `torch.ops._C_ascend.<op>(...)`（走 `_C_ascend_extra` 的 adapter）
  vs  `实现函数直接调用`（按实现本来的参数顺序）
逐个元素比对 —— 一致即证明 adapter 的参数搬运/返回搬运没错位。
（数值正确性由各算子自带的 `test_*.py`（fp64 参考）覆盖，本探针不重复。）

用法：
    EXTRA_MODE=triton DSH_NPU=npu:14 python3 -W ignore _probe_extra_oracle.py
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

DEV = os.environ.get("DSH_NPU", "npu:14")
torch.npu.set_device(DEV)
MODE = os.environ.get("EXTRA_MODE", "triton")
SP = "/usr/local/python3.12.13/lib/python3.12/site-packages"
SO = SP + "/vllm_fl/_C_ascend.cpython-312-aarch64-linux-gnu.so"
SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.ops.load_library(SO)
print(f"[oracle] dev={DEV} mode={MODE} .so 已加载（schema 就位）", flush=True)

import triton_ops._C_ascend_extra as extra  # noqa: E402

schemas, adapters = extra.build()
print(f"[oracle] build(): schemas={len(schemas)} adapters={len(adapters)}", flush=True)

if MODE == "triton":
    lib = torch.library.Library("_C_ascend", "IMPL")
    n = 0
    for name, fn in adapters.items():
        lib.impl(name, fn, "PrivateUse1")     # 与 patch_triton_ops._DISPATCH_KEY 一致
        n += 1
    print(f"[oracle] 已按 patch 的方式注册 {n} 个 adapter 到 _C_ascend(PrivateUse1)", flush=True)
else:
    print("[oracle] ascendc 模式：只用 .so 原生实现（预期 aclnn 缺失报错）", flush=True)

OUT, INFO = [], []


def rec(name, t):
    if torch.is_tensor(t):
        OUT.append((name, t.detach().float().cpu().numpy()))


def cmp_ab(tag, a, b):
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        INFO.append(f"{tag}: 非张量，跳过逐元素比对 (a={type(a).__name__}, b={type(b).__name__})")
        return
    if tuple(a.shape) != tuple(b.shape):
        INFO.append(f"{tag}: SHAPE 不一致 {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    d = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    md = float(d.max()) if d.numel() else 0.0
    exact = float((d == 0).float().mean()) * 100 if d.numel() else 100.0
    INFO.append(f"{tag}: max|Δ|={md:.3e} exact={exact:.4f}% n={d.numel()}")


# ------------------------------------------------------ A) npu_hc_post
def case_hc_post():
    n, hc, d = 8, 4, 64
    torch.manual_seed(11)
    x = (torch.randn(1, n, d, device=DEV) * 2 - 1).mul(0.4).to(torch.bfloat16)
    r = (torch.randn(1, n, hc, d, device=DEV) * 2 - 1).mul(0.4).to(torch.bfloat16)
    p = ((torch.randn(1, n, hc, device=DEV) * 2 - 1).mul(0.4)).to(torch.bfloat16)
    c = ((torch.randn(1, n, hc, hc, device=DEV) * 2 - 1).mul(0.4)).to(torch.bfloat16)
    y_op = torch.ops._C_ascend.npu_hc_post(x, r, p, c)
    torch.npu.synchronize()
    import triton_ops.hc_post as impl_mod
    y_impl = impl_mod.hc_post(x, r, p, c)
    torch.npu.synchronize()
    INFO.append(f"npu_hc_post: op.out.shape={tuple(y_op.shape)} dtype={y_op.dtype}")
    cmp_ab("npu_hc_post(op vs impl)", y_op, y_impl)
    rec("hc_post_y", y_op)


# ------------------------------------------------------ B) npu_hc_pre_inv_rms
def case_inv_rms():
    import triton_ops.hc_pre_inv_rms as impl_mod
    torch.manual_seed(12)
    x = (torch.randn(64, 4, 512, device=DEV) * 1.3).to(torch.bfloat16)
    y_op = torch.ops._C_ascend.npu_hc_pre_inv_rms(x, 1e-6)
    torch.npu.synchronize()
    y_impl = impl_mod.hc_pre_inv_rms(x, 1e-6)
    torch.npu.synchronize()
    INFO.append(f"npu_hc_pre_inv_rms: op.out.shape={tuple(y_op.shape)} dtype={y_op.dtype}")
    cmp_ab("npu_hc_pre_inv_rms(op vs impl)", y_op, y_impl)
    rec("inv_rms_y", y_op)


# ------------------------------------------------------ C) npu_causal_conv1d_custom
def case_conv1d():
    import triton_ops.causal_conv1d as impl_mod
    nb, sl, dim, width = 2, 1, 64, 4
    dtype = torch.float16
    torch.manual_seed(21)
    x = (torch.randn(nb * sl, dim, device=DEV) * 1.5).to(dtype)
    w = (torch.randn(width, dim, device=DEV) * 0.7).to(dtype)
    b = (torch.randn(dim, device=DEV) * 0.3).to(dtype)
    st = (torch.randn(2, width - 1, dim, device=DEV) * 0.5).to(dtype)
    ci = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    SENT = -777.0

    out_op = torch.full((nb * sl, dim), SENT, dtype=dtype, device=DEV)
    res_op = torch.ops._C_ascend.npu_causal_conv1d_custom(
        out_op, x, w, st, b, None, ci, None, None, 1, -1, 1)
    torch.npu.synchronize()
    wrote = int((out_op != SENT).sum())
    same_obj = torch.is_tensor(res_op) and res_op.data_ptr() == out_op.data_ptr()
    INFO.append(f"npu_causal_conv1d_custom: 传入 output 被原地写入 {wrote}/{out_op.numel()} 元素; "
                f"返回值就是同一个张量={same_obj}; 返回 shape={tuple(res_op.shape) if torch.is_tensor(res_op) else type(res_op).__name__}")

    st2 = st.clone()
    out_impl = torch.full((nb * sl, dim), SENT, dtype=dtype, device=DEV)
    res_impl = impl_mod.causal_conv1d_fn(
        x, w, conv_state=st2, bias_opt=b, query_start_loc_opt=None, cache_indices_opt=ci,
        initial_state_mode_opt=None, num_accepted_tokens_opt=None,
        activation_mode=1, pad_slot_id=-1, run_mode=1, out=out_impl)
    torch.npu.synchronize()
    cmp_ab("npu_causal_conv1d_custom(op vs impl)", res_op, res_impl)
    cmp_ab("conv1d conv_state(op 路径 vs impl 路径)", st, st2)
    rec("conv1d_out", res_op if torch.is_tensor(res_op) else out_op)


# ------------------------------------------------------ D) npu_recurrent_gated_delta_rule
def case_rgdr():
    import triton_ops.recurrent_gated_delta_rule as impl_mod
    T, NK, NV, DK, DV = 32, 2, 2, 32, 32
    torch.manual_seed(3)

    def U(*s, lo=0.2):
        return (torch.rand(*s, dtype=torch.float32, device=DEV) * 2 - 1) * lo

    query = U(T, NK, DK).to(torch.bfloat16)
    key = U(T, NK, DK).to(torch.bfloat16)
    value = U(T, NV, DV).to(torch.bfloat16)
    beta = U(T, NV).to(torch.bfloat16)
    g = U(T, NV, lo=0.1) - 0.05
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=DEV)
    ssm = torch.arange(T, dtype=torch.int32, device=DEV) * NV
    sc = float(DK) ** -0.5

    state_op = U(int(ssm.max().item()) + NV, NV, DV, DK).to(torch.bfloat16)
    before_op = state_op.clone()
    res_op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query, key=key, value=value, g=g, beta=beta, state=state_op,
        scale=sc, actual_seq_lengths=cu, ssm_state_indices=ssm)
    torch.npu.synchronize()
    is_tuple = isinstance(res_op, tuple)
    INFO.append(f"npu_recurrent_gated_delta_rule: 返回类型={type(res_op).__name__} "
                f"is_tuple={is_tuple} len={len(res_op) if is_tuple else '-'}")
    if is_tuple:
        INFO.append(f"   元素 shape: {[tuple(t.shape) for t in res_op if torch.is_tensor(t)]}")
    d_state_op = float((state_op.float() - before_op.float()).abs().max())
    INFO.append(f"   ★ schema 标的 state（Tensor(a!)）是否被**原地更新**: "
                f"max|Δ|={d_state_op:.3e}，变化元素={int((state_op != before_op).sum())}/{state_op.numel()}")

    state_impl = before_op.clone()
    res_impl = impl_mod.recurrent_gated_delta_rule(
        query, key, value, beta, state_impl, cu, ssm, g,
        scale_value=sc)
    torch.npu.synchronize()
    d_state_impl = float((state_impl.float() - before_op.float()).abs().max())
    INFO.append(f"   （对照）实现直调时 state 是否被原地更新: max|Δ|={d_state_impl:.3e}")
    if is_tuple and isinstance(res_impl, tuple):
        cmp_ab("rgdr out(op vs impl)", res_op[0], res_impl[0])
        cmp_ab("rgdr final_state(op vs impl)", res_op[1], res_impl[1])
    if is_tuple:
        rec("rgdr_out", res_op[0])
        rec("rgdr_final", res_op[1])
    rec("rgdr_state_after_op", state_op)
    rec("rgdr_state_before", before_op)


CASES = {"hc_post": case_hc_post, "inv_rms": case_inv_rms,
         "conv1d": case_conv1d, "rgdr": case_rgdr}


def main():
    want = sys.argv[1:] or list(CASES)
    for name in want:
        print(f"== case {name} ==", flush=True)
        try:
            CASES[name]()
        except Exception as ex:  # noqa: BLE001
            INFO.append(f"{name}: EXC {type(ex).__name__}: {str(ex)[:230]}")
            print(f"   EXC {type(ex).__name__}: {str(ex)[:230]}", flush=True)
    print("---- 结论行 ----", flush=True)
    for line in INFO:
        print(f"  [info] {line}", flush=True)
    np.savez(f"/tmp/extra_oracle_{MODE}.npz", **dict(OUT))
    print(f"[oracle] saved /tmp/extra_oracle_{MODE}.npz keys={[k for k, _ in OUT]}", flush=True)
    print("[oracle] done", flush=True)


if __name__ == "__main__":
    main()
