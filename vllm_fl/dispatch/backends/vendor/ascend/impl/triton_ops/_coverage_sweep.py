# -*- coding: utf-8 -*-
"""_coverage_sweep.py —— 生产 shape 覆盖扫描 + 判据框架

目的：把「单测过 ≠ 生产对」变成可执行的检查。三类判据（从便宜到贵）：
  J1 静态覆盖判据  —— 生产轴取值是否被现有用例覆盖（不需要 NPU）
  J2 编译/执行判据 —— 生产取值下 kernel 能否编译+执行（UB 溢出 / FLIR 断言）
  J3 数值判据      —— 与参考逐元素比对（需要参考；本框架只对**自带参考**的算子跑）

用法：
    python _coverage_sweep.py static                 # 只出静态覆盖矩阵
    python _coverage_sweep.py sweep [算子名...]       # J2：生产轴下编译+执行
    python _coverage_sweep.py sweep --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 1. 生产轴登记表 —— 来源全部标注（调用点行号 / config.json / 算子 def.cpp）
#    每条 = (轴名, 生产取值列表, 依据)
#    ⚠️ 只有写进这里是**有出处**的；推断一律标 [推断]
# ===========================================================================
PRODUCTION: dict[str, dict] = {
    "grouped_matmul_swiglu_quant_v2": {
        "src": "impl/moe/moe_mlp.py:343 + config.json moe_intermediate_size",
        "axes": {
            "K": [2048, 7168],                 # hidden / moe hidden
            "I": [512, 1024, 1408, 1536, 2048, 2304],   # 全模型扫描值（BACKEND_NOTES 3.2）
            "BLK_N": [512, 1024, 2048, 4096],  # = next_pow2(I)，I>=512 ⇒ 全部 >256（已修，task-6）
            "M": [1, 16, 64, 4096],
        },
    },
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": {
        "src": "impl/moe/moe_mlp.py:360（W8A8 融合分支）",
        "axes": {"K": [2048, 7168], "I": [512, 1024, 1536, 2048, 2304], "M": [1, 16, 64, 4096]},
    },
    "mla_preprocess": {
        "src": "impl/attention/*（DeepSeek-V4 MLA）",
        "axes": {"KVD": [512], "N": [16, 128], "D": [576], "BLK_D": [64, 256, 512]},
    },
    "dequant_swiglu_quant": {
        "src": "dispatch 默认接入；H = moe_intermediate_size*2",
        "axes": {"M": [1, 33, 4096, 65536], "H": [1024, 2048, 4096, 4608, 512], "G": [1, 2, 4, 8]},
    },
    "rms_norm_dynamic_quant": {
        "src": "impl/linearnorm/",
        "axes": {"R": [1, 33, 4096], "D": [512, 2048, 4096, 7168, 1024, 1000]},
    },
    "compressor": {
        "src": "impl/（DeepSeek-V4 真实 dispatch）",
        "axes": {"D": [512], "R": [4, 128], "RD": [64], "S": [384, 512, 4096], "H": [128, 512]},
    },
    "compressor_bigr": {
        "src": "同 compressor（bigr = block interleave gather rope）",
        "axes": {"D": [512], "R": [128], "S": [300, 384, 512], "H": [128]},
    },
    "compressor_coff2": {
        "src": "coff=2（OVERLAP）",
        "axes": {"D": [128, 512], "R": [4], "S": [22, 24, 16, 4]},
    },
    "moe_gating_top_k_hash": {
        "src": "impl/moe/experts_selector.py",
        "axes": {"E": [256, 257, 512], "k": [8], "G": [8], "P": [32], "KG": [4]},
    },
    "moe_gating_top_k": {
        "src": "task-4 探针：E=256 k=8 G=8 KG=4（生产）",
        "axes": {"E": [256, 512], "k": [8], "G": [8], "KG": [4], "rows": [1, 32, 4096]},
    },
    "hc_pre": {"src": "impl/hc/", "axes": {"bs": [4096], "hm": [4], "d": [512, 7168]}},
    "hc_post": {"src": "impl/hc/", "axes": {"B": [4096], "H": [4], "D": [512, 7168]}},
    "hc_pre_inv_rms": {"src": "impl/hc/", "axes": {"R": [4096], "hc": [4], "D": [512, 7168]}},
    "hc_pre_sinkhorn": {"src": "impl/hc/", "axes": {"bs": [4096], "hm": [4], "d": [512, 7168]}},
    "kv_compress_epilog": {
        "src": "DeepSeek-V4 KV cache",
        "axes": {"d": [576], "group": [64, 128], "rows": [9, 4096], "rscale": [0, 1]},
    },
    "indexer_compress_epilog_v2": {"src": "DeepSeek-V4 indexer", "axes": {"d": [576], "group": [64, 128]}},
    "sparse_flash_attention": {"src": "impl/attention/", "axes": {"S1": [4, 4096], "S2": [48, 8192], "N1": [4, 64]}},
    "sparse_attention_score": {"src": "impl/attention/", "axes": {"K": [8, 2048], "N1": [64]}},
    "lightning_indexer": {"src": "impl/attention/", "axes": {"K": [2048], "N1": [64]}},
    "lightning_indexer_quant": {"src": "impl/attention/", "axes": {"K": [2048], "N1": [64]}},
    "vllm_quant_lightning_indexer": {"src": "impl/attention/", "axes": {"K": [2048], "N1": [64]}},
}


# ===========================================================================
# 2. 从 test_*.py 里提取「实际测过的取值」——正则扫 run_case/case 调用
# ===========================================================================
CALL_RE = re.compile(r"(?:run_case|case|run_case_saskv_pa)\s*\(\s*\"([^\"]*)\"(.*?)\)\s*$", re.S)
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+)(?![0-9])")


def tested_numbers(test_path: str) -> set[int]:
    if not os.path.exists(test_path):
        return set()
    txt = open(test_path, encoding="utf-8", errors="replace").read()
    return {int(m.group(1)) for m in NUM_RE.finditer(txt)}


# ===========================================================================
# 3. 静态覆盖判据 J1
# ===========================================================================
def static_report() -> list[dict]:
    """对每个登记算子，判断生产轴取值是否出现在测试文件的数字集合里。

    这是**粗判据**（只看数字有没有出现过），价值在于把"从没出现过"的取值挑出来；
    数字出现过不代表语义正确，但**没出现过一定没被覆盖**。
    """
    rows = []
    for op, info in PRODUCTION.items():
        tf = os.path.join(HERE, f"test_{op}.py")
        nums = tested_numbers(tf)
        for axis, vals in info["axes"].items():
            for v in vals:
                rows.append({
                    "op": op, "axis": axis, "value": v,
                    "covered": v in nums,
                    "test_file": os.path.basename(tf) if os.path.exists(tf) else "(缺)",
                })
    return rows


# ===========================================================================
# 4. 编译/执行判据 J2 —— 生产轴下直接调算子，只看「能不能跑 + 输出健全性」
#    ⚠️ 不判断数值对错（那需要参考）；判据是：
#       OK / UB 溢出 / FLIR 断言 / 其它异常 / 输出含 nan|inf / 输出 shape 不符
#    产出 shape 与 dtype 也核对，因为"静默丢写"常表现为大面积未写（0/初值）。
# ===========================================================================
def classify_exc(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "ub overflow" in low or "ub overflow" in s:
        return "UB_OVERFLOW"
    if "assertion" in low or "unreachable" in low or "interleave" in low:
        return "COMPILER_ASSERT"
    if "notimplemented" in low:
        return "NOT_IMPLEMENTED"
    if "507035" in s or "507057" in s:
        return "DEVICE_TRANSIENT"
    if "compilation" in low or "compile" in low:
        return "COMPILE_ERROR"
    return "OTHER_EXC"


def sweep_case(fn, build) -> dict:
    """build() 造输入（在设备上）；返回 {status, detail, shape, dtype, nan, inf, absmax}。"""
    out = {"status": "?", "detail": ""}
    try:
        args = build()
    except Exception as e:  # 造输入本身失败不算算子问题
        out.update(status="INPUT_BUILD_FAIL", detail=classify_exc(e), tb=traceback.format_exc()[-500:])
        return out
    try:
        res = fn(*args) if isinstance(args, tuple) else fn(**args)
        try:
            import torch
            torch.npu.synchronize()
        except Exception:
            pass
        tensors = []
        if isinstance(res, (tuple, list)):
            tensors = [t for t in res if hasattr(t, "shape") and hasattr(t, "dtype")]
        elif hasattr(res, "shape"):
            tensors = [res]
        if not tensors:
            out.update(status="OK", detail="无张量输出")
            return out
        import torch
        t = tensors[0]
        tf = t.detach().float()
        out.update(status="OK", shape=list(t.shape), dtype=str(t.dtype),
                   nan=int(torch.isnan(tf).sum().item()),
                   inf=int(torch.isinf(tf).sum().item()),
                   absmax=float(tf[~torch.isnan(tf)].abs().max().item()) if tf.numel() else 0.0)
    except Exception as e:
        out.update(status=classify_exc(e), detail=f"{type(e).__name__}: {str(e)[:300]}")
    return out


# ===========================================================================
# 5. 数值判据 J3 —— **大形状下的分块抽样比对**
#
# 为什么需要：J2 只判"编译/执行/形状/健全性"，它的 OK **不代表数值正确**。
# D1（dequant_swiglu_quant @ M=65536 的 coreDim 越界）就是靠"生产轴取值从未被测试过"
# 漏掉的。J3 补这一层：**在真生产形状上判数值**。
#
# 做法：① 按生产形状造大输入；② 只取若干**探针行**（必含首行/末行 = 边界行 + 等距中间行）；
#      ③ 只对这些行做**主机 fp64 独立复算**（成本 O(采样行) 而非 O(M)）；
#      ④ 逐元素给 4 个判据：max|Δ| / >1LSB 计数 / **完全一致率** / 边界行单独报。
# **不用均值**——均值会把"整行错、别行对"平均掉（compressor 踩过这个坑）。
#
# 本段**不 import torch/npu**，可直接被设备端脚本 import 复用；
# 设备端示例见 `_coverage_j3.py`（dequant_swiglu_quant）。
# ===========================================================================
J3_RESULTS: list = []


def probe_rows(n_rows: int, n_sample: int = 8) -> list:
    """首行 + 末行 + 等距中间行（**保证边界行必测**）。"""
    if n_rows <= n_sample:
        return list(range(n_rows))
    mids = [int(round(i * (n_rows - 1) / (n_sample - 1))) for i in range(n_sample)]
    return sorted(set([0, n_rows - 1] + mids))


def run_j3(name, fn, build_inputs, ref_rows, n_rows, out_index=0, tol=1, exact_min=0.999,
           n_sample=8, scale_index=None, scale_tol=2e-6, floor_all_zero=False,
           float_floor=None, results=None):
    """跑一个 J3 用例。

    Args:
        name:          用例名
        fn:            被测算子，`fn(*inputs)`
        build_inputs:  `() -> tuple`，在设备上构造输入
        ref_rows:      `(inputs, rows[, want_scale=True]) -> fp64 张量`（**只算采样行**）
        n_rows:        形状的第 0 维（用于挑边界行）
        out_index:     输出元组里要比对的那一项
        tol:           允许的 max|Δ|（int8 量化 ⇒ 1 LSB）
        exact_min:     要求的最低**完全一致率**
        scale_index:   若给定，额外比对该输出的相对误差
        results:       结果列表（默认 J3_RESULTS）
    Returns: 结果 dict
    """
    import torch
    res = J3_RESULTS if results is None else results
    rows = probe_rows(n_rows, n_sample)
    rec = {"name": name, "n_rows": n_rows, "rows": rows, "status": "?"}
    try:
        ins = build_inputs()
        out = fn(*ins)
        torch.npu.synchronize()
    except Exception as e:
        rec.update(status="EXC", detail=f"{type(e).__name__}: {str(e)[:200]}")
        res.append(rec)
        print(f"[J3 EXC ] {name}  {rec['detail']}", flush=True)
        return rec
    tensors = out if isinstance(out, (tuple, list)) else (out,)
    got = tensors[out_index].detach().cpu()
    # ⚠️ 约定统一：`build_inputs()` 返回输入元组，`fn` 与 `ref_rows` **都**用 `*ins` 展开，
    #    两者签名一致，不会出现"一个收元组、一个收展开"的错位（我踩过这个坑）。
    ref = None
    ref_err = None
    try:
        ref = ref_rows(*ins, rows=rows)        # 只算采样行的 fp64 参考
    except TypeError as e:
        try:
            ref = ref_rows(ins, rows)          # 兼容旧的 (ins, rows) 位置参数写法
        except Exception as e2:
            ref_err = f"{type(e2).__name__}: {str(e2)[:200]}  (首次尝试: {e})"
    except Exception as e:
        ref_err = f"{type(e).__name__}: {str(e)[:200]}"
    if ref is None:
        rec.update(status="REF_EXC", detail=ref_err or "ref_rows 返回 None")
        res.append(rec)
        print(f"[J3 REF!] {name}  {rec['detail']}", flush=True)
        return rec
    ref = ref if torch.is_tensor(ref) else torch.tensor(ref)
    sub = got[rows]
    d = (sub.double() - ref.double()).abs()
    md = float(d.max()) if d.numel() else 0.0
    ex = float((d == 0).double().mean()) if d.numel() else 1.0
    over1 = int((d > 1).sum().item())
    # ⚠️ 判据必须按**输出 dtype** 分两类（我在这里错过一次，把正确的 hc_post 判成 FAIL）：
    #   ① 整型输出（int8 量化等）⇒ 逐元素**完全一致** + 最多 1 LSB；
    #   ② 浮点输出（bf16/fp16/fp32）⇒ 用**该 dtype 的量化地板**判 max|Δ|。
    #      **绝不能要求浮点输出"逐位完全一致"**：输入是 bf16 时，
    #      fp64 参考与 kernel 的舍入路径天然不同，"exact率"只反映 dtype 精度，
    #      不反映对错（hc_post bf16 的 exact 只有 0.3%，但 max|Δ|≈1e-3 < 地板 4e-3，是对的）。
    is_int_out = not sub.is_floating_point()
    if is_int_out:
        crit = f"逐元素(max|Δ|≤{tol:g}, 完全一致≥{exact_min:.1%})"
        ok = (md <= tol) and (ex >= exact_min)
    else:
        if float_floor is None:
            float_floor = {torch.float32: 2e-6, torch.float16: 1e-3,
                           torch.bfloat16: 4e-3}.get(got.dtype, 4e-3)
        crit = f"浮点(max|Δ|≤地板 {float_floor:g}，不看 exact)"
        ok = md <= float_floor
    b_bad = []
    for r in sorted(set([0, n_rows - 1]) & set(rows)):
        dd = d[rows.index(r)]
        bad = (float(dd.max()) > (tol if is_int_out else float_floor)) or \
              (is_int_out and float((dd != 0).double().mean()) > (1 - exact_min))
        if bad:
            b_bad.append((r, float(dd.max()), float((dd == 0).double().mean())))
    sc_txt = ""
    if scale_index is not None and len(tensors) > scale_index:
        # ⚠️ 与上面的 ref 调用**必须同约定**（*ins 展开 + 关键字 rows/want_scale）；
        #    这里曾是旧写法 `ref_rows(ins, rows, want_scale=True)`，改成展开后漏改此处，
        #    导致只有"带 scale 的用例"报 missing arguments —— 两处必须同步改。
        try:
            sc_ref = ref_rows(*ins, rows=rows, want_scale=True)
        except TypeError:
            sc_ref = ref_rows(ins, rows, want_scale=True)
        if sc_ref is not None:
            sc_got = tensors[scale_index].detach().cpu().double()[rows]
            erel = float(((sc_got - sc_ref).abs() / sc_ref.abs().clamp_min(1e-30)).max())
            sc_txt = f" scale_rel={erel:.2e}"
            ok = ok and erel <= scale_tol
    allz = int((sub == 0).sum().item()) == sub.numel()
    if floor_all_zero and allz:
        ok = False
        sc_txt += " [采样行全 0 ⇒ 判定失败]"
    rec.update(status="PASS" if ok else "FAIL", max_d=md, exact=ex, over1=over1,
               boundary_bad=b_bad, n_elems=int(d.numel()), all_zero=allz, extra=sc_txt,
               crit=crit)
    res.append(rec)
    print(f"[J3 {'PASS' if ok else 'FAIL'}] {name:44s} n={n_rows:6d} 采样{len(rows)}行 "
          f"max|Δ|={md:.3e} exact={ex * 100:.3f}% >1LSB={over1}{sc_txt}  [{crit}]", flush=True)
    if b_bad:
        print(f"          ⚠️ 边界行异常: {b_bad}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["static", "sweep", "list"], default="static", nargs="?")
    ap.add_argument("ops", nargs="*")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    if a.mode == "list":
        for op, info in PRODUCTION.items():
            print(f"{op:60s} src={info['src']}")
        return

    rows = static_report()
    if a.mode == "static":
        print(f"{'算子':48s} {'轴':8s} {'生产值':>10s}  覆盖?  测试文件")
        print("-" * 110)
        gap = 0
        for r in rows:
            mark = "已测" if r["covered"] else "**未覆盖**"
            if not r["covered"]:
                gap += 1
            print(f"{r['op']:48s} {r['axis']:8s} {r['value']:>10d}  {mark:8s} {r['test_file']}")
        print("-" * 110)
        print(f"合计 {len(rows)} 条生产轴取值，其中**测试文件里从未出现的数字**有 {gap} 条 "
              f"（判据方向：没出现 ⇒ 一定没测过；出现 ≠ 测对了）")
        if a.json:
            json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    print("sweep 模式需要在设备上跑；本文件只提供框架，具体算子的 build() 见 _probe_cov_*.py",
          file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
