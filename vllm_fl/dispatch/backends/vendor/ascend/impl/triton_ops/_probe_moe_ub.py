# -*- coding: utf-8 -*-
"""诊断 B（已更新到 1-D 写法）：生产列宽（cols=7168）下的 BiShengIR UB 需求。

用法（容器内）：
    python3 -W ignore _probe_moe_ub.py                # 公开入口 + 生产参数逐内核
    python3 -W ignore _probe_moe_ub.py 1024:4 8192:4  # 指定 (BLK_C:BLK_R) 找 UB 断裂点
环境：MOE_COLS 列宽；MOE_TOTAL 行数（=N*k）；MOE_DEV 设备
"""
import os
import re
import sys

import torch
import torch_npu  # noqa: F401
import triton

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from triton_ops.moe_init_routing_custom import (  # noqa: E402
    DYNAMIC_QUANT, GATHER, KEY_VALUE, NON_QUANT,
    _apply_quant_kernel, _gather_kernel, _rowscale_kernel, moe_init_routing_custom,
)

DEV = os.environ.get("MOE_DEV", "npu:1")
COLS = int(os.environ.get("MOE_COLS", "7168"))
TOTAL = int(os.environ.get("MOE_TOTAL", "64"))
K = int(os.environ.get("MOE_K", "2"))
E = int(os.environ.get("MOE_E", "8"))
LIMIT = 1572864
print(f"[probe-ub] dev={DEV} cols={COLS} total={TOTAL} k={K} E={E} limit={LIMIT} bits", flush=True)


def catch(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        print(f"  [{name}] OK  (compiled & ran)", flush=True)
        return None
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        bits = None
        m = re.search(r"requires\s+([0-9]+)\s*bits", msg)
        if m:
            bits = int(m.group(1))
        else:
            m2 = re.search(r"([0-9]{5,})\s*bits", msg)
            if m2:
                bits = int(m2.group(1))
        hl = " | ".join(l.strip() for l in msg.splitlines()
                        if ("overflow" in l.lower() or "bits" in l.lower()))
        ratio = f"{bits / LIMIT:.2f}x" if bits else "?"
        print(f"  [{name}] FAIL bits={bits} ({ratio}) :: {hl[:400]}", flush=True)
        return bits


def tensors(cols, total, k, dtype=torch.float16):
    n = total // k
    x = torch.randn(n, cols, dtype=torch.float32, device=DEV).to(dtype)
    src = torch.arange(total, dtype=torch.int32, device=DEV)
    de = torch.zeros(total, dtype=torch.int32, device=DEV)
    smat = torch.zeros(1, dtype=torch.float32, device=DEV)
    esc = torch.zeros(total, dtype=torch.float32, device=DEV)
    xq = torch.empty((total, cols), dtype=torch.int8, device=DEV)
    return x, src, de, smat, esc, xq


def kernels(blk_c, blk_r, dtype=torch.float16):
    x, src, de, smat, esc, xq = tensors(COLS, TOTAL, K, dtype)
    g = (triton.cdiv(TOTAL, blk_r),)
    catch(f"rowscale BLK_C={blk_c} BLK_R={blk_r}",
          lambda: _rowscale_kernel[g](x, src, de, smat, esc, TOTAL, K, COLS,
                                      BLK_R=blk_r, BLK_C=blk_c, HAS_SMOOTH=False))
    catch(f"apply    BLK_C={blk_c} BLK_R={blk_r}",
          lambda: _apply_quant_kernel[g](x, src, de, smat, esc, xq, TOTAL, K, COLS,
                                         BLK_R=blk_r, BLK_C=blk_c, HAS_SMOOTH=False))
    catch(f"gather   BLK_C={blk_c} BLK_R={blk_r}",
          lambda: _gather_kernel[g](x, src, xq, TOTAL, K, COLS,
                                    BLK_R=blk_r, BLK_C=blk_c))


def public(dtype=torch.float16, quant=True):
    x, src, de, smat, esc, xq = tensors(COLS, TOTAL, K, dtype)
    ei = torch.randint(0, E, (TOTAL // K, K), dtype=torch.int32, device=DEV)
    return catch(f"public quant={quant} dtype={str(dtype)[6:]}",
                 lambda: moe_init_routing_custom(
                     x, ei, None, None, expert_num=E, expert_tokens_num_type=KEY_VALUE,
                     expert_tokens_num_flag=True,
                     quant_mode=(DYNAMIC_QUANT if quant else NON_QUANT),
                     row_idx_type=GATHER))


if __name__ == "__main__":
    cands = []
    for a in sys.argv[1:]:
        if ":" in a:
            c, d = a.split(":")
            cands.append((int(c), int(d)))
    print("\n== public entry (production shape, cols=%d) ==" % COLS, flush=True)
    public(torch.float16, True)
    public(torch.bfloat16, True)
    public(torch.float16, False)
    if not cands:
        cands = [(1024, 4)]
    print("\n== per-kernel (BLK_C:BLK_R) ==", flush=True)
    for c, d in cands:
        kernels(c, d)
    print("\n[probe-ub] done", flush=True)
