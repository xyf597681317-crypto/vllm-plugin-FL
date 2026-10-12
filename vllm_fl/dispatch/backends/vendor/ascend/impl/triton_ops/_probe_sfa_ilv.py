# -*- coding: utf-8 -*-
"""task: sparse_flash_attention 编译期崩溃（FLIR InterleaveOptimization 断言）最小复现。

现象（fix-moe 复测）：
  test_sparse_flash_attention 第 5 例 `B2 S1=4 S2=48 N1=4 N2=1 K=8 fp16 block_size=2`
  → RC=134(SIGABRT)，FLIR
     `InterleaveOptimization.cpp:635: Assertion 'llvm::isa<Attribute>(extractSizes.back())' failed`

怀疑触发点：`_sfa_score_kernel` 里那条 **1-D 掩码 store + 步长 = BS 的跨步写**：
    tl.store(SB + row*(K_IDX*BS) + slot*BS + sub, s, mask=sm)   # sub ∈ static_range(BS)

本探针把「访问形态」当变量逐个隔离。因为崩溃是进程级 SIGABRT，**必须每个变体一个进程**：
    for v in stride2_masked contig_masked stride2_nomask twod_masked real; do
        python3 -W ignore _probe_sfa_ilv.py $v; echo "RC=$?"
    done

用法（容器内）：DSH_NPU=npu:12 python3 -W ignore _probe_sfa_ilv.py <variant>
"""
import os
import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

SRC = os.environ.get("MOE_SRC", "/workspace/FlagGems/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = os.environ.get("DSH_NPU", "npu:12")
torch.npu.set_device(DEV)


# ---------------------------------------------------------------- 最小 kernel
@triton.jit
def k_stride2(Y, N, BLK: tl.constexpr, BS: tl.constexpr, MASKED: tl.constexpr):
    """1-D 掩码 store，元素步长 = BS（跨步/交织写）—— 怀疑触发点。"""
    row = tl.program_id(0)
    slot = tl.arange(0, BLK)
    sm = slot < N
    v = slot.to(tl.float32)
    for sub in tl.static_range(BS):
        if MASKED:
            tl.store(Y + row * (N * BS) + slot * BS + sub, v, mask=sm)
        else:
            tl.store(Y + row * (N * BS) + slot * BS + sub, v)


@triton.jit
def k_contig(Y, N, BLK: tl.constexpr, BS: tl.constexpr, MASKED: tl.constexpr):
    """同样 1-D 掩码 store，但每个 sub 段内**连续**（步长 1，无交织）。"""
    row = tl.program_id(0)
    slot = tl.arange(0, BLK)
    sm = slot < N
    v = slot.to(tl.float32)
    for sub in tl.static_range(BS):
        if MASKED:
            tl.store(Y + row * (N * BS) + sub * N + slot, v, mask=sm)
        else:
            tl.store(Y + row * (N * BS) + sub * N + slot, v)


@triton.jit
def k_twod(Y, N, BLK: tl.constexpr, BS: tl.constexpr):
    """2-D [BLK, BS] 形态的同样写（交织用 2-D 表达）。"""
    row = tl.program_id(0)
    slot = tl.arange(0, BLK)
    ar = tl.arange(0, BS)
    sm = slot < N
    v = slot.to(tl.float32)
    tl.store(Y + row * (N * BS) + slot[:, None] * BS + ar[None, :],
             tl.broadcast_to(v[:, None], [BLK, BS]),
             mask=sm[:, None] & (ar < BS)[None, :])


def run_minimal(kind, N=8, BLK=16, BS=2, rows=4):
    # ⚠️ 无掩码变体必须让 BLK == N，否则 `slot*BS` 会写出缓冲区（首版探针在这里越界写了
    #    16 个 float，把 npu:12 的设备池搞脏了 —— 教训：探针自己也要先算好边界）。
    if kind.endswith("nomask"):
        BLK = N
    size = rows * max(N, BLK) * BS
    y = torch.full((size,), -1.0, dtype=torch.float32, device=DEV)
    if kind == "twod":
        k_twod[(rows,)](y, N, BLK=BLK, BS=BS)
    elif kind.startswith("stride2"):
        k_stride2[(rows,)](y, N, BLK=BLK, BS=BS, MASKED=(kind == "stride2_masked"))
    elif kind.startswith("contig"):
        k_contig[(rows,)](y, N, BLK=BLK, BS=BS, MASKED=(kind == "contig_masked"))
    else:
        raise SystemExit(f"unknown kind {kind}")
    torch.npu.synchronize()
    print(f"  [OK] {kind}: 编译+执行通过, 前 6 = {y[:6].detach().cpu().tolist()}", flush=True)


def run_real():
    """真实失败用例：B=2 S1=4 S2=48 N1=4 N2=1 K(k_idx)=8 fp16 block_size(bs)=2。"""
    from triton_ops.sparse_flash_attention import sparse_flash_attention
    B, S1, S2, N1, N2, K, bs, D, DR = 2, 4, 48, 4, 1, 8, 2, 512, 64
    torch.manual_seed(0)
    dev = DEV
    q = torch.randn(B, S1, N1, D, device=dev).to(torch.float16) * 0.2
    k = torch.randn(B, S2, N2, D, device=dev).to(torch.float16) * 0.2
    v = torch.randn(B, S2, N2, D, device=dev).to(torch.float16) * 0.2
    qr = torch.randn(B, S1, N1, DR, device=dev).to(torch.float16) * 0.2
    kr = torch.randn(B, S2, N2, DR, device=dev).to(torch.float16) * 0.2
    idx = torch.zeros(B, S1, N2, K, dtype=torch.int32, device=dev)
    for b in range(B):
        for s in range(S1):
            idx[b, s, 0, :] = torch.arange(K, dtype=torch.int32, device=dev) * bs
    out, lse_max, lse_sum = sparse_flash_attention(
        q, k, v, idx, query_rope=qr, key_rope=kr, scale_value=0.5,
        sparse_block_size=bs, sparse_mode=0, layout_query="BSND", layout_kv="BSND")
    torch.npu.synchronize()
    print(f"  [OK] real: out{tuple(out.shape)} 非零={int((out != 0).sum())}", flush=True)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "stride2_masked"
    print(f"[sfa-ilv] dev={DEV} variant={kind} triton={triton.__version__}", flush=True)
    if kind == "real":
        run_real()
    else:
        run_minimal(kind)
    print(f"[sfa-ilv] {kind} DONE", flush=True)


if __name__ == "__main__":
    main()
