"""第七轮：用真实 recurrent 内核，尝试不同 BLK_V/BLK_K 是否让 fp32 路径编过（IR 规模假设）。"""
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/tops")
from triton_ops.recurrent_gated_delta_rule import _rgdr_kernel  # noqa: E402

dev = "npu:0"
T, B, NK, NV, DK, DV = 32, 2, 2, 2, 32, 32
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


def trial(name, state_dtype, blk_k, blk_v, use_g=True, use_gk=True):
    st = U(64, NV, DV, DK).to(state_dtype)
    out = torch.zeros(T, NV, DV, dtype=torch.bfloat16, device=dev)
    fin = torch.zeros(64, NV, DV, DK, dtype=state_dtype, device=dev)
    grid = (NV, (DV + blk_v - 1) // blk_v, B)
    try:
        _rgdr_kernel[grid](q, k, v, b, g, gk, st, cu, ssm, out, fin, nacc,
                           T, NK, NV, DK, DV, 1.0,
                           HAS_G=use_g, HAS_GK=use_gk, HAS_ACC=False,
                           FINAL_F32=(state_dtype == torch.float32),
                           STATE_F32=(state_dtype == torch.float32),
                           BLK_K=blk_k, BLK_V=blk_v)
        torch.npu.synchronize()
        print(f"  {name:44s} OK   |fin|max={float(fin.abs().max()):.3f}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        tag = "CRASH" if ("LLVM" in msg or "bug report" in msg or "ConvertLinalgRToBinary" in msg) else "ERR"
        print(f"  {name:44s} {tag}: {msg[:52]}", flush=True)
        return False


print("=" * 84, flush=True)
print("第七轮：真实 recurrent 内核 —— 不同 BLK_V/BLK_K × state dtype", flush=True)
print("=" * 84, flush=True)
trial("fp32 state BLK_K=32 BLK_V=32（基线）", torch.float32, 32, 32)
trial("fp32 state BLK_K=32 BLK_V=16", torch.float32, 32, 16)
trial("fp32 state BLK_K=32 BLK_V=8", torch.float32, 32, 8)
trial("fp32 state BLK_K=16 BLK_V=16", torch.float32, 16, 16)
trial("bf16 state BLK_K=32 BLK_V=32（对照）", torch.bfloat16, 32, 32)
print("done", flush=True)
