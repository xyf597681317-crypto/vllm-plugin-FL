# -*- coding: utf-8 -*-
"""_probe_cov_j3_gmm3.py —— 把 glist_type=1 的差异钉死到"谁错"

已知（`_probe_cov_j3_gmm2.py`）：
  M=20 K=32 I=16 G=2 glist_type=1：
    group_list=[10,10]（计数语义）⇒ 边界应为 [10,20]
    kernel:  行 10..19 有值（±127）
    我的参考: 行 10..19 = 0
  kernel 侧 `_ends_from_group_list`（v2 实现 L220-225）与我的参考**语义一致**（都 cumsum）。
  ⇒ 至少有**一方**在别处错了。本探针逐层对照，定位分歧点。

对照设计（同一份输入张量）：
  E1) G=1：group_list=[M]（计数=全部）⇒ 若 kernel 与参考一致，说明"计数语义"通路本身没问题
  E2) G=2：[M/2, M/2] ⇒ 复现 A 的分歧，并逐行打印"第 2 组的行"两侧的 y[0..3] 与 scale
  E3) 手工 fp64：把第 2 组按 [g=1] 的权重算一遍，看**kernel 的值**是不是"第 1 组的权重算出来的"
      （即 kernel 把计数当成了累计偏移）—— 这是本探针的关键判据
"""
import importlib.util
import os
import sys

IDX = int(os.environ.get("DSH_NPU", "npu:12").split(":")[-1])
import torch  # noqa: E402
import torch_npu  # noqa: F401

torch.npu.set_device(IDX)
DEV = "npu:%d" % IDX
sys.path.insert(0, "/workspace/FlagGems/src")

_s = importlib.util.spec_from_file_location(
    "cov", "/workspace/FlagGems/src/triton_ops/_coverage_sweep.py")
cov = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cov)

from triton_ops.grouped_matmul_swiglu_quant_v2 import grouped_matmul_swiglu_quant_v2 as gv2  # noqa: E402


def build(M, K, I, G, cnts, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=DEV)
    w = [torch.randint(-40, 41, (K, 2 * I), dtype=torch.int8, device=DEV) for _ in range(G)]
    ws = [torch.rand(2 * I, dtype=torch.float32, device=DEV) * 0.01 + 0.005 for _ in range(G)]
    xs = torch.rand(M, dtype=torch.float32, device=DEV) * 0.01 + 0.005
    smooth = torch.rand(G, I, dtype=torch.float32, device=DEV) * 0.8 + 0.6
    gl = torch.tensor(cnts, dtype=torch.int64, device=DEV)      # 计数语义
    return x, w, ws, xs, gl, smooth


def call(ins, glist_type=1):
    x, w, ws, xs, gl, smooth = ins
    return gv2(x, weight=w, weight_scale=ws, x_scale=xs, group_list=gl, smooth_scale=smooth,
               weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0,
               quant_mode=0, quant_dtype=0, transpose_weight=False,
               group_list_type=glist_type, tuning_config=[0], swiglu_limit=0.0)


def _rint_fp16(t):
    q16 = t.float().half().float().double()
    return torch.where(q16 >= 0, torch.floor(q16 + 0.5), torch.ceil(q16 - 0.5))


def ref_group(ins, r0, r1, g):
    """用**第 g 组**的权重/scale 手工算 [r0,r1) 行的 fp64 参考（y 与 scale）。"""
    x, w, ws, xs, gl, smooth = ins
    I = w[0].shape[1] // 2
    xd = x.detach().cpu().double()[r0:r1]
    XS = xs.detach().cpu().double()[r0:r1]
    C = (xd * XS[:, None]) @ (w[g].detach().cpu().double() * ws[g].detach().cpu().double()[None, :])
    gate, up = C[:, :I], C[:, I:]
    sw = (gate * torch.sigmoid(gate)) * up * smooth.detach().cpu().double()[g][None, :]
    mx = sw.abs().amax(dim=1)
    sc = mx / 127.0
    return _rint_fp16(sw * (127.0 / mx)[:, None]).clamp(-128, 127).to(torch.int64), sc


def show(tag, ins, rows, groups):
    """rows: 要看的行；groups: 每行应当属于哪一组（按计数语义）。"""
    x, w, ws, xs, gl, smooth = ins
    y, ys = call(ins)
    torch.npu.synchronize()
    yc = y.detach().cpu()
    print(f"\n  [{tag}] group_list={gl.detach().cpu().tolist()} (计数语义)",
          flush=True)
    for r in rows:
        g = groups(r)
        ry_g, rsc_g = ref_group(ins, r, r + 1, g)
        # 其它组的"错口径"参考，用来判 kernel 是否用错了组
        alt = [gg for gg in range(len(w)) if gg != g]
        alt_txt = ""
        for gg in alt:
            ra, _ = ref_group(ins, r, r + 1, gg)
            if int(ra[0].abs().max()) > 0:
                alt_txt += f" | 用组{gg}权重算={int(ra[0].abs().max())}"
        print(f"    row={r:3d} 应为组{g}: kernel_max|y|={int(yc[r].abs().max()):3d} "
              f"ref(组{g})_max|y|={int(ry_g[0].abs().max()):3d} "
              f"kernel_sc={float(ys.detach().cpu()[r]):.4e} ref_sc={float(rsc_g[0]):.4e}{alt_txt}",
              flush=True)


print(f"J3-GMM3 | DEV={DEV}", flush=True)

print("\n===== E1) G=1, 计数=[M]（整批一组）=====", flush=True)
ins1 = build(20, 32, 16, 1, [20])
show("E1", ins1, [0, 1, 19], lambda r: 0)

print("\n===== E2) G=2, 计数=[10,10] —— 复现分歧 =====", flush=True)
ins2 = build(20, 32, 16, 2, [10, 10])
show("E2", ins2, [0, 5, 9, 10, 11, 19], lambda r: 0 if r < 10 else 1)

print("\n===== E3) G=2 但用 glist_type=0 + 累计偏移[10,20]（对照：同一分组）=====", flush=True)
x, w, ws, xs, gl, smooth = ins2
gl0 = torch.tensor([10, 20], dtype=torch.int64, device=DEV)
ins2b = (x, w, ws, xs, gl0, smooth)
y, ys = call(ins2b, glist_type=0)
torch.npu.synchronize()
yc = y.detach().cpu()
print(f"  group_list={gl0.detach().cpu().tolist()} (累计偏移语义)", flush=True)
for r in [0, 5, 9, 10, 11, 19]:
    g = 0 if r < 10 else 1
    ry_g, rsc_g = ref_group(ins2b, r, r + 1, g)
    print(f"    row={r:3d} 应为组{g}: kernel_max|y|={int(yc[r].abs().max()):3d} "
          f"ref_max|y|={int(ry_g[0].abs().max()):3d} kernel_sc={float(ys.detach().cpu()[r]):.4e} "
          f"ref_sc={float(rsc_g[0]):.4e}", flush=True)

print("\n===== E4) 单测形状 M=20 K=32 I=16 G=2 glist_type=1（单测里标 PASS）=====", flush=True)
print("  注：单测 run_case 里 cnt=[min(M,per*(i+1))-min(M,per*i)] 且 cnt[-1]=M-sum(前) "
      "⇒ 与 [10,10] 相同；END 见 unittest 结论", flush=True)

print("J3GMM3DONE", flush=True)
