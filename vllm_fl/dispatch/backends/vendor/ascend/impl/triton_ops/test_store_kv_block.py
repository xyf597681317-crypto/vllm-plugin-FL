"""验证 store_kv_block 的 Triton 实现 vs AscendC 语义。

语义：逐 group 连续行块拷贝 keyCache[dst:dst+len] = keyIn[src:src+len]；
       len<=0 / src<0 / dst<0 的组跳过。
纯拷贝算子 → 输出必须**逐位相等**（exact，不用误差地板）。
调用点风格：vllm_ascend/attention/indexer.py L261-268 的 6 个位置参数。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_ops.store_kv_block import store_kv_block  # noqa: E402

DEV = "npu:0"
L = []


def emit(s=""):
    L.append(str(s))


def ref_grouped(key_in, cache0, glen, gsrc, gdst, token):
    """AscendC 语义参考（纯拷贝）：在展平后的 [rows, token] 上做块拷贝。"""
    ki = key_in.reshape(-1)
    out = cache0.reshape(-1).clone()
    for g in range(glen.numel()):
        ln, s, d = int(glen[g]), int(gsrc[g]), int(gdst[g])
        if ln <= 0 or s < 0 or d < 0:
            continue
        out[d * token:(d + ln) * token] = ki[s * token:(s + ln) * token]
    return out.reshape(cache0.shape)


def scatter_ref(key_in, cache0, slot_mapping, token):
    """把每个 token 按 slot 散布进 cache（= 分组拷贝应等价的那件事）。"""
    ki = key_in.reshape(-1)
    out = cache0.reshape(-1).clone()
    for i, s in enumerate(slot_mapping.tolist()):
        if s < 0:
            continue
        out[s * token:(s + 1) * token] = ki[i * token:(i + 1) * token]
    return out.reshape(cache0.shape)


def build_groups(slot_mapping, block_size):
    """模拟 store_kv_block_metadata：把连续 slot 的 token 合成 block_size 上限的 run。"""
    n = len(slot_mapping)
    glen, gsrc, gdst = [], [], []
    i = 0
    while i < n:
        if slot_mapping[i] < 0:
            glen.append(0)
            gsrc.append(-1)
            gdst.append(-1)
            i += 1
            continue
        j = i + 1
        while (j < n and j - i < block_size and slot_mapping[j] >= 0
               and slot_mapping[j] == slot_mapping[j - 1] + 1):
            j += 1
        glen.append(j - i)
        gsrc.append(i)
        gdst.append(slot_mapping[i])
        i = j
    return (torch.tensor(glen, dtype=torch.int32, device=DEV),
            torch.tensor(gsrc, dtype=torch.int32, device=DEV),
            torch.tensor(gdst, dtype=torch.int32, device=DEV))


def run_case(name, num_tokens, head_size, dtype, num_blocks=8, block_size=4,
             num_head=1, holes=False, seed=3):
    torch.manual_seed(seed)
    key_in = torch.randint(0, 100, (num_tokens, num_head, head_size)).to(dtype).to(DEV)
    cache0 = torch.zeros(num_blocks, block_size, num_head, head_size).to(dtype).to(DEV)
    cache0 += 7                                  # 非零底，便于发现"没写"的位置
    n_cache = num_blocks * block_size
    token = num_head * head_size
    # 连续 slot（模拟正常 prefill），可选插入空洞/负值
    slots = list(range(1, 1 + num_tokens))
    if holes:
        for k in range(0, num_tokens, 5):
            slots[k] = -1
    slots = [s if s < n_cache else -1 for s in slots]
    sm = torch.tensor(slots, dtype=torch.int32, device=DEV)
    glen, gsrc, gdst = build_groups(slots, block_size)

    cache = cache0.clone()
    store_kv_block(key_in, cache, glen, gsrc, gdst, block_size)   # 调用点风格（位置参数）
    torch.npu.synchronize()

    r_grp = ref_grouped(key_in, cache0, glen, gsrc, gdst, token)
    r_sca = scatter_ref(key_in, cache0, sm, token)
    ok_grp = torch.equal(cache.detach().cpu(), r_grp.detach().cpu())
    ok_sca = torch.equal(cache.detach().cpu(), r_sca.detach().cpu())
    ok = ok_grp and ok_sca
    line = (f"  {name:52s} vs_grouped={'OK' if ok_grp else 'DIFF'} "
            f"vs_scatter={'OK' if ok_sca else 'DIFF'}  {'PASS' if ok else 'FAIL'}")
    emit(line)
    print(line, flush=True)
    return ok


emit("=" * 96)
emit("store_kv_block: Triton vs AscendC 语义（逐 group 连续行块拷贝；纯拷贝 → 必须逐位相等）")
emit("=" * 96)

ok = True
F16 = torch.float16
BF = torch.bfloat16
I8 = torch.int8

ok &= run_case("int8  tokens=52 head=2 span=1 blk=1", 52, 2, I8, num_blocks=1773, block_size=1)
ok &= run_case("int8  tokens=32 head=4 span=2 blk=4", 32, 4, I8, num_blocks=16, block_size=4, num_head=2)
ok &= run_case("fp16  tokens=20 head=8 span=1 blk=8", 20, 8, F16, num_blocks=8, block_size=8)
ok &= run_case("bf16  tokens=20 head=8 span=1 blk=8", 20, 8, BF, num_blocks=8, block_size=8)
ok &= run_case("int8  tokens=17 非整除 head=3 span=5", 17, 3, I8, num_blocks=12, block_size=5)
ok &= run_case("int8  含负 slot(-1) 跳过 head=2", 24, 2, I8, num_blocks=8, block_size=4, holes=True)

# 退化/边界：空组、负索引、group_len=0
torch.manual_seed(11)
ki = torch.arange(12, dtype=torch.int32, device=DEV).to(torch.int8).reshape(6, 2)
c0 = torch.zeros(8, 2, dtype=torch.int8, device=DEV) + 5
glen = torch.tensor([2, 0, -3, 2], dtype=torch.int32, device=DEV)
gsrc = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEV)
gdst = torch.tensor([0, 1, -1, 4], dtype=torch.int32, device=DEV)
c = c0.clone()
store_kv_block(ki, c, glen, gsrc, gdst, 4)
torch.npu.synchronize()
r = ref_grouped(ki, c0, glen, gsrc, gdst, 2)
edge_ok = torch.equal(c.detach().cpu(), r.detach().cpu())
emit(f"  {'边界：len=0/len<0/dst<0 全跳过':52s} {'PASS' if edge_ok else 'FAIL'}")
ok &= edge_ok

# 调用点风格：scatter 语义等价（indexer.py 里非 reshape-optim 路径就是 scatter_cache）
emit()
emit(f"总体: {'全部 PASS' if ok else '存在 FAIL'}")
emit("ALLDONE")

txt = "\n".join(L)
try:
    open("/root/dsh_work/store_kv_block_test.txt", "w", encoding="utf-8").write(txt + "\n")
except OSError:
    pass
print(txt)
