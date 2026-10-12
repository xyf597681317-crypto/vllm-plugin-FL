# sparse_attn_sharedkv TND+PA 数值错误 —— 交接材料（fix-saskv，2026-10-12）

## 0. 一句话状态
**未修好**。`总体: 存在 FAIL`（17 PASS / 9 FAIL），失败的 4 个 TND+PA 用例
成因**未指认到具体代码环节**。下面是我已排除的和未排除的，以及可直接复用的工具链。

## 1. 当前算子文件状态（已是"改动最小且无回归"的形态）
文件：`_analysis/triton_ops/sparse_attn_sharedkv.py`（本地 md5 `7188d0f1a0af852e2b9b8470935a1a01`）

改动只有两处，**数学完全没动**：
1. `_row_of` → 拆成 `_kv_elem_off`（只接收"已保证合法"的行号，内部保留 min/max 兜底）。
2. 两个 kernel 的滑窗/cmps 索引改为 **Lead 建议的逐 lane `tl.where` 消除负下标**：
   ```python
   j   = base_o + lane
   vld = (j >= 0) & (j < kvlo)
   rw  = tl.where(vld, j, 0)        # 负值不进入取址
   ```
   稀疏 cmp：`vld = (i0 >= 0) & (i0 < min(kvlc, n_valid))`，`tl.where(vld, i0, 0)`。

**分块完全未动**：`blk_d_out=128`（`SASKV_BLK_D_OUT`）、`blk_k=8`（`SASKV_BLK_K`）保持，
本轮所有编译/运行都通过，**没有 UB 溢出回归**。

实测（本轮真单测）：
```
总体(BSND 等价性): 全部 PASS        ← BSND 8/8 + 调用点 7/7 稳定全 PASS
  TND+PA fp16      err=1.455e-01   FAIL
  TND+PA bf16      err=3.651e-01   FAIL
  TND+PA N1=8 无sink err=1.098e-01  FAIL
  TND+PA cmp_ratio=4 err=nan        FAIL
总体: 存在 FAIL
```
与任务描述里的原始数字（1.463e-01 / 3.651e-01 / 1.093e-01）**基本一致** ⇒
**我这两处改动对 TND+PA 的数值没有影响**，即：既没修好，也没弄坏。

## 2. 最关键的实测事实（比错误数字有用）
用测试夹具同 seed 跑算子 + 测试自带参考，**按 batch 看误差**：
```
t=0..3  err=nan（全 512 维都是 nan）
t=4     err=1.131e-04   ← 正确
t=5..7  err=nan
t=8     err=1.163e-04   ← 正确
```
`t=0..4` 属于 batch 0（kvolen=64），`t=5..8` 属于 batch 1（kvolen=48）。
⇒ **batch 1（每个 batch 的最后一个 token）正确，其余 token 全 nan**。
⇒ 不是"整个布局全错"，而是**特定 token 的 out kernel 结果全 nan**（acc 或 lse 变 nan）。

## 3. 已用设备实验排除的（都是逐 lane 精确对照，不是推断）
| 假设 | 实验 | 结论 |
|---|---|---|
| 负下标 clamp 没生效 | 4 种窗口写法（shift+minmax / 常量分支 / where / max）+ 运行时 base | **全部正确**，不是根因 |
| `_kv_elem_off` 元素偏移算错 | dump off 数组 vs 主机期望 | 逐项一致 |
| 2D tile load `KV+off[:,None]+d[None,:]` 错 | 真实块表 + 负 base + BLK_K=8 分块 + 128 窗口，与设备端 ground-truth 逐 lane 比 | **0 不符** |
| score 计算 `tl.sum(kv*q,axis=1)` 错 | 干净复刻 score kernel（一次运行 dump ECHO/SC/LANE 三路） | **0 错** |
| SB 写入丢 lane | 同上（ECHO 与 LANE 两路都精确） | **0 错** |
| 块表读取 `BT + b*MB + bidx` 错 | 3 种取法对比 | 行为符合语义 |
| NaN 来自 clamp | 撤掉 clamp 前后数值不变 | 不是（与任务描述一致） |

## 4. 定位到但未解释的偏差形态（下一手从这里接）
用 `SASKV_DUMP_SCORES=1` 取算子内部 SB，逐列对参考（t=0, b=0, s1=0,
kvolo=64, kvolc=24, n_valid=24）：
```
ori 段: nan=0  数值错=4   首个错 lane=68 j=0  got=71.51546  exp=0.56019
cmp 段: nan=0  数值错=8
   cmp 原始 idx = [0, 8, 16, 23, 4, 15, 21, 17]
   SB 里读到的  = [-1.0140, 0.2935, 0.1260, -0.1558, -0.2611, -0.0286, 0.3634, -0.0167]
   逐 idx 期望   = [ 0.2935, 0.1260, -0.1558, -0.2611, -0.0286, 0.3634, -0.0167, -0.4828]
```
形态：**cmp 段整体"前移一格"**（`SB[slot] ≈ q·dd[idx[slot+1]]`，slot1..7 有 6 个精确吻合），
**ori 段只有紧邻第一个有效 lane 的少数列错**（lane68 量级差 128 倍，lane69+ 全部精确）。

注意：我的"干净复刻 score kernel"探针里这两处都是 **0 错**。两者唯一的结构差别是
**复刻版用 1 维块表 `bt=[1,2,3,0]`，真算子用 2 维 `b*MAX_BLOCKS + bidx`**。
所以下一手最该查：**score kernel 在"2 维块表 + 每 token 不同的 b"下，
是否只有每 batch 的第一个 token 取到了错位的行**（这与第 2 节的
"只有最后一个 token 对"高度呼应——`t=0` 恰好是 `base_o` 最负的那个 token）。

## 5. 两条可复用的后端"坑"（实测，建议写进项目结论）
1. **被 mask 的地址会真实访问**：不带 mask 的 tile load 直接触发
   `507035 vector core exception`，设备日志 "The DDR address of the MTE instruction
   is out of range"。⇒ 非法行号**绝不能**进入 `KV + off[:,None]`，mask 挡不住地址。
2. **不要用 0 维 tensor 当运行时标量传参**：`tl.where(BASEV < 0, ...)` 会报
   `invalid operands of type pointer<int32> and triton.language.int32`
   （0 维 tensor 在这个后端被当成 pointer）。用 Python int / constexpr。
   同理 `tile[:, 0]` 切片不支持（`unsupported tensor index: constexpr[0]`），
   要用 `tl.sum(tile * mask[None,:], axis=1)` 之类替代。
3. **就地 `tl.maximum(j, 0)` 之后再用 `j` 做除法/取模不可靠**（旧 `_row_of` 的写法），
   可靠做法是让负值根本不进取址（现在文件里就是这个形态）。

## 6. 工具链（下次直接用，省 1 小时）
### 部署（三跳 + 每跳 md5）
```powershell
# 1) 上传到跳板机 relay 目录
python _analysis/handoff_1011/up.py <本地文件> /root/dsh_work/handoff_1011/relay/<文件名>
# 2) 跳板机 → 114 → 容器两处（src + site-packages），自带 md5 核对
python _analysis/q.py "SASKV_FILES='<文件名>' bash /root/dsh_work/handoff_1011/deploy_relay.sh"
```
⚠️ `_analysis/q.py` 只到跳板机；容器在 **114 的第二跳**，必须用
`ssh root@172.16.10.114 "docker exec flaggems114 ..."`（见 `saskv_*.sh` 的 heredoc 写法）。
⚠️ **本地文件 md5 与容器 md5 不一致是常态**：`up.py` 走 base64，容器里是 LF；
本地如果是 CRLF 会差。以 `deploy_relay.sh` 打印的三方 md5 为准。

### 跑单测（结果在容器内 /tmp/t_saskv.log，判定看 `总体:` 行）
```bash
python _analysis/q.py 'bash /root/dsh_work/handoff_1011/saskv_launch.sh'   # 后台启动
python _analysis/q.py 'bash /root/dsh_work/handoff_1011/saskv_poll.sh'     # 轮询（约 3 分钟出结果）
```

### 跑探针
```bash
python _analysis/q.py 'bash /root/dsh_work/handoff_1011/probe_run.sh <文件名>'        # 前台（易被 q.py 超时切断）
python _analysis/q.py 'bash /root/dsh_work/handoff_1011/saskv_deploy.sh <文件名>; bash /root/dsh_work/handoff_1011/probe_bg.sh <文件名> 40'
python _analysis/q.py 'bash /root/dsh_work/handoff_1011/probe_poll.sh <文件名>'
```
**重要**：输出一定落到容器内 `/tmp/probe_<名>.log`，用 `probe_poll.sh` 读；
`q.py` read() 有 120s 上限，长探针会被切断，必须后台跑。

### 探针清单（都在 `_analysis/triton_ops/`，都属于我的写范围）
| 文件 | 用途 |
|---|---|
| `_probe_saskv_ab.py` | ori / cmp 分支单独跑（A/B） |
| `_probe_saskv_diag.py` | SB vs 参考、逐列 | 
| `_probe_saskv_map.py` | 身份块表 vs 随机块表 |
| `_probe_saskv_2d.py` / `_2d2.py` | 最小 2D gather + off/page/row dump |
| `_probe_saskv_gt.py` | **设备端 ground truth**（不依赖 CPU 参考） |
| `_probe_saskv_tile.py` / `tile2.py` | 2D tile load 形态对比 |
| `_probe_saskv_store.py` | ECHO/SC/LANE 三路区分 store vs compute |
| `_probe_saskv_sb.py` | **算子内部 SB vs 参考 logits（最有信息量）** |
| `_probe_saskv_vsref.py` | 算子 + 测试自带参考，逐 token 误差 |
| `_probe_saskv_rt.py` / `iso2.py` / `op.py` / `blk.py` / `real.py` / `bt.py` | 各种隔离实验 |

## 7. 建议的下一步（按性价比排序）
1. **在 score kernel 里 dump `b`、`base_o`、`js`（每个 token 一个值），看 `t=0..4` 是否取了错行。**
   一次运行即可判定（`_probe_saskv_idx.py` 已有雏形，但它无 mask 触发过 507035，
   记得给 tile load 加 `vld[:,None] & dm[None,:]`）。
2. 若 score 正确而 out 全 nan：**在 out kernel 里 dump `mx`/`lse`**，看是不是
   `lse` 被写成 0/负导致 `acc/lse` nan（注意 `res = tl.where(lse > 0.0, acc / lse, 0)`，
   除零在 Ascend 上可能污染结果）。
3. 若确认是"每 batch 第一个 token 错"：试**把 `b` 从 kernel 内的 `tl.load(TOK_B+t)`
   改成 grid 轴**（或按 batch 分段启动 kernel），绕开"运行时标量 b 参与取址"。

---

# 第二轮进展（fix-tndpa，2026-10-12 17:24~17:55）

## 8. 一句话结论
**已修好。** 根因是 **Ascend 后端上「2-D tile + 间接寻址」的 load/store 在窗口里
*最早几个有效 lane* 上会丢写/读脏**（与 `BACKEND_NOTES.md` 坑 1 同族）。
修法：两个 kernel 的 KV 访问全部改成**逐行 1-D load + 广播相乘**
（即坑 1 的「可靠写法」），数学与分块完全不动。

单测：`总体: 全部 PASS`（BSND 8/8 + 调用点 7/7 + TND+PA 4/4），
新文件 md5 `25d48acca7d1e1ca3c0eae18ba969943`（旧 `7188d0f1a0af852e2b9b8470935a1a01`）。

## 9. ⚠️ 先纠正第一轮第 2 节的一个结论（重要）
第一轮写「batch0 除最后一个 token 外**全 512 维都是 nan**」。第二轮用
`_probe_saskv_judge.py`（同进程三路：Triton / AscendC / 测试参考）复跑同一夹具后实测：

| 用例 | nan 数 | 说明 |
|---|---|---|
| TND+PA fp16 | `nan=3584` | = 7 token × 4 head × 512 维（t=4、t=8 两行好） |
| TND+PA bf16 | `nan=0` | **bf16 一个 nan 都没有**，只有 err=3.65e-01 |
| TND+PA N1=8 无 sink | `nan=0` | 同上 |
| TND+PA cmp_ratio=4 | `nan=0` | 同上 |

⇒ **"全 512 维 nan" 只在 fp16 那一条出现，不是普遍形态**；真正的普遍形态是
"**每行的误差量级≈参考值本身**"（即整行基本没算对），bf16 下表现为纯错值而非 nan。
（`_probe_saskv_loc.py` 逐行打印也证实：`out_nan=0`，`out|max|` 在 1e-6~0.3 之间跳。）

## 10. 决定性证据：错的不是"行号"，而是"写不进去"
`_probe_saskv_errmap.py`（逐 (t,n1) 误差 + SB 逐 lane 对**设备张量独立复算**，非测试参考）：

```
 t n1 |      err      out_nan  ref_absmax     t n1 |      err
 0  0 | 2.4350e-01        0   2.4350e-01      4  0 | 9.1656e-05   ← 正确
 0  1 | 2.4221e-01        0   2.4221e-01      4  1 | 1.1308e-04   ← 正确
 0  2 | 5.2288e-02        0   3.0791e-01      8  0 | 8.6107e-05   ← 正确
 0  3 | 3.0420e-01        0   3.0420e-01      8  2 | 1.1627e-04   ← 正确
 1..3 | 2.6e-01~3.3e-01  0   ~3e-01           5..7 | 5.6e-03~3.2e-01
```
**err ≈ ref_absmax** ⇒ 那些行输出接近 0/无关值（不是"差一点"）。

SB 逐 lane（`_probe_saskv_errmap.py`，与**设备张量在主机上 fp64 复算**对照）：
```
t=0 n1=0: ori base_o=-68 js0=0 n_js=60  ori_mismatch=4
      lane=68 j=0 got=27.924423 exp=0.560187 ratio=49.85
      lane=69 j=1 got=14.330708 exp=1.323371 ratio=10.83
      lane=70 j=2 got=-65.849411 exp=0.630235 ratio=-104.48
      lane=71 j=3 got=-55.575199 exp=0.335249 ratio=-165.77
      cmp_mismatch=0/8        ← cmp 段 8/8 全对
t=4 n1=0: ori base_o=-64 n_js=64  ori_mismatch=0   cmp_mismatch=0/8   ← 全对
t=5 n1=0: ori base_o=-83 n_js=45  ori_mismatch=3
      lane=85 j=2 got=-1.783840 exp=0.590071
      lane=86 j=3 got=43.868546 exp=1.378420
      lane=87 j=4 got=25.845211 exp=-0.478840
      cmp_mismatch=0/8
t=8 n1=0: ori base_o=-80 n_js=48  ori_mismatch=0   cmp_mismatch=0/8   ← 全对
```
⇒ 形态固定：**ori 段"窗口里最早的有效行"那 3~4 个 lane 错，其余（含 cmp 段全部）精确**。

`_probe_saskv_which.py`（把这些 lane 的值与**全部候选 (kv 张量, 页, 块内行, head)** 的
q·kv 逐一比对）给出：
```
lane=68 j=0 got=  45.241875 exp= 0.560187 hits=[]     ← 不在任何合法行里
lane=72 j=4 got=  -0.317138 exp=-0.317138 hits=[('ori_raw',3,4,0)]  ← 精确
lane=83 j=0 got=   0.161158 exp= 0.161158 hits=[('ori_raw',7,0,0)]  ← 精确
lane=85 j=2 got=   0.000000 exp= 0.590071 hits=[]     ← **0.0**
lane=88 j=5 got=   0.095035 exp= 0.095035 hits=[('ori_raw',7,5,0)]  ← 精确
```
两条硬结论：
1. 坏 lane 的值**不在任何合法行里**（`hits=[]`）⇒ 不是"取错行"（第一轮第 4 节的
   "前移一格 / 取错行"假设被否证）；
2. 同一份 SB 里出现 **`got=0.000000`** ⇒ **写入被丢掉**（`torch.empty` 或未写入），
   而不是算错。同一 `(rw, off)` 在其它 block 位置（如 lane 68 与 lane 4）完全正常
   ⇒ 与"地址算错"无关，是**这条 2-D 间接寻址 tile 通路本身在特定 lane 上不可靠**。

## 11. 根因 / 修法
**根因**：`tl.load(OKV + off[:, None] + d[None, :], mask=vld[:, None] & dm[None, :])`
与对应的 2-D 掩码 store —— `off` 由**运行时标量行号**经块表间接寻址算出
（`_kv_elem_off`），在 Ascend 后端上属于 `BACKEND_NOTES.md` 坑 1 的
「2-D tile + 间接寻址」组合：**窗口里最早几个有效 lane 会被丢写/写脏**。
（注：坑 1 已记录"只有 2-D 掩码 + 长值链、没有间接寻址"不触发；
本轮是**有**间接寻址、且命中位置固定在窗口起始处，属同一族。）

**修法**（`sparse_attn_sharedkv.py`，新增 `ROW_MODE` constexpr，默认 1）：
```python
for c in range(0, tl.cdiv(N_ORI, BLK_K)):
    for i in range(BLK_K):
        li = c * BLK_K + i
        ji = base_o + li
        vi = (ji >= 0) & (ji < kvlo) & (li < N_ORI)
        off_i = _kv_elem_off(OKV, BTO, b, tl.where(vi, ji, 0), n2, ...)
        kvi = tl.load(OKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
        tl.store(SB + row * total + li, tl.where(vi, tl.sum(kvi * q, axis=0) * SCALE, NEG),
                 mask=li < N_ORI)
```
out kernel 同理，把 `acc += tl.sum(pc[:,None] * kv, axis=0)` 改成
`acc += tl.where(vi, tl.exp(pci - mx), 0.0) * kvi`（**纯 1-D**，不再有 `[BLK_K, BLK_D]` tile）。
- **分块未动**：`blk_d_out=128` / `blk_k=8` 保持，无 UB 回归。
- **数学未动**：`vi` 为真时 `ji == base_o + li`，逐 lane 语义与原来逐位相同。
- `SASKV_ROW_MODE=0` 可切回 2-D tile 形态做 A/B（仅调试）。

## 12. 验收（本轮实测）
```
总体(BSND 等价性): 全部 PASS
  TND+PA fp16        err=1.187e-04 (<= 1e-03)  PASS
  TND+PA bf16        err=9.738e-04 (<= 4e-03)  PASS
  TND+PA N1=8 无sink err=1.220e-04 (<= 1e-03)  PASS
  TND+PA cmp_ratio=4 err=1.214e-04 (<= 1e-03)  PASS
  （BSND 8/8 + 调用点 7/7 全部 PASS，无回归）
总体: 全部 PASS
```
跑法：`bash /root/dsh_work/tnd/tnd_test.sh npu:2 1800` + `tnd_tpoll.sh`（容器内 `/tmp/tnd_test.log`）。
修前同命令为 `err=1.455e-01 / 3.651e-01 / 1.098e-01 / nan`。

**连跑 2 次的原始数字（两次逐位一致）**：
- 第 1 次 `npu:2`（`tnd_test.sh`）：1.187e-04 / 9.738e-04 / 1.220e-04 / 1.214e-04，`总体: 全部 PASS`
- 第 2 次 `npu:3`（`tnd_test2.sh`）：1.187e-04 / 9.738e-04 / 1.220e-04 / 1.214e-04，`总体: 全部 PASS`
两次都同时含 `总体(BSND 等价性): 全部 PASS` 与 `调用点风格合计: PASS`。

## 12b. 性能（副作用是大幅变快，非回归）
`_probe_saskv_perf.py`（TND+PA，fp16，先 warmup 再取 3 次均值）：
```
== 测试形状 B2 S1=5/4 N1=4 D=512 ==     ROW_MODE=1:   1.64 ms/call | ROW_MODE=0:   5.62 ms/call
== 放大  B8 S1=64  N1=4 D=512 ==     ROW_MODE=1:  14.98 ms/call | ROW_MODE=0: 327.76 ms/call
```
⇒ 逐行 1-D 形态不但正确，**还比原来快 3.4×（小形状）/ 22×（放大形状）**：
原来的 `[BLK_K, BLK_D]` tile + `tl.sum(..., axis)` 归约在这个后端上代价极高。
（`ROW_MODE=0` 那一列顺便证明：旧形态在放大形状下**不再报错、也不崩**，
只是数值错 —— 与第 10/11 节的"静默丢写"判定一致。）

## 13. 本轮新增的工具与探针（都可复用）
部署/跑测（跳板机 `/root/dsh_work/tnd/`）：
| 脚本 | 用途 |
|---|---|
| `tnd_deploy.sh <名字...>` | relay → 114 → 容器 src + site-packages，四方 md5 核对 |
| `tnd_run.sh <py> [dev] [timeout] [env]` | 容器内后台跑探针，日志 `/tmp/tnd_<名>.log` |
| `tnd_poll.sh <py>` | 取探针状态+日志 |
| `tnd_test.sh [dev] [timeout]` / `tnd_tpoll.sh` | 跑真实单测 |
| `tnd_test2.sh [dev]` | 第二次连跑（不同卡） |
| `tnd_pull.sh` | 从容器 base64 取回文件（**文件被本地工具弄坏时的救命通道**） |

探针（`_analysis/triton_ops/`）：
| 文件 | 用途 |
|---|---|
| `_probe_saskv_judge.py` | **同进程三路**：Triton / AscendC / 测试参考，逐 token nan 与误差 |
| `_probe_saskv_loc.py` | SB 逐 token 统计 + out 逐 token nan/absmax（定位 nan 来源） |
| `_probe_saskv_errmap.py` | 逐 (t,n1) 误差地图 + SB 逐 lane 对**设备张量独立复算** |
| `_probe_saskv_which.py` | 把坏 lane 的值与全部候选 (张量,页,行,head) 比对，判定"取错行"还是"没写入" |
| `_probe_saskv_perf.py` | `ROW_MODE` 1/0 性能对照 |
| `_probe_saskv_env.py` / `_env2.py` / `_so.py` | 设备占用、`_C_ascend` 算子清单、.so 探查 |

## 14. 三条新踩的后端/工具坑（建议进 BACKEND_NOTES）
1. **容器里没有 AscendC 裁判**：`dir(torch.ops._C_ascend)` == `['name']`，
   `torch.ops._C_ascend.npu_sparse_attn_sharedkv` 抛 `AttributeError`；
   本机也没有源码 op（`/vllm-workspace/vllm-ascend/csrc/` 无 `attention/sparse_attn_sharedkv/`）。
   ⇒ **"用 AscendC 同输入裁判"这条路在本环境走不通**，别再花时间；
   判定只能靠"设备张量独立复算"（`_probe_saskv_errmap.py` 的做法）。
2. **`SASKV_BLK_K=1` 让编译器直接崩**：`arith.maxsi` + `AddPtrOp` →
   `BlockPtrAnalysis.cpp:494 UNREACHABLE executed`（RC=134，core dumped）。
   `BLK_K=8` 正常。⇒ 调 `BLK_K` 时不要低于 8。
3. **不要用 PowerShell 读写这个仓库的 .py**：`Get-Content -Raw | Set-Content`
   会把 UTF-8 中文注释打成乱码（文件直接无法 decode）。
   本次已从容器 base64 取回原文件修复（`tnd_pull.sh` + `tnd_restore.py`）。
   一律用编辑工具改文件。

