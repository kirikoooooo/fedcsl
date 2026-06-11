# §7.7 显存约束尺度选择实验 — 执行手册 (RUNBOOK)

> 本文件是**给执行者(可能是能力较弱的模型或人)照着做的操作手册**。
> 每一步都给了：要运行什么、预期看到什么、出错怎么办、做完怎么自检。
> 不要跳步。遇到与「预期」不符时，**停下来报告**，不要猜测性地继续。

---

## 0. 这个实验在干什么(一句话)

论文 §7.7 把弱设备的 shapelet 尺度选择建模为 **0-1 背包**：在显存预算 $G_k$ 下，
最大化本地周期评分之和 $\sum_{r\in\mathcal{R}} s_r$，约束 $g_0+\sum_{r\in\mathcal{R}} g_r \le G_k$。
本实验要：
1. **实测**每个尺度 $r$ 的显存增量 $g_r$（GPU 上跑）；
2. **拟合**组合显存预测函数 $\widehat{\mathrm{Mem}}(\mathcal{R})=g_0+\sum g_r$ 并验证它准不准；
3. 用**背包 DP** 在各显存预算档位求最优尺度组合，和纯 top-m 对照；
4. 把数值**回填**到论文 §7.7 的表格。

符号对照（与论文一致）：$R=8$ 个尺度；$g_r$=激活尺度 $r$ 的峰值显存增量(MB)；
$g_0$=与尺度无关的固定开销；$s_r$=本地 STFT/ACF 周期评分(价值)；$G_k$=设备显存预算。

---

## 1. 前置条件检查（做之前先确认）

| 检查项 | 命令 | 预期 |
|---|---|---|
| 有 GPU | `nvidia-smi` | 列出至少一张卡，有空闲显存 |
| 在项目根 | `ls FedCSL_All.py train.py blocks.py` | 三个文件都在 |
| 脚本存在 | `ls scripts/system_efficiency/measure_scale_memory.py scripts/system_efficiency/fit_scale_memory.py scripts/system_efficiency/run_scale_memory.sh` | 三个都在 |
| Python 能 import torch | `python -c "import torch; print(torch.cuda.is_available())"` | 打印 `True` |
| HAR 数据可加载 | 见下方 step 2 第一次运行时会自动加载 | 日志出现 `N=..., C=9, T=128` |

**如果 `torch.cuda.is_available()` 是 `False`**：本实验测的是 GPU 峰值显存，CPU 上跑没有意义。
停下来，报告"无可用 GPU，无法标定 $g_r$"。不要继续。

---

## 2. 第一步：标定每个尺度的显存 $g_r$（GPU）

### 运行

```bash
# 用 GPU 0，10 个客户端，alpha=0.1，batch=32，并验证 3 个组合的可加性
GPU=0 NUM_CLIENTS=10 ALPHA=0.1 BATCH_SIZE=32 \
VERIFY_SUBSETS="0,1;2,4,6;0,3,7" \
bash scripts/system_efficiency/run_scale_memory.sh
```

这一条命令会**自动跑完两步**（标定 + 拟合 DP）。下面分别讲两步各自的预期。

### 标定阶段预期输出

日志里应出现 8 行，每个尺度一行，类似：

```
  [scale 0] len=  12  mean=XX.XMB  max=XX.XMB
  [scale 1] len=  24  mean=XX.XMB  max=XX.XMB
  ...
  [scale 7] len= 102  mean=XX.XMB  max=XX.XMB
```

**自检点 A**：
- 8 个尺度的 `mean` 都应是**正数**且在**几 MB ~ 几十 MB**量级（取决于 batch 和 scale_aux）。
- 显存**不一定随尺度长度单调**：中等尺度(scale 2/3，len 37/50)可能最高，最长尺度(scale 7，len 102)可能反而较低。这是**正常的、预期的**（滑窗数 $W=L-\ell+1$ 随长度变小抵消了参数增长）。
- 如果某个尺度的 `mean` 是 0 或 NaN → 该尺度测量失败，去看上面有没有 `[err]` 堆栈，报告它。

### 产物

```
data/scale_memory_HAR_partials/per_scale.json
```

**自检点 B**：确认文件存在且非空：
```bash
python -c "import json; d=json.load(open('data/scale_memory_HAR_partials/per_scale.json',encoding='utf-8')); print('R=',d['R'],'尺度数=',len(d['per_scale']),'verify=',len(d.get('verify_subsets',[])))"
```
预期打印 `R= 8 尺度数= 8 verify= 3`。

---

## 3. 第二步：拟合 + 背包 DP（自动接在第一步后，无需 GPU）

`run_scale_memory.sh` 会自动调用 `fit_scale_memory.py`。预期日志：

```
  [budget 64MB] DP选中=[...] 价值=... 显存=...MB | top4=[...] 显存=...MB OK/OOM
  [budget 128MB] ...
  [budget 256MB] ...
[ok] 拟合 + 背包 DP 完成
     可加性 MAE = X.XX MB, max rel err = X.XXX
```

### 产物

```
data/scale_memory_HAR.json   # 机读：g0, g_r, 可加性误差, DP 结果
data/scale_memory_HAR.md     # 可读报告（直接可看的表格）
```

### 自检点 C —— 可加性误差(最关键)

打开 `data/scale_memory_HAR.md`，看「可加性验证」一节的 **MAE** 和 **最大相对误差**。

- **MAE < 5 MB 且 最大相对误差 < 0.10（10%）**：可加模型成立，方案有效。继续。
- **误差较大（相对误差 > 15%）**：说明 $g_0+\sum g_r$ 这个可加近似不够准。
  **不要假装没看见**。报告这个误差，并说明：可能需要给模型加修正项，或重新审视
  "stitched 子模型让各尺度激活同时驻留"这一可加性假设是否被实现细节破坏。
  （这是科学发现，不是失败——如实报告即可。）

### 自检点 D —— DP vs top-m 的对照

在 DP 选择表里确认：
- 当某档预算下 top-4 显示 **OOM**（超预算）时，DP 那一行应给出**预算内可行**的尺度组合（显存 ≤ 预算）。
- DP 的显存列**永远不超过**该行的预算 $G_k$。若超了 → DP 实现有 bug，报告。

---

## 4. 第三步：注入真实评分(可选但推荐)

上面第二步缺省用**占位评分**(演示用)。论文里 DP 的价值应是**客户端真实的本地周期评分 $s_r$**。

### 怎么拿到真实 $s_r$
真实评分来自 §周期感知模块的 STFT/ACF 评分(`train.py` 里 `Cached_Scale_Scores` / pscore)。
如果你能从某个客户端导出一组长度为 8 的评分向量，用 `--scores` 注入：

```bash
python scripts/system_efficiency/fit_scale_memory.py \
  --partial data/scale_memory_HAR_partials/per_scale.json \
  --budgets 64,128,256 --topm 4 \
  --scores 0.42,0.31,0.05,0.03,0.02,0.08,0.04,0.05   # ← 换成真实的 8 个评分
```

**如果暂时拿不到真实评分**：跳过这步，用占位评分的结果先把管线跑通，并在回填论文时**明确标注**"评分为占位/示意"。不要把占位评分当成真实结果写进论文结论。

---

## 5. 第四步：回填论文 §7.7（人工或交给更强的模型)

论文文件：
`论文撰写/论文overleaf[不含omp，含spilter].tex`

要回填两处：

### (a) 表 `tab:scale_mem_har`（每尺度显存）
当前是**解析估计值**。用 `data/scale_memory_HAR.json` 里的实测 `g_r_mb` 替换，
并把表注从"解析显存估计"改成"实测峰值显存"。

### (b) 表 `tab:resource_hetero`（资源不平衡对比）
当前数值是 `---`(待补)。用 DP 结果填入：各档预算下 DP 选中组合、显存、参与率等。

### 回填后重新编译 PDF
```bash
cd 论文撰写
# 用 tectonic（本机已验证可用）
"$USERPROFILE/.codex/.tmp/bundled-marketplaces/openai-bundled/plugins/latex/bin/tectonic.exe" \
  --keep-intermediates --keep-logs "论文overleaf[不含omp，含spilter].tex"
```
预期：末尾出现 `Writing ...pdf`，生成 `论文overleaf[不含omp，含spilter].pdf`。

> 注意：`build_pdf.ps1` 经 PowerShell 传中文文件名有编码坑；直接用上面的 tectonic 命令更稳。

---

## 6. dashboard 查看(可选)

```bash
python -m dashboard.app          # 启动后端（需要 psutil/fastapi/uvicorn）
```
- 「脚本」面板能看到 `run_scale_memory.sh`，点击可启动；
- API：`/api/scale-memory/status`、`/api/scale-memory/report.md`、`/api/scale-memory/result`。

---

## 7. 环境变量速查（run_scale_memory.sh）

| 变量 | 默认 | 含义 |
|---|---|---|
| `GPU` | 0 | 用哪张卡 |
| `NUM_CLIENTS` | 10 | 客户端数（标定 $g_r$ 用少量即可） |
| `ALPHA` | 0.1 | Dirichlet 异质度 |
| `BATCH_SIZE` | 32 | 与 `config/configSpilter.yml` 对齐 |
| `MAX_CLIENTS` | 0 | 每尺度最多测几个客户端（0=全部） |
| `VERIFY_SUBSETS` | 空 | 分号分隔的尺度子集，验证可加性，如 `"0,1;2,4,6"` |
| `BUDGETS` | `64,128,256` | 背包预算档位(MB)，对应小/中/大显存设备 |
| `TOPM` | 4 | 对照的固定 top-m |
| `SCALE_AUX` | 1 | 1=开 per-scale CL/KD，0=关 |

---

## 8. 常见失败与处理

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 卡太小或被占用 | 换更空的卡 `GPU=N`，或减小 `BATCH_SIZE` |
| 某尺度 `mean=0/NaN` | 该尺度测量抛异常 | 看 `[err]` 堆栈，报告具体错误，别忽略 |
| 可加性 MAE 很大 | 可加假设不成立 | **如实报告**，可能需加修正项，不要硬凑 |
| DP 显存 > 预算 | DP 实现 bug | 报告，停下 |
| `找不到标定结果` | 第一步没成功 | 先确认 `per_scale.json` 存在 |
| 中文乱码（控制台） | 终端码页 | 只是显示问题，文件内容是对的，看 `.md`/`.json` 即可 |

---

## 9. 绝对不要做的事

- ❌ 不要修改 `measure_har_epoch.py` / `aggregate_results.py` / `run_har_efficiency.sh`
  （那是 §7.6 已发表实验，本实验与它独立）。
- ❌ 不要把占位评分的 DP 结果当真实结论写进论文。
- ❌ 可加性误差大时不要假装通过——如实报告。
- ❌ 没有 GPU 不要硬跑标定（CPU 显存数无意义）。

---

## 10. 完成标准(Definition of Done)

全部满足才算完成：
- [ ] `data/scale_memory_HAR_partials/per_scale.json` 存在，8 个尺度 $g_r$ 都是有效正数；
- [ ] `data/scale_memory_HAR.md` 存在，可加性 MAE 和相对误差已报告；
- [ ] DP 选择表中每行显存 ≤ 对应预算；
- [ ] （若能拿到真实评分）已用 `--scores` 注入；否则已标注"占位评分"；
- [ ] （回填阶段）论文两表已更新且 PDF 重新编译成功。

执行完把以上清单逐项打勾，连同 `scale_memory_HAR.md` 的关键数字一起报告。
