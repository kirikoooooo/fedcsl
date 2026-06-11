# HAR 系统效率实验

测量 HAR_results 表里每个联邦算法的**最慢客户端 1 个 epoch 耗时**，
作为同步联邦协议的 round-time 下界代理。

## 设计要点

| 要求 | 实现 |
| --- | --- |
| 串行计算 1 epoch | `measure_har_epoch.py` 内对客户端逐个 `client.fit/train`，不并发 |
| 覆盖所有 HAR_results 算法 | FedAvg、FedProx、FedBYOL、FedU2、Orchestra、FedCSL、Spilter-m1/m2/m4 |
| 单 GPU 独占 | `run_har_efficiency.sh` 通过 `CUDA_VISIBLE_DEVICES=$GPU` 锁定一张卡；同一时刻只跑一个算法进程，进程内同一时刻只跑一个客户端 |
| 输出表格放 `data/` | 汇总后生成 `data/system_efficiency_HAR.{json,csv,md}` |

## 文件清单

- `measure_har_epoch.py` — 单算法测量进程：对 K 个客户端串行测时，写
  `data/system_efficiency_HAR_partials/<algo>.json`。
- `aggregate_results.py` — 把所有 partial 合并为最终报告。
- `run_har_efficiency.sh` — 一键串行跑全部算法 + 汇总（dashboard 自动可见的入口）。

## 用法

### 1. dashboard 启动（推荐）

打开 dashboard 后端：

```bash
python -m dashboard.app
```

在前端「脚本」面板里能看到 `scripts/system_efficiency/run_har_efficiency.sh`，
点击即可启动。环境变量在「附加环境变量」里加（如 `GPU=1`, `NUM_CLIENTS=30`）。

跑完后调用：

- `GET /api/system-efficiency/status` 查看产物是否就绪；
- `GET /api/system-efficiency/result` 返回机读 JSON；
- `GET /api/system-efficiency/report.md` 返回可读 markdown。

### 2. 命令行直接跑

```bash
# 默认全部算法、GPU 0、K=50、alpha=0.1、batch=8
bash scripts/system_efficiency/run_har_efficiency.sh

# 自定义：用 GPU 1、alpha=0.5、K=30、子集算法
GPU=1 ALPHA=0.5 NUM_CLIENTS=30 \
ALGOS="fedcsl spilter-m1 spilter-m2 spilter-m4" \
bash scripts/system_efficiency/run_har_efficiency.sh
```

### 3. 只补跑单个算法

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/system_efficiency/measure_har_epoch.py \
  --algo spilter-m2 --num-clients 50 --alpha 0.1 --batch-size 8

python scripts/system_efficiency/aggregate_results.py
```

## 输出字段

`data/system_efficiency_HAR.md` 表格列：

| 列 | 含义 |
| --- | --- |
| `Slowest Client Epoch (s)` | $\max_k T_k^{\text{epoch}}$（核心指标） |
| `#Samples@Slowest` | 最慢客户端的本地样本数（异质性诊断） |
| `Mean / Median / Min (s)` | 所有客户端 1-epoch 耗时统计量 |
| `Mean / Max Peak Mem (MB)` | 所有客户端 1-epoch 峰值显存的均值 / 最大值 |
| `Timed/Skipped` | 计时成功 / 因样本数 < batch 跳过 的客户端数 |

## Spilter 显存节省专题

同一次运行**额外**生成 `data/spilter_memory_HAR.md`，聚焦 Spilter 不同 `top-m`
与 FedCSL 全尺度的客户端**平均峰值显存**对比，含 `Saving vs FedCSL` 与
`Compression Ratio` 两列。指标定义：

```
\bar M(algo) = (1/K) Σ_k max_t |GPU_mem_k(t)|
Saving(m)   = 1 - \bar M(Spilter-m) / \bar M(FedCSL)
```

dashboard 端：

- `GET /api/spilter-memory/report.md` — 直接拿 markdown 报告
- `GET /api/spilter-memory/result` — 精简 JSON（仅 FedCSL + Spilter-m1/m2/m4）

## 注意

- 本实验在每个客户端开始前做 1 个 batch 的 cudnn warmup，避免首次卷积算子选择影响首个客户端计时；可通过 `WARMUP_BATCHES=0` 关闭。
- FedCSL / Spilter / FedProx 会构造一个 frozen teacher (server 初始权重副本)，模拟 round ≥ 1 的稳态训练成本。FedAvg、BYOL、FedU2、Orchestra 不需要 teacher。
- Spilter 的 `m1/m2/m4` 直接把 `Selected_Scales` 固定为 `[0..m-1]`，走 stitched；默认 **`selected_scales_only`**：`encode_mix_forward_selected_scales`（与 FedCSL `forward` 同结构的 LN→拼接，仅所选尺度），辅助损失复用。
- 实验默认只跑 1 个本地 epoch；若想测多个 epoch 的累积耗时，可设 `NUM_EPOCH=5` 等。
- **默认 `BATCH_SIZE=32`** 与所有 `config/configXxx.yml` 主流程实验对齐；如修改 yml 里的 `batch_size` 想测时与训练匹配，请显式 `BATCH_SIZE=N` 覆盖（脚本不读 yml）。

## Spilter 显存与 `SCALE_AUX`（更新说明）

默认 **`selected_scales_only`**：`encode_mix_forward_selected_scales`（FedCSL ``forward`` 同源顺序，LN 仅基于所选 m 个尺度）；与 **`forward_slices`**（全 8 尺度 LN 再切片）数值可能不同。

可选 **`forward_slices`**（全模 forward + 切片）用于与 FedCSL 表征严格对齐的对照实验。

测时脚本写入 `stitched_feature_source: selected_scales_only`，与 `config/configSpilter.yml` 一致。

**`SCALE_AUX=0`**（`--no-scale-aux`）：测时时关闭 `UseScaleCL` / `UseScaleKD`（FedCSL / Spilter 均受影响），观察去掉多尺度对齐项后的下界；**仅影响测时命令**。

```bash
SCALE_AUX=0 bash scripts/system_efficiency/run_har_efficiency.sh
```

---

# 尺度显存标定 + 背包 DP 选择（§7.7 支撑实验）

为论文 §7.7「系统资源不平衡场景」提供**显存约束尺度选择**的实测输入。
与上面的系统效率实验**完全独立**（不同脚本、不同 partial 目录、不同产物），
但共用 `LearningShapeletsCL` 客户端与同一套显存测量口径，结果可直接对比。

## 动机

§7.7 把弱设备的尺度选择建模为 0-1 背包：在显存预算 $G_k$ 下，
最大化本地周期评分 $\sum_{r\in\mathcal{R}} s_r$，约束 $g_0+\sum_{r\in\mathcal{R}} g_r \le G_k$。
其中每个尺度的显存增量 $g_r$ **并非随尺度长度单调**（中等尺度因滑窗数 $W_r=L-\ell_r+1$
与长度 $\ell_r$ 的乘积最大而显存最高），因此需要先**实测标定** $g_r$，
再对每种组合用可加模型预测显存，最后用 DP 求最优组合。

## 两步流程

1. **标定** `measure_scale_memory.py`：逐个测单尺度子模型 1-epoch 训练峰值显存，
   得到每个 $g_r$；可选 `--verify-subsets` 实测若干组合验证可加性。
   产物：`data/scale_memory_HAR_partials/per_scale.json`。
2. **拟合 + DP** `fit_scale_memory.py`：拟合 $\widehat{\mathrm{Mem}}(\mathcal{R})=g_0+\sum_{r\in\mathcal{R}} g_r$，
   报告可加性误差（MAE / 最大相对误差），在各显存预算档位下用背包 DP（`O(R\cdot G)` 精确解）
   求最优尺度组合并与纯 top-m 对照。产物：`data/scale_memory_HAR.{json,md}`。

## 用法

```bash
# 一键：标定 + 拟合 + DP（默认 HAR、GPU 0、K=10、batch=32、预算档 64/128/256 MB）
bash scripts/system_efficiency/run_scale_memory.sh

# 换数据集（与主流程一致：HAR / Epilepsy-TSTCC / SleepEDF / FD-A / 任意 UEA 名）
DATASET=FaceDetection bash scripts/system_efficiency/run_scale_memory.sh
DATASET=Epilepsy-TSTCC GPU=1 bash scripts/system_efficiency/run_scale_memory.sh

# 带可加性验证（实测 3 个组合与可加预测对比）
VERIFY_SUBSETS="0,1;2,4,6;0,3,7" bash scripts/system_efficiency/run_scale_memory.sh

# 自定义预算档位与对照 top-m
BUDGETS="48,96,192" TOPM=4 GPU=1 bash scripts/system_efficiency/run_scale_memory.sh
```

产物按数据集命名（`/ 空格 -` → `_`）：
`data/scale_memory_<数据集>_partials/per_scale.json`、`data/scale_memory_<数据集>.{json,md}`，
不同数据集互不覆盖。

单独重跑拟合（已有标定 json 时，不必再占 GPU）：

```bash
python scripts/system_efficiency/fit_scale_memory.py --dataset FaceDetection \
  --budgets 64,128,256 --topm 4
# 或显式指定路径：
python scripts/system_efficiency/fit_scale_memory.py \
  --partial data/scale_memory_HAR_partials/per_scale.json \
  --budgets 64,128,256 --topm 4
```

注入真实评分（缺省用占位评分演示管线）：`--scores` 传 R 个本地周期评分。

## dashboard 端

- 入口脚本 `scripts/system_efficiency/run_scale_memory.sh` 自动出现在「脚本」面板（环境变量里可加 `DATASET=...`）；
- `GET /api/scale-memory/datasets` — 列出已有标定产物的数据集；
- `GET /api/scale-memory/status?dataset=HAR` — 该数据集产物是否就绪；
- `GET /api/scale-memory/report.md?dataset=HAR` — 可读 markdown（可加模型 + DP 选择表）；
- `GET /api/scale-memory/result?dataset=HAR` — 机读 JSON（$g_0$、$g_r$、可加性误差、DP 结果）。

`dataset` 查询参数缺省为 `HAR`，并做了目录穿越防御（拒绝含 `/ \ ..` 的名字）。

## 对已有实验的影响

无。新增脚本均为新文件，partial 写入独立的 `data/scale_memory_HAR_partials/`，
产物为新的 `data/scale_memory_HAR.*`，不读写也不覆盖 `system_efficiency_HAR.*`。
`measure_har_epoch.py` / `aggregate_results.py` / `run_har_efficiency.sh` 未改动。

