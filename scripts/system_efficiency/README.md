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
- Spilter 的 `m1/m2/m4` 直接把 `Selected_Scales` 固定为 `[0..m-1]`，走 stitched 子模型分支；与主流程一致时 `spilter.stitched_feature_source` 默认为 `forward_slices`（单次 `forward` + 切片）。
- 实验默认只跑 1 个本地 epoch；若想测多个 epoch 的累积耗时，可设 `NUM_EPOCH=5` 等。
- **默认 `BATCH_SIZE=32`** 与所有 `config/configXxx.yml` 主流程实验对齐；如修改 yml 里的 `batch_size` 想测时与训练匹配，请显式 `BATCH_SIZE=N` 覆盖（脚本不读 yml）。

## Spilter 显存与 `SCALE_AUX`（更新说明）

主代码已默认 **`forward_slices`**：`stitched` 主损失与 `UseScaleCL` / `UseScaleKD` 辅助项共用同一次 `forward(optimize=None)` 的表征，按尺度 **切片** 计算，**不再**像旧版 `subset_ln_forward_scale` 那样对同一视图重复 `forward_scale`。因此 **m=4 不应再出现「显存反常高于 FedCSL」**（二者同为「学生 2×forward + 老师 2×forward + 循环内切片」量级）。

测时脚本显式写入 `stitched_feature_source: forward_slices`，与 `config/configSpilter.yml` 对齐。

若需对照旧实现（论文附录 / 消融），在主配置里设 `stitched_feature_source: subset_ln_forward_scale`，或在代码侧单独讨论。

**`SCALE_AUX=0`**（`--no-scale-aux`）仍可用于测时：**关闭** `UseScaleCL` / `UseScaleKD` 两项（FedCSL / Spilter 均受影响），观察「去掉多尺度对齐损失」后的算力/显存下界；**仅影响本测时命令**，不等价于主训练默认配置。

```bash
SCALE_AUX=0 bash scripts/system_efficiency/run_har_efficiency.sh
```
