# 论文实验待办清单（fedcsl_spilter_acm）

> 生成日期：2026-11-03
> 依据：`论文撰写/acm_article/fedcsl_spilter_acm.pdf`（11 页）与仓库结果记录 md 逐项对照。
> 用法：每做完一项把 `- [ ]` 改成 `- [x]`，并在对应结果 md（`data/*.md`）回填数值。

## 状态总览

| 内容 | 状态 |
|---|---|
| Fig 1 框架图、Fig 2 HAR 收敛曲线 | ✅ 有 |
| Tab 2 HAR α 矩阵（含 Spilter m1/m2/m4） | ✅ 完整 |
| Tab 5 HAR 显存/时间微基准、Tab 6 Local vs Random top-m | ✅ 有 |
| Tab 1 FaceDetection α 矩阵 | ⚠️ 无 Spilter 行；FedProx α=1.0 单元格为空 |
| Tab 3 LSST α 矩阵 | ⚠️ Spilter 只有 m=4 |
| Tab 4 模块消融 | ❌ "Local" 列是占位符 |
| §7.6 MAST | ⚠️ 只有文字，无表格无图 |
| §7.7 可解释性分析 | ❌ 整节零图零数据 |

---

## P0 — 论文正文明确承认"没做/占位"的（优先）

### 1. Tab 4 "Local" 列（本地非联邦消融）

论文原文：*"The local-only column is not reported in this run and is shown as a placeholder."*

- [ ] 跑 Epilepsy / HAR / SleepEDF 的 Local-only 对比学习
      （关闭 UseJointCL / UseJointKD / UseScaleCL / UseScaleKD，即 Eq. (20) 中 𝕀_Fed=0）
- [ ] 回填论文 Tab 4（`tab:ablation`），删掉 placeholder 表述

**脚本**：`FedCSL_All.py`（基于 configAblation.yml 体系，关闭 Fed 开关）
**记录**：`data/ablation_local_only.md`（新建）

### 2. 覆盖感知背包实验（§7.5 / limitation ii）

论文原文：*"the coverage-aware knapsack is formulated but not yet deployed at scale."*

- [ ] 跑 `scripts/coverage_sensitivity.sh`（coverage_target × α 网格，无结果日志）
- [ ] 跑显存标定 + DP 管线（`scripts/system_efficiency/run_scale_memory.sh`），HAR 完全没跑
      （`data/scale_memory/` 只有 Epilepsy-TSTCC / LSST 的 per-scale json，无 DP 产物）
- [ ] 注入客户端真实 STFT/ACF 评分（`--scores`，runbook 强调占位评分不得写进论文结论）
- [ ] 回填论文 §7.5，替换"not yet deployed"表述

**脚本**：`scripts/coverage_sensitivity.sh`、`scripts/system_efficiency/run_scale_memory.sh`
**手册**：`scripts/system_efficiency/RUNBOOK_scale_memory.md`（含自检点 A–D）
**记录**：`data/scale_memory_HAR.md`（runbook 定义的产品）

### 3. Mixed m_k 实验（同轮内客户端异构 m，limitation ii "not evaluated"）

- [ ] 跑 `scripts/spilter_mixed_har.sh`（10 客户端 = 3×m4 + 4×m2 + 3×m1，α=0.1 默认）
- [ ] （可选）`SPILTER_RANDOM=1 DIR_ALPHA=1.0` 扩展
- [ ] 结果写入 `data/mixed_m_HAR.md`（新建），回填论文 §7.5

**脚本**：`scripts/spilter_mixed_har.sh` + `config/configSpilterMixed.yml`（均就绪，无结果）

### 4. §7.7 可解释性分析（整节无图）

正文承诺了四类可视化，均未产出：

- [ ] class-wise embedding 结构（t-SNE/UMAP，表征用 `train.py` 的 `transform()` 提取）
- [ ] per-scale 响应强度图
- [ ] 选中尺度集 ℛ_k 在不同客户端周期结构下的可视化
- [ ] period-aware 权重 π 的训练轨迹
- [ ] 插入论文 §7.7 成图

**脚本**：无，需新建（如 `scripts/plot_interpretability.py`）

### 5. §7.6 MAST 无表格/图

- [ ] 汇总 4 个 UEA 数据集（Cricket / DuckDuckGeese / MotorImagery / PEMS-SF）× 预算档的
      MAST vs Mimose / Checkmate / MONet 时间与显存对比表
- [ ] MotorImagery 16GB profiling 曲线图（`CSL/MAST/profiler_trace_plot.py` 已存在，直接出图）
- [ ] 插入论文 §7.6

**脚本**：`CSL/MAST/UEA_MAST.py`、`CSL/MAST/profiler_trace_plot.py`

---

## P1 — 主实验补跑（数据集 × 方法矩阵缺格）

### 6. FaceDetection Spilter 全矩阵

- [ ] Spilter m1/m2/m4 × 6 个 α 全部补跑（论文 Tab 1 目前无 Spilter 行）
- [ ] FedProx α=1.0 补跑（md 为 "-"，论文表格该格为空）
- [ ] 回填 `data/FaceDetection_results.md` 与论文 Tab 1

**记录**：`data/FaceDetection_results.md:24`（"待补跑"标注处）

### 7. LSST Spilter m1/m2

- [ ] 补跑 Spilter m1、m2 × 6 个 α（现有仅 m=4）
- [ ] 回填 `data/LSST_results.md` 与论文 Tab 3

**记录**：`data/LSST_results.md:23`（"仅有 local_top_m=4 的日志"标注处）

---

## P2 — 下游微调 / 标签效率实验

**设计文档**：`docs/downstream_finetuning_ablation.md`（状态：设计稿，待导师确认后实施）

已做（均为 20 轮快速验证，**非正式口径**）：
- Epilepsy：Spilter-m2 → `data/fine_tuning_Epilepsy.md`
- LSST：Spilter-m2 + FedAvg 对照 → `data/fine_tuning_LSST.md`

### 8. HAR 下游微调主矩阵（完全没做）

- [ ] 协议 A 主矩阵：FedCSL / Spilter-m2 / Spilter-m4 / FedAvg / FedBYOL × 6 比例 × 3 种子
      = 90 组（α=0.1）
- [ ] 扩展 α=1.0（再 90 组，可选）
- [ ] 结果写入 `data/fine_tuning_HAR.md`（新建，格式对齐 HAR_results.md）

**脚本**：仿照 `scripts/run_ft_epilepsy.sh` 写 `scripts/run_ft_har.sh`
（参数：`--ft-ratios "0.01,0.05,0.1,0.2,0.5,1.0" --ft-seeds 3`）

### 9. Epilepsy / LSST 补方法 + 200 轮正式复跑

- [ ] Epilepsy：补 FedCSL（configACF.yml）、Spilter-m4、FedAvg，200 轮
- [ ] LSST：补 FedCSL、Spilter-m4、FedU2，200 轮
- [ ] 用 200 轮正式结果替换两个 md 里的 20 轮快速验证数值
- [ ] （可选）低标注档（5%/10%）种子数 3 → 5，降方差

**脚本**：`scripts/run_ft_epilepsy.sh`（3 方法并行）；LSST 复现命令见 `data/fine_tuning_LSST.md` 末尾
**记录**：`data/fine_tuning_Epilepsy.md` / `data/fine_tuning_LSST.md` 的「待办」节

### 10. B 组解冻消融（代码未实现）

- [ ] 实现 `_finetune_e2e_eval()`（全解冻/分层解冻，`--ft-unfreeze {none,all,shapelets}`）
      ——设计文档 §8 标注"待实现（B 组）"
- [ ] 跑 HAR α=0.1：FedCSL + Spilter-m4 × 1/10/100% × B1/B2/B3（B4 可选）
- [ ] 结果写入 `data/fine_tuning_HAR.md` 的 B 组表

**设计**：`docs/downstream_finetuning_ablation.md` §4.2 / §8

### 11. 论文新增"下游微调与标签效率"小节

- [ ] 论文目前**完全没有微调章节**；新表 `tab:fine_tuning_har` / `tab:fine_tuning_epilepsy` /
      `tab:fine_tuning_lsst` / `tab:unfreeze_ablation`
- [ ] 正文说明三套评估口径（SVM / 冻结线性头 / 全微调）各自的适用问题

---

## 其他（低优先 / 可选）

- [ ] 收敛曲线目前仅 HAR（Fig 2）；评估协议提到的 average rank、rounds-to-convergence 未在正文报出
- [ ] 通信开销实验未覆盖全部五个数据集（limitation i）
- [ ] SleepEDF 微调矩阵（设计文档 §6.1 标为可选，60 组）
- [ ] `\email{author@example.com}` 投稿前替换（tex 第 43 行 TODO）

---

## 速查表：缺口 ↔ 脚本 ↔ 记录文件

| 缺口 | 实验脚本 | TODO/记录 md |
|---|---|---|
| Tab 4 Local 列 | `FedCSL_All.py`（关 Fed 开关） | `data/ablation_local_only.md`（新建） |
| 背包/coverage | `scripts/coverage_sensitivity.sh`、`scripts/system_efficiency/run_scale_memory.sh` | `scripts/system_efficiency/RUNBOOK_scale_memory.md` |
| Mixed m_k | `scripts/spilter_mixed_har.sh` | `data/mixed_m_HAR.md`（新建） |
| MAST 表/图 | `CSL/MAST/UEA_MAST.py`、`profiler_trace_plot.py` | `CSL/MAST/README.md` |
| 可解释性图 | 需新建 `scripts/plot_interpretability.py` | — |
| FaceDetection Spilter 补跑 | `scripts/spilter_har.sh`（同款，换 -dataset） | `data/FaceDetection_results.md` |
| LSST Spilter m1/m2 | 同上 | `data/LSST_results.md` |
| HAR 微调主矩阵 | 仿 `scripts/run_ft_epilepsy.sh` 新建 run_ft_har.sh | `docs/downstream_finetuning_ablation.md` §6.1 |
| Epilepsy/LSST 微调补方法+200轮 | `scripts/run_ft_epilepsy.sh` | `data/fine_tuning_Epilepsy.md` / `data/fine_tuning_LSST.md` 待办节 |
| B 组解冻消融 | 待实现 `_finetune_e2e_eval()` | `docs/downstream_finetuning_ablation.md` §4.2/§8 |
