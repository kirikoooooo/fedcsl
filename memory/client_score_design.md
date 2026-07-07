# client_score 尺度得分方案

## 两层决策架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: 尺度选择 — 训练前确定 (一次分配, 不再变更)                   │
│   主算法: Unified MILP (线性整数规划, 一阶段联合优化)                 │
│   fallback: NSV 背包 + DFS swap (scipy 不可用时)                    │
├──────────────────────────────────────────────────────────────────┤
│ Phase 2: 尺度内 loss 加权 — 待实现                                  │
│   NSV + loss_contribution → 已选 scale 之间的训练强度               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Unified MILP（主算法）

将 per-client budget 约束 + 全局覆盖均衡**合并为一个整数线性规划**，`_plan_unified_milp()` 实现。

### 变量

```
x[c][s] ∈ {0,1}      客户端 c 是否选尺度 s (N×R, 最多 80 binary)
sl_lo[s] ∈ [0,∞)     尺度 s 欠覆盖的松弛量 (R continuous, 软约束)
sl_hi[s] ∈ [0,∞)     尺度 s 过覆盖的松弛量 (R continuous, 软约束)
```

### 优化目标

```
max  Σ NSV[c][s] · x[c][s]  −  λ · Σ (sl_lo[s] + sl_hi[s])
     c,s                          s

λ = strength / max(1 − strength, ε)
```

| strength | λ | 效果 |
|----------|---|------|
| 0 | 0 | 纯 knapsack（不调覆盖） |
| 0.5 | 1 | 平衡 NSV 和覆盖 |
| 1 | ≈∞ | 强引导趋近 target（但非硬约束，sl_∗ 可无限大吸收偏差） |

### 约束

```
(1) Σ weight[s] · x[c][s] ≤ budget[c]     ∀c  (hard — 仅此硬约束)
    s

(2) Σ x[c][s] + sl_lo[s] ≥ target_lo       ∀s  (soft coupling → 目标函数)
    c

(3) Σ x[c][s] − sl_hi[s] ≤ target_hi       ∀s  (soft coupling → 目标函数)
    c
```

sl_lo/sl_hi ∈ [0,∞)，所以 (2)/(3) 永远可行。覆盖引导完全通过目标函数的 λ 惩罚项实现，无硬限制。

### 求解

SciPy `milp`（HiGHS 后端），5 秒 time limit。80 binary + 16 continuous，
~200 个线性约束，秒解。

### 调用链

```
strength > 0 + knapsack_lagrangian 模式
  → _plan_unified_milp(nsv_scores, scale_costs, budgets, cov_target, strength)
  → scipy.milp → 直接输出 client_selected + coverage
  → 跳过 per-client knapsack + DFS swap

scipy 不可用 or non-knapsack_lagrangian 模式
  → fallback: NSV 背包 (per-client) + _balance_coverage_optimal (DFS)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `COVERAGE_TARGET` | `max(2, round(estimated_total / R))` | 每 scale 目标覆盖数 |
| `COVERAGE_STRENGTH` | 0.5 | 覆盖均衡强度 0-1 |
| `COVERAGE_TOLERANCE` | 1 | gap 区间宽度 (±tol) |

### 与两阶段对比

| | 两阶段 (per-client knapsack + DFS swap) | 统一 MILP |
|---|---|---|
| 全局信息 | Step 1 不感知覆盖, Step 2 事后补救 | 一次看到所有约束 |
| 优化保证 | Step 2 DFS 可达 swap 集内全局最优 | 整个可行域全局最优 |
| 复杂度 | O(R×N) knapsack + DFS | HiGHS LP solve |
| 依赖 | 无额外依赖 | scipy ≥ 1.9 |

---

## Phase 1 Fallback: 两阶段（scipy 不可用时）

### Step 1: NSV 背包

```
NSV[c][s] = period_score[c][s] / (marginal_cost[s] + ε)

# 每个客户端独立 DP knapsack
maximize Σ NSV[c][s]  s.t.  Σ weight[s] · x[s] ≤ budget[c]
                           s
```

### Step 2: DFS 覆盖均衡

`_balance_coverage_optimal` 在两阶段模式下被调用，
对 knapsack 分配做带覆盖约束的 DFS swap 调整。

---

## 显存代价

### 边际成本（per scale — MILP weight）

```
scale_marginal[s] = 2 × retained[s]       # q/k 双通路激活保留
                  + max(fwd[s], bwd[s])   # 峰值取大
                  + 2 × param[s]          # 参数 + SGD momentum buffer
```

### 背包约束

```
base_cost = overhead_mb                   # 模型参数 + CUDA 运行时 + LN 共享激活
cap = budget - base_cost
Σ weight[selected] ≤ cap
```

### 显存持久化

首次训练自动保存到 `data/scale_memory/<dataset>_scale_memory.json`。
后续实验通过 `--scale-memory-cache`（或 `SCALE_MEMORY_CACHE=1`）加载。

---

## Coverage Sensitivity 实验

独立脚本 `scripts/coverage_sensitivity.sh`，对每个 `(cov, alpha)` 组合使用 `--coverage-strength 1`（硬约束）并行启动：

```
COVERAGE_TARGET="1,2,3,4,5" DIR_ALPHA="0.1,0.5" DATASET=HAR \
  bash scripts/coverage_sensitivity.sh
```

命名：`HAR_spilter_cov2_dir0.5`。

Dashboard 中通过 `--coverage-strength` 和 `--coverage-target` 控件调整实验参数。
