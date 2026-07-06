# client_score 尺度得分方案

## 两层决策架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: 尺度选择 (knapsack) — 训练前确定                           │
│   Step 1: NSV 背包 → 初始分配                                      │
│   Step 2: DFS branch-and-bound 覆盖均衡 → 最终分配                  │
│   → 训练周期内不再变更                                              │
├──────────────────────────────────────────────────────────────────┤
│ Phase 2: 尺度内 loss 加权 — 待实现                                  │
│   NSV + loss_contribution → 已选 scale 之间的训练强度               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 Step 1: NSV 背包

```
NSV[c][s] = period_score[c][s] / (marginal_cost[s] + ε)

# knapsack (per-client):
value[s]  = NSV[c][s]                     # 单位显存的周期价值
weight[s] = marginal_cost[s]              # 边际显存开销
cap       = budget[c] - base_cost

maximize Σ value[selected]  s.t.  Σ weight[selected] ≤ cap
```

每个客户端独立背包，拿到初始分配 `selected_scales[c]`。

---

## Phase 1 Step 2: 覆盖均衡

目标：按可调节强度 `λ` 将覆盖分布向 `[target_lo, target_hi]` 拉近，
同时全局 NSV 损失最小。

**主算法**: `_balance_coverage_optimal` — 带惩罚项的 DFS branch-and-bound 全局最优。

```
objective = Σ NSV_delta + penalty_weight × Σ max(0, gap)²

penalty_weight = strength / (1 - strength)
gap = |coverage[s] - target_lo|  if coverage[s] < target_lo
    | coverage[s] - target_hi|   if coverage[s] > target_hi
    0                            otherwise
```

**strength 控制覆盖强度**:

| strength | penalty_weight | 效果 |
|----------|---------------|------|
| 0 | 0 | 不 swap，保留 knapsack 结果 |
| 0.5 | 1 | 平衡 NSV 损失和覆盖均匀 |
| 1 | ∞ | 硬约束（最大可达均匀度） |

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `COVERAGE_TARGET` | `max(2, round(Σ coverage/R))` | 每 scale 目标覆盖数 |
| `COVERAGE_STRENGTH` | 0.5 | 覆盖均衡强度 0-1 |
| `COVERAGE_TOLERANCE` | 1 | gap 区间宽度 (±tol) |

---

## 显存代价

### 边际成本（per scale — 背包 weight）

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

---

### 显存持久化

首次训练自动保存到 `data/scale_memory/<dataset>_scale_memory.json`。后续实验通过 `--scale-memory-cache`（或 `SCALE_MEMORY_CACHE=1`）加载，避免并行 GPU 采集冲突。

---

## Coverage Sensitivity 实验

独立脚本 `scripts/coverage_sensitivity.sh`，对每个 `(cov, alpha)` 组合并行启动，使用持久化显存：

```
COVERAGE_TARGET="1,2,3,4,5" DIR_ALPHA="0.1,0.5" DATASET=HAR \
  bash scripts/coverage_sensitivity.sh
```

命名：`HAR_spilter_cov2_dir0.5`。
