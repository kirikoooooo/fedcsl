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

目标：`coverage[s] ∈ [target_lo, target_hi]`，全局 NSV 总损失最小。

**主算法**: `_balance_coverage_optimal` — DFS branch-and-bound 全局最优。

1. 枚举候选 swap: `(cid, s_out→s_in, cost = KSV[c][s_out] - KSV[c][s_in])`
2. 按 cost 升序 + DFS + 剪枝（cost≥best → 跳过 / 覆盖已满足 → 终止）
3. DFS 无解 → `raise RuntimeError`（不允许 fallback，必须排查原因）
4. 一个客户端允许多次 swap（无 used_clients 限制），通过追踪当前路径的
   实际 scale 集合验证可行性

**复杂度**: N≤10, R≤8, 候选 swap ≤200, 因覆盖约束 DFS 快速收敛至 <1000 节点。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `COVERAGE_TARGET` | `max(2, round(sum(coverage)/R))` | 每 scale 目标覆盖数（基于实际总分配量） |
| `COVERAGE_TOLERANCE` | 1 | 允许偏差 (±tol) |

**默认 target 计算**: `target = round(Σ coverage / R)`。knapsack 输出总分配到各 scale 后，target 取实际均值保证数学可行。例: 10 clients × 平均 2.7 scales = 27 次分配，target = round(27/8) = 3, range = [2,4]。

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
