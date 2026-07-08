# client_score 尺度得分方案

## 两层决策架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: 尺度选择 — 训练前确定 (一次分配, 不再变更)                   │
│   主算法: Unified MILP (整数线性规划, gurobipy 一阶段联合优化)        │
├──────────────────────────────────────────────────────────────────┤
│ Phase 2: 尺度内 loss 加权 — （动态，待实现）                         │
│   NSV + loss_contribution → 已选 scale 之间的训练强度               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Unified MILP

将 per-client budget 约束 + 全局覆盖均衡合并为一个整数线性规划，`_plan_unified_milp()` 实现（gurobipy 求解）。

### 变量

```
x[c][r] ∈ {0,1}      客户端 c 是否选尺度 r (N×R, 最多 80 binary)
sl_lo[r] ∈ [0,∞)     尺度 r 欠覆盖的松弛量 (R continuous, soft)
sl_hi[r] ∈ [0,∞)     尺度 r 过覆盖的松弛量 (R continuous, soft)
```



### 优化目标

```
max  Σ NSV[c][r] · x[c][r]  −  λ · Σ (sl_lo[r] + sl_hi[r])
     c,r                          r

λ = strength / max(1 − strength, ε)
```


| strength | λ   | 效果                                |
| -------- | --- | --------------------------------- |
| 0        | 0   | 纯 knapsack（不调覆盖）                  |
| 0.5      | 1   | 平衡 NSV 和覆盖                        |
| 1        | ≈∞  | 强引导趋近 target（soft coupling, 永不无解） |




### 约束

```
(1) Σ weight[r] · x[c][r] ≤ budget[c]      ∀c  (hard — 仅此硬约束)
    r

(2) Σ x[c][r] + sl_lo[r] ≥ target_lo        ∀r  (soft coupling)
    c

(3) Σ x[c][r] − sl_hi[r] ≤ target_hi        ∀r  (soft coupling)
    c
```

sl_lo/sl_hi ∈ [0,∞)，coverage 引导完全通过目标函数的 λ 惩罚项实现。

**target_lo / target_hi 计算方式**:

```
estimated_total = Σ_c min(budget[c] / mean_weight, R)      ← 预估总分配次数
target         = max(2, round(estimated_total / R))         ← 平均每 scale 分配数
target_lo      = max(1, target − COVERAGE_TOLERANCE)        ← 下限
target_hi      = target + COVERAGE_TOLERANCE                 ← 上限
```

当手动指定 `--coverage-target` 时，直接用该值替代 `target`。

### 参数


| 参数                   | 默认值                                  | 含义              |
| -------------------- | ------------------------------------ | --------------- |
| `COVERAGE_TARGET`    | 见上方 `target` 计算                      | 每 scale 目标覆盖数   |
| `COVERAGE_STRENGTH`  | 0.5                                  | 覆盖均衡强度 0-1      |
| `COVERAGE_TOLERANCE` | 1                                    | gap 区间宽度 (±tol) |




### NSV: 单位显存价值

```
NSV[c][r] = normalize( period_score[c][r] / (marginal[r] + ε) )
```



### 边际显存

```
marginal[r] = 2 × retained[r]       # q/k 双通路激活保留
            + max(fwd[r], bwd[r])   # 峰值取大
            + 2 × param[r]          # 参数 + SGD momentum buffer
```



### 与两阶段对比


|      | 两阶段 (per-client knapsack + DFS swap) | 统一 MILP            |
| ---- | ------------------------------------ | ------------------ |
| 全局信息 | Step 1 不感知覆盖, Step 2 事后补救            | 一次看到所有约束           |
| 优化保证 | Step 2 DFS 可达 swap 集内全局最优            | 整个可行域全局最优          |
| 求解器  | 纯 Python (DP + DFS)                  | gurobipy           |
| 依赖   | 无                                    | gurobipy (MAST 同款) |


---



## 显存持久化

首次训练自动保存到 `data/scale_memory/<dataset>_scale_memory.json`。
后续实验通过 `--scale-memory-cache`（或 `SCALE_MEMORY_CACHE=1`）加载。

---

