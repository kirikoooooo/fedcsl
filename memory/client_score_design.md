# client_score 调整方案评估与设计

## 一、两层决策架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: 尺度选择 (knapsack) — 训练前确定                           │
│   Step 1: NSV 背包 → 初始分配                                      │
│   Step 2: 最小 NSV 损失覆盖均衡 → 最终分配                           │
│   → 训练周期内不再变更 (除非手动触发)                                  │
├──────────────────────────────────────────────────────────────────┤
│ Phase 2: 尺度内 loss 加权 — 每轮动态                                  │
│   NSV + loss_contribution → 已选 scale 间的训练强度分配               │
│   → 每 N 轮重算一次分数, 但不再重分配 scale                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、Phase 1: 尺度选择

### Step 1: NSV 背包

```
NSV[c][s] = period_score[c][s] / (marginal_cost[s] + ε)

# knapsack (per-client):
value[s]  = NSV[c][s]                     # 单位显存的周期价值
weight[s] = marginal_cost[s]              # 边际显存开销
cap       = budget[c] - base_cost

maximize Σ value[selected]  s.t.  Σ weight[selected] ≤ cap
```

每个客户端独立背包，拿到初始分配 `selected_scales[c]`。

### Step 2: 最小 NSV 损失覆盖均衡

第一步完成后统计全局覆盖 `coverage[s] = Σ 1{s ∈ selected[c]}`。

目标：将覆盖偏差控制在 `target ± tolerance` 内，同时最小化全局 NSV 损失。

```
输入:
  selected_scales[c]: Step 1 的各客户端分配
  NSV[c][s]:          各客户端各 scale 的 NSV 值

目标:
  ∀s: coverage[s] ∈ [target_lo, target_hi]     (覆盖约束)
  minimize Σ NSV_loss                            (全局 NSV 损失最小)
```

**算法: Greedy Swap** (类似你的方案 2.2)

```
# 迭代直到覆盖满足或无法改进:
1. 找超覆盖 scale over_s   (coverage[over_s] > target_hi)
   找欠覆盖 scale under_s  (coverage[under_s] < target_lo)

2. 对每个拥有 over_s 的客户端 c:
   计算: swap_gain = NSV[c][under_s] - NSV[c][over_s]
   如果 under_s 已在 c 的候选集但未被选中 → 增加选择

3. 取 swap_gain 最大的那个客户端, 执行 swap
   (对 over_s 来说, 替换 NSV 损失最小; 对 under_s 来说, 增益最大)

4. 更新 coverage, 重复直到满足或 max_rounds 耗尽
```

**最坏复杂度**: `O(max_rounds × numClients × R)`，约 `20 × 10 × 8 = 1600` 次操作，可忽略。

**参数**:

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `target_coverage` | `max(2, ceil(numClients/R))` | 每 scale 目标覆盖数 |
| `tolerance` | 1 | 允许偏差 (±1) |
| `max_rounds` | 20 | 最多 swap 轮数 |

---

## 三、Phase 2: 尺度内 loss 加权

**不重新分配 scale**。只调整已选 scale 之间的训练强度。

```
scale_weight[si] = 5.0 × normalize(
    λ × NSV[si]                               # 静态: 单位显存价值
  + (1-λ) × loss_contrib[si]                   # 动态: 累积 loss 贡献
)

λ = 0.7 (默认)
```

`loss_contrib` 使用累积 loss（和当前实现一样，无 EMA）：

```python
client_cum_loss[c]  += total_loss          # per-client 累积
client_cum_total[c] += 1.0

loss_contrib[c] = cum_loss[c] / max(cum_total[c], ε)
loss_contrib = normalize_minmax(loss_contrib)   # → [0,1]
```

**注意**: `loss_contrib` 是 client-level 的（不是 per-scale），因为 Phase 2 作用于**已选 scale** 内部，loss 信号无法分解到未选中的 scale。

**Phase 2 每 N 轮重算一次分数并打印, 但不改 scale 分配**:

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `λ` | 0.7 | NSV 权重 |
| `recalc_interval` | 10 | Phase 2 重算间隔 (轮) |
| `warmup_rounds` | 10 | loss_contrib 线性 ramp |

---

## 四、和旧方案的对比

| | 旧 L1-A+ (删除) | 新方案 |
|---|---|---|
| 覆盖处理 | cov_boost 在 score 中隐式引导 | Step 2 显式 swap, 有保证 |
| 尺度重分配 | 每 10 轮重新 knapsack | 不重分配 |
| 动态项位置 | Layer 1 (尺度选择) | Phase 2 (loss 加权) |
| NSV 权重 | per-client 归一化 | Step 1 独立背包 |
| 实现复杂度 | 已实现 (~100行) | Phase 1 约 80 行, Phase 2 约 20 行 |

---

## 五、推荐实现顺序

| 优先级 | Phase | 组件 | 改动量 |
|--------|-------|------|--------|
| **P0** | 1-1 | NSV 背包 (替换现有 period_score) | ~20行 |
| **P0** | 1-2 | Greedy Swap 覆盖均衡 | ~60行 |
| **P0** | 2 | NSV + loss_contrib loss 加权 | ~20行 |

---

## 六、实现要点

### Phase 1 Step 1: NSV 背包

```python
# 计算 per-client NSV
nsv_scores = []
for c in range(numClients):
    nsv = period_scores[c] / (np.array(scale_costs) + 1e-6)
    nsv = (nsv - nsv.min()) / (nsv.max() - nsv.min() + 1e-6)   # normalize per-client
    nsv_scores.append(nsv.astype(np.float32))

# 用 nsv_scores 替代 cached_client_scale_scores 跑 knapsack
cached_client_scale_scores = nsv_scores
cached_client_scale_plans, cached_scale_hist, ... = _plan_...
```

### Phase 1 Step 2: Greedy Swap

```python
def _balance_coverage_greedy_swap(
    client_selected, nsv_scores, coverage_hist,
    target_lo, target_hi, max_rounds=20
):
    """用最小 NSV 损失做 per-scale 覆盖均衡。"""
    coverage = coverage_hist.copy()
    R = len(coverage)
    for _ in range(max_rounds):
        overs  = [s for s in range(R) if coverage[s] > target_hi]
        unders = [s for s in range(R) if coverage[s] < target_lo]
        if not overs or not unders:
            break
        best_loss = float('inf')
        best_swap = None  # (cid, over_s, under_s)
        for over_s in overs:
            for under_s in unders:
                for cid, sel in enumerate(client_selected):
                    if over_s not in sel or under_s in sel:
                        continue
                    loss = nsv_scores[cid][over_s] - nsv_scores[cid][under_s]
                    if loss < best_loss:
                        best_loss = loss
                        best_swap = (cid, over_s, under_s)
        if best_swap is None:
            break
        cid, over_s, under_s = best_swap
        sel = client_selected[cid]
        sel[sel.index(over_s)] = under_s
        coverage[over_s] -= 1
        coverage[under_s] += 1
    return client_selected, coverage
```

### Phase 2: Loss Weighting

```python
# train.py _selected_scale_loss_weights
loss_contrib = client_cum_loss[c] / max(client_cum_total[c], 1e-6)
loss_contrib = normalize_minmax_global(loss_contrib)  # 跨 client 归一化

nsv = period_score / (scale_costs + 1e-6)   # local per-client
nsv_norm = normalize(nsv)

scale_weight[si] = 5.0 * normalize(
    0.7 * nsv_norm[si] + 0.3 * loss_contrib
)
```

---

## 七、代码清理

需要删除的旧实现:
- `_compute_client_nsv_scores()` — 替换为 Phase 1 直接 NSV 计算
- `_format_nsv_breakdown()` — 替换为 Phase 1 + Phase 2 各自的打印
- `cov_boost` / `coverage_boost` / `beta` 退火 — 改为 Step 2 显式贪心
- 每 10 轮的 knapsack replan — Phase 1 不再重分配 scale
