# Knapsack-Lagrangian 全局覆盖感知的尺度分配

## §1 问题背景

在 Spilter 架构中，每个客户端仅训练一部分 shapelet 尺度（scale），以节省 GPU 显存。
给定 $K$ 个客户端、$R$ 个尺度（默认 $R=8$），需要为每个客户端 $k$ 分配一个尺度子集
$\mathcal{R}_k \subseteq \{0,\dots,R-1\}$，使得：

1. **本地适配**：客户端 $k$ 优先训练其数据周期模式匹配的尺度；
2. **全局覆盖**：每个尺度 $r$ 至少被 $C_{\min}$ 个客户端训练，避免某些尺度"失传"；
3. **OOM 约束**：客户端 $k$ 的显存用量不超过其预算 $B_k$。

这是一个带全局覆盖约束的多背包问题。

## §2 形式化定义

### 2.1 优化问题

$$
\begin{aligned}
\max_{x_{k,r} \in \{0,1\}} \quad & \sum_{k=1}^{K} \sum_{r=1}^{R} v_{k,r} \cdot x_{k,r} \\[4pt]
\text{s.t.} \quad & g_0 + \sum_{r=1}^{R} g_r \cdot x_{k,r} \le B_k, \quad \forall k \in [K] \tag{1} \\[4pt]
& \sum_{k=1}^{K} x_{k,r} \ge C_{\min}, \quad \forall r \in [R] \tag{2}
\end{aligned}
$$

其中：

| 符号 | 含义 |
|------|------|
| $x_{k,r}$ | 客户端 $k$ 是否训练尺度 $r$ |
| $v_{k,r}$ | 尺度 $r$ 对客户端 $k$ 的价值（无监督：周期评分 `period_score`） |
| $g_0$ | 固定显存开销（基模型 + 优化器状态） |
| $g_r$ | 尺度 $r$ 的边际显存开销 |
| $B_k$ | 客户端 $k$ 的显存预算 |
| $C_{\min}$ | 每个尺度的最少覆盖客户端数 |

### 2.2 价值函数（无监督）

当前使用 `period_score`（STFT + ACF 周期检测），完全无监督。后续可扩展：

- $v_{k,r} = \alpha \cdot V_{\text{period}}(k,r) + \beta \cdot \lambda_r + \delta \cdot V_{\text{learning\_gap}}(k,r)$
- $V_{\text{learning\_gap}}$ 可用 contrastive loss 或梯度范数量化，天然无监督
- $\lambda_r$ 为 Lagrange 乘子，隐式编码全局稀缺性

## §3 Lagrange 松弛

### 3.1 分解

将全局覆盖约束 (2) 用 Lagrange 乘子 $\lambda_r \ge 0$ 松弛到目标函数：

$$
L(x, \lambda) = \sum_{k} \sum_{r} (v_{k,r} + \lambda_r) \cdot x_{k,r} \;-\; \sum_{r} \lambda_r \cdot C_{\min}
$$

此时问题分解为 $K$ 个**独立**的 0-1 背包问题：

> 对每个客户端 $k$：
> $$
> \max \sum_{r} (v_{k,r} + \lambda_r) \cdot x_{k,r}
> \quad \text{s.t.} \quad g_0 + \sum_{r} g_r \cdot x_{k,r} \le B_k
> $$

### 3.2 乘子更新（次梯度方法）

$$
\lambda_r^{(t+1)} \leftarrow \max\!\Big(0,\; \lambda_r^{(t)} + \frac{\eta}{\sqrt{t+1}} \cdot \big(C_{\min} - \text{coverage}_r\big)\Big)
$$

- 覆盖不足（$\text{coverage}_r < C_{\min}$）→ $\lambda_r$ 升高 → 更多客户端被"激励"选择该尺度
- 覆盖充足（$\text{coverage}_r \ge C_{\min}$）→ $\lambda_r$ 降低 → 释放激励
- 衰减学习率 $\eta / \sqrt{t+1}$ 避免震荡

### 3.3 直观解释

$\lambda_r$ 可以理解为尺度 $r$ 的**全局影子价格**（shadow price）：

- $\lambda_r \approx 0$：该尺度"供过于求"，不需要额外激励
- $\lambda_r > 0$：该尺度"供不应求"，调高其"价值"以吸引更多客户端选择
- 这一信号完全从全局覆盖需求中自动涌现，无需人工干预

## §4 显存代价估计

### 4.1 可加显存模型

显存预测采用线性可加模型，与 `scripts/system_efficiency/fit_scale_memory.py` 完全一致：

$$
\widehat{\text{Mem}}(\mathcal{R}_k) = g_0 + \sum_{r \in \mathcal{R}_k} g_r
$$

其中 $g_r$ 为尺度 $r$ 的**边际显存**（单尺度训练峰值 − $g_0$），$g_0$ 为固定开销。

### 4.2 三种获取方式

| 优先级 | 方式 | 说明 |
|--------|------|------|
| 1 | Config 显式指定 `scale_memory_costs_mb` | 用户提供实测值，最精确 |
| 2 | **自动标定**（首次运行） | 创建 `LearningShapeletsCL` 逐尺度训练 1 epoch，测量 `max_memory_allocated()`，结果缓存到 `data/knapsack_calibration/<dataset>_scale_memory.json`，后续运行直接加载 |
| 3 | `_scale_system_costs` 代理 | CPU 场景或标定失败时的 fallback，使用参数量 + 滑窗计算量作为相对权重（无量纲） |

### 4.3 HAR 数据集标定示例（RTX 3090, batch_size=32, mix distance）

| Scale $r$ | Shapelet 长度 $\ell_r$ | 单尺度峰值显存 (MB) | 边际显存 $g_r$ (MB) |
|:---:|:---:|:---:|:---:|
| 0 | 12 | 59.4 | 29.7 |
| 1 | 24 | 82.0 | 52.3 |
| 2 | 37 | 97.0 | 67.3 |
| 3 | 50 | 107.2 | 77.5 |
| 4 | 63 | 109.9 | 80.2 |
| 5 | 76 | 108.1 | 78.4 |
| 6 | 89 | 98.9 | 69.2 |
| 7 | 102 | 84.3 | 54.6 |

- **固定开销** $g_0 = 29.7$ MB
- 显存随长度先增后减（长 shapelet 的滑窗数量减少，激活显存下降）
- 峰值在 $\ell=63$（Scale 4），之后缓慢下降

## §5 0-1 背包精确求解

### 5.1 DP 算法

$R=8$ 规模下采用整数化动态规划，时间复杂度 $O(R \cdot \lceil B / q \rceil)$，
空间复杂度 $O(\lceil B / q \rceil)$，其中 $q=0.5$ MB 为量化粒度。

```
dp[0] = 0
dp[w] = -∞  for w > 0

for r in 0..R-1:
    for w in W down to w_r:
        if dp[w - w_r] + v_r > dp[w]:
            dp[w] = dp[w - w_r] + v_r
            sel[w] = sel[w - w_r] ∪ {r}

return max(sel[w] for w in 0..W if dp[w] feasible)
```

### 5.2 预算未约束时的退化

当 $B_k \ge g_0 + \sum_r g_r$ 时，背包退化为覆盖感知的 top-m 选择：
- 所有客户端都能装下全部尺度
- Lagrange 乘子 $\lambda_r$ 仍通过全局覆盖约束引导选择
- 等价于"确保每个尺度至少被 $C_{\min}$ 个客户端选中"的分配问题

## §6 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│ Round 0: 一次性标定 + 规划                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. 加载/标定 per-scale 显存 g_r (缓存优先)                    │
│ 2. 计算 per-client period_score v_{k,r} (STFT + ACF)        │
│ 3. λ_r ← 0,  ∀r                                            │
│ 4. for t = 1..max_iter:                                    │
│      for each client k (parallel):                          │
│        adj_v_{k,r} ← v_{k,r} + λ_r                         │
│        x_{k,*} ← knapsack_dp(adj_v, g_r, g_0, B_k)         │
│      coverage_r ← Σ_k x_{k,r}                              │
│      if coverage_r ≥ C_min, ∀r: break                       │
│      λ_r ← max(0, λ_r + η/√t · (C_min − coverage_r))       │
│ 5. 返回最优分配 {R_k}                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Round 1..T: 联邦训练（分配结果复用）                          │
├─────────────────────────────────────────────────────────────┤
│ for each client k:                                         │
│   Selected_Scales ← R_k                                    │
│   train locally → upload scale params                      │
│ server aggregates per-scale                                │
└─────────────────────────────────────────────────────────────┘
```

## §7 关键优势

| 特性 | 说明 |
|------|------|
| **全局覆盖保证** | Lagrange 乘子自动调节，确保每个尺度至少被 $C_{\min}$ 个客户端覆盖 |
| **自适应本地偏好** | 每个客户端优先选择自身数据周期匹配的尺度（本地背包最大化本地价值） |
| **OOM 硬约束** | 0-1 背包精确解保证预算永远不被突破（DP 约束 $g_0 + \sum g_r \le B_k$） |
| **异构设备友好** | 支持 per-client 差异化的显存预算 $B_k$ |
| **隐私友好** | 客户端只上传 `x_{k,*}`（选中的尺度索引），不上传评分 $v_{k,r}$ 或预算 $B_k$ |
| **通信开销极小** | 服务端广播 $R$ 个 float ($\lambda_r$)，客户端上传 $R$ 个 bit ($x_{k,r}$) |
| **收敛保证** | 次梯度方法在凸松弛下收敛；衰减步长 + plateau 早停确保有限迭代 |
| **可扩展** | 价值函数可插入 $V_{\text{learning\_gap}}$ 等无监督信号，不改总体架构 |

## §8 配置与使用

### 8.1 配置参数（`config/*.yml`）

```yaml
spilter:
  allocation_mode: "knapsack_lagrangian"

  # ── 显存预算（MB），可选 ──
  memory_budget_mb: 500          # 500MB per client
  # memory_budget_mb: [500, 400, 300, ...]  # 或 per-client

  # ── 显存成本（MB），可选 ──
  # 不设则首次自动标定 + 缓存
  scale_memory_costs_mb: [29.7, 52.3, 67.3, 77.5, 80.2, 78.4, 69.2, 54.6]
  base_memory_mb: 29.7

  knapsack_lagrangian:
    coverage_min: 2              # 每尺度最少覆盖数（默认 ceil(K/R)）
    lambda_lr: 0.1               # Lagrange 步长
    max_iter: 50                 # 最大迭代次数
```

### 8.2 命令行

```bash
python FedCSL_All.py --config config/configSpilter.yml --spilter-knapsack
python FedCSL_All.py --config config/configSpilter.yml --spilter-knapsack --spilter-memory-budget 256
```

### 8.3 Log 解读

```
[spilter] knapsack_lagrangian: loaded cached calibration: g_0=29.7MB, g_r=['29.7' '52.3' ...] MB
[spilter] knapsack_lagrangian: 3 iters, converged=True, coverage=[10,10,10,5,10,10,10,8]
[round 0] per-client scales: c0:[0,1,2,4,5,6,7] 477/500MB | c1:[0,1,2,3,4,5,6] 455/500MB | ...
```

- `converged=True`：所有尺度都满足 $C_{\min}$ 覆盖要求
- `coverage=[10,10,10,5,...]`：尺度 3 只被 5 个客户端覆盖（因为它的边际显存 $g_3=77.5$ MB 最大，预算紧的客户端装不下）
- `c0:[...] 477/500MB`：客户端 0 选了 7 个尺度，用了 477 MB（在 500 MB 预算内）

## §9 与 Top-m 的对比

| | Local Top-m | Knapsack-Lagrangian |
|---|---|---|
| **尺度选择** | 按 period_score 降序取前 m | 0-1 背包最大化 value |
| **全局覆盖** | 无保证，依赖客户端间自然差异 | Lagrange 乘子强制 $C_{\min}$ |
| **显存约束** | 无（可能 OOM） | 硬约束 $g_0 + \sum g_r \le B_k$ |
| **预算感知** | 否 | 是，支持异构设备 |
| **可解释性** | 简单 | $\lambda_r$ 可解释为稀缺性信号 |

## §10 参考文献

- Fisher, M.L. (1981). The Lagrangian Relaxation Method for Solving Integer Programming Problems. *Management Science*, 27(1), 1–18.
- Kellerer, H., Pferschy, U., & Pisinger, D. (2004). *Knapsack Problems*. Springer.
- Boyd, S., Xiao, L., & Mutapcic, A. (2003). Subgradient Methods. *Lecture Notes, Stanford University*.
