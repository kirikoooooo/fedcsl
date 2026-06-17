# MAST：面向 Shapelet 模型的显存感知 Checkpointing 规划器

> 来源：Liang et al., "Towards GPU Memory-Aware Efficient Contrastive Shapelet
> Learning for Unsupervised Representation Learning in Multivariate Time Series"

---

## §1 研究背景：CSL 面临的显存瓶颈

### 1.1 CSL 框架简介

CSL（Contrastive Shapelet Learning）是无监督 MTS 表征学习的 shapelet-based 框架。核心组件：

- **Shapelet Transformer (ST) 编码器**：$R$ 个尺度 × $M$ 个距离度量 的 sub-module 拼接
  - 每个 sub-module $f_{r,m}$：尺度 $r$ + 距离度量 $m$（Euclidean / Cosine / Cross-Correlation）
  - 默认 $R=8, M=3$，共 24 个 sub-module
  - 所有 sub-module 的 shapelet 长度不同（从短到长覆盖多种时间尺度）
- **多粒度对比学习**：coarse-grained（拼接特征）+ fine-grained（per-scale 特征）
- **多尺度对齐**：不同尺度表征之间的一致性约束

### 1.2 显存瓶颈

ST 编码器在训练时，每个 sub-module $f_{r,m}$ 的**中间激活**需要常驻 GPU 显存以用于反向传播。激活显存主要由 sub-module 主导：

$$
\text{Mem}(f_{r,m}) \propto BD(T-L_r+1)L_r + BD_{\text{repr}}(T-L_r+1)
$$

其中 $B$=batch size, $D$=变量数, $T$=序列长度, $L_r$=shapelet 长度。

当 $B$、$D$、$T$ 较大时，所有 sub-module 的激活同时驻留 → **OOM**。

---

## §2 MAST 核心思路

### 2.1 两个关键观察

**O1：显存由 sub-module 主导。** ST 编码器的激活显存集中在 $R\times M$ 个 shapelet sub-module 上，后续的 loss 计算等操作显存开销极小（$\Theta(BD_{\text{repr}})$，$D_{\text{repr}}$ 为常数）。

**O2：sub-module 之间相互独立。** 不同 $(r,m)$ 的 sub-module 之间没有数据依赖——它们分别对输入做距离计算后拼接输出。因此任意 sub-module 的激活可以**独立丢弃和重算**，不会产生递推依赖（recursion）。

### 2.2 设计思想

通用 AC（Activation Checkpointing）规划器的困境：
- 启发式方法（Mimose、Monet）：逐层贪心，缺乏全局信息
- 优化方法（Checkmate）：存在非线性约束（层间依赖 → 多项式约束），只能近似求解

MAST 利用 ST 的 sub-module 独立性，将规划空间从"所有计算层"缩小到"$R\times M$ 个独立 sub-module"，从而将问题建模为**可精确求解的二元线性规划**。

---

## §3 形式化定义

### 3.1 决策变量

$$
y_{r,m} \in \{0,1\}, \quad \forall r \in [R],\; m \in [M]
$$

- $y_{r,m}=1$：对 sub-module $f_{r,m}$ 使用 checkpointing（前向时丢弃激活，反向时重算）
- $y_{r,m}=0$：正常存储激活

### 3.2 优化目标

最小化由 checkpointing 引入的额外重算开销：

$$
\arg\min_{y} \sum_{r} \sum_{m} y_{r,m} \cdot c_{r,m}
$$

其中 $c_{r,m}$ 为 sub-module $f_{r,m}$ 的前向计算成本（在目标硬件上 profiled）。

### 3.3 显存约束

训练过程中任意时刻的峰值显存不得超过预算 $\mathcal{B}$：

$$
p \le \mathcal{B}, \quad \forall p \in \{p_{r,m}, p_L, \tilde{p}_{r,m}, \tilde{p}_L, p^\uparrow_{r,m}\}
$$

五种峰值场景：

| 符号 | 含义 | 关键公式 |
|------|------|----------|
| $p_{r,m}$ | 前向执行 $f_{r,m}$ 时 | $\sum_{(r',m')\in S_{r,m}} y_{r',m'}o_{r',m'} + w_{r,m} + \varsigma$ |
| $p_L$ | 前向执行 loss 时 | $\sum_{r,m} y_{r,m}o_{r,m} + w_L + \varsigma$ |
| $\tilde{p}_{r,m}$ | 反向传播 $f_{r,m}$ 时 | 含存储的激活 + 梯度缓存 |
| $\tilde{p}_L$ | 反向传播 loss 时 | $\sum_{r,m} y_{r,m}o_{r,m} + o_L + \tilde{w}_L + \varsigma$ |
| $p^\uparrow_{r,m}$ | 重算 $f_{r,m}$ 激活时 | 临时分配前向 workspace |

符号说明：
- $o_{r,m}$：sub-module $f_{r,m}$ 的输出激活显存
- $w_{r,m}, \tilde{w}_{r,m}$：前向/反向 workspace 显存
- $\varsigma$：模型参数显存（常量）
- $\phi_L, \phi_{r,m}$：loss 层 / sub-module 的梯度显存

### 3.4 模型性质

由于所有符号除 $y$ 外均为**常量**（hardware-independent 常数可提前计算，hardware-dependent 常数通过首轮 profiling 获取），该问题是**二元线性规划（Binary Linear Program）**，可用 Gurobi 等标准求解器精确求解。

---

## §4 算法流程

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Profiling（首轮训练时在线完成）                     │
├────────────────────────────────────────────────────────────┤
│ 1. Checkpoint 所有 sub-module（y_{r,m}=1）                  │
│ 2. 逐 sub-module 测量：                                     │
│    - c_{r,m}: time.perf_counter() 计时                      │
│    - w_{r,m}, w̃_{r,m}: reset_peak_memory_stats +            │
│      max_memory_allocated - baseline                        │
│ 3. 计算 hardware-independent 常数：                          │
│    - o_{r,m}: BD(T-L_r+1)L_r + BD_repr(T-L_r+1)           │
│    - ς, φ_L, φ_{r,m}: 参数量 × 4 bytes                      │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 2: 求解 BLP                                           │
├────────────────────────────────────────────────────────────┤
│ 1. 给定显存预算 B，求解：                                    │
│    min Σ y_{r,m}·c_{r,m}                                   │
│    s.t. all peak memory ≤ B                                 │
│ 2. 使用 Gurobi / SCIP 等 ILP solver 精确求解                 │
│    (R×M = 24 变量，规模极小，求解时间可忽略)                  │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 3: 应用 Plan                                          │
├────────────────────────────────────────────────────────────┤
│ 1. 后续训练迭代中，对 y_{r,m}=1 的 sub-module 丢弃激活       │
│ 2. 反向传播时按需重算                                       │
│ 3. plan 跨轮复用（模型结构不变）                              │
└────────────────────────────────────────────────────────────┘
```

---

## §5 对比 Baseline

MAST 与三类通用 AC 规划器对比：

| 规划器 | 类型 | 策略 | 局限性 |
|--------|------|------|--------|
| **Mimose** [47] | 启发式 | 贪心选择 memory/compute 比值最高的层 | 忽略全局最优 |
| **Monet** [32] | 启发式 | 基于层的显存和计算特征贪心决策 | 同上 |
| **Checkmate** [28] | 优化 | 建模为优化问题，含多项式非线性约束 | 只能近似求解（线性化），解质量受限 |
| **MAST (Ours)** | 优化 | 利用 sub-module 独立性，建模为 BLP | 精确解，但仅适用于 CSL 类 shapelet 模型 |

### 实验结果（HAR 等 4 个 UEA 数据集）

- **训练时间**：MAST 比 Mimose / Checkmate / Monet **减少 1.43%–7.65%**
- **显存节省**：相同时间开销下，MAST 比无 checkpointing **节省 6.4%–21.9% GPU 显存**
- **规划开销**：MAST 规划时间仅为 Checkmate 的 **0.06%**（24 变量 vs 全部计算层）
- **计划质量**：显存极紧时（如 4 GB），MAST 计划的实际显存更接近预算上限（用得更满），而 Checkmate 因近似误差留有过大冗余

---

## §6 与我们的 Knapsack-Lagrangian 方案对比

| 维度 | MAST (AC Checkpointing) | 我们的方案 (Knapsack-Lagrangian) |
|------|--------------------------|----------------------------------|
| **优化层面** | 训练过程（激活存储策略） | 架构层面（选择哪些 scale 参与训练） |
| **决策粒度** | sub-module 级别的激活保留/丢弃 | scale 级别的 sub-model 选择 |
| **显存模型** | 精确分析（$o_{r,m}, w_{r,m}$ 等 profiled） | 可加模型（$g_0 + \sum g_r$） |
| **优化方法** | 二元线性规划（Gurobi 精确解） | Lagrange 松弛 + 0-1 背包 DP |
| **约束类型** | 每时刻峰值 ≤ 预算（多时刻约束） | 总显存 ≤ 预算（单一约束） |
| **求解时间** | ILP solver (~ms)，需 profiling | DP $O(R\cdot B/q)$，极快 |
| **适用范围** | CSL 模型（依赖 sub-module 独立性） | 通用（只要有 per-scale 显存成本） |
| **互补性** | 与我们的方案正交，可叠加 | 选好 scale 子集后，MAST 可进一步压缩显存 |

**关键洞察**：两者可以叠加使用。先用 Knapsack-Lagrangian 为每个客户端选择适合的 scale 子集（架构层面），再对选中的 sub-module 应用 MAST 的 checkpointing（训练过程层面），实现双层显存优化。

---

## §7 关键技术要点

1. **独立性子模块假设**：不同 $(r,m)$ 的 sub-module 无数据依赖，使得规划空间从"所有层（数十到数百）"降到"$R\times M$（24）"，消除了非线性约束的根源。

2. **在线 profiling**：首轮训练时 checkpoint 全部 sub-module 来获取 $c_{r,m}$ 和 $w_{r,m}$，不引入额外测量开销。

3. **硬件独立性**：$o_{r,m}$（激活大小）、$\varsigma$（参数量）等常数可纯由数据维度和模型结构计算，不需 profiling。

4. **Plan 复用**：模型结构不变 → plan 跨轮复用，只在模型变化时重新求解。

---

## §8 参考

1. Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost. *arXiv:1604.06174*. (Activation Checkpointing 基础)
2. Jain, P. et al. (2020). Checkmate: Breaking the Memory Wall in Optimal Tensor Rematerialization. *MLSys*.
3. Kusumoto, M. et al. (2019). Mimose: Memory-Aware Optimal Sample Selection for Efficient Training. *SC*.
4. Kumar, R. et al. (2019). Monet: Memory Optimization for Neural Network Training. *NeurIPS Workshop*.
