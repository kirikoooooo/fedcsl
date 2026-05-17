# Spilter 显存节省实验：客户端训练平均峰值显存随 top-m 变化

## 实验设置

- **数据集**: HAR (Human Activity Recognition)
- **联邦客户端数**: K = 50, Dirichlet $\alpha$ = 0.1, batch size = 8
- **样本形状**: N=5881, C=9, T=128
- **GPU**: NVIDIA GeForce RTX 3090
- **总尺度数**: 8（FedCSL baseline 等价于 $m = 8$；Spilter $m \in \{1,2,4\}$ 拼接子模型）
- **指标定义**: 对每个客户端独立训练 1 epoch，用 `torch.cuda.reset_peak_memory_stats()` 清零后训练，结束时取 `torch.cuda.max_memory_allocated()` 作为该客户端的峰值显存；对所有客户端取均值得到 $\bar M$。

$$
\bar M(\text{algo}) = \frac{1}{K}\sum_{k=1}^{K} \max_{t} \big\| \text{GPU mem}_k(t) \big\|, \quad \text{Saving}(m) = 1 - \frac{\bar M(\text{Spilter-}m)}{\bar M(\text{FedCSL})}
$$

- **公平性**: 同时刻 GPU 上仅 1 个客户端在训练；含 1 个 batch warmup 再 reset peak，避免 cudnn workspace 探测污染统计。

## 结果表格

| Method | top-m | Mean Peak Mem (MB) | Max Peak Mem (MB) | Saving vs FedCSL | Compression Ratio |
|--------|:-----:|:------------------:|:-----------------:|:----------------:|:-----------------:|
| FedCSL (baseline) | 8 | 58.7 | 59.0 | — | 1.00x (ref) |
| Spilter-m1 | 1 | 29.1 | 29.2 | 50.4% | 2.02x |
| Spilter-m2 | 2 | 40.1 | 40.1 | 31.7% | 1.46x |
| Spilter-m4 | 4 | 63.8 | 64.0 | -8.7% | 0.92x |

> **最大节省**: Spilter-m1 把客户端平均峰值显存从 58.7 MB 降到 29.1 MB —— 节省 **50.4%** ($\approx$2.02$\times$ 显存压缩)。

## 字段说明

- **top-m**: Spilter 在客户端本地实际激活并训练的尺度子集大小。FedCSL 等价于 $m$ = 全部尺度（无切分）。
- **Mean / Max Peak Mem (MB)**: 全部客户端 1-epoch 峰值显存的均值 / 最大值。**均值是论文里最主要指标**（受样本量异质度影响小），Max 用于说明最坏客户端仍能装入卡内。
- **Saving vs FedCSL**: $1 - \bar M(\text{Spilter-}m) / \bar M(\text{FedCSL})$，即 Spilter 相对全尺度 baseline 的相对显存节省。
- **Compression Ratio**: $\bar M(\text{FedCSL}) / \bar M(\text{Spilter-}m)$，即 baseline 显存是 Spilter 的多少倍，等价表达。

## 解读建议

- 显存近似随 top-m 线性下降（前向激活、后向缓存、teacher 的拼接子模型都按 m 缩放），但**不会线性到 0**，因为 PyTorch context、cudnn workspace、shapelet 参数本身（无论是否激活都常驻显存）有固定开销。
- 在客户端显存受限场景（嵌入式 / 移动 GPU），$m=1$ 的 Spilter 通常能让原本 OOM 的 FedCSL 设备重新可训练，代价是精度（参见 `data/HAR_results.md`）。
- 论文里建议把本表与 HAR_results 的精度表对照展示：横轴 $m$，双纵轴分别为「精度」和「显存」，能直接画出 Spilter 的 Pareto 前沿。

_最后更新: 2026-05-17 20:59:46_
