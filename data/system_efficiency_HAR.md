# HAR 数据集系统效率实验：最慢客户端 1-epoch 耗时

## 实验设置

- **数据集**: HAR (Human Activity Recognition)
- **联邦客户端数**: K = 50, Dirichlet $\alpha$ = 0.1
- **本地 epoch 数**: 1（一次客户端调用 = 1 个 epoch）
- **batch size**: 8
- **样本形状**: N=5881, C=9, T=128
- **GPU / 设备**: NVIDIA GeForce RTX 3090
- **测量方式**: 对每个算法、每个客户端**串行**独占 GPU 跑 1 epoch，`torch.cuda.synchronize()` 前后用 `time.perf_counter()` 计时；对所有客户端取最大值作为该算法的最慢客户端单 epoch 耗时。
- **公平性**: 同时刻 GPU 上仅 1 个 client/算法在跑；含 1 个 batch warmup 避免首次 cudnn 算子选择影响。

## 结果表格

| Method | Slowest Client Epoch (s) | #Samples@Slowest | Mean (s) | Median (s) | Min (s) | Mean Peak Mem (MB) | Max Peak Mem (MB) | Timed/Skipped |
|--------|:------------------------:|:----------------:|:--------:|:----------:|:-------:|:------------------:|:-----------------:|:-------------:|
| FedAvg | 4.887 | 336 | 1.647 | 1.012 | 0.096 | 57.6 | 57.9 | 50/0 |
| FedProx | 5.445 | 336 | 1.878 | 1.200 | 0.105 | 59.8 | 60.1 | 50/0 |
| FedBYOL | 6.578 | 323 | 2.310 | 1.565 | 0.264 | 61.3 | 61.3 | 50/0 |
| FedU2 | 7.393 | 323 | 2.465 | 1.637 | 0.274 | 61.3 | 61.3 | 50/0 |
| Orchestra | 7.049 | 323 | 2.489 | 1.664 | 0.297 | 74.0 | 74.0 | 50/0 |
| FedCSL | **10.876** | 336 | 3.671 | 2.357 | 0.226 | 58.7 | 59.0 | 50/0 |
| Spilter-m1 | **4.580** | 336 | 1.624 | 1.045 | 0.103 | 29.1 | 29.2 | 50/0 |
| Spilter-m2 | 6.647 | 336 | 2.262 | 1.425 | 0.127 | 40.1 | 40.1 | 50/0 |
| Spilter-m4 | 10.135 | 336 | 3.428 | 2.186 | 0.209 | 63.8 | 64.0 | 50/0 |

> 最快算法（最慢客户端 epoch 最短）: **Spilter-m1**；最慢算法（最慢客户端 epoch 最长）: **FedCSL**。

## 字段说明

- **Slowest Client Epoch (s)**: $\max_k T_k^{\text{epoch}}$，该算法所有客户端中跑完 1 epoch 用时最长的那个。对应 round 同步联邦的 min--max 时延分析里的拖尾客户端。
- **#Samples@Slowest**: 最慢客户端的本地样本数。对照样本数与耗时可粗略反映「耗时随样本量线性增长」/「客户端样本异质度」的影响。
- **Mean / Median / Min**: 全部客户端 1-epoch 耗时的统计量。
- **Timed/Skipped**: 实际计时的客户端数 / 因样本数 < batch_size 跳过的客户端数。

## 解读建议

- 同步 FedAvg 类联邦协议的每轮 wall-clock 至少为 `Slowest Client Epoch × local_epoch + 通信/聚合`。因此该指标可作为 round-time 下界的代理。
- Spilter (m=1/2/4) 与 FedCSL 的差距反映尺度切分对客户端本地算力开销的削减。
- BYOL / FedU2 / Orchestra 走 SSL 路径，模型结构与 forward 不同，时间不直接可比；但相对 FedCSL 仍可作为同 backbone 不同自监督方法的代价对比。

_最后更新: 2026-05-17 20:59:46_
