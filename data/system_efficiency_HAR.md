# HAR 数据集系统效率实验：最慢客户端 1-epoch 耗时

## 实验设置

- **数据集**: HAR (Human Activity Recognition)
- **联邦客户端数**: K = 50, Dirichlet $\alpha$ = 0.1
- **本地 epoch 数**: 1（一次客户端调用 = 1 个 epoch）
- **batch size**: 32
- **样本形状**: N=5881, C=9, T=128
- **GPU / 设备**: NVIDIA GeForce RTX 3090
- **测量方式**: 对每个算法、每个客户端**串行**独占 GPU 跑 1 epoch，`torch.cuda.synchronize()` 前后用 `time.perf_counter()` 计时；对所有客户端取最大值作为该算法的最慢客户端单 epoch 耗时。
- **公平性**: 同时刻 GPU 上仅 1 个 client/算法在跑；含 1 个 batch warmup 避免首次 cudnn 算子选择影响。

## 结果表格

| Method | Slowest Client Epoch (s) | #Samples@Slowest | Mean (s) | Median (s) | Min (s) | Mean Peak Mem (MB) | Max Peak Mem (MB) | Timed/Skipped |
|--------|:------------------------:|:----------------:|:--------:|:----------:|:-------:|:------------------:|:-----------------:|:-------------:|
| FedAvg | **1.283** | 332 | 0.521 | 0.496 | 0.087 | 177.0 | 180.3 | 36/14 |
| FedProx | 1.467 | 323 | 0.641 | 0.542 | 0.124 | 179.2 | 182.4 | 36/14 |
| FedBYOL | 1.886 | 336 | 0.879 | 0.814 | 0.267 | 182.9 | 183.4 | 36/14 |
| FedU2 | 1.928 | 336 | 0.915 | 0.788 | 0.312 | 182.9 | 183.4 | 36/14 |
| Orchestra | 2.283 | 336 | 1.102 | 0.963 | 0.345 | 231.3 | 233.5 | 36/14 |
| FedCSL | **4.695** | 336 | 2.049 | 1.927 | 0.373 | 178.1 | 181.3 | 36/14 |
| Spilter-m1 | 3.169 | 332 | 1.337 | 1.078 | 0.218 | 51.5 | 51.7 | 36/14 |
| Spilter-m2 | 3.434 | 332 | 1.444 | 1.250 | 0.244 | 78.7 | 79.1 | 36/14 |
| Spilter-m4 | 3.758 | 332 | 1.616 | 1.514 | 0.342 | 130.3 | 131.5 | 36/14 |

> 最快算法（最慢客户端 epoch 最短）: **FedAvg**；最慢算法（最慢客户端 epoch 最长）: **FedCSL**。

## 字段说明

- **Slowest Client Epoch (s)**: $\max_k T_k^{\text{epoch}}$，该算法所有客户端中跑完 1 epoch 用时最长的那个。对应 round 同步联邦的 min--max 时延分析里的拖尾客户端。
- **#Samples@Slowest**: 最慢客户端的本地样本数。对照样本数与耗时可粗略反映「耗时随样本量线性增长」/「客户端样本异质度」的影响。
- **Mean / Median / Min**: 全部客户端 1-epoch 耗时的统计量。
- **Timed/Skipped**: 实际计时的客户端数 / 因样本数 < batch_size 跳过的客户端数。

## 解读建议

- 同步 FedAvg 类联邦协议的每轮 wall-clock 至少为 `Slowest Client Epoch × local_epoch + 通信/聚合`。因此该指标可作为 round-time 下界的代理。
- Spilter (m=1/2/4) 与 FedCSL 的差距反映尺度切分对客户端本地算力开销的削减。
- BYOL / FedU2 / Orchestra 走 SSL 路径，模型结构与 forward 不同，时间不直接可比；但相对 FedCSL 仍可作为同 backbone 不同自监督方法的代价对比。

_最后更新: 2026-05-17 23:02:38_
