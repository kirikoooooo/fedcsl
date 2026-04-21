import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, find_peaks
from utils.dataset_utils import LoadDataset_HAR 
# 示例：生成时间序列数据及加载 HAR 数据
# ts_data = generate_sampled_time_series(batch_size=3, num_dimensions=2, sequence_length=512, fs=10.0)
X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(1, 0.1)
har_data = X_all
print("Total samples in X_all:", len(X_all))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks
from statsmodels.tsa.stattools import acf

# 输入数据 (假设 har_data 是一个三维数组，形状为 (样本数, 维度数, 时间步长))
# 例如：har_data.shape = (N, K, L)

# 调整系数（0 <= alpha <= 1）
# alpha=0.5 表示 STFT 和 ACF 各占 50%
# alpha=0.7 表示 STFT 占 70%，ACF 占 30%
alpha = 0.4  # 用户可根据需求调整

# 目标点（归一化到 [0,1]）
list_points = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# 初始化总得分
total_scores_stft = np.zeros(len(list_points))  # STFT 得分
total_scores_acf = np.zeros(len(list_points))   # ACF 得分

# ACF 峰值检测门限
acf_threshold = 0.2  # 可根据需要调整

# 遍历所有 HAR 样本
for sample_idx in range(har_data.shape[0]):  # 遍历样本数 N
    sample_data = har_data[sample_idx]  # 提取单个样本，形状为 (K, L)
    K, L = sample_data.shape  # K: 维度数, L: 时间步长
    list_points_scaled = (list_points * L).astype(int)  # 按 L 计算真实目标点

    # 选择前 k 维度
    k = min(9, K)  # 最多选择 9 个维度

    # 1. STFT 分析
    stft_scores = []

    for i in range(k):
        # 重新计算 STFT 相关参数
        window_size = max(4, L // 2)
        n_fft = 8 * window_size
        hop_size = window_size // 4

        # 计算 STFT
        f, _, Zxx = stft(sample_data[i, :], fs=1, nperseg=window_size, noverlap=window_size - hop_size, nfft=n_fft)

        # 计算功率谱
        power_spectrum = np.abs(Zxx) ** 2

        # 找到功率最大的前 5%
        threshold = np.percentile(power_spectrum, 95)
        bright_region = power_spectrum >= threshold

        # 计算加权中心频率
        if np.sum(bright_region) > 0:
            weighted_freq = np.sum(f[:, None] * power_spectrum * bright_region) / np.sum(power_spectrum * bright_region)
        else:
            weighted_freq = 0  # 防止除零

        # 计算当前维度的 STFT 得分
        current_score = 1 / weighted_freq if weighted_freq != 0 else 0
        stft_scores.append(current_score)

    # 转换为数组并计算高斯核得分
    stft_scores = np.array(stft_scores)
    stft_gauss = np.exp(-((stft_scores[:, None] - list_points_scaled) ** 2) / (2 * (L * 0.1) ** 2))
    stft_normalized = stft_gauss / stft_gauss.sum(axis=1, keepdims=True)
    total_scores_stft += stft_normalized.sum(axis=0)  # 累加所有维度的得分

    # 2. ACF 分析
    acf_scores = []

    for i in range(k):
        # 计算当前维度的 ACF
        time_series = sample_data[i, :]
        acf_values = acf(time_series, nlags=L-1, fft=True)
        
        # 找到 lag > L/2 的峰值点
        half_lag = L // 2
        peaks, _ = find_peaks(acf_values[half_lag:], height=acf_threshold)
        peak_lags = peaks + half_lag

        if len(peak_lags) == 0:
            continue

        # 计算每个峰值点与参考点的距离得分
        current_scores = np.zeros(len(list_points))
        for lag in peak_lags:
            distances = np.abs(lag - list_points_scaled)
            current_scores += 1 / (distances + 1e-6)
        
        # 归一化当前维度的得分
        current_scores /= current_scores.sum()
        acf_scores.append(current_scores)

    if acf_scores:
        acf_scores = np.array(acf_scores)
        acf_normalized = acf_scores / acf_scores.sum(axis=1, keepdims=True)
        total_scores_acf += acf_normalized.sum(axis=0)  # 累加所有维度的得分

# 合并得分：前4个用STFT，后4个用ACF，并根据alpha调整权重
stft_part = total_scores_stft[:4]
acf_part = total_scores_acf[4:]

# 归一化各自的得分（防止分母为零）
stft_sum = stft_part.sum()
acf_sum = acf_part.sum()

stft_part_normalized = stft_part / stft_sum if stft_sum != 0 else np.zeros_like(stft_part)
acf_part_normalized = acf_part / acf_sum if acf_sum != 0 else np.zeros_like(acf_part)

# 加权合并并归一化
final_scores = np.concatenate([
    stft_part_normalized * alpha,
    acf_part_normalized * (1 - alpha)
])
final_scores /= final_scores.sum()

# 打印结果
print("Final Normalized Scores for Each Reference Point:")
for i, score in enumerate(final_scores):
    print(f"Point {list_points[i]:.1f}L: Score = {score:.4f}")

# 可视化
plt.figure(figsize=(8, 5))
plt.bar(list_points * 100, final_scores, width=5, color='green', alpha=0.7, edgecolor='k')
plt.xlabel("Reference Points (% of L)")
plt.ylabel("Final Normalized Score")
plt.title("Final Scores Across Reference Points (α={:.1f})".format(alpha))
plt.xticks(list_points * 100, labels=[f"{p*100:.0f}%" for p in list_points])
plt.grid(alpha=0.3)
#plt.show()
plt.savefig("acf2.png")  # 减少dpi值以降低图像质量/大小