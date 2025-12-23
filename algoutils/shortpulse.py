import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def generate_ecg_like_signal(N, heart_rate=72, noise_level=0.1):
    """生成类似ECG的信号"""
    t = np.linspace(0, 10, N)
    ecg = np.zeros(N)
    heart_period = int(N / (heart_rate * 10 / 60))
    for i in range(0, N, heart_period):
        if i + 5 < N:
            ecg[i:i+5] += np.exp(-((np.arange(5) - 2) / 0.5) ** 2)
    ecg += noise_level * np.random.randn(N)
    return ecg

def generate_simulated_data(N, D, T):
    """生成N个样本，D维特征，T时间点的数据"""
    signals = np.zeros((N, D, T))
    for i in range(N):
        for j in range(D):
            signals[i, j, :] = generate_ecg_like_signal(T)
    return signals

def periodicity_score(signal, nlags=40):
    """计算周期性得分"""
    acf_values = sm.tsa.acf(signal, nlags=nlags, fft=True)[1:]
    peak_acf = max(acf_values[1:])
    
    N = len(signal)
    fft_vals = np.abs(np.fft.fft(signal))[:N // 2]
    fft_power = fft_vals ** 2
    fft_power /= np.sum(fft_power)
    peak_fft = np.max(fft_power)
    
    score = (peak_acf + peak_fft) / 2
    main_period = np.argmax(acf_values) + 1
    
    return score, main_period

# 参数设置
N = 3  # 样本数量
D = 2  # 维度数量
T = 500  # 时间点数量

# 数据生成
signals = generate_simulated_data(N, D, T)

# 分析并绘制图形
fig_width = 16
fig_height_per_subplot = 4
fig_height = fig_height_per_subplot * D * N
fig, axes = plt.subplots(N * D, 2, figsize=(fig_width, fig_height))

for sample_idx in range(N):
    for feature_idx in range(D):
        signal = signals[sample_idx, feature_idx, :]
        score, main_period = periodicity_score(signal)
        
        # 时间序列图
        time_series_ax = axes[sample_idx * D + feature_idx, 0]
        time_series_ax.plot(signal, color='b')
        time_series_ax.set_title(f"Sample {sample_idx+1}, Feature {feature_idx+1} (Score: {score:.2f}, Period: {main_period})")
        time_series_ax.set_xlabel("Time")
        time_series_ax.set_ylabel("Value")

        # ACF图
        acf_ax = axes[sample_idx * D + feature_idx, 1]
        sm.graphics.tsa.plot_acf(signal, lags=40, ax=acf_ax)
        acf_ax.set_title(f"ACF of Sample {sample_idx+1}, Feature {feature_idx+1}")

plt.tight_layout()
plt.show()