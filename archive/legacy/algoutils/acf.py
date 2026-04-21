import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.fftpack import fft

from utils.dataset_utils import LoadDataset_HAR 
# **Set Matplotlib font for English display**

def periodicity_score(signal, nlags=100):
    acf_values = sm.tsa.acf(signal, nlags=nlags, fft=True)[1:]
    peak_acf = max(acf_values[1:])
    
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N // 2]
    fft_power = fft_vals ** 2
    fft_power /= np.sum(fft_power)
    peak_fft = np.max(fft_power)
    
    score = (peak_acf + peak_fft) / 2
    main_period = np.argmax(acf_values) + 1
    
    return score, acf_values, main_period

# 假设已经正确加载了数据
scores = {}
acf_results = {}
X_all, y_all, X_test, y_test, X_fed, y_fed  = LoadDataset_HAR(1,0.1)
signals = X_all
N = len(signals)
D =signals.shape[1]
T = signals.shape[2]
for sample_idx in range(N):
    sample = signals[sample_idx]  # 形状 (D, T)
    feature_scores = []
    feature_main_periods = []
    feature_acfs = []

    for feature_idx in range(D):
        signal = sample[feature_idx]  # 形状 (T,)
        score, acf_values, main_period = periodicity_score(signal)
        feature_scores.append(score)
        feature_main_periods.append(main_period)
        feature_acfs.append(acf_values)

    #scores[sample_idx] = (np.mean(feature_scores), np.mean(feature_main_periods))
    scores[sample_idx] = np.mean(feature_scores)
    acf_results[sample_idx] = feature_acfs

#print(scores)

def custom_weights(lengths, r, k=0.1):
    """
    根据归一化自相关性 r 和 shapelet 长度生成权重：
    
    当 r >= 0.5 时：采用递增后饱和函数，即：
        f(L) = 1 / (1 + exp(-k*(L - L0)))
    当 r < 0.5 时：采用先稳定后递减函数，即：
        f(L) = 1 - 1 / (1 + exp(-k*(L - L0)))
    
    参数:
        lengths: list 或 numpy array，shapelet 长度列表，例如 [10, 20, 30, ..., 80]
        r: 归一化自相关性（范围 [0,1]）
        k: 控制 sigmoid 曲线陡峭程度的参数
    返回:
        weights: 归一化后的权重数组，和为 1
    """
    lengths = np.array(lengths, dtype=float)
    # 选择 L0 为 lengths 的中位数
    L0 = np.median(lengths)
    
    if r >= 0.6:
        # 高 r：权重随着 L 增加，先递增后趋于饱和
        f = 1 / (1 + np.exp(-k * (lengths - L0)))
    else:
        # 低 r：权重随 L 增加而下降，先保持较高，再递减
        f = 1 - 1 / (1 + np.exp(-k * (lengths - L0)))
    
    weights = f / np.sum(f)
    return weights

def custom_weights_piecewise(lengths, r, k=0.1):
    """
    根据归一化自相关性 r 和 shapelet 长度采用分段策略生成权重：
    
    当 r >= 0.6 时，采用递增后饱和函数：
        f_inc(L) = 1 / (1 + exp(-k*(L - L0)))
        
    当 r <= 0.4 时，采用先稳定后递减函数：
        f_dec(L) = 1 - 1 / (1 + exp(-k*(L - L0)))
        
    当 r 在 (0.4, 0.6) 区间内时，使用线性插值：
        alpha = (r - 0.4)/0.2，
        f(L) = alpha * f_inc(L) + (1 - alpha) * f_dec(L)
    
    参数:
        lengths: list 或 numpy array，shapelet 长度列表，例如 [10, 20, 30, ..., 80]
        r: 归一化自相关性（范围 [0,1]）
        k: 控制 sigmoid 曲线陡峭程度的参数
    返回:
        weights: 归一化后的权重数组，和为 1
    """
    lengths = np.array(lengths, dtype=float)
    # 选取 L0 为 lengths 的中位数（你也可以根据实际需求选择其他中心值）
    L0 = np.median(lengths)
    
    # 计算递增函数
    f_inc = 1 / (1 + np.exp(-k * (lengths - L0)))
    # 计算递减函数
    f_dec = 1 - f_inc
    
    if r >= 0.6:
        f = f_inc
    elif r <= 0.4:
        f = f_dec
    else:
        # 对于 r 在 (0.4, 0.6) 内进行线性插值
        alpha = (r - 0.4) / 0.2  # 当 r=0.4 时 alpha=0，当 r=0.6 时 alpha=1
        f = alpha * f_inc + (1 - alpha) * f_dec
    
    weights = f / np.sum(f)
    return weights

# lengths = [10, 20, 30, 40,50,60,70,80]  # 示例 shapelet 长度
# beta = 0.03                # 调节参数

# all_exp_weights = []
# for sample_idx in range(N):
#     r = scores[sample_idx] # 归一化的自相关性
#     exp_weights = custom_weights_piecewise(lengths, r, beta)
#     all_exp_weights.append(exp_weights)
    
# print(all_exp_weights[:10])
# all_exp_weights = np.array(all_exp_weights)

# # 权重按数据集级别，not 样本级别
# summed_array = np.sum(all_exp_weights, axis=0)  # 结果形状为 (8,)
# l1_norm = np.abs(summed_array).sum()
# print(summed_array/l1_norm)

# np.save('shapelet_weight_All.npy', summed_array/l1_norm)



# 绘制前五个样本的所有维度的时间序列图和ACF图
# 绘制前五个样本的所有维度的时间序列图和ACF图
num_samples_to_plot = min(N, 5)
fig_width = 16  # 图形宽度
fig_height_per_subplot = 4  # 每个子图的高度
fig_height = fig_height_per_subplot * D * num_samples_to_plot  # 总高度根据子图数量动态计算

fig, axes = plt.subplots(num_samples_to_plot * D, 2, figsize=(fig_width, fig_height))

for i, sample_idx in enumerate(range(num_samples_to_plot)):
    sample = signals[sample_idx]
    
    for feature_idx in range(D):
        # 计算当前子图的索引
        time_series_ax = axes[i * D + feature_idx, 0]
        acf_ax = axes[i * D + feature_idx, 1]
        
        # 绘制第一个特征的时间序列
        time_series_ax.plot(sample[feature_idx], color='b')
        #time_series_ax.set_title(f"Sample {sample_idx+1}, Feature {feature_idx+1} (Score: {scores[sample_idx][0]:.2f}, Period: {scores[sample_idx][1]})")
        time_series_ax.set_xlabel("Time")
        time_series_ax.set_ylabel("Value")

        # 绘制第一个特征的ACF
        sm.graphics.tsa.plot_acf(sample[feature_idx], lags=120, ax=acf_ax)  # 使用statsmodels内置函数绘制ACF
        acf_ax.set_title(f"ACF of Sample {sample_idx+1}, Feature {feature_idx+1}")

plt.tight_layout()
plt.savefig("periodicity_analysis_tensor.png", dpi=150, bbox_inches='tight')  # 减少dpi值以降低图像质量/大小
plt.show()