import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import origin_pic

def plot_psd_to_file(time_series, fs=1.0, num_bands=8, filename="psd_plot.png"):
    """
    绘制时间序列的功率谱密度（PSD）和频段划分，并保存为图片文件。
    :param time_series: 单个时间序列
    :param fs: 采样频率
    :param num_bands: 频段数量
    :param filename: 保存的图片文件名
    """
    frequencies, power_spectrum = welch(time_series, fs=fs, nperseg=len(time_series))
    
    # 确定最大频率
    f_max = frequencies[-1]
    
    # 划分频段
    band_edges = np.linspace(0, f_max, num_bands + 1)
    
    # 绘制 PSD 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, power_spectrum, label="Power Spectral Density")
    
    # 标记频段划分
    for edge in band_edges:
        plt.axvline(edge, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Power Spectral Density with Band Divisions")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    
    # 保存图片
    plt.savefig(filename)
    plt.close()  # 关闭图像以释放内存

def plot_weights_to_file(weights, title="Adaptive Weights", filename="weights_plot.png"):
    """
    绘制自适应权重分布，并保存为图片文件。
    :param weights: 权重数组，形状为 (batch_size, num_bands)
    :param title: 图表标题
    :param filename: 保存的图片文件名
    """
    batch_size, num_bands = weights.shape
    
    plt.figure(figsize=(10, 6))
    for i in range(batch_size):
        plt.plot(range(num_bands), weights[i], label=f"Sample {i+1}")
    
    plt.title(title)
    plt.xlabel("Frequency Bands")
    plt.ylabel("Weight")
    plt.xticks(range(num_bands), [f"Band {i+1}" for i in range(num_bands)])
    plt.legend()
    
    # 保存图片
    plt.savefig(filename)
    plt.close()  # 关闭图像以释放内存

def plot_energy_vs_weights_to_file(energy_per_band, weights, title="Energy vs Weights", filename="energy_vs_weights_plot.png"):
    """
    绘制能量占比与权重的关系，并保存为图片文件。
    :param energy_per_band: 能量占比数组，形状为 (batch_size, num_bands)
    :param weights: 权重数组，形状为 (batch_size, num_bands)
    :param title: 图表标题
    :param filename: 保存的图片文件名
    """
    batch_size, num_bands = weights.shape
    
    plt.figure(figsize=(10, 6))
    for i in range(batch_size):
        plt.scatter(energy_per_band[i], weights[i], label=f"Sample {i+1}", alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Energy Proportion per Band")
    plt.ylabel("Weight")
    plt.legend()
    
    # 保存图片
    plt.savefig(filename)
    plt.close()  # 关闭图像以释放内存

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

def compute_significant_frequency_range(frequencies, power_spectrum, threshold_percentile=90):
    """
    确定显著频率分量的范围。
    :param frequencies: 频率数组
    :param power_spectrum: 功率谱密度数组
    :param threshold_percentile: 用于确定显著频率范围的百分位数
    :return: 显著频率范围 [f_low, f_high]
    """
    # 计算功率谱密度的阈值
    threshold = np.percentile(power_spectrum, threshold_percentile)
    
    # 找到超过阈值的频率索引
    significant_mask = power_spectrum >= threshold
    significant_frequencies = frequencies[significant_mask]
    
    if len(significant_frequencies) == 0:
        raise ValueError("未检测到显著频率分量，请调整阈值或数据预处理。")
    
    # 确定显著频率范围
    f_low = np.min(significant_frequencies)
    f_high = np.max(significant_frequencies)
    
    return f_low, f_high

def compute_spectral_weights_batch(time_series_batch, fs=1.0, num_bands=8, alpha=2.0, threshold_percentile=90):
    """
    基于频域分析计算批量多维时间序列的自适应 Shapelet 权重。
    :param time_series_batch: 批量多维时间序列，形状为 (batch_size, num_dimensions, sequence_length)
    :param fs: 采样频率
    :param num_bands: 频段数量
    :param alpha: 超参数，控制高频分量的增强程度
    :param threshold_percentile: 用于确定显著频率范围的百分位数
    :return: 自适应权重列表，形状为 (batch_size, num_bands)
    """
    batch_size, num_dimensions, sequence_length = time_series_batch.shape
    
    # 初始化权重存储
    final_weights_batch = np.zeros((batch_size, num_bands))
    
    for sample_idx in range(batch_size):  # 遍历每个样本
        energy_per_band_sample = np.zeros(num_bands)  # 存储当前样本的能量分布
        
        for dim_idx in range(num_dimensions):  # 遍历每个维度
            time_series = time_series_batch[sample_idx, dim_idx, :]  # 当前维度的时间序列
            
            # 计算功率谱密度
            frequencies, power_spectrum = welch(time_series, fs=fs, nperseg=sequence_length)
            
            # 确定显著频率范围
            try:
                f_low, f_high = compute_significant_frequency_range(frequencies, power_spectrum, threshold_percentile)
            except ValueError as e:
                print(f"样本 {sample_idx}, 维度 {dim_idx}: {e}")
                continue
            
            # 划分频段（仅在显著频率范围内）
            band_edges = np.linspace(f_low, f_high, num_bands + 1)
            print(f_low,f_high)
            print("\n")
            # 计算每个频段的能量
            energy_per_band_dim = []
            for i in range(num_bands):
                low, high = band_edges[i], band_edges[i + 1]
                mask = (frequencies >= low) & (frequencies < high)
                energy = np.trapz(power_spectrum[mask], frequencies[mask])
                energy_per_band_dim.append(energy)
            
            # 累加当前维度的能量分布
            energy_per_band_sample += np.array(energy_per_band_dim)
        
        # 归一化能量
        if np.sum(energy_per_band_sample) > 0:  # 避免除零错误
            normalized_energy = energy_per_band_sample / np.sum(energy_per_band_sample)
        else:
            normalized_energy = np.ones(num_bands) / num_bands
        
        # 应用映射函数
        weights = normalized_energy ** alpha
        
        # 归一化权重
        final_weights_batch[sample_idx, :] = weights / np.sum(weights)
    
    return final_weights_batch

# 示例数据
np.random.seed(42)  # 固定随机种子以确保结果一致
batch_size = 3      # 样本数量
num_dimensions = 1  # 时间序列维度数
sequence_length = 128  # 时间序列长度
fs = 5.0            # 采样频率

# 生成频率区分明显的时间序列数据
time_series_batch = origin_pic.generate_sampled_time_series(batch_size, num_dimensions, sequence_length, fs)

# 计算自适应权重
weights = compute_spectral_weights_batch(time_series_batch, fs=1.0, num_bands=8, alpha=1, threshold_percentile=95)
print(weights)

# 可视化 PSD 并保存为图片
plot_psd_to_file(time_series_batch[2, 0, :], fs=1.0, num_bands=8, filename="psd_plot.png")

# 可视化权重分布并保存为图片
plot_weights_to_file(weights, title="Adaptive Weights for Each Sample", filename="weights_plot.png")

# 验证高频与权重关系并保存为图片
energy_per_band_batch = []
for sample_idx in range(batch_size):
    energy_per_band_sample = np.zeros(8)
    for dim_idx in range(num_dimensions):
        time_series = time_series_batch[sample_idx, dim_idx, :]
        frequencies, power_spectrum = welch(time_series, fs=1.0, nperseg=sequence_length)
        
        # 确定显著频率范围
        try:
            f_low, f_high = compute_significant_frequency_range(frequencies, power_spectrum, threshold_percentile=90)
        except ValueError as e:
            print(f"样本 {sample_idx}, 维度 {dim_idx}: {e}")
            continue
        
        # 划分频段
        band_edges = np.linspace(f_low, f_high, 9)
        energy_per_band_dim = []
        for i in range(8):
            low, high = band_edges[i], band_edges[i + 1]
            mask = (frequencies >= low) & (frequencies < high)
            energy = np.trapz(power_spectrum[mask], frequencies[mask])
            energy_per_band_dim.append(energy)
        energy_per_band_sample += np.array(energy_per_band_dim)
    energy_per_band_batch.append(energy_per_band_sample / np.sum(energy_per_band_sample))

energy_per_band_batch = np.array(energy_per_band_batch)
plot_energy_vs_weights_to_file(energy_per_band_batch, weights, title="Energy Proportion vs Weights", filename="energy_vs_weights_plot.png")