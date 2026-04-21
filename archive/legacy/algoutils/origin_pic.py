import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, find_peaks
from utils.dataset_utils import LoadDataset_HAR 

def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def generate_sampled_time_series(batch_size, num_dimensions, sequence_length, fs=1.0):
    time = np.linspace(0, sequence_length / fs, sequence_length)
    time_series_batch = np.zeros((batch_size, num_dimensions, sequence_length))
    
    for sample_idx in range(batch_size):
        for dim_idx in range(num_dimensions):
            freq = [0.5, 2.0, 5.0][min(sample_idx, 2)] + dim_idx * 0.5
            time_series = np.sin(2 * np.pi * freq * time) + np.random.normal(0, 0.1, sequence_length)
            time_series_batch[sample_idx, dim_idx, :] = time_series
    
    return time_series_batch

def find_dominant_freq_range(data, fs, threshold=0.9):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    cumulative_psd = np.cumsum(psd) / np.sum(psd)
    low_idx = np.argmax(cumulative_psd >= (1 - threshold))
    high_idx = np.argmax(cumulative_psd >= threshold)
    return freqs[low_idx], freqs[high_idx]

def find_top_n_frequencies(data, fs, n=3):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    peaks, _ = find_peaks(psd)
    top_n_idx = np.argsort(psd[peaks])[-n:][::-1]
    top_n_freqs = freqs[peaks][top_n_idx]
    return top_n_freqs

def score_shapelets(shapelet_lengths, top_frequencies, freq_range=(0.01, 0.1)):
    valid_frequencies = [f for f in top_frequencies if freq_range[0] <= f <= freq_range[1]]
    if not valid_frequencies:
        return np.zeros(len(shapelet_lengths))
    top_periods = 1 / np.array(valid_frequencies)
    scores = []
    for L in shapelet_lengths:
        min_distance = np.min(np.abs(L - top_periods))
        score = 1 / (1 + min_distance)
        scores.append(score)
    return np.array(scores)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-3)
    high = min(highcut / nyq, 0.99)
    if low >= high:
        low = max(1e-3, low / 2)
        high = min(0.99, high * 1.5)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def plot_multi_dim_sample(time_series, fs, sample_idx, save_prefix="sample_plot"):
    """
    对于单个样本（形状 (num_dimensions, sequence_length)），生成一个图像文件，
    图中采用2行，每一列对应一个维度：
      - 第一行：该维度的时间序列（原始和滤波后）
      - 第二行：该维度的 PSD（原始和滤波后）
    """
    num_dimensions, sequence_length = time_series.shape
    time = np.linspace(0, sequence_length / fs, sequence_length)
    
    # 创建2行、num_dimensions列的子图，图像尺寸可根据维度数调整
    fig, axs = plt.subplots(2, num_dimensions, figsize=(6 * num_dimensions, 10))
    
    # 若只有一个维度时，axs 维度可能降为1，需要调整为二维数组
    if num_dimensions == 1:
        axs = np.expand_dims(axs, axis=1)
    
    for dim_idx in range(num_dimensions):
        # 时间序列部分
        raw_series = time_series[dim_idx, :]
        lowcut, highcut = find_dominant_freq_range(raw_series, fs, threshold=0.8)
        filtered_series = butter_bandpass_filter(raw_series, lowcut, highcut, fs)
        axs[0, dim_idx].plot(time, raw_series, label="Raw")
        axs[0, dim_idx].plot(time, filtered_series, linestyle='--', 
                             label=f"Filtered ({lowcut:.2f}-{highcut:.2f} Hz)")
        axs[0, dim_idx].set_xlabel("Time (s)")
        axs[0, dim_idx].set_ylabel("Amplitude")
        axs[0, dim_idx].set_title(f"Sample {sample_idx+1}, Dim {dim_idx+1}")
        axs[0, dim_idx].legend()
        
        # PSD 部分
        freqs, psd = welch(raw_series, fs=fs, nperseg=min(256, sequence_length))
        freqs_filt, psd_filt = welch(filtered_series, fs=fs, nperseg=min(256, sequence_length))
        axs[1, dim_idx].semilogy(freqs, psd, label="Raw PSD")
        axs[1, dim_idx].semilogy(freqs_filt, psd_filt, linestyle='--', label="Filtered PSD")
        axs[1, dim_idx].set_xlabel("Frequency (Hz)")
        axs[1, dim_idx].set_ylabel("PSD")
        axs[1, dim_idx].set_title(f"Sample {sample_idx+1}, Dim {dim_idx+1} PSD")
        axs[1, dim_idx].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_sample_{sample_idx+1}.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_time_series_and_psd(time_series, fs, save_prefix="time_series"):
    time_series = ensure_numpy(time_series)
    batch_size, num_dimensions, sequence_length = time_series.shape
    # 针对每个样本绘图，每个样本生成一个文件（多子图，每个子图对应一个维度）
    for sample_idx in range(batch_size):
        sample = time_series[sample_idx]  # shape: (num_dimensions, sequence_length)
        plot_multi_dim_sample(sample, fs, sample_idx, save_prefix=save_prefix)

# 示例：生成时间序列数据及加载 HAR 数据
ts_data = generate_sampled_time_series(batch_size=3, num_dimensions=2, sequence_length=512, fs=10.0)
X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(1, 0.1)
har_data = X_all[:10]
print("Total samples in X_all:", len(X_all))

# 绘制 HAR 数据（假设 har_data 的形状为 (batch_size, num_dimensions, sequence_length)）
# 这里 fs 根据 HAR 数据实际采样率设置，此处示例中设为 1.0 Hz
plot_time_series_and_psd(har_data, fs=1.0, save_prefix="har_sample")
