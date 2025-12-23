import numpy as np
import torch
from torch import nn, optim

import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
#import umap
def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters


class MaskBlock(nn.Module):
    def __init__(self, p=0.1):
        super(MaskBlock, self).__init__()

        self.net = nn.Dropout(p=p)
    def forward(self, X):
        return self.net(X)



class LinearBlock(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(LinearBlock, self).__init__()

        #self.linear = nn.Sequential(nn.Linear(in_channel, 256), nn.ReLU(), nn.Linear(256, n_classes))
        self.linear = nn.Linear(in_channel, n_classes)

    def forward(self, X):
        return self.linear(X)

class LinearClassifier():
    def __init__(self, in_channel, n_classes, batch_size=256, lr=1e-3, wd=1e-4, max_epoch=200):
        super(LinearClassifier, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.lr = lr

        self.wd = wd
        self.max_epoch = max_epoch

        self.net = LinearBlock(in_channel, n_classes)


    def train(self, X, y):
        X = torch.from_numpy(X)
        X = X.float()

        y = torch.from_numpy(y)
        y = y.long()

        self.net.cuda()

        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, min_lr=0.0001)

        # build dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=True)




        self.net.train()

        for epoch in range(self.max_epoch):
            losses = []
            for (x, y) in train_loader:
                x = x.cuda()
                y = y.cuda()
                logits = self.net(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step(loss)


    def predict(self, X):
        X = torch.from_numpy(X)
        X = X.float()
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=False)

        predict_list = np.array([])

        self.net.eval()

        for (x, ) in loader:
            x = x.cuda()
            y_predict = self.net(x)
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)

        return predict_list



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def z_normalize(a, eps=1e-7):
    return (a - np.mean(a, axis=-1, keepdims=True)) / (eps + np.std(a, axis=-1, keepdims=True))


#def replace_nan_with_row_mean(a):
#    out = np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(axis=1)[:, np.newaxis], a)
#    return np.float32(out)

def replace_nan_with_near_value(a):
    mask = np.isnan(a)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = a[np.arange(idx.shape[0])[:,None], idx]
    return np.float32(out)

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)


def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i

                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length

        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)

        return np.float32(X), Y




def TSC_multivariate_data_loader(dataset_path, dataset_name):

    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)

    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

def generate_binomial_mask(size, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=size)).cuda()


def eval_accuracy(model, X, Y, X_test, Y_test, normalize=False, lr=1e-3, wd=1e-4):
    transformation = model.transform(X, result_type='numpy', normalize=normalize)
    clf = LinearClassifier(transformation.shape[1], len(set(Y)), lr=lr, wd=wd)
    clf.train(transformation, Y)
    acc_train = accuracy_score(clf.predict(transformation), Y)
    acc_test = accuracy_score(clf.predict(model.transform(X_test, result_type='numpy', normalize=normalize)), Y_test)
    return acc_train, acc_test



def direct_kl_loss(x1, x2, t=0.1):
    """Direct KL divergence loss objective function between x1 and x2"""

    # Normalize the inputs to ensure they are comparable.
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    # Compute the similarity matrices for each pair of samples in x1 and x2 with themselves.
    pred_sim1 = torch.mm(x1, x1.t())  # Similarity matrix for x1
    pred_sim2 = torch.mm(x2, x2.t())  # Similarity matrix for x2

    # Apply temperature scaling and convert similarities into log probabilities.
    log_prob_x1 = F.log_softmax(pred_sim1 / t, dim=1)
    prob_x2 = F.softmax(pred_sim2 / t, dim=1)

    # Calculate the KL divergence from log_prob_x1 to prob_x2.
    kl_loss = F.kl_div(log_prob_x1, prob_x2, reduction="batchmean")

    return kl_loss


# 选择降维方法之一
def reduce_dimensions(vectors, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000, random_state=42)
    # elif method == 'umap':
    #     reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unknown dimensionality reduction method")

    reduced_vectors = reducer.fit_transform(vectors)
    return reduced_vectors


def draw_scatter_plot(vectors, labels=None):
    # 应用降维
    reduced_vectors = reduce_dimensions(vectors, method='tsne')  # 你可以选择 'pca', 'tsne' 或 'umap'

    # 创建 DataFrame 以便于绘图
    df = pd.DataFrame(reduced_vectors, columns=['Component 1', 'Component 2'])
    if labels is not None:
        df['Label'] = labels

    # 使用 Seaborn 绘制散点图
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='Component 1', y='Component 2',
        hue='Label' if 'Label' in df.columns else None,
        palette=sns.color_palette("hls", len(set(labels)) if labels is not None else 1),
        data=df,
        s=50, alpha=0.6
    )

    plt.title('Scatter Plot of Reduced Dimensional Vectors')
    plt.savefig('scatter_plot_high_quality.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()



# 对batch 内样本进行多周期提取，多尺度评分
# har_data: (batch_size, n_dimensions, time_steps)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks
from statsmodels.tsa.stattools import acf

def period_score(x,alpha=0.4):
    # 输入数据 (假设 har_data 是一个三维数组，形状为 (样本数, 维度数, 时间步长))
    # 例如：har_data.shape = (N, K, L)
    #print(x.shape)
    har_data = x.cpu().numpy()
    # 调整系数（0 <= alpha <= 1）
    # alpha=0.5 表示 STFT 和 ACF 各占 50%
    # alpha=0.7 表示 STFT 占 70%，ACF 占 30%
    #alpha = 0.4  # 用户可根据需求调整

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

    #打印结果
    # print("Final Normalized Scores for Each Reference Point:")
    # for i, score in enumerate(final_scores):
    #     print(f"Point {list_points[i]:.1f}L: Score = {score:.4f}")
    return final_scores * 10




## 只有两个得分
def period_score_onehot(x, alpha=0.4):

    """
    输入数据 x 是 PyTorch 张量，形状为 (N, K, L)
    N: 样本数
    K: 维度数
    L: 时间步长
    输出：长度为 8 的数组，表示每个参考点的得分（集中在 STFT 和 ACF 各一个点）
    """

    # 将输入张量转换为 NumPy 数组
    har_data = x.cpu().numpy()
    N, K, L = har_data.shape

    # 参考点（归一化到 [0,1] 区间）
    list_points = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    list_points_scaled = (list_points * L).astype(int)  # 按 L 计算真实目标点

    # 初始化总得分
    total_scores_stft = np.zeros(len(list_points[:4]))  # 前4个用 STFT
    total_scores_acf = np.zeros(len(list_points[4:]))   # 后4个用 ACF

    # ACF 峰值检测门限
    acf_threshold = 0.2

    # 遍历所有样本
    for sample_idx in range(N):  # 遍历样本数 N
        sample_data = har_data[sample_idx]  # 提取单个样本，形状为 (K, L)

        # 选择前 k 个维度（最多9个）
        k = min(9, K)

        # --- 1. STFT 分析 ---
        stft_scores = []

        for i in range(k):
            window_size = max(4, L // 2)
            n_fft = 8 * window_size
            hop_size = window_size // 4

            f, _, Zxx = stft(sample_data[i, :], fs=1, nperseg=window_size,
                             noverlap=window_size - hop_size, nfft=n_fft)
            power_spectrum = np.abs(Zxx) ** 2

            if np.sum(power_spectrum) > 0:
                weighted_freq = np.sum(f[:, None] * power_spectrum) / np.sum(power_spectrum)
            else:
                weighted_freq = 0

            current_score = 1 / weighted_freq if weighted_freq != 0 else 0
            stft_scores.append(current_score)

        stft_scores = np.array(stft_scores)
        stft_gauss = np.exp(-((stft_scores[:, None] - list_points_scaled[:4]) ** 2) /
                            (2 * (L * 0.1) ** 2))
        stft_normalized = stft_gauss / stft_gauss.sum(axis=1, keepdims=True)
        total_scores_stft += stft_normalized.sum(axis=0)

        # --- 2. ACF 分析 ---
        acf_scores = []

        for i in range(k):
            time_series = sample_data[i, :]
            acf_values = acf(time_series, nlags=L - 1, fft=True)

            half_lag = L // 2
            peaks, _ = find_peaks(acf_values[half_lag:], height=acf_threshold)
            peak_lags = peaks + half_lag

            if len(peak_lags) == 0:
                continue

            current_scores = np.zeros(len(list_points_scaled[4:]))
            for lag in peak_lags:
                distances = np.abs(lag - list_points_scaled[4:])
                current_scores += 1 / (distances + 1e-6)

            current_scores /= current_scores.sum()
            acf_scores.append(current_scores)

        if acf_scores:
            acf_scores = np.array(acf_scores)
            acf_normalized = acf_scores / acf_scores.sum(axis=1, keepdims=True)
            total_scores_acf += acf_normalized.sum(axis=0)

    # 归一化 STFT 和 ACF 得分
    stft_part_normalized = total_scores_stft / total_scores_stft.sum() if total_scores_stft.sum() > 0 else np.zeros_like(total_scores_stft)
    acf_part_normalized = total_scores_acf / total_scores_acf.sum() if total_scores_acf.sum() > 0 else np.zeros_like(total_scores_acf)

    # === 新增：只保留 STFT 和 ACF 的最强点，进行加权融合 ===
    new_final_scores = np.zeros_like(list_points)

    # 找到 STFT 和 ACF 的最强点索引
    stft_main_idx = np.argmax(stft_part_normalized)
    acf_main_idx = np.argmax(acf_part_normalized)

    # 赋值最强点得分
    new_final_scores[stft_main_idx] += alpha
    new_final_scores[4 + acf_main_idx] += (1 - alpha)

    # 归一化（可选）
    final_scores = new_final_scores / new_final_scores.sum()

    # 打印结果
    print("Final Normalized Scores for Each Reference Point:")
    for i, score in enumerate(final_scores):
        print(f"Point {list_points[i]:.1f}L: Score = {score:.4f}")

    return final_scores  # 返回放大后的得分



import numpy as np
from sklearn.model_selection import train_test_split

def sample_with_min_per_class(X, y, percentage=1.0):
    """
    从数据中随机选取指定百分比的样本，并确保每个类别至少有一个样本。

    参数:
        X (np.ndarray): 特征数据，形状为 (n_samples, ...)
        y (np.ndarray): 标签数据，形状为 (n_samples,)
        percentage (float): 要选取的数据比例（1 表示 1%）

    返回:
        X_sampled (np.ndarray): 采样后的特征数据
        y_sampled (np.ndarray): 对应的标签数据
    """
    n_samples = X.shape[0]
    n_to_sample = max(int(n_samples * percentage / 100), 1)

    unique_classes = np.unique(y)
    sampled_indices = []

    # 确保每个类至少有一个样本
    for cls in unique_classes:
        idxs = np.where(y == cls)[0]
        if len(idxs) > 0:
            sampled_idx = np.random.choice(idxs, size=1, replace=False)
            sampled_indices.extend(sampled_idx)

    # 剩下的样本从所有样本中随机选取，排除已选样本
    remaining = n_to_sample - len(unique_classes)
    if remaining > 0:
        all_indices = set(range(n_samples))
        already_sampled = set(sampled_indices)
        remaining_indices = list(all_indices - already_sampled)
        additional_samples = np.random.choice(remaining_indices, size=remaining, replace=False)
        sampled_indices.extend(additional_samples)

    sampled_indices = np.array(sampled_indices)
    return X[sampled_indices], y[sampled_indices]



def compute_gap_scores(similarity, shapelets_length=None, mode="mean"):
    """
    输入:
        similarity: [B, C, T-L+1, K] 余弦相似度张量
        shapelets_length: int 或 list/tensor of shape [K]，每个 shapelet 的长度
        mode: str, "mean" 或 "weighted"，表示使用哪种方式聚合 gap score

    输出:
        gap_score_vector: [K] 每个 shapelet 的 gap score
        aggregated_score: 标量，gap score 的聚合结果（加权或简单平均）
    """
    B, C, W, K = similarity.shape

    # Step 1: 取绝对值
    similarity_abs = torch.abs(similarity)

    # Step 2: 对通道维度合并（可选）
    similarity_abs = similarity_abs.mean(dim=1)  # [B, W, K]

    # Step 3: 取最短子序列距离（min over window dim）→ [B, K]
    shortest_distances = similarity_abs.min(dim=1).values  # [B, K]

    # Step 4: 计算每个 shapelet 的 gap score
    gap_scores = []
    for k in range(K):
        dists = shortest_distances[:, k]
        sorted_indices = torch.argsort(dists)
        split_idx = len(sorted_indices) // 2
        DA_indices = sorted_indices[:split_idx]
        DB_indices = sorted_indices[split_idx:]

        if len(DA_indices) == 0 or len(DB_indices) == 0:
            gap_scores.append(torch.tensor(0.0, device=similarity.device))
            continue

        DA_dists = dists[DA_indices]
        DB_dists = dists[DB_indices]

        mu_A = DA_dists.mean()
        sigma_A = DA_dists.std()
        mu_B = DB_dists.mean()
        sigma_B = DB_dists.std()

        gap = (mu_B - sigma_B) - (mu_A + sigma_A)
        gap_scores.append(gap)

    gap_score_vector = torch.stack(gap_scores)  # [K]

    # Step 5: 聚合 gap scores
    if mode == "mean":
        aggregated_score = gap_score_vector.mean()
    elif mode == "weighted":
        assert shapelets_length is not None, "必须提供 shapelets_length 才能使用加权平均"
        weights = torch.as_tensor(shapelets_length, device=similarity.device, dtype=torch.float32)
        weights = weights / weights.sum()
        aggregated_score = (gap_score_vector * weights).sum()
    else:
        raise ValueError(f"mode 必须是 'mean' 或 'weighted'，但得到 {mode}")

    return gap_score_vector, aggregated_score






def compute_gap_scores_from_transformation(transformation):
    """
    输入:
        transformation: [N, K] numpy array，每个元素是某个 shapelet 对应的距离或匹配分数
    输出:
        gap_score_vector: [K] 每个 shapelet 的 gap score
    """
    N, K = transformation.shape
    gap_scores = []

    for k in range(K):  # 遍历每个 shapelet
        dists = transformation[:, k]  # [N]，当前 shapelet 与所有样本的距离

        # 排序并划分 DA/DB（按距离升序排列）
        sorted_indices = np.argsort(dists)
        split_idx = len(sorted_indices) // 2
        DA_indices = sorted_indices[:split_idx]
        DB_indices = sorted_indices[split_idx:]

        if len(DA_indices) == 0 or len(DB_indices) == 0:
            gap_scores.append(0.0)
            continue

        DA_dists = dists[DA_indices]
        DB_dists = dists[DB_indices]

        mu_A = DA_dists.mean()
        sigma_A = DA_dists.std()
        mu_B = DB_dists.mean()
        sigma_B = DB_dists.std()

        gap = (mu_B - sigma_B) - (mu_A + sigma_A)
        gap_scores.append(gap)

    gap_score_vector = np.array(gap_scores)  # [K]

    return gap_score_vector