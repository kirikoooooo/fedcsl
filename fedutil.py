import numpy as np
from sklearn.utils import check_random_state
import torch
import process_har
#IID SPilt
def federated_data_split(data, *, num_parties=3, ratio=None, random_state=None):
    """ Split input sktime formatted time series data into federated parties.

    Parameters:
    ----------------------------------
    data:                    Input data, 2D array or dataframe
    num_parties:             The number of federated parties
    ratio:                   List, the ratio of data in each party.
                             Default = None, split the data evenly.
    random_state:            The random seed

    Return:
    ----------------------------------
    T_fed:                  List of dataframe, federated training data
    y_fed:                  List of ndarray, federated targets
    """


    if ratio is None:
        ratio = [1 / num_parties] * num_parties
    else:
        ratio = [r / sum(ratio) for r in ratio]

    cum_ratio = np.cumsum([0] + ratio)

    rst = check_random_state(random_state)

    #T = data.iloc[:,0].to_frame()
    #y = data.iloc[:,1].to_numpy()
    T = data.iloc[:, :-1]
    y = data.iloc[:, -1].to_numpy()
    #classes = np.unique(data.iloc[:, 1].to_numpy())
    classes = np.unique(data.iloc[:, -1].to_numpy())
    n_classes = len(classes)

    data_idx_per_class = []
    data_splits_per_class = []
    for c in classes:
        data_idx = np.where(y == c)[0]
        rst.shuffle(data_idx)
        data_idx_per_class.append(data_idx)
        data_splits_per_class.append(np.int32(np.round(cum_ratio * len(data_idx))))

    T_fed = []
    y_fed = []

    for i in range(num_parties):

        idx = np.concatenate(
            [data_idx_per_class[c][data_splits_per_class[c][i]:data_splits_per_class[c][i+1]]\
             for c in range(n_classes)]
        )
        T_fed.append(T.iloc[idx])
        y_fed.append(y[idx])

    return T_fed, y_fed

from typing import List

def uniformly_split_list(l: list, num_sublists: int) -> List[list]:
    # 计算每个子列表的基本大小
    num_items_per_shard, extra_items = divmod(len(l), num_sublists)

    sublists = []
    start_index = 0

    for i in range(num_sublists):
        # 如果还有额外的项需要分配，则当前子列表多分一个
        end_index = start_index + num_items_per_shard + (1 if i < extra_items else 0)
        sublists.append(l[start_index:end_index])
        start_index = end_index

    return sublists

def federated_data_split_XD(data, *, num_parties=3, ratio=None, random_state=None):
    """
    将输入的符合sktime格式的时间序列数据分割成多个联邦学习参与方的数据集。

    参数：
    ----------------------------------
    data:                    输入数据，二维数组或DataFrame，其中每行代表一个样本，含有序列化的时序特征及目标标签。
    num_parties:             联邦学习参与方的数量。
    ratio:                   各参与方数据占比的列表，默认为None，表示均匀分割数据。
    random_state:            随机种子，确保结果的可复现性。

    返回：
    ----------------------------------
    T_fed:                  联邦学习参与方数据集的DataFrame列表，每个DataFrame对应一个参与方的训练数据。
    y_fed:                  联邦学习参与方标签的ndarray列表，每个ndarray对应一个参与方的训练目标。
    """

    # 如果没有指定分配比例，那么数据将被均匀分割
    if ratio is None:
        ratio = [1 / num_parties] * num_parties
    else:
        # 如果指定了比例，确保这些比例归一化总和为1
        ratio = [r / sum(ratio) for r in ratio]

    # 计算累计比例，用于后续分割数据
    cum_ratio = np.cumsum([0] + ratio)

    # 初始化随机状态以确保随机行为的可复现性
    rst = check_random_state(random_state)

    # 从输入数据中分离特征（T）和标签（y）
    T = data.iloc[:, :-1]  # 特征转换为DataFrame
    y = data.iloc[:, -1].to_numpy()  # 标签转换为numpy数组

    # 获取唯一的目标类别及其数量
    classes = np.unique(y)
    n_classes = len(classes)

    # 初始化列表存储每个类别的样本索引及分割点
    data_idx_per_class = []  # 每个类别的样本索引
    data_splits_per_class = []  # 每个类别内分割点

    # 遍历每个类别
    for c in classes:
        # 获取属于该类别的样本索引并随机打乱
        data_idx = np.where(y == c)[0]
        rst.shuffle(data_idx)
        data_idx_per_class.append(data_idx)
        # 计算该类别内数据分割点
        data_splits_per_class.append(np.int32(np.round(cum_ratio * len(data_idx))))

    # 初始化联邦学习数据和标签列表
    T_fed = []
    y_fed = []

    # 遍历每个参与方
    for i in range(num_parties):
        # 拼接每个类别中分配给当前参与方的样本索引
        idx = np.concatenate(
            [data_idx_per_class[c][data_splits_per_class[c][i]:data_splits_per_class[c][i + 1]] \
             for c in range(n_classes)]
        )
        # 将索引对应的数据和标签添加到各自的列表中
        T_fed.append(T.iloc[idx])  # T_fed 存放的是DataFrame列表
        y_fed.append(y[idx])

    # 返回分割后的数据集和标签
    return T_fed, y_fed



import pandas as pd
import numpy as np
# 定义一个函数来处理 DataFrame 并转换为三维数组

# # 假设 df 是 shape (12, 4) 的 DataFrame，且每个cell包含一个长度为 T 的 pandas.Series
# def reshape_dataframe_to_tensor(df):
#     rows, cols = df.shape
#     # 获取每个cell中的时间序列长度（假设所有时间序列长度相同）
#     T = df.iloc[0, 0].shape[0]

#     # 创建一个空的 numpy 数组，来存放最后的 (12, 4, T) 结构数据
#     tensor = np.empty((rows, cols, T),dtype=np.float32)

#     # 填充 numpy 数组
#     for i in range(rows):
#         for j in range(cols):
#             tensor[i, j, :] = df.iloc[i, j].values  # 将 Series 转换为数值数组并填入

#     return tensor

# v2 版本，自适应截断或者补0
def reshape_dataframe_to_tensor(df, mode='max'):
    """
    将 DataFrame 中的时间序列转换为统一长度的 3D tensor。

    参数:
        df (pd.DataFrame): 每个 cell 是一个时间序列（pandas.Series 或 numpy.array）
        mode (str): 'min' 表示所有时间序列统一为最短长度（截断）；
                    'max' 表示统一为最长长度（补零）

    返回:
        numpy.ndarray: 形状为 (rows, cols, T) 的 tensor
    """
    rows, cols = df.shape

    # 找出所有时间序列的长度
    lengths = [df.iloc[i, j].shape[0] for i in range(rows) for j in range(cols)]

    if mode == 'min':
        T = min(lengths)
    elif mode == 'max':
        T = max(lengths)
    else:
        raise ValueError("mode must be 'min' or 'max'")

    tensor = np.zeros((rows, cols, T), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            values = df.iloc[i, j].values.astype(np.float32)
            cur_len = len(values)
            if cur_len >= T:
                tensor[i, j, :] = values[:T]
            else:
                tensor[i, j, :cur_len] = values  # 自动补零到 T 长度

    return tensor

# 检查是否定长
def check_series_lengths(df):
    lengths = set()
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            series = df.iloc[i, j]
            if isinstance(series, (pd.Series, list, np.ndarray)):
                lengths.add(len(series))
            else:
                raise ValueError(f"Unexpected data type at ({i}, {j}): {type(series)}")
    return lengths





#noiid的数据分割，来自gpt
def federated_data_split_XD_non_iid(data, labels, *, num_parties=3, ratio=None, random_state=None):
    """
    将输入的多维时间序列数据和标签分割成多个联邦学习参与方的数据集。

    参数：
    ----------------------------------
    data:                    输入数据，形状为 (N, C, T) 的 PyTorch Tensor，其中 N 是样本数量，C 是特征通道数，T 是时间步长。
    labels:                  输入标签，形状为 (N,) 的 PyTorch Tensor。
    num_parties:             联邦学习参与方的数量。
    ratio:                   各参与方数据占比的列表，默认为None，表示均匀分割数据。
    random_state:            随机种子，确保结果的可复现性。

    返回：
    ----------------------------------
    T_fed:                  联邦学习参与方数据集的 Tensor 列表，每个 Tensor 对应一个参与方的训练数据。
    y_fed:                  联邦学习参与方标签的 Tensor 列表，每个 Tensor 对应一个参与方的训练目标。
    """

    # 如果没有指定分配比例，那么数据将被均匀分割
    if ratio is None:
        ratio = [1 / num_parties] * num_parties
    else:
        # 如果指定了比例，确保这些比例归一化总和为1
        ratio = [r / sum(ratio) for r in ratio]

    # 计算累计比例，用于后续分割数据
    cum_ratio = np.cumsum([0] + ratio)

    # 初始化随机状态以确保随机行为的可复现性
    rst = check_random_state(random_state)

    # 获取样本总数
    num_samples = data.shape[0]

    # 获取所有样本的索引并随机打乱
    indices = np.arange(num_samples)
    rst.shuffle(indices)

    # 初始化联邦学习数据和标签列表
    T_fed = []
    y_fed = []

    # 遍历每个参与方
    for i in range(num_parties):
        # 计算当前参与方的数据起始和结束索引
        start_idx = int(cum_ratio[i] * num_samples)
        end_idx = int(cum_ratio[i + 1] * num_samples)

        # 获取当前参与方的数据和标签
        party_indices = indices[start_idx:end_idx]
        T_fed.append(data[party_indices])
        y_fed.append(labels[party_indices])

    # 返回分割后的数据集和标签
    return T_fed, y_fed


def set_diagonal_to_zero(matrix):
    """
    将给定矩阵的对角线元素设置为零。

    参数:
    matrix (torch.Tensor): 输入的二维张量。

    返回:
    torch.Tensor: 对角线元素被设置为零的矩阵。
    """
    if len(matrix.shape) != 2:
        raise ValueError("输入必须是一个二维张量")

    # 获取对角线元素
    diag = torch.diag(matrix)

    # 将对角线元素嵌入到新的对角矩阵中
    diag_matrix = torch.diag_embed(diag)

    # 从原矩阵中减去对角矩阵，从而将对角线元素设置为零
    result = matrix - diag_matrix

    return result


## 将样本索引集合划分为n_clients个子集
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''

    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs





def channel_discrete_metric(sequences):
    """
    计算样本集群的离散程度，并映射到 [0, 1] 范围内。

    参数:
        sequences: 输入时间序列，形状为 (batch_size, num_channels, sequence_length)

    返回:
        discrete_score: 样本集群的离散程度得分，范围 [0, 1]
    """
    # 将多维时间序列展平为特征向量
    # batch_size, num_channels, sequence_length = sequences.shape
    batch_size = len(sequences)
    num_channels,sequence_length = sequences[0].shape
    sequences = torch.tensor(sequences)
    features = sequences.view(batch_size, -1)  # 形状: (batch_size, num_channels * sequence_length)

    # 计算样本间的方差
    mean_features = torch.mean(features, dim=0)  # 计算特征均值
    variance = torch.mean((features - mean_features) ** 2)  # 计算方差

    # 使用 Sigmoid 函数将方差映射到 [0, 1]
    discrete_score = torch.sigmoid(variance)

    return discrete_score.item()  # 返回标量值

# # 示例用法
# sequences = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],  # 样本1
#                           [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]])  # 样本2
# discrete_score = channel_discrete_metric(sequences)
# print("Channel Discrete Score:", discrete_score)


def skewness_time_series(sequences):
    """
    计算多维时间序列的偏度。

    参数:
        sequences: 输入时间序列，形状为 (batch_size, num_channels, sequence_length)

    返回:
        skewness: 偏度值，形状为 (batch_size, num_channels)
    """
    mean = torch.mean(sequences, dim=2, keepdim=True)  # 计算时间维度的均值
    std = torch.std(sequences, dim=2, keepdim=True)    # 计算时间维度的标准差
    z = (sequences - mean) / (std + 1e-9)              # 标准化
    skew = torch.mean(z ** 3, dim=2)                   # 计算偏度
    return skew
def inter_sample_distance_time_series(features, alpha=2.0, beta=4.0):
    """
    计算多维时间序列的离散程度分数，结果反向，范围 [0, 2]。

    参数:
        features: 输入时间序列，形状为 (batch_size, num_channels, sequence_length)
        alpha (float): 控制映射范围的参数。
        beta (float): 控制差异放大的参数。

    返回:
        score: 离散程度分数，范围 [1 - alpha/2, 1 + alpha/2]
    """
    # 转化为 tensor
    features = torch.tensor(features)

    # 计算样本间的欧氏距离矩阵
    distance_matrix = torch.cdist(features, features, p=2)  # 形状: (batch_size, batch_size)

    # 计算平均距离
    mean_distance = torch.mean(distance_matrix)  # 标量

    # 对距离进行归一化处理
    max_distance = torch.max(distance_matrix)  # 最大距离
    normalized_distance = mean_distance / (max_distance + 1e-9)  # 归一化到 [0, 1]

    # 结果反向：将归一化后的距离转换为 1 - d
    reversed_distance = 1 - normalized_distance

    # 非线性变换：映射到 1 附近，并拉开差距
    score = 1 + alpha * (reversed_distance - 0.5)  # 映射到 [1 - alpha/2, 1 + alpha/2]
    score = torch.pow(score, beta)  # 进一步放大差异

    return score.item()



def count_labels(labels):
    """
    统计标签类别的数量。

    参数:
        labels: 输入样本的标签集合，形状为 (batch_size,)

    返回:
        label_counts: 每个标签类别的数量
    """
    num_unique_labels = torch.unique(labels).size(0)  # 使用 torch.unique 统计唯一值
    return num_unique_labels

import math
def map_to_near_one(labels, total_classes):
    """
    将不同标签类别的数量映射到 1 附近。

    参数:
        labels: 输入样本的标签集合，形状为 (batch_size,)
        total_classes: 总类别数

    返回:
        mapped_value: 映射后的值，范围在 1 附近
    """
    # 统计不同标签类别的数量
    num_unique_labels = torch.unique(labels).size(0)

    # 使用对数变换映射到 1 附近
    mapped_value = math.log(num_unique_labels + 1)

    return mapped_value

import torch
from sklearn.cluster import KMeans

def map_cluster_count_to_near_one(features, max_clusters=10, alpha=2.0, beta=2.0):
    """
    对特征向量进行聚类，并将类别数量映射到 1 附近，且拉开距离。

    参数:
        features (np.array): 输入特征向量，形状为 (num_samples, num_features)。
        max_clusters (int): 最大聚类数。
        alpha (float): 控制映射范围的参数。
        beta (float): 控制差异放大的参数。

    返回:
        mapped_score (float): 映射后的分数，范围 [1 - alpha/2, 1 + alpha/2]。
    """
    # 将特征向量转化为 numpy 数组
    features = np.array(features)

    # 使用肘部法确定最佳聚类数
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        distortions.append(kmeans.inertia_)

    # 计算肘部点（最佳聚类数）
    k = np.argmin(np.diff(distortions)) + 1  # 肘部点对应的聚类数

    # 归一化聚类数到 [0, 1] 范围
    normalized_k = k / max_clusters

    # 线性变换：映射到 1 附近
    score = 1 + alpha * (normalized_k - 0.5)  # 映射到 [1 - alpha/2, 1 + alpha/2]

    # 非线性变换：拉开距离
    mapped_score = np.power(score, beta)  # 使用幂函数放大差异

    return mapped_score

def normalize_to_near_one(scores, target_sum=3.0):
    """
    将分数归一化到 1 附近，并限制输出结果之和为 target_sum。

    参数:
        scores (list or np.array): 输入分数列表或数组。
        target_sum (float): 目标总和。

    返回:
        normalized_scores (np.array): 归一化后的分数，总和为 target_sum。
    """
    scores = np.array(scores)

    # 计算均值
    mean_score = np.mean(scores)

    # 归一化到 1 附近
    normalized_scores = scores / mean_score

    # 限制总和为 target_sum
    normalized_scores = normalized_scores / np.sum(normalized_scores) * target_sum

    return normalized_scores
def cal_score(features):

    return inter_sample_distance_time_series(features)



import torch
from sklearn.linear_model import OrthogonalMatchingPursuit

def omp_from_state_dicts(client_state_dicts, global_state_dict, n_nonzero_coefs=None, selected_keys=None):
    """
    用 OMP 从客户端模型中选择最具代表性的模型组合重构全局模型。

    参数:
        client_state_dicts: List[Dict[str, Tensor]]，客户端模型参数字典
        global_state_dict: Dict[str, Tensor]，全局模型参数字典
        n_nonzero_coefs: int，OMP中允许的非零系数个数（稀疏度）
        selected_keys: List[str]，指定只使用部分参数层（可选）

    返回:
        x: ndarray, 稀疏系数向量
    """

    def flatten_state_dict(state_dict):
        flat_params = []
        keys = sorted(state_dict.keys())
        if selected_keys is not None:
            keys = [k for k in keys if k in selected_keys]
        for key in keys:
            flat_params.append(state_dict[key].flatten())
        return torch.cat(flat_params)

    # 构造字典矩阵 A 和目标向量 Y
    A = torch.stack([flatten_state_dict(sd) for sd in client_state_dicts], dim=1)  # [num_params, num_clients]
    Y = flatten_state_dict(global_state_dict)  # [num_params]

    # 使用 OMP 解稀疏系数
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(A.cpu().numpy(), Y.cpu().numpy())
    x = omp.coef_

    return x


import numpy as np

# def get_sampling_probs_from_omp(x, prev_probs, selection_mask):
#     """
#     根据 OMP 的稀疏系数向量和掩码更新下一轮客户端采样概率。

#     参数:
#         x: numpy array，长度为总客户端数，OMP输出的稀疏向量（含0和负值）
#         prev_probs: numpy array，上一轮的采样概率（总客户端数）
#         selection_mask: List[int] or np.ndarray，0/1 向量，表示哪些客户端被选中（长度与 x 相同）

#     返回:
#         updated_probs: numpy array，更新后的下一轮采样概率（总客户端数）
#     """
#     selection_mask = np.asarray(selection_mask).astype(int)
#     abs_x = np.abs(x) * selection_mask  # 只保留选中客户端的贡献
#     total_contribution = abs_x.sum()

#     updated_probs = prev_probs.copy()
#     if total_contribution > 0:
#         # 当前总权重份额（只考虑被选中的客户端）
#         prob_share = prev_probs[selection_mask == 1].sum()
#         # 归一化分配
#         for i in range(len(x)):
#             if selection_mask[i] == 1:
#                 updated_probs[i] = abs_x[i] / total_contribution * prob_share

#     return updated_probs
def get_sampling_probs_from_omp(x, prev_probs, selection_mask, min_selection_prob=0.01, ema_alpha=0.3):
    """
    根据 OMP 的稀疏系数向量和掩码更新下一轮客户端采样概率（使用指数移动平均平滑）。

    参数:
        x: numpy array，长度为总客户端数，OMP输出的稀疏向量（含0和负值）
        prev_probs: numpy array，上一轮的采样概率（总客户端数）
        selection_mask: List[int] or np.ndarray，0/1 向量，表示哪些客户端被选中（长度与 x 相同）
        min_selection_prob: float，最低选择概率，避免概率完全为0（默认0.01，即1%）
        ema_alpha: float，指数移动平均的平滑系数，范围[0,1]，值越大更新越快（默认0.3）

    返回:
        updated_probs: numpy array，更新后的下一轮采样概率（总客户端数）
    """
    # 确保 x 是 numpy 数组
    x = np.asarray(x)
    # 确保 prev_probs 是 numpy 数组，以便可以修改
    prev_probs = np.asarray(prev_probs, dtype=float)
    
    # 长度检查
    if len(x) != len(prev_probs) or len(prev_probs) != len(selection_mask):
        raise ValueError("输入向量 x, prev_probs, selection_mask 的长度必须相同。")

    num_clients = len(prev_probs)
    
    # 1. 处理负值：使用 ReLU 函数，只保留非负值，并加上小的正数避免完全为0
    # 对于负值，我们将其视为较小的贡献（使用绝对值但降低权重）
    x_processed = np.abs(x)  # 使用绝对值处理负值
    # 对于被选中的客户端，如果贡献为0，给予最小贡献值
    epsilon = 1e-8
    for i in range(num_clients):
        if selection_mask[i] == 1 and x_processed[i] < epsilon:
            x_processed[i] = epsilon
    
    # 2. 计算被选中客户端的贡献总和
    total_contribution = 0.0
    for i in range(num_clients):
        if selection_mask[i] == 1:
            total_contribution += x_processed[i]
    
    # 3. 计算新的目标概率（基于贡献比例）
    target_probs = np.zeros_like(prev_probs)
    
    if total_contribution > 0:
        # 计算被选中客户端的总概率份额
        prob_share = 0.0
        for i in range(num_clients):
            if selection_mask[i] == 1:
                prob_share += prev_probs[i]
        
        # 根据贡献比例分配概率给被选中的客户端
        for i in range(num_clients):
            if selection_mask[i] == 1:
                # 根据贡献比例计算目标概率
                contribution_ratio = x_processed[i] / total_contribution
                target_probs[i] = contribution_ratio * prob_share
            else:
                # 未被选中的客户端保持原概率（后续会通过EMA平滑）
                target_probs[i] = prev_probs[i]
    else:
        # 如果总贡献为0，保持原概率
        target_probs = prev_probs.copy()
    
    # 4. 使用指数移动平均（EMA）平滑更新概率
    # EMA公式: new_prob = alpha * target_prob + (1 - alpha) * prev_prob
    updated_probs = ema_alpha * target_probs + (1 - ema_alpha) * prev_probs
    
    # 5. 应用最小概率保护
    updated_probs = np.maximum(updated_probs, min_selection_prob)
    
    # 6. 重新归一化，确保概率总和为1
    updated_probs = updated_probs / updated_probs.sum()
    
    return updated_probs


def sample_clients_mask_by_probability(probs, num_to_sample, seed=None):
    """
    按概率采样客户端并输出一个0/1稀疏向量表示。

    参数:
        probs: List[float] or np.ndarray，客户端采样概率（总和应为1）
        num_to_sample: int，需要采样的客户端数
        seed: int or None，用于控制随机性（可选）

    返回:
        mask: List[int]，与客户端数量等长的0/1向量，表示是否被采样
    """
    if seed is not None:
        np.random.seed(seed)

    probs = np.asarray(probs)
    assert np.isclose(probs.sum(), 1.0), "采样概率总和必须为1"
    assert len(probs) >= num_to_sample, "采样数量超过客户端总数"

    num_clients = len(probs)
    
    # 检查非零概率的客户端数量
    non_zero_count = np.count_nonzero(probs)
    
    # 如果非零概率的客户端数量少于需要采样的数量，进行概率平滑处理
    if non_zero_count < num_to_sample:
        # 计算最小概率值，确保有足够的客户端可以采样
        min_prob = 1e-6  # 设置一个很小的最小概率
        # 对概率进行平滑：probs = (1 - smoothing_factor) * probs + smoothing_factor * uniform
        # smoothing_factor 根据不足的数量动态调整
        smoothing_factor = min(0.1, (num_to_sample - non_zero_count) / num_clients)
        uniform_prob = 1.0 / num_clients
        probs = (1 - smoothing_factor) * probs + smoothing_factor * uniform_prob
        
        # 确保所有概率至少为最小概率
        probs = np.maximum(probs, min_prob)
        
        # 重新归一化
        probs = probs / probs.sum()
        
        print(f"警告: 非零概率客户端数量({non_zero_count})少于采样数量({num_to_sample})，已进行概率平滑处理")
    
    # 如果平滑后仍然不足（理论上不应该发生，但为了安全起见）
    non_zero_count_after = np.count_nonzero(probs)
    if non_zero_count_after < num_to_sample:
        num_to_sample = non_zero_count_after
        print(f"警告: 采样数量已调整为非零概率客户端数量: {num_to_sample}")
    
    indices = np.random.choice(np.arange(num_clients), size=num_to_sample, replace=False, p=probs)

    mask = [1 if i in indices else 0 for i in range(num_clients)]
    return mask