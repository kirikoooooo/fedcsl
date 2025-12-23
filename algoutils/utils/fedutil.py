import numpy as np
from sklearn.utils import check_random_state
import torch
#import utils.process_har
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

# 假设 df 是 shape (12, 4) 的 DataFrame，且每个cell包含一个长度为 T 的 pandas.Series
def reshape_dataframe_to_tensor(df):
    rows, cols = df.shape
    # 获取每个cell中的时间序列长度（假设所有时间序列长度相同）
    T = df.iloc[0, 0].shape[0]  
    
    # 创建一个空的 numpy 数组，来存放最后的 (12, 4, T) 结构数据
    tensor = np.empty((rows, cols, T),dtype=np.float32)
    
    # 填充 numpy 数组
    for i in range(rows):
        for j in range(cols):
            tensor[i, j, :] = df.iloc[i, j].values  # 将 Series 转换为数值数组并填入
    
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
    