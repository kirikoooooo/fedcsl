# import pandas as pd
# import numpy as np

# # 假设你的 DataFrame 是这样的
# # df = pd.DataFrame({
# #     'col1': ['t1,t2,t3', 't4,t5,t6'],
# #     'col2': ['t7,t8,t9', 't10,t11,t12']
# # })

# # 定义一个函数来处理 DataFrame 并转换为三维数组
# def dataframe_to_3d_array(df, num_timestamps):
#     # 获取行数和列数
#     num_rows, num_cols = df.shape
    
#     # 初始化一个空的三维数组
#     array_3d = np.empty((num_rows, num_cols, num_timestamps), dtype=float)
    
#     # 遍历 DataFrame 的每一列
#     for col_idx, col in enumerate(df.columns):
#         # 对于每一列，遍历每一个元素
#         for row_idx, cell in enumerate(df[col]):
#             # 分割字符串并转换为浮点数
#             timestamps = [float(ts) for ts in cell.split(',')]
            
#             # 检查时间戳数量是否符合预期
#             if len(timestamps) != num_timestamps:
#                 raise ValueError(f"Expected {num_timestamps} timestamps, but got {len(timestamps)} in column {col}, row {row_idx}")
            
#             # 将时间戳放入三维数组
#             array_3d[row_idx, col_idx, :] = timestamps
    
#     return array_3d

# # 示例 DataFrame
# df = pd.DataFrame({
#     'col1': ['1.0,2.0,3.0', '4.0,5.0,6.0'],
#     'col2': ['7.0,8.0,9.0', '10.0,11.0,12.0'],
#     'col3': ['13.0,14.0,15.0', '16.0,17.0,18.0'],
#     'col4': ['19.0,20.0,21.0', '22.0,23.0,24.0']
# })

# # 假设每个时间序列有 3 个时间戳
# num_timestamps = 3

# # 调用函数并打印结果
# array_3d = dataframe_to_3d_array(df, num_timestamps)
# print(array_3d)
import torch
import numpy as np
from sklearn.utils import check_random_state

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

# 示例使用
if __name__ == "__main__":
    # 假设 data 和 labels 已经定义
    data = torch.randn(1000, 9, 128)  # 示例数据，形状为 (1000, 9, 128)
    labels = torch.randint(0, 6, (1000,))  # 示例标签，形状为 (1000,)

    # 调用函数进行数据分割
    T_fed, y_fed = federated_data_split_XD_non_iid(data, labels, num_parties=3, ratio=[5, 3, 2], random_state=42)

    # 打印结果
    for i in range(len(T_fed)):
        print(f"Party {i+1} - Data shape: {T_fed[i].shape}, Labels shape: {y_fed[i].shape}")