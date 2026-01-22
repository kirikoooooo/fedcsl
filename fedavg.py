import torch
import copy
import numpy as np

# 使用score
def fedavg(w, y_fed, score_list):
    """
    Returns the average of the weights.
    
    Args:
        w: List of model state dicts from clients
        y_fed: List of labels for each client (used to calculate sample counts)
        score_list: List of scores/masks for each client (0 or 1 for client selection)
    
    Returns:
        w_avg: Averaged model weights
    """
    # 只计算被选中客户端的总样本数（score_list[i] != 0 的客户端）
    total_samples = sum([len(y_fed[i]) * (1 if score_list[i] != 0 else 0) for i in range(len(y_fed))])
    
    # 如果没有客户端被选中，返回第一个客户端的权重（或抛出错误）
    if total_samples == 0:
        print("警告：没有客户端被选中，返回第一个客户端的权重")
        return copy.deepcopy(w[0])
    
    w_avg = copy.deepcopy(w[0])
    
    # 只聚合可训练参数（跳过buffers如num_batches_tracked等整数类型）
    # 获取第一个模型的参数键（排除buffers）
    param_keys = [k for k in w_avg.keys() if not k.endswith('.num_batches_tracked')]
    
    # 初始化第一个客户端的权重
    for k in param_keys:
        if score_list[0] != 0:
            w_avg[k] = w_avg[k] * len(y_fed[0]) / total_samples * score_list[0]
        else:
            # 如果第一个客户端未被选中，权重为0
            w_avg[k] = w_avg[k] * 0
    
    # 累加其他客户端的权重
    for key in param_keys:
        for i in range(1, len(w)):
            if score_list[i] != 0:
                contribution = w[i][key] * len(y_fed[i]) / total_samples * score_list[i]
                w_avg[key] += contribution
            # 如果 score_list[i] == 0，该客户端不参与聚合（权重为0，不需要累加）
    
    # 对于buffers（如num_batches_tracked），直接使用第一个客户端的值（不进行聚合）
    buffer_keys = [k for k in w_avg.keys() if k.endswith('.num_batches_tracked')]
    for k in buffer_keys:
        w_avg[k] = w[0][k]  # 直接使用第一个客户端的buffer值
    
    return w_avg

def fedavg2(w,y_fed):
        # FedAvg with weight
        total_samples = sum([len(row) for row in y_fed])
        print(total_samples)
        base = [0] * len(w[0])
        for i, client_weight in enumerate(w):
            #total_samples += len(y_fed[i])
            for j, v in enumerate(client_weight):
                base[j] += (len(y_fed[i])/ total_samples * v.astype(np.float64))

        # Update the model
        return base

#修改后加权平均
def FedAvg3(w, dict_users):
    """
    使用加权平均的方式对每个用户的模型权重进行聚合。

    参数：
    ----------------------------------
    w:                    包含每个用户模型权重的列表。
    dict_users:           每个用户的样本索引字典，dict_users[i] 表示第i号用户持有的样本索引集合。

    返回：
    ----------------------------------
    w_avg:                聚合后的模型权重。
    """
    # 初始化权重总和为第一个用户的权重
    w_avg = copy.deepcopy(w[0])

    # 计算总样本数
    total_samples = sum(len(user_idxs) for user_idxs in dict_users)

    for k in w_avg.keys():
        # 计算加权总和
        w_avg[k] = sum(w[i][k] * len(dict_users[i]) / total_samples for i in range(len(w)))

    return w_avg