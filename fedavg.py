import torch
import copy
import numpy as np

# 使用score
def fedavg(w, y_fed,score_list):
    """
    Returns the average of the weights.
    """
    total_samples = sum([len(row) for row in y_fed])
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] =  w_avg[k] * len(y_fed[0]) / total_samples * score_list[0]
        #print(w_avg[k].shape)
    # print(len(w))
    # print(len(y_fed))
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * len(y_fed[i]) / total_samples * score_list[i] #key 是model_state_dict 的key
        #w_avg[key] = torch.div(w_avg[key], len(w))
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