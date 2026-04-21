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

X_train = np.random.rand(10000, 32, 32, 3)  # 模拟图像数据
y_train = np.random.randint(0, 10, size=(10000,))  # 模拟 10 类标签

# 取 1% 数据，即 100 个样本，且每个类至少有 1 个
X_1_percent, y_1_percent = sample_with_min_per_class(X_train, y_train, percentage=1.0)
print("1% 样本数量:", len(y_1_percent))
print("各类别出现次数:\n", np.bincount(y_1_percent))

# 取 5% 数据
X_5_percent, y_5_percent = sample_with_min_per_class(X_train, y_train, percentage=5.0)
print("5% 样本数量:", len(y_5_percent))


