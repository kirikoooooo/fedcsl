import numpy as np

def add_gaussian_noise(R, epsilon, delta, sensitivity=1.0):
    """
    为相关矩阵 R 添加高斯噪声以确保差分隐私。

    参数:
    R (np.ndarray): 相关矩阵。
    epsilon (float): 隐私参数 ε  
    delta (float): 隐私参数 δ
    sensitivity (float): 函数 f 的 l2 敏感度，默认值为 1.0。

    返回:
    np.ndarray: 加了噪声的相关矩阵。
    """
    # 计算标准差 sigma
    sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))

    # 生成与 R 形状相同的高斯噪声
    noise = np.random.normal(loc=0.0, scale=sigma, size=R.shape)
    
    # 将噪声添加到相关矩阵
    noisy_R = R + noise
    
    return noisy_R

# 示例：创建一个相关矩阵 R
R = np.array([[1.0, 0.5], [0.5, 1.0]])

# 设置隐私参数
epsilon = 3
delta = 1e-2

# 为 R 添加高斯噪声
noisy_R = add_gaussian_noise(R, epsilon, delta)

print("原始相关矩阵:\n", R)
print("加噪声后的相关矩阵:\n", noisy_R)