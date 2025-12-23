import matplotlib.pyplot as plt
import pandas as pd
import re

def extract_data(filename, step=1):
    # 读取文件内容并提取数据
    with open(filename, "r") as file:
        data = file.readlines()

    rounds = []
    accuracy = []
    avg_loss = []

    for line in data:
        match = re.search(r"round:(\d+).*?acc:(\d+\.\d+).*?avg_loss:(\d+\.\d+)", line)
        if match:
            rounds.append(int(match.group(1)))
            accuracy.append(float(match.group(2)))
            avg_loss.append(float(match.group(3)))

    # 选择每隔一定步长的数据点
    rounds_sampled = rounds[::step]
    accuracy_sampled = accuracy[::step]
    avg_loss_sampled = avg_loss[::step]

    # 平滑数据
    window_size = 3  # 滚动窗口大小，可以调节
    accuracy_smooth = pd.Series(accuracy_sampled).rolling(window=window_size, center=True).mean()
    avg_loss_smooth = pd.Series(avg_loss_sampled).rolling(window=window_size, center=True).mean()

    return rounds_sampled, accuracy_smooth, avg_loss_smooth

# 从两个文件中提取数据
rounds1, accuracy_smooth1, avg_loss_smooth1 = extract_data("data.txt")
rounds2, accuracy_smooth2, avg_loss_smooth2 = extract_data("data2.txt")
rounds3, accuracy_smooth3, avg_loss_smooth3 = extract_data("data3.txt")
rounds4, accuracy_smooth4, avg_loss_smooth4 = extract_data("data4.txt")
rounds5, accuracy_smooth5, avg_loss_smooth5 = extract_data("data5.txt")
# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 绘制第一个子图：Accuracy
ax1.plot(rounds1, accuracy_smooth1, label='FedAVG', color='tab:blue', linestyle='-')
ax1.plot(rounds2, accuracy_smooth2, label='jointKD', color='tab:green', linestyle='--')
ax1.plot(rounds3, accuracy_smooth3, label='jointCL', color='tab:red', linestyle='-.')
ax1.plot(rounds4, accuracy_smooth4, label='jointKD+CL', color='tab:orange', linestyle=':')
ax1.plot(rounds5, accuracy_smooth5, label='jointKD_jointCL_scaleKD_scaleCL', color='tab:purple', linestyle='--')

ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Comparison')
#ax1.set_ylim([0.87, 0.91])
ax1.legend()

# 绘制第二个子图：Average Loss
#ax2.plot(rounds1, avg_loss_smooth1, label='FedAVG', color='tab:blue', linestyle='-')
ax2.plot(rounds2, avg_loss_smooth2, label='jointKD', color='tab:green', linestyle='--')
ax2.plot(rounds3, avg_loss_smooth3, label='jointCL', color='tab:red', linestyle='-.')
ax2.plot(rounds4, avg_loss_smooth4, label='jointKD+CL', color='tab:orange', linestyle=':')
ax2.plot(rounds5, avg_loss_smooth5, label='jointKD_jointCL_scaleKD_scaleCL', color='tab:purple', linestyle='--')



ax2.set_xlabel('Round')
ax2.set_ylabel('Average Loss')
ax2.set_title('Average Loss Comparison')
ax2.legend()

# 调整布局并保存图像
plt.tight_layout()
plt.savefig("comparison_accuracy_loss_separate.png")

# 显示图像
plt.show()
