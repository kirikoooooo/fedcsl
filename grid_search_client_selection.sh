#!/bin/bash

# 客户端选择超参数网格搜索脚本
# 批量并行训练，每次4个任务

# 设置基础参数
DATASET="Epilepsy-TSTCC"
BASE_CONFIG="configEpilepsy.yml"
NUM_ROUNDS=50  # 50轮训练
MAX_PARALLEL=4  # 最大并行数

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "错误: 未找到Python，请安装Python3"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# 超参数网格
CLIENT_SELECTION_RATIOS=(0.5 0.6 0.7 0.8)  # 采样比例
MIN_SELECTION_PROBS=(0.005 0.01 0.02 0.05)  # 最低选择概率
EMA_ALPHAS=(0.1 0.2 0.3 0.5)  # EMA平滑系数

# 创建结果目录
RESULT_DIR="./grid_search_results"
mkdir -p ${RESULT_DIR}

# 创建任务列表
TASK_FILE="${RESULT_DIR}/tasks.txt"
> ${TASK_FILE}  # 清空文件

task_id=0
for ratio in "${CLIENT_SELECTION_RATIOS[@]}"; do
    for min_prob in "${MIN_SELECTION_PROBS[@]}"; do
        for ema_alpha in "${EMA_ALPHAS[@]}"; do
            task_id=$((task_id + 1))
            # 使用task_id确保配置文件名唯一，避免并行任务冲突
            config_name="config_grid_task${task_id}_ratio${ratio}_min${min_prob}_ema${ema_alpha}.yml"
            echo "${task_id}|${ratio}|${min_prob}|${ema_alpha}|${config_name}" >> ${TASK_FILE}
        done
    done
done

echo "总共生成 $task_id 个任务"

# 函数：创建配置文件（直接复制完整配置文件，只修改需要的部分）
create_config() {
    local task_id=$1
    local ratio=$2
    local min_prob=$3
    local ema_alpha=$4
    local config_name=$5
    
    # 配置文件保存在RESULT_DIR中，避免污染项目根目录
    local config_path="${RESULT_DIR}/${config_name}"
    
    # 确保结果目录存在
    mkdir -p "${RESULT_DIR}"
    
    # 直接复制完整的基础配置文件
    cp "${BASE_CONFIG}" "${config_path}"
    
    # 使用Python修改配置文件（只修改需要的部分，保留所有原有配置）
    ${PYTHON_CMD} << EOF
import yaml
import os

config_file = "${config_path}"

# 读取完整配置（保留所有原有配置项）
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 确保federated键存在
if 'federated' not in config:
    config['federated'] = {}

# 只修改客户端选择相关的参数，其他配置保持不变
config['federated']['use_client_selection'] = True
config['federated']['client_selection_ratio'] = ${ratio}
config['federated']['min_selection_prob'] = ${min_prob}
config['federated']['ema_alpha'] = ${ema_alpha}
config['federated']['numRound'] = ${NUM_ROUNDS}

# 只修改description，其他字段保持不变
config['description'] = f"GridSearch_task${task_id}_ratio${ratio}_min${min_prob}_ema${ema_alpha}"

# 保存配置（保留所有原有配置项，不排序以保持原有顺序）
with open(config_file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print(f"Created config: {config_file}")
EOF
}

# 函数：运行单个任务
run_task() {
    local task_id=$1
    local ratio=$2
    local min_prob=$3
    local ema_alpha=$4
    local config_name=$5
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始任务 ${task_id}: ratio=${ratio}, min_prob=${min_prob}, ema_alpha=${ema_alpha}"
    
    # 创建配置文件（传入task_id确保唯一性）
    create_config ${task_id} ${ratio} ${min_prob} ${ema_alpha} ${config_name}
    
    # 配置文件路径（在RESULT_DIR中）
    local config_path="${RESULT_DIR}/${config_name}"
    
    # 运行训练
    log_file="${RESULT_DIR}/task_${task_id}_ratio${ratio}_min${min_prob}_ema${ema_alpha}.log"
    
    ${PYTHON_CMD} FedCSL_Epilepsy.py -dataset ${DATASET} --config ${config_path} > ${log_file} 2>&1
    
    # 提取最终准确率（兼容不同grep版本）
    if [ -f ${log_file} ]; then
        # 尝试使用grep -oP（Perl正则，Linux）
        best_acc=$(grep "best round" ${log_file} | tail -1 | grep -oP 'acc is \K[0-9.]+' 2>/dev/null || \
                   grep "best round" ${log_file} | tail -1 | sed -n 's/.*acc is \([0-9.]*\).*/\1/p' || \
                   echo "N/A")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 任务 ${task_id} 完成: best_acc=${best_acc}"
        echo "${task_id}|${ratio}|${min_prob}|${ema_alpha}|${best_acc}" >> ${RESULT_DIR}/results_summary.txt
    fi
    
    # 清理临时配置文件（确保不会留下临时文件）
    if [ -f "${config_path}" ]; then
        rm -f "${config_path}"
        echo "已清理临时配置文件: ${config_path}"
    fi
}

# 导出函数以便并行使用（如果支持）
if [ "$(type -t export)" = "builtin" ]; then
    export -f create_config run_task 2>/dev/null || true
fi
export DATASET BASE_CONFIG NUM_ROUNDS RESULT_DIR PYTHON_CMD

# 初始化结果摘要文件
echo "task_id|ratio|min_prob|ema_alpha|best_acc" > ${RESULT_DIR}/results_summary.txt

# 读取任务列表并并行执行
total_tasks=$(wc -l < ${TASK_FILE})
current_task=0

while IFS='|' read -r task_id ratio min_prob ema_alpha config_name; do
    # 等待直到有可用槽位
    while [ $(jobs -r | wc -l) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done
    
    # 启动任务（后台运行）
    run_task ${task_id} ${ratio} ${min_prob} ${ema_alpha} ${config_name} &
    
    current_task=$((current_task + 1))
    echo "进度: ${current_task}/${total_tasks}"
    
done < ${TASK_FILE}

# 等待所有后台任务完成
echo "等待所有任务完成..."
wait

echo "所有任务完成！结果摘要保存在: ${RESULT_DIR}/results_summary.txt"

# 生成排序后的结果
sort -t'|' -k5 -rn ${RESULT_DIR}/results_summary.txt > ${RESULT_DIR}/results_sorted.txt
echo "排序后的结果保存在: ${RESULT_DIR}/results_sorted.txt"

