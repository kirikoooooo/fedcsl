#!/bin/bash

# 检查是否提供了输出目录参数
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
  echo "Usage: $0 <output_directory> <dataset_root_directory> <start_pos> <end_pos>"
  exit 1
fi

OUTPUT_DIR=$1
DATASET_ROOT=$2
START_POS=$3
END_POS=$4
mkdir -p ${OUTPUT_DIR}

# 获取数据集列表
DATASETS=($(ls -d ${DATASET_ROOT}/*/ | xargs -n 1 basename))

# 限制数据集范围
DATASETS=(${DATASETS[@]:${START_POS}:${END_POS}-${START_POS}+1})

# 逐个数据集串行执行实验
for dataset in "${DATASETS[@]}"; do
  CONFIG_FILE="${OUTPUT_DIR}/config_${dataset}.yml"
  cp configAVG.yml ${CONFIG_FILE}
  
  # 修改数据集参数
  sed -i "s/dataset:.*$/dataset: ${dataset}/" ${CONFIG_FILE}
  
  LOG_FILE="${OUTPUT_DIR}/${dataset}_experiments.log"
  echo "Running experiment with dataset: ${dataset}, config: ${CONFIG_FILE}" | tee -a ${LOG_FILE}
  python FedCSL_All.py --config "${CONFIG_FILE}" >> ${LOG_FILE} 2>&1

done

echo "All experiments have been completed."
