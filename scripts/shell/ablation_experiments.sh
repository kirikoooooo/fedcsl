#!/bin/bash
# 客户端选择消融实验脚本（Shell版本）
# 使用方法: bash ablation_experiments.sh

BASE_CONFIG="configAVG.yml"
DATASET="HAR"
LOG_DIR="logs"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "========================================"
echo "客户端选择消融实验"
echo "========================================"
echo "数据集: ${DATASET}"
echo "配置文件: ${BASE_CONFIG}"
echo "日志目录: ${LOG_DIR}"
echo "========================================"
echo ""

# ============================================
# Baseline: FedAvg without client selection
# ============================================
# echo "=== Baseline (FedAvg without client selection) ==="
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --description "Baseline_FedAvg" > ${LOG_DIR}/baseline_fedavg.log 2>&1

# ============================================
# Ablation 1: 不同采样比例 (ratio)
# 固定: min_prob=0.01, ema_alpha=0.2
# ============================================
# echo "=== Ablation 1: Different client selection ratios ==="
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.5 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.5" > ${LOG_DIR}/ablation_ratio0.5.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.6 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.6" > ${LOG_DIR}/ablation_ratio0.6.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.7" > ${LOG_DIR}/ablation_ratio0.7.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.8 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.8" > ${LOG_DIR}/ablation_ratio0.8.log 2>&1 &
# wait

# ============================================
# Ablation 2: 不同最小选择概率 (min_prob)
# 固定: ratio=0.7, ema_alpha=0.2
# ============================================
echo "=== Ablation 2: Different minimum selection probabilities ==="
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.005 --ema-alpha 0.2 --description "Ablation_minprob0.005" > ${LOG_DIR}/ablation_minprob0.005.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_minprob0.01" > ${LOG_DIR}/ablation_minprob0.01.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.02 --ema-alpha 0.2 --description "Ablation_minprob0.02" > ${LOG_DIR}/ablation_minprob0.02.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.05 --ema-alpha 0.2 --description "Ablation_minprob0.05" > ${LOG_DIR}/ablation_minprob0.05.log 2>&1 &
wait

# ============================================
# Ablation 3: 不同EMA平滑系数 (ema_alpha)
# 固定: ratio=0.7, min_prob=0.01
# ============================================
echo "=== Ablation 3: Different EMA smoothing coefficients ==="
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.1 --description "Ablation_ema0.1" > ${LOG_DIR}/ablation_ema0.1.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ema0.2" > ${LOG_DIR}/ablation_ema0.2.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.3 --description "Ablation_ema0.3" > ${LOG_DIR}/ablation_ema0.3.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.5 --description "Ablation_ema0.5" > ${LOG_DIR}/ablation_ema0.5.log 2>&1 &
wait

echo ""
echo "========================================"
echo "所有消融实验完成！"
echo "========================================"
