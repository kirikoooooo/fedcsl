#!/bin/bash
# 客户端选择消融实验命令列表
# 使用方法: bash ablation_experiments.sh

BASE_CONFIG="configAVG.yml"
DATASET="HAR"

# 基础命令（不使用客户端选择，作为baseline）
# echo "=== Baseline (FedAvg without client selection) ==="
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} > logs/baseline_fedavg.log 2>&1

# # 消融实验1: 不同采样比例
# echo "=== Ablation 1: Different client selection ratios ==="
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.5 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_ratio0.5.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.6 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_ratio0.6.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_ratio0.7.log 2>&1 &
# python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.8 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_ratio0.8.log 2>&1 &
# wait

# 消融实验2: 不同最小选择概率
echo "=== Ablation 2: Different minimum selection probabilities ==="
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.005 --ema-alpha 0.2 > logs/ablation_minprob0.005.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_minprob0.01.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.02 --ema-alpha 0.2 > logs/ablation_minprob0.02.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.05 --ema-alpha 0.2 > logs/ablation_minprob0.05.log 2>&1 &
wait

# 消融实验3: 不同EMA平滑系数
echo "=== Ablation 3: Different EMA smoothing coefficients ==="
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.1 > logs/ablation_ema0.1.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 > logs/ablation_ema0.2.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.3 > logs/ablation_ema0.3.log 2>&1 &
python FedCSL_Epilepsy.py -dataset ${DATASET} --config ${BASE_CONFIG} --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.5 > logs/ablation_ema0.5.log 2>&1 &
wait

echo "=== All ablation experiments completed ==="

