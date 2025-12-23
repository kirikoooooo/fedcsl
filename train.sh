#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# 定义数据集列表
datasets=(
    "ArticularyWordRecognition"
    "AtrialFibrillation"
    "BasicMotions"
    "CharacterTrajectories"
    "Cricket"
    "DuckDuckGeese"
    "EigenWorms"
    "Epilepsy"
    "ERing"
    "EthanolConcentration"
    "FaceDetection"
    "FingerMovements"
    "HandMovementDirection"
    "Handwriting"
    "Heartbeat"
    "InsectWingbeat"
    "JapaneseVowels"
    "Libras"
    "LSST"
    "MotorImagery"
    "NATOPS"
    "PEMS-SF"
    "PenDigits"
    "PhonemeSpectra"
    "RacketSports"
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "SpokenArabicDigits"
    "StandWalkJump"
    "UWaveGestureLibrary"
)
datasetsBig=(
    "CharacterTrajectories"
    "DuckDuckGeese"
    "EigenWorms"
    "FaceDetection"
    "InsectWingbeat"
    "JapaneseVowels"
    "LSST"
    "MotorImagery"
    "PenDigits"
    "PhonemeSpectra"
    "SpokenArabicDigits"
)
# 遍历数据集并顺序执行训练
for dataset in "${datasetsBig[@]}"; do
    echo "Starting training for dataset: $dataset"

    # 执行训练命令，并捕获退出状态码
    if nohup python FedCSL_All.py -dataset "$dataset" --config ./configACF.yml > "./log/log_${dataset}.txt" 2>&1; then
        echo "Training for dataset $dataset completed successfully."
    else
        echo "Error occurred during training for dataset $dataset. Skipping to the next dataset."
    fi

    # 等待当前任务完成后再继续下一个
    wait
done

echo "All datasets have been processed."