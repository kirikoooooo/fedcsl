#!/usr/bin/env bash
# Coverage Sensitivity 实验：greedy swap 的 coverage_target 参数敏感度分析。
#
# 对每个 coverage_target 做一次独立 run，使用持久化显存数据避免并行 GPU 冲突。
#
# 用法（网格）：
#   COVERAGE_TARGET="1,2,3,4,5" DIR_ALPHA="0.1,0.5,1.0" DATASET=HAR \
#     bash scripts/coverage_sensitivity.sh
#
# 用法（单次，默认 cov=1..5 alpha=0.5）：
#   bash scripts/coverage_sensitivity.sh

set -euo pipefail

CONFIG="${CONFIG:-config/configSpilter.yml}"
DATASET="${DATASET:-HAR}"
SEED="${SEED:-42}"

IFS=',' read -ra COVS <<< "${COVERAGE_TARGET:-1,2,3,4,5}"
IFS=',' read -ra ALPHAS <<< "${DIR_ALPHA:-0.5}"

for alpha in "${ALPHAS[@]}"; do
  a="$(echo "$alpha" | xargs)"
  for cov in "${COVS[@]}"; do
    c="$(echo "$cov" | xargs)"
    echo "[coverage_sensitivity] cov=${c} alpha=${a} dataset=${DATASET}"
    python -u FedCSL_All.py \
      -dataset "${DATASET}" \
      --config "${CONFIG}" \
      --dirichlet-alpha "${a}" \
      --seed "${SEED}" \
      --description "${DATASET}_spilter_cov${c}_dir${a}" \
      --spilter-knapsack \
      --scale-memory-cache \
      --coverage-target "${c}" &
  done
done
wait
echo "[coverage_sensitivity] done"
