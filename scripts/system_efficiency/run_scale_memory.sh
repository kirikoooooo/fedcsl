#!/usr/bin/env bash
# HAR 尺度显存标定 + 背包 DP 选择实验（§7.7 支撑，不影响已有实验）。
#
# 行为：
#   1) measure_scale_memory.py —— 逐个测量单尺度子模型 1-epoch 训练峰值显存，
#      可选验证若干尺度组合的可加性；产出 data/scale_memory_HAR_partials/per_scale.json；
#   2) fit_scale_memory.py —— 拟合可加显存模型 g0+sum(g_r)，报告可加性误差，
#      在给定显存预算档位下用背包 DP 求最优尺度组合并与 top-m 对照；
#      产出 data/scale_memory_HAR.{json,md}.
#
# 与 §7.6 system_efficiency 实验共用 LearningShapeletsCL 与显存测量口径
# （warmup 后 reset_peak_memory_stats + max_memory_allocated），结果可直接对比。
#
# 环境变量（可覆盖默认值）：
#   GPU                GPU index，默认 0
#   NUM_CLIENTS        客户端数，默认 10（标定 g_r 用少量客户端即可）
#   ALPHA              Dirichlet 异质度，默认 0.1
#   BATCH_SIZE         默认 32（与 config/configSpilter.yml 对齐）
#   MAX_CLIENTS        每个尺度最多测多少客户端，默认 0=全部
#   VERIFY_SUBSETS     分号分隔的尺度子集做可加性验证，如 "0,1;0,3,7;2,4,6"，默认空
#   BUDGETS            背包预算档位 (MB)，默认 "64,128,256"（小/中/大显存设备）
#   TOPM               对照的固定 top-m，默认 4
#   SCALE_AUX          1=开 per-scale CL/KD（默认），0=关
#
# 用法：
#   bash scripts/system_efficiency/run_scale_memory.sh
#   GPU=1 NUM_CLIENTS=10 VERIFY_SUBSETS="0,1;2,4,6;0,3,7" bash scripts/system_efficiency/run_scale_memory.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${GPU:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ALPHA="${ALPHA:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_CLIENTS="${MAX_CLIENTS:-0}"
VERIFY_SUBSETS="${VERIFY_SUBSETS:-}"
BUDGETS="${BUDGETS:-64,128,256}"
TOPM="${TOPM:-4}"
SCALE_AUX="${SCALE_AUX:-1}"

NO_SCALE_AUX_FLAG=""
if [[ "$SCALE_AUX" == "0" ]]; then
  NO_SCALE_AUX_FLAG="--no-scale-aux"
fi

PARTIALS_DIR="data/scale_memory_HAR_partials"
mkdir -p "$PARTIALS_DIR"

echo "================================================================"
echo "[run_scale_memory] HAR 尺度显存标定 + 背包 DP"
echo "  GPU=$GPU  NUM_CLIENTS=$NUM_CLIENTS  ALPHA=$ALPHA  BATCH_SIZE=$BATCH_SIZE"
echo "  MAX_CLIENTS=$MAX_CLIENTS  SCALE_AUX=$SCALE_AUX"
echo "  VERIFY_SUBSETS='$VERIFY_SUBSETS'"
echo "  BUDGETS=$BUDGETS  TOPM=$TOPM"
echo "================================================================"

echo
echo "[$(date +%H:%M:%S)] >>> step 1/2: 逐尺度显存标定"
CUDA_VISIBLE_DEVICES="$GPU" \
python scripts/system_efficiency/measure_scale_memory.py \
  --num-clients "$NUM_CLIENTS" \
  --alpha "$ALPHA" \
  --batch-size "$BATCH_SIZE" \
  --max-clients-per-scale "$MAX_CLIENTS" \
  --verify-subsets "$VERIFY_SUBSETS" \
  $NO_SCALE_AUX_FLAG \
  --output "${PARTIALS_DIR}/per_scale.json"

echo
echo "[$(date +%H:%M:%S)] >>> step 2/2: 拟合 + 背包 DP"
python scripts/system_efficiency/fit_scale_memory.py \
  --partial "${PARTIALS_DIR}/per_scale.json" \
  --budgets "$BUDGETS" \
  --topm "$TOPM" \
  --out-json data/scale_memory_HAR.json \
  --out-md   data/scale_memory_HAR.md

echo
echo "================================================================"
echo "[summary] outputs:"
echo "  - data/scale_memory_HAR_partials/per_scale.json  (逐尺度峰值)"
echo "  - data/scale_memory_HAR.json                     (可加模型 + DP 结果)"
echo "  - data/scale_memory_HAR.md                       (可读报告)"
echo "================================================================"
