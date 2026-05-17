#!/usr/bin/env bash
# HAR 系统效率实验：串行测每个联邦算法 "最慢客户端 1 epoch" 耗时。
#
# 行为：
#   - 对 HAR_results 表里的全部算法（FedAvg/FedProx/FedBYOL/FedU2/Orchestra/
#     FedCSL/Spilter-m1/m2/m4）依次启动一个独立 python 进程；
#   - 每个进程内部对 K 个客户端串行计时（同时刻 GPU 只有一个客户端在跑）；
#   - 通过 CUDA_VISIBLE_DEVICES 把每个进程锁到一张卡上（默认 0）；
#   - 全部跑完后调用 aggregate_results.py 生成 data/system_efficiency_HAR.{json,csv,md}.
#
# 环境变量（可覆盖默认值）：
#   GPU                指定使用的 GPU index，默认 0
#   NUM_CLIENTS        客户端数，默认 50
#   ALPHA              Dirichlet 异质度，默认 0.1
#   BATCH_SIZE         默认 8
#   LR                 默认 0.05
#   NUM_EPOCH          每客户端本地 epoch 数，默认 1
#   WARMUP_BATCHES     cudnn warmup 个 batch 数，默认 1
#   ALGOS              空格分隔的算法子集，默认全部
#
# 用法：
#   bash scripts/system_efficiency/run_har_efficiency.sh
#   GPU=1 ALPHA=0.5 NUM_CLIENTS=30 bash scripts/system_efficiency/run_har_efficiency.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${GPU:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
ALPHA="${ALPHA:-0.1}"
# 与 config/configSpilter.yml 等主流程 yml 保持一致 (model.params.batch_size = 32)；
# 想测不同 batch 的扫描，可在命令行覆盖 BATCH_SIZE。
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.01}"
NUM_EPOCH="${NUM_EPOCH:-1}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
DEFAULT_ALGOS="fedavg fedprox byol fedu2 orchestra fedcsl spilter-m1 spilter-m2 spilter-m4"
ALGOS="${ALGOS:-$DEFAULT_ALGOS}"

# SCALE_AUX=0 关闭 UseScaleCL/UseScaleKD 的 per-scale 辅助 loss。
# Spilter stitched 模式下，per-scale 辅助 loss 会让显存几乎随 m 翻倍
# （是 m=4 显存反而比 FedCSL 大的根因）。设为 0 测「纯尺度切分」显存上界。
SCALE_AUX="${SCALE_AUX:-1}"
NO_SCALE_AUX_FLAG=""
if [[ "$SCALE_AUX" == "0" ]]; then
  NO_SCALE_AUX_FLAG="--no-scale-aux"
fi

PARTIALS_DIR="data/system_efficiency_HAR_partials"
mkdir -p "$PARTIALS_DIR"

echo "================================================================"
echo "[run_har_efficiency] HAR 系统效率实验"
echo "  GPU=$GPU  NUM_CLIENTS=$NUM_CLIENTS  ALPHA=$ALPHA"
echo "  BATCH_SIZE=$BATCH_SIZE  LR=$LR  NUM_EPOCH=$NUM_EPOCH"
echo "  WARMUP_BATCHES=$WARMUP_BATCHES  SCALE_AUX=$SCALE_AUX"
echo "  PARTIALS_DIR=$PARTIALS_DIR"
echo "  ALGOS=$ALGOS"
echo "================================================================"

TOTAL=0
FAILED=()
for algo in $ALGOS; do
  TOTAL=$((TOTAL + 1))
  echo
  echo "----------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] >>> algo=$algo  (GPU=$GPU)"
  echo "----------------------------------------------------------------"
  # 单 GPU 独占：CUDA_VISIBLE_DEVICES 限定可见卡，CUDA_LAUNCH_BLOCKING=0 保留默认调度
  CUDA_VISIBLE_DEVICES="$GPU" \
  python scripts/system_efficiency/measure_har_epoch.py \
    --algo "$algo" \
    --num-clients "$NUM_CLIENTS" \
    --alpha "$ALPHA" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-epoch "$NUM_EPOCH" \
    --warmup-batches "$WARMUP_BATCHES" \
    $NO_SCALE_AUX_FLAG \
    --output "${PARTIALS_DIR}/{algo}.json" \
    || { echo "[err] algo=$algo failed"; FAILED+=("$algo"); }
done

echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] aggregating results"
echo "================================================================"
python scripts/system_efficiency/aggregate_results.py \
  --partials "$PARTIALS_DIR" \
  --out-json data/system_efficiency_HAR.json \
  --out-csv  data/system_efficiency_HAR.csv \
  --out-md   data/system_efficiency_HAR.md

echo
echo "================================================================"
echo "[summary] $((TOTAL - ${#FAILED[@]}))/$TOTAL algorithms timed."
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[summary] failed: ${FAILED[*]}"
fi
echo "[summary] outputs:"
echo "  - data/system_efficiency_HAR.md"
echo "  - data/system_efficiency_HAR.csv"
echo "  - data/system_efficiency_HAR.json"
echo "================================================================"
