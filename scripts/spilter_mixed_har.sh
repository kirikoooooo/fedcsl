#!/usr/bin/env bash
# Spilter Mixed-m：10 客户端 heterogeneous m 实验（3×m4, 4×m2, 3×m1）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configSpilterMixed.yml
#   SEED=42
#   SPILTER_RANDOM=0   (0=top-m selection, 1=random non-topM selection)
#   PY_BIN=python3
#
# 当前模式：
#   - 10 个客户端，m 值按 [4,4,4,2,2,2,2,1,1,1] 分配；
#   - 默认按本地周期评分选择 top-m 尺度（local_score_topm）；
#   - 设置 SPILTER_RANDOM=1 可切换为随机选尺度（local_score_random_topm）；
#   - 所选尺度拼接为一个局部子模型表示进行训练；
#   - 只下发/上传这些被选中的尺度参数，并按尺度聚合。
#
# 用法:
#   bash scripts/spilter_mixed_har.sh
#   SPILTER_RANDOM=1 DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/spilter_mixed_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

PY_BIN="${PY_BIN:-python3}"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configSpilterMixed.yml}"
SEED="${SEED:-42}"
SPILTER_RANDOM="${SPILTER_RANDOM:-0}"

MODE_SUFFIX="topm"
CMD_EXTRA_ARGS=()
if [[ "${SPILTER_RANDOM}" == "1" ]]; then
  MODE_SUFFIX="random"
  CMD_EXTRA_ARGS+=(--spilter-random)
fi

DESC="${DESC:-${DATASET}_spilter_mixed_${MODE_SUFFIX}_dir${DIR_ALPHA}}"
# 追加用户通过 $@ 传入的额外参数（dashboard 的 extra args 通过此通道传入）
CMD_EXTRA_ARGS+=("$@")

exec "${PY_BIN}" -u FedCSL_All.py \
  -dataset "${DATASET}" \
  --config "${CONFIG}" \
  --dirichlet-alpha "${DIR_ALPHA}" \
  --seed "${SEED}" \
  --description "${DESC}" \
  "${CMD_EXTRA_ARGS[@]}"
