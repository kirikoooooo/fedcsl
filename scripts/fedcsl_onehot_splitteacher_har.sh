#!/usr/bin/env bash
# FedCSL onehot splitteacher：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configFedCSL_OneHot_SplitTeacher.yml
#   SEED=42
#
# 用法:
#   bash scripts/fedcsl_onehot_splitteacher_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/fedcsl_onehot_splitteacher_har.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY_BIN="${PY_BIN:-python}"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configFedCSL_OneHot_SplitTeacher.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_onehot_splitteacher_dir${DIR_ALPHA}}"

exec "${PY_BIN}" -u FedCSL_All.py \
  -dataset "${DATASET}" \
  --config "${CONFIG}" \
  --dirichlet-alpha "${DIR_ALPHA}" \
  --seed "${SEED}" \
  --description "${DESC}"
