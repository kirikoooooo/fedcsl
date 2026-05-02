#!/usr/bin/env bash
# FedAvg：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configAVG.yml
#   SEED=42
#   PY_BIN=python3
#
# 用法:
#   bash scripts/fedavg_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/fedavg_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configAVG.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_fedavg_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
