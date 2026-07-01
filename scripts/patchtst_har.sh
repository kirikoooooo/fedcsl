#!/usr/bin/env bash
# PatchTST self-supervised + FedAvg：HAR 单脚本启动入口。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configPatchTST.yml
#   SEED=42
#   PY_BIN=python3
#
# 用法:
#   bash scripts/patchtst_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/patchtst_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configPatchTST.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_patchtst_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
