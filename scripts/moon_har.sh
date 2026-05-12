#!/usr/bin/env bash
# MOON：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 用法:
#   bash scripts/moon_har.sh
#   DIR_ALPHA=0.5 CUDA_VISIBLE_DEVICES=0 bash scripts/moon_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configMOON.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_moon_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
