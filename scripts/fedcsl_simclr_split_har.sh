#!/usr/bin/env bash
# FedCSL-SimCLR-Split：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 保留 SimCLR 风格本地对比目标，同时启用 Spilter 的尺度计划、
# 只下发/上传被选中的尺度参数，并按尺度聚合。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configFedCSL_SimCLR_Split.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_fedcsl_simclr_split_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
