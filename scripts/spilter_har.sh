#!/usr/bin/env bash
# Spilter：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configSpilter.yml
#   SEED=42
#   PY_BIN=python3
#
# 当前默认模式：
#   - 每个 client 根据本地周期评分选择 local_top_m 个尺度；
#   - 所选尺度拼接为一个局部子模型表示进行训练；
#   - 只下发/上传这些被选中的尺度参数，并按尺度聚合。
#
# 用法:
#   bash scripts/spilter_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/spilter_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configSpilter.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_spilter_local_topm_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
