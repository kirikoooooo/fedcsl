#!/usr/bin/env bash
# FedCSL onehot splitteacher：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configFedCSL_OneHot_SplitTeacher.yml
#   SEED=42
#   PY_BIN=python3
#
# 当前算法说明：
#   - 每个 client 每轮训练 4 个尺度：本地评分保留 2 个，server 额外补 2 个；
#   - splitteacher 仍会只下发/上传这些被选中的尺度参数，并按尺度分组聚合。
#
# 用法:
#   bash scripts/fedcsl_onehot_splitteacher_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/fedcsl_onehot_splitteacher_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configFedCSL_OneHot_SplitTeacher.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_onehot_splitteacher_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
