#!/usr/bin/env bash
# FedCSL onehot fullteacher：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 默认参数：
#   DATASET=HAR
#   DIR_ALPHA=0.1
#   CONFIG=config/configFedCSL_OneHot_FullTeacher.yml
#   SEED=42
#   PY_BIN=python3
#
# 当前算法说明：
#   - 每个 client 每轮不再只训练 1 个尺度；
#   - 本地先按现有评分在前半段/后半段各选 1 个尺度；
#   - server 再补 2 个尽量均衡的尺度，缓解周期选择过度集中。
#
# 用法:
#   bash scripts/fedcsl_onehot_fullteacher_har.sh
#   DIR_ALPHA=1.0 CUDA_VISIBLE_DEVICES=0 bash scripts/fedcsl_onehot_fullteacher_har.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configFedCSL_OneHot_FullTeacher.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_onehot_fullteacher_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
