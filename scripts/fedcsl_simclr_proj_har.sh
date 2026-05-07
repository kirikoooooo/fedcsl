#!/usr/bin/env bash
# FedCSL-SimCLR-Proj：HAR 单脚本启动入口（dashboard 可直接发现并一键运行）。
#
# 用 projector-space InfoNCE 替代 encoder-space 主损失，
# 同时保留多尺度 shapelet 与 ACF/STFT 周期评分。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
CONFIG="${CONFIG:-config/configFedCSL_SimCLR_Proj.yml}"
SEED="${SEED:-42}"
DESC="${DESC:-${DATASET}_fedcsl_simclr_proj_dir${DIR_ALPHA}}"

# shellcheck disable=SC1091
source "${HERE}/_run_har_algo.sh"
