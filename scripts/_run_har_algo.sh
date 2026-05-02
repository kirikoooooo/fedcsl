#!/usr/bin/env bash
# 内部 helper：供 scripts/*_har.sh 复用的 HAR 单脚本启动模板。
# 文件名以下划线开头，dashboard 扫描脚本时会自动跳过。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY_BIN="${PY_BIN:-python3}"
DATASET="${DATASET:-HAR}"
DIR_ALPHA="${DIR_ALPHA:-0.1}"
SEED="${SEED:-42}"

if [[ -z "${CONFIG:-}" ]]; then
  echo "CONFIG 未设置" >&2
  exit 2
fi

if [[ -z "${DESC:-}" ]]; then
  echo "DESC 未设置" >&2
  exit 2
fi

exec "${PY_BIN}" -u FedCSL_All.py \
  -dataset "${DATASET}" \
  --config "${CONFIG}" \
  --dirichlet-alpha "${DIR_ALPHA}" \
  --seed "${SEED}" \
  --description "${DESC}"
