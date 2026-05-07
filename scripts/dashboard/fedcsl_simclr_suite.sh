#!/usr/bin/env bash
# FedCSL-SimCLR 方案筛选套件：dashboard 可直接发现并一键启动。
#
# 作用：
#   1. 串行跑一批候选方案；
#   2. 若首轮全局评估精度低于阈值，则提前停止该方案，转入后续方案；
#   3. 在同目录持久化 RUN_HISTORY.md，便于后续调参记录。
#
# 用法：
#   bash scripts/dashboard/fedcsl_simclr_suite.sh
#   GATE_ACC=0.92 LIMIT=6 bash scripts/dashboard/fedcsl_simclr_suite.sh
#   DATASET=Epilepsy-TSTCC bash scripts/dashboard/fedcsl_simclr_suite.sh
#   bash scripts/dashboard/fedcsl_simclr_suite.sh --dataset Epilepsy-TSTCC --limit 4

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${ROOT}"

PY_BIN="${PY_BIN:-python3}"
DATASET="${DATASET:-}"
PLANS="${PLANS:-${HERE}/fedcsl_simclr_suite_plans.yml}"
HISTORY="${HISTORY:-${HERE}/RUN_HISTORY.md}"
LOG_DIR="${LOG_DIR:-${HERE}/logs}"
ARGS=()

if [[ -n "${DATASET}" ]]; then
  ARGS+=(--dataset "${DATASET}")
fi
if [[ -n "${GATE_ACC:-}" ]]; then
  ARGS+=(--gate-acc "${GATE_ACC}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

exec "${PY_BIN}" "${HERE}/fedcsl_simclr_suite_runner.py" \
  --plans "${PLANS}" \
  --history "${HISTORY}" \
  --log-dir "${LOG_DIR}" \
  "${ARGS[@]}" \
  "$@"
