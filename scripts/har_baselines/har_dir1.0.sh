#!/usr/bin/env bash
# HAR × Dirichlet α = 1.0 × {fedavg, fedprox, scaffold, fedproto, fedcsl}
#
# 自动在空闲 GPU 上并行派发，核心调度逻辑见 _common.sh + gpu_sched.py。
# 关键约束：2 号卡禁用；空闲 GPU ≥ 3 才挂新任务；预留至少 1 张空闲。
#
# 用法:
#   bash scripts/har_baselines/har_dir1.0.sh
#
# 常用环境变量（都可选）:
#   HAR_MIN_FREE=3          启动门槛（空闲 GPU 数下限），默认 3
#   HAR_GPU_EXCLUDE="2"     禁用的 GPU id（空格分隔），默认 "2"
#   HAR_POLL_INTERVAL=20    轮询间隔（秒）
#   HAR_WARMUP_SEC=40       任务启动后等待多少秒再挑下一张 GPU
#   HAR_SEED=42             随机种子
#   HAR_EXTRA_ARGS="..."    追加到 FedCSL_All.py 的额外参数
#   PY_BIN=python           Python 可执行文件

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

ALPHA=1.0
echo "========================================"
echo "HAR  dirichlet_alpha = ${ALPHA}  (all baselines)"
echo "========================================"
run_all_baselines_for_alpha "${ALPHA}"

har_wait_all
