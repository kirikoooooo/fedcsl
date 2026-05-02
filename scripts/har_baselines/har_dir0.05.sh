#!/usr/bin/env bash
# HAR × Dirichlet α = 0.05 × {fedavg, fedprox, scaffold, fedproto, fedcsl}
#
# 自动在空闲 GPU 上并行派发，核心调度逻辑见 _common.sh + gpu_sched.py。
# 关键约束：2 号卡禁用；调度策略默认 "mem"（显存空闲比 ≥ HAR_MEM_FREE_RATIO=0.70 即可派发）。
#
# 用法:
#   bash scripts/har_baselines/har_dir0.05.sh
#
# 常用环境变量（都可选）:
#   HAR_STRATEGY=mem        调度策略: "mem"（显存阈值）或 "idle"（严格空闲），默认 mem
#   HAR_MEM_FREE_RATIO=0.70 mem 模式下显存空闲比阈值（默认 0.70；即已用 ≤ 30%）
#   HAR_GPU_EXCLUDE="2"     禁用的 GPU id（空格分隔），默认 "2"
#   HAR_POLL_INTERVAL=20    轮询间隔（秒）
#   HAR_WARMUP_SEC=15       任务启动后等待多少秒再挑下一张 GPU
#   HAR_SEED=42             随机种子
#   HAR_EXTRA_ARGS="..."    追加到 FedCSL_All.py 的额外参数
#   PY_BIN=python           Python 可执行文件

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

ALPHA=0.05
echo "========================================"
echo "HAR  dirichlet_alpha = ${ALPHA}  (all baselines)"
echo "========================================"
run_all_baselines_for_alpha "${ALPHA}" "$@"

har_wait_all
