# shellcheck shell=bash
# -----------------------------------------------------------------------------
# HAR 数据集 × 不同 dirichlet 异质性 × 联邦 baseline 公共脚本。
# 被 scripts/har_baselines/har_dir*.sh 以及 run_all.sh 以 ``source`` 方式引用。
#
# 职责：
#   1. 通过 gpu_sched.py 监听可用 GPU，按 "显存空闲比 ≥ HAR_MEM_FREE_RATIO" 派给本次训练；
#      —— 允许**同一张 GPU 运行多个自家任务**，只要显存还剩得够。
#   2. 派发前仅要求可用 GPU 数 ≥ HAR_MIN_FREE（默认 1）；不再额外保留"严格空闲卡"。
#      mem 模式下会优先把新任务分散到本会话里当前负载更低的 GPU，避免集中启动到一张卡。
#   3. 每个任务在后台启动（&），脚本末尾用 har_wait_all 等待全部完成；
#   4. 2 号卡永远排除。
#
# 可调环境变量（均可选）：
#   HAR_STRATEGY         调度策略: "mem" (默认, 按显存阈值) 或 "idle" (严格空闲)
#   HAR_MEM_FREE_RATIO   mem 策略下的显存空闲比阈值 (默认 0.30; 即 "已用 ≤ 70%")
#   HAR_MIN_FREE         派发前必须的 "可用 GPU" 下限 (默认 1)
#   HAR_GPU_EXCLUDE      禁用 GPU id，空格分隔，默认 "2"
#   HAR_POLL_INTERVAL    GPU 轮询间隔（秒），默认 20
#   HAR_WARMUP_SEC       启动任务后等待的时间（秒），让 GPU 真正吃显存再挑下一张，默认 15
#   HAR_TIMEOUT          单次 wait 的最大等待秒数，0=无限（默认 0）
#   HAR_SEED             随机种子，默认 42
#   HAR_EXTRA_ARGS       追加到 FedCSL_All.py 命令末尾的自由参数（按空白切分）
#   PY_BIN               训练用 Python 可执行文件，默认 python3
#   SCHED_PY_BIN         调度器用 Python，默认 python3
# -----------------------------------------------------------------------------

set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_HERE}/../.." && pwd)"
LOG_ROOT="${PROJECT_ROOT}/result/har_baselines"
DATASET="${DATASET:-HAR}"
PY_BIN="${PY_BIN:-python3}"              # 用于训练（FedCSL_All.py）
SCHED_PY_BIN="${SCHED_PY_BIN:-python3}"   # 用于调度器（gpu_sched.py）
SEED="${HAR_SEED:-42}"

STRATEGY="${HAR_STRATEGY:-mem}"
MEM_FREE_RATIO="${HAR_MEM_FREE_RATIO:-0.30}"
MIN_FREE="${HAR_MIN_FREE:-1}"
GPU_EXCLUDE="${HAR_GPU_EXCLUDE:-2}"
POLL_INTERVAL="${HAR_POLL_INTERVAL:-20}"
WARMUP_SEC="${HAR_WARMUP_SEC:-15}"
TIMEOUT="${HAR_TIMEOUT:-0}"

# 本会话专属 reserve-file：用 PPID 隔离不同 run_all.sh 实例。
RESERVE_FILE="${HAR_RESERVE_FILE:-${TMPDIR:-/tmp}/har_sched_${PPID}.reserved}"
: > "${RESERVE_FILE}"  # 清空

mkdir -p "${LOG_ROOT}"

# 跟踪所有后台任务的 PID，便于 har_wait_all
HAR_PIDS=()

# (algo:config) 组合，第一列仅用于日志/描述，真正决定算法的是 config.algo 字段。
BASELINES=(
  "fedavg:configAVG.yml"
  "fedprox:configFedProx.yml"
  "scaffold:configSCAFFOLD.yml"
  "fedproto:configFedProto.yml"
  "fedcsl:configACF.yml"
  "fedcsl-onehot:configFedCSL_OneHot.yml"
  "fedcsl-onehot-fullteacher:configFedCSL_OneHot_FullTeacher.yml"
  "fedcsl-onehot-splitteacher:configFedCSL_OneHot_SplitTeacher.yml"
)

# ---------------------------------------------------------------------------
# 内部：挑一张 GPU（阻塞直到满足条件）
# ---------------------------------------------------------------------------
_pick_gpu() {
  # shellcheck disable=SC2206
  local exclude_args=(${GPU_EXCLUDE})
  "${SCHED_PY_BIN}" "${_HERE}/gpu_sched.py" wait \
    --strategy "${STRATEGY}" \
    --mem-free-ratio "${MEM_FREE_RATIO}" \
    --min-free "${MIN_FREE}" \
    --exclude "${exclude_args[@]}" \
    --reserve-file "${RESERVE_FILE}" \
    --poll-interval "${POLL_INTERVAL}" \
    --timeout "${TIMEOUT}"
}

_release_gpu() {
  local gpu_id="$1"
  "${SCHED_PY_BIN}" "${_HERE}/gpu_sched.py" release \
    --gpu "${gpu_id}" \
    --reserve-file "${RESERVE_FILE}" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# 对外：启动单个 baseline（后台）。
#   run_baseline <algo-name> <config-file> <dirichlet-alpha>
# ---------------------------------------------------------------------------
run_baseline() {
  local algo="$1"
  local cfg="$2"
  local alpha="$3"
  shift 3
  local extra_args=("$@")
  local desc="har_${algo}_dir${alpha}"
  local log="${LOG_ROOT}/${desc}.log"

  cd "${PROJECT_ROOT}"

  if [[ "${STRATEGY}" == "mem" ]]; then
    echo "[$(date +%H:%M:%S)] ⏳ ${desc}  等待可用 GPU (strategy=mem, mem_free≥${MEM_FREE_RATIO}, min_free=${MIN_FREE}, exclude=${GPU_EXCLUDE}, spread=on)"
  else
    echo "[$(date +%H:%M:%S)] ⏳ ${desc}  等待空闲 GPU (strategy=idle, min_free=${MIN_FREE}, exclude=${GPU_EXCLUDE})"
  fi
  local gpu_id
  gpu_id="$(_pick_gpu)" || {
    echo "[$(date +%H:%M:%S)] ✗ ${desc} 调度失败，跳过"
    return 1
  }

  if [[ "${gpu_id}" == "-1" ]]; then
    echo "[$(date +%H:%M:%S)] ⚠ nvidia-smi 不可用，本任务将不设置 CUDA_VISIBLE_DEVICES"
  fi

  # 组装命令；HAR_EXTRA_ARGS 可选追加
  local cmd=("${PY_BIN}" -u FedCSL_All.py
    -dataset "${DATASET}"
    --config "${cfg}"
    --dirichlet-alpha "${alpha}"
    --seed "${SEED}"
    --description "${desc}")
  if [[ -n "${HAR_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local _extra=(${HAR_EXTRA_ARGS})
    cmd+=("${_extra[@]}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi

  echo "[$(date +%H:%M:%S)] ▶ ${desc}  on GPU ${gpu_id}  (config=${cfg}, alpha=${alpha})"
  (
    # 子 shell 内启动真实训练；退出前释放 GPU 预留。
    if [[ "${gpu_id}" != "-1" ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
    fi
    # 确保子 shell 退出时释放（无论成败、被 kill）
    trap '_release_gpu '"${gpu_id}"'' EXIT INT TERM

    {
      echo "# $(date '+%Y-%m-%d %H:%M:%S')  ${desc}  GPU=${gpu_id}  cwd=$(pwd)"
      echo "# CMD: ${cmd[*]}"
      echo
    } > "${log}"

    if "${cmd[@]}" >> "${log}" 2>&1; then
      echo "[$(date +%H:%M:%S)] ✓ ${desc}  on GPU ${gpu_id}  (log=${log})"
    else
      local rc=$?
      echo "[$(date +%H:%M:%S)] ✗ ${desc}  on GPU ${gpu_id}  rc=${rc}  (log=${log})"
      # 不吞错误码，仍正常退出子 shell 让 wait 能收到 rc
      exit "${rc}"
    fi
  ) &
  local pid=$!
  HAR_PIDS+=("${pid}")
  echo "[$(date +%H:%M:%S)]   pid=${pid} (GPU ${gpu_id})"

  # 等 GPU 真正占用再去挑下一张，避免 nvidia-smi 延迟导致重复派发。
  if [[ "${WARMUP_SEC}" -gt 0 ]]; then
    sleep "${WARMUP_SEC}"
  fi
}

# 给一个 dirichlet 值跑全部 baseline
run_all_baselines_for_alpha() {
  local alpha="$1"
  shift 1
  local extra_args=("$@")
  local spec algo cfg
  for spec in "${BASELINES[@]}"; do
    algo="${spec%%:*}"
    cfg="${spec##*:}"
    run_baseline "${algo}" "${cfg}" "${alpha}" "${extra_args[@]}" || true
  done
}

# 等待本脚本派发的全部后台任务
har_wait_all() {
  if [[ "${#HAR_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "[$(date +%H:%M:%S)] 等待 ${#HAR_PIDS[@]} 个后台任务结束..."
  local pid failed=0
  for pid in "${HAR_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=$((failed + 1))
    fi
  done
  echo "[$(date +%H:%M:%S)] 全部任务完成 (失败 ${failed} 个)"
  return 0
}

# 脚本被 Ctrl-C / 异常退出时，尝试 kill 所有子任务
_har_trap_shutdown() {
  echo ""
  echo "[$(date +%H:%M:%S)] 收到终止信号，kill 所有后台任务..."
  local pid
  for pid in "${HAR_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  exit 130
}
trap _har_trap_shutdown INT TERM
