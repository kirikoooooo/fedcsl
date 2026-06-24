#!/usr/bin/env bash
# auto_pull_restart.sh — 持续监控 git 远程更新，有更新则 pull 并重启 dashboard
#
# 用法:
#   chmod +x scripts/auto_pull_restart.sh
#   nohup bash scripts/auto_pull_restart.sh >> /tmp/auto_pull.log 2>&1 &
#
# 停止:
#   pkill -f auto_pull_restart.sh
#   pkill -f "dashboard.app"
#
# 查看日志:
#   tail -f /tmp/auto_pull.log

set -euo pipefail

# ====== 配置 ======
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${AUTO_PULL_BRANCH:-feat/muti-routine}"
CHECK_INTERVAL="${AUTO_PULL_INTERVAL:-60}"          # 检测间隔（秒）
DASHBOARD_HOST="${AUTO_PULL_HOST:-0.0.0.0}"
DASHBOARD_PORT="${AUTO_PULL_PORT:-8765}"
PYTHON_BIN="${AUTO_PULL_PY:-python}"

cd "${PROJECT_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 杀死旧 dashboard 进程
kill_dashboard() {
    # 按端口杀
    local old_pid
    old_pid=$(lsof -ti ":${DASHBOARD_PORT}" 2>/dev/null || true)
    if [[ -n "${old_pid}" ]]; then
        log "stopping old dashboard (pid=${old_pid})"
        kill "${old_pid}" 2>/dev/null || true
        sleep 2
        # 如果还在，强杀
        kill -9 "${old_pid}" 2>/dev/null || true
    fi
    # 兜底：按进程名杀
    pkill -f "dashboard.app" 2>/dev/null || true
}

# 启动 dashboard
start_dashboard() {
    log "starting: ${PYTHON_BIN} -m dashboard.app --host ${DASHBOARD_HOST} --port ${DASHBOARD_PORT}"
    cd "${PROJECT_DIR}"
    nohup "${PYTHON_BIN}" -m dashboard.app \
        --host "${DASHBOARD_HOST}" \
        --port "${DASHBOARD_PORT}" \
        >> /tmp/dashboard.log 2>&1 &
    local pid=$!
    log "dashboard started (pid=${pid})"
}

# ====== 主循环 ======
log "auto_pull_restart started"
log "  project:   ${PROJECT_DIR}"
log "  branch:    ${BRANCH}"
log "  interval:  ${CHECK_INTERVAL}s"
log "  dashboard: ${DASHBOARD_HOST}:${DASHBOARD_PORT}"

# 首次启动 dashboard
kill_dashboard
start_dashboard

while true; do
    sleep "${CHECK_INTERVAL}"

    # 拉取远程信息
    if ! git fetch origin "${BRANCH}" 2>/dev/null; then
        log "git fetch failed — 网络问题？跳过本轮"
        continue
    fi

    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    REMOTE=$(git rev-parse "origin/${BRANCH}" 2>/dev/null || echo "unknown")

    if [[ "${LOCAL}" == "unknown" || "${REMOTE}" == "unknown" ]]; then
        log "git rev-parse 失败"
        continue
    fi

    if [[ "${LOCAL}" == "${REMOTE}" ]]; then
        # 没有更新
        continue
    fi

    log "更新检测到: ${LOCAL:0:8} -> ${REMOTE:0:8}"

    # Pull
    if ! git pull origin "${BRANCH}" 2>&1; then
        log "git pull 失败，跳过重启"
        continue
    fi

    NEW_LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    log "pull 完成: ${NEW_LOCAL:0:8}"

    # 重启 dashboard
    kill_dashboard
    start_dashboard
done
