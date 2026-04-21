"""FedCSL Dashboard — 独立的轻量 FastAPI 控制台。

设计原则：
  * 只读项目根与 ``config/``、``scripts/`` 目录；写文件范围仅限 ``config/*.yml``
    与 ``dashboard/logs/*.log``、``dashboard/runs.json``，不会动其他代码。
  * 所有路径做了目录穿越防御（必须位于 ``PROJECT_ROOT`` 之内）。
  * 无数据库，使用 ``runs.json`` 以 JSON 记录任务元数据。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = DASHBOARD_DIR / "logs"
RUNS_FILE = DASHBOARD_DIR / "runs.json"
STATIC_DIR = DASHBOARD_DIR / "static"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
_runs_lock = threading.Lock()


def _safe_under(root: Path, target: Path) -> Path:
    """确保 ``target`` 解析后位于 ``root`` 之下，否则抛 404。"""
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="invalid path")
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes project root")
    return resolved


def _load_runs() -> list[dict[str, Any]]:
    with _runs_lock:
        if not RUNS_FILE.exists():
            return []
        try:
            return json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []


def _save_runs(runs: list[dict[str, Any]]) -> None:
    with _runs_lock:
        RUNS_FILE.write_text(
            json.dumps(runs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _pid_alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _refresh_run_status(run: dict[str, Any]) -> dict[str, Any]:
    """根据 pid 是否存活更新 run 的 status 字段（in-place）。"""
    if run.get("status") == "running":
        pid = run.get("pid")
        if pid and not _pid_alive(pid):
            run["status"] = "exited"
            run["ended_at"] = run.get("ended_at") or time.time()
    return run


# ---------------------------------------------------------------------------
# nvidia-smi 解析
# ---------------------------------------------------------------------------
def _run_nvidia_smi(fields: str, *, mode: str = "gpu") -> list[list[str]] | None:
    """``mode``: ``"gpu"`` -> --query-gpu, ``"apps"`` -> --query-compute-apps。"""
    flag = "--query-gpu" if mode == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"{flag}={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [
        [c.strip() for c in line.split(",")]
        for line in out.strip().splitlines()
        if line.strip()
    ]


def gpu_snapshot() -> dict[str, Any]:
    gpus_raw = _run_nvidia_smi(
        "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    )
    procs_raw = _run_nvidia_smi(
        "gpu_uuid,pid,process_name,used_memory", mode="apps"
    )
    if gpus_raw is None:
        return {"available": False, "gpus": []}

    uuid_map_raw = _run_nvidia_smi("index,uuid")
    index_by_uuid: dict[str, int] = {}
    if uuid_map_raw:
        for row in uuid_map_raw:
            if len(row) >= 2:
                try:
                    index_by_uuid[row[1]] = int(row[0])
                except ValueError:
                    pass

    procs_by_idx: dict[int, list[dict[str, Any]]] = {}
    if procs_raw:
        for row in procs_raw:
            if len(row) < 4:
                continue
            gpu_uuid, pid, pname, mem = row
            idx = index_by_uuid.get(gpu_uuid)
            if idx is None:
                continue
            try:
                pid_int = int(pid)
                mem_mb = int(mem.split()[0]) if mem else 0
            except ValueError:
                continue
            user = "?"
            try:
                user = psutil.Process(pid_int).username()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            procs_by_idx.setdefault(idx, []).append(
                {"pid": pid_int, "name": pname, "mem_mb": mem_mb, "user": user}
            )

    gpus: list[dict[str, Any]] = []
    for row in gpus_raw:
        if len(row) < 6:
            continue
        try:
            idx = int(row[0])
            util = int(row[2])
            mem_used = int(row[3])
            mem_total = int(row[4])
            temp = int(row[5])
        except ValueError:
            continue
        gpus.append(
            {
                "index": idx,
                "name": row[1],
                "utilization": util,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "temperature_c": temp,
                "processes": procs_by_idx.get(idx, []),
            }
        )

    return {"available": True, "gpus": gpus}


# ---------------------------------------------------------------------------
# 脚本扫描
# ---------------------------------------------------------------------------
_DESC_RE = re.compile(r"^\s*#\s?(.*)$")


def _extract_script_doc(path: Path, max_lines: int = 30) -> str:
    """抽取脚本首部连续注释块作为说明（跳过 shebang）。"""
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, raw in enumerate(f):
                if i == 0 and raw.startswith("#!"):
                    continue
                m = _DESC_RE.match(raw)
                if not m:
                    if lines:
                        break
                    if raw.strip() == "":
                        continue
                    break
                lines.append(m.group(1).rstrip())
                if len(lines) >= max_lines:
                    break
    except OSError:
        return ""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def list_scripts() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    patterns = [
        (SCRIPTS_DIR, "**/*.sh"),
        (PROJECT_ROOT, "*.py"),
    ]
    allowed_root_py = {"run_tasks.py", "generate_tasks.py"}
    for root, pat in patterns:
        if not root.exists():
            continue
        for p in sorted(root.glob(pat)):
            if p.suffix == ".py" and p.name not in allowed_root_py:
                continue
            if p.name.startswith("_") and p.suffix == ".sh":
                continue  # 跳过 _common.sh 等仅被 source 的脚本
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            result.append(
                {
                    "path": rel,
                    "name": p.name,
                    "kind": "shell" if p.suffix == ".sh" else "python",
                    "description": _extract_script_doc(p),
                    "size": p.stat().st_size,
                }
            )
    return result


# ---------------------------------------------------------------------------
# Config 扫描
# ---------------------------------------------------------------------------
def list_configs() -> list[dict[str, Any]]:
    if not CONFIG_DIR.exists():
        return []
    return [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(CONFIG_DIR.glob("*.yml"))
    ]


def read_config(name: str) -> str:
    if "/" in name or "\\" in name:
        raise HTTPException(400, "invalid config name")
    target = _safe_under(CONFIG_DIR, CONFIG_DIR / name)
    if not target.exists():
        raise HTTPException(404, "config not found")
    return target.read_text(encoding="utf-8")


def write_config(name: str, content: str) -> None:
    if "/" in name or "\\" in name or not name.endswith(".yml"):
        raise HTTPException(400, "invalid config name")
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"invalid yaml: {e}")
    target = _safe_under(CONFIG_DIR, CONFIG_DIR / name)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Run 管理
# ---------------------------------------------------------------------------
class LaunchRequest(BaseModel):
    script: str = Field(..., description="相对项目根的脚本路径")
    args: list[str] = Field(default_factory=list, description="附加参数")
    env: dict[str, str] = Field(default_factory=dict, description="附加环境变量")


def launch_script(req: LaunchRequest) -> dict[str, Any]:
    script_rel = req.script.lstrip("/")
    script_path = _safe_under(PROJECT_ROOT, PROJECT_ROOT / script_rel)
    if not script_path.exists() or not script_path.is_file():
        raise HTTPException(404, f"script not found: {req.script}")

    run_id = uuid.uuid4().hex[:12]
    log_path = LOGS_DIR / f"{run_id}.log"

    if script_path.suffix == ".sh":
        cmd = ["bash", str(script_path), *req.args]
    elif script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path), *req.args]
    else:
        raise HTTPException(400, "unsupported script type")

    env = os.environ.copy()
    for k, v in req.env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            raise HTTPException(400, f"invalid env var name: {k}")
        env[k] = str(v)

    log_f = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,  # 独立进程组，便于整组 kill
    )

    run = {
        "id": run_id,
        "script": script_rel,
        "args": req.args,
        "env": req.env,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "pid": proc.pid,
        "pgid": os.getpgid(proc.pid),
        "log": str(log_path.relative_to(DASHBOARD_DIR)),
        "started_at": time.time(),
        "ended_at": None,
        "status": "running",
    }
    runs = _load_runs()
    runs.insert(0, run)
    _save_runs(runs[:200])  # 最多保留 200 条历史
    return run


def stop_run(run_id: str) -> dict[str, Any]:
    runs = _load_runs()
    target = next((r for r in runs if r["id"] == run_id), None)
    if target is None:
        raise HTTPException(404, "run not found")

    pid = target.get("pid")
    pgid = target.get("pgid")
    killed = False
    if pid and _pid_alive(pid):
        try:
            if pgid:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            pass
        except PermissionError as e:
            raise HTTPException(500, f"kill failed: {e}")

        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.25)
        if _pid_alive(pid):
            try:
                if pgid:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    target["status"] = "killed" if killed else "exited"
    target["ended_at"] = time.time()
    _save_runs(runs)
    return target


def tail_log(run_id: str, tail: int = 500) -> str:
    runs = _load_runs()
    target = next((r for r in runs if r["id"] == run_id), None)
    if target is None:
        raise HTTPException(404, "run not found")
    log_path = DASHBOARD_DIR / target["log"]
    if not log_path.exists():
        return ""
    with log_path.open("rb") as f:
        try:
            f.seek(0, 2)
            size = f.tell()
            block = 64 * 1024
            data = b""
            while size > 0 and data.count(b"\n") <= tail:
                read_size = min(block, size)
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
        except OSError:
            f.seek(0)
            data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-tail:])


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="FedCSL Dashboard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "project_root": str(PROJECT_ROOT)}


@app.get("/api/gpus")
def api_gpus() -> dict[str, Any]:
    return gpu_snapshot()


@app.get("/api/scripts")
def api_scripts() -> list[dict[str, Any]]:
    return list_scripts()


@app.get("/api/configs")
def api_configs() -> list[dict[str, Any]]:
    return list_configs()


@app.get("/api/configs/{name}", response_class=PlainTextResponse)
def api_config_get(name: str) -> str:
    return read_config(name)


class ConfigPut(BaseModel):
    content: str


@app.put("/api/configs/{name}")
def api_config_put(name: str, body: ConfigPut) -> dict[str, Any]:
    write_config(name, body.content)
    return {"ok": True}


@app.get("/api/runs")
def api_runs_list() -> list[dict[str, Any]]:
    runs = _load_runs()
    for r in runs:
        _refresh_run_status(r)
    _save_runs(runs)
    return runs


@app.post("/api/runs")
def api_runs_launch(req: LaunchRequest) -> dict[str, Any]:
    return launch_script(req)


@app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
def api_run_log(run_id: str, tail: int = 500) -> str:
    return tail_log(run_id, tail=tail)


@app.post("/api/runs/{run_id}/stop")
def api_run_stop(run_id: str) -> dict[str, Any]:
    return stop_run(run_id)


# 静态前端
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(404, "index.html missing")
    return FileResponse(str(idx))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="FedCSL Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "dashboard.app:app" if __package__ else "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
