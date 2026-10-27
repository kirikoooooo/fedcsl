"""Stage wrapper: run one pipeline phase, stream logs, and persist its status.

Usage:
    python -m observability.stage --run-dir <dir> --phase pretrain -- \\
        python -u FedCSL_All.py ...

The wrapper is deliberately CLI-based so the same command works locally and as
an Argo Workflows step. It writes ``<run-dir>/stages/<phase>.json`` and
``<run-dir>/logs/<phase>.log`` which the exporter consumes.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import (
    atomic_write_json,
    read_json,
    relpath,
    utc_now_iso,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run and record one FedCSL pipeline stage")
    parser.add_argument("--run-dir", required=True, help="Directory holding run metadata")
    parser.add_argument("--phase", required=True, help="prepare | pretrain | finetune | evaluate")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.stderr.write("no command supplied; use -- <command...>\n")
        return 2

    run_dir = Path(args.run_dir).resolve()
    (run_dir / "stages").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    stage_path = run_dir / "stages" / ("%s.json" % args.phase)
    log_path = run_dir / "logs" / ("%s.log" % args.phase)
    run_path = run_dir / "run.json"

    started_at = utc_now_iso()
    started_ts = time.time()
    stage = {
        "run_id": run_dir.name,
        "phase": args.phase,
        "status": "running",
        "started_at": started_at,
        "command": command,
        "command_display": subprocess.list2cmdline(command),
        "log": relpath(log_path),
    }
    atomic_write_json(stage_path, stage)

    meta = read_json(run_path, {}) or {}
    meta.setdefault("run_id", run_dir.name)
    meta.setdefault("created_at", started_at)
    meta["updated_at"] = started_at
    meta.setdefault("status", "running")
    meta.setdefault("stages", {})[args.phase] = {
        "status": "running",
        "started_at": started_at,
    }
    atomic_write_json(run_path, meta)

    env = os.environ.copy()
    env["FEDCSL_RUN_DIR"] = str(run_dir)
    env["FEDCSL_PHASE"] = args.phase
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    exit_code = 127
    try:
        with open(log_path, "a", encoding="utf-8") as log_handle:
            header = (
                "\n===== %s | %s =====\ncommand: %s\n"
                % (args.phase, started_at, subprocess.list2cmdline(command))
            )
            log_handle.write(header)
            log_handle.flush()
            sys.stdout.write(header)
            sys.stdout.flush()

            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_handle.write(line)
                    log_handle.flush()
                exit_code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    exit_code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    exit_code = 130
                raise
    except KeyboardInterrupt:
        exit_code = exit_code or 130
    except FileNotFoundError as exc:
        message = "command not found: %s\n" % exc
        sys.stderr.write(message)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(message)
        exit_code = 127
    except Exception as exc:  # keep the stage record even for wrapper failures
        message = "stage wrapper failed: %s: %s\n" % (type(exc).__name__, exc)
        sys.stderr.write(message)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(message)
        exit_code = 1
    finally:
        ended_at = utc_now_iso()
        duration = round(time.time() - started_ts, 3)
        status = "success" if exit_code == 0 else "failed"
        stage.update(
            {
                "status": status,
                "ended_at": ended_at,
                "duration_s": duration,
                "exit_code": int(exit_code),
            }
        )
        atomic_write_json(stage_path, stage)

        meta = read_json(run_path, {}) or {}
        meta.setdefault("run_id", run_dir.name)
        meta["updated_at"] = ended_at
        meta.setdefault("stages", {})[args.phase] = {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration,
            "exit_code": int(exit_code),
        }
        if status == "failed":
            meta["status"] = "failed"
        elif args.phase == "evaluate":
            meta["status"] = "succeeded"
            meta["ended_at"] = ended_at
        else:
            meta.setdefault("status", "running")
        atomic_write_json(run_path, meta)

        sys.stdout.write(
            "[stage] %s %s in %.1fs (exit=%s)\n"
            % (args.phase, status, duration, exit_code)
        )
        sys.stdout.flush()

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
