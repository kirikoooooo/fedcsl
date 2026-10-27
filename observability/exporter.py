"""Dependency-free Prometheus exporter for FedCSL runs.

Reads ``observability/runs/<run_id>/`` (written by the stage wrapper and the
training hooks) and exposes:

    http://0.0.0.0:9108/metrics
    http://0.0.0.0:9108/healthz
    http://0.0.0.0:9108/api/runs

Only the standard library is used so the exporter can run inside the training
image without installing ``prometheus_client``.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .common import (
    DEFAULT_RUNS_DIR,
    epoch_now,
    parse_iso_epoch,
    read_json,
    relpath,
    utc_now_iso,
)

_GPU_CACHE = {"ts": 0.0, "rows": []}
_GPU_LOCK = threading.Lock()
_START_TS = time.time()


def _escape(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _fmt(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return repr(value)
    return str(value)


class MetricBuffer:
    def __init__(self):
        self._samples = {}
        self._types = {}

    def add(self, name, value, labels=None, mtype="gauge"):
        formatted = _fmt(value)
        if formatted is None:
            return
        self._types[name] = mtype
        label_text = ",".join(
            '%s="%s"' % (key, _escape(val))
            for key, val in sorted((labels or {}).items())
        )
        line = name
        if label_text:
            line += "{%s}" % label_text
        line += " " + formatted
        self._samples.setdefault(name, []).append(line)

    def render(self):
        chunks = []
        for name in sorted(self._samples):
            chunks.append("# TYPE %s %s" % (name, self._types.get(name, "gauge")))
            chunks.extend(self._samples[name])
        chunks.append("")
        return "\n".join(chunks)


def _gpu_rows():
    now = time.time()
    with _GPU_LOCK:
        if now - _GPU_CACHE["ts"] < 5.0:
            return _GPU_CACHE["rows"]
    rows = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    rows.append(
                        {
                            "index": int(float(parts[0])),
                            "memory_used_mb": float(parts[1]),
                            "memory_total_mb": float(parts[2]),
                            "utilization_pct": float(parts[3]),
                            "temperature_c": float(parts[4]) if len(parts) > 4 else None,
                        }
                    )
                except ValueError:
                    continue
    except (OSError, subprocess.SubprocessError):
        rows = []
    with _GPU_LOCK:
        _GPU_CACHE["ts"] = now
        _GPU_CACHE["rows"] = rows
    return rows


def _load_run(run_dir):
    meta = read_json(run_dir / "run.json", {}) or {}
    stages = {}
    stages_dir = run_dir / "stages"
    if stages_dir.is_dir():
        for path in stages_dir.glob("*.json"):
            stages[path.stem] = read_json(path, {}) or {}
    progresses = []
    for path in sorted(run_dir.glob("progress*.json")):
        data = read_json(path, {}) or {}
        if data:
            data.setdefault("phase", "pretrain" if path.stem == "progress" else path.stem[len("progress_"):])
            progresses.append(data)
    summary = read_json(run_dir / "eval_summary.json", {}) or {}
    dataset_manifest = read_json(run_dir / "dataset.json", {}) or {}

    status = "pending"
    if summary.get("status") == "passed":
        status = "succeeded"
    elif summary.get("status") == "failed":
        status = "failed"
    elif any(stage.get("status") == "failed" for stage in stages.values()):
        status = "failed"
    elif any(stage.get("status") == "running" for stage in stages.values()):
        status = "running"
    elif meta.get("status") in ("running", "succeeded", "failed"):
        status = meta["status"]

    return {
        "run_id": meta.get("run_id") or run_dir.name,
        "meta": meta,
        "stages": stages,
        "progresses": progresses,
        "summary": summary,
        "dataset_manifest": dataset_manifest,
        "dataset": (
            meta.get("dataset")
            or summary.get("dataset")
            or dataset_manifest.get("dataset")
            or (progresses[0].get("dataset") if progresses else "")
        ),
        "status": status,
        "path": run_dir,
    }


def collect_runs(root, max_age_hours):
    root = Path(root)
    if not root.is_dir():
        return []
    now = time.time()
    runs = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            run = _load_run(run_dir)
        except Exception:
            continue
        created = (
            parse_iso_epoch(run["meta"].get("created_at"))
            or parse_iso_epoch(run["meta"].get("started_at"))
            or run_dir.stat().st_mtime
        )
        run["created_ts"] = created
        if max_age_hours and now - created > max_age_hours * 3600:
            continue
        runs.append(run)
    return runs


def _metric_buffer(runs):
    buf = MetricBuffer()
    buf.add("fedcsl_exporter_up", 1)
    buf.add("fedcsl_exporter_scrape_timestamp_seconds", epoch_now())
    buf.add(
        "fedcsl_exporter_build_info",
        1,
        {"version": "0.1.0", "mode": "file-collector"},
    )

    status_counts = {}
    for run in runs:
        labels = {
            "run_id": run["run_id"],
            "dataset": run["dataset"] or "unknown",
            "status": run["status"],
        }
        buf.add("fedcsl_run_info", 1, labels)
        status_counts[run["status"]] = status_counts.get(run["status"], 0) + 1

        created = run["created_ts"]
        buf.add("fedcsl_run_start_timestamp_seconds", created, {"run_id": run["run_id"]})
        ended = (
            parse_iso_epoch(run["meta"].get("ended_at"))
            or parse_iso_epoch(run["summary"].get("generated_at"))
        )
        if ended:
            buf.add("fedcsl_run_end_timestamp_seconds", ended, {"run_id": run["run_id"]})

        for phase, stage in run["stages"].items():
            labels = {"run_id": run["run_id"], "phase": phase}
            if stage.get("status") in ("success", "failed"):
                buf.add(
                    "fedcsl_stage_status",
                    1 if stage.get("status") == "success" else 0,
                    labels,
                )
            if stage.get("duration_s") is not None:
                buf.add("fedcsl_stage_duration_seconds", stage["duration_s"], labels)

        for progress in run["progresses"]:
            phase = progress.get("phase") or "pretrain"
            labels = {"run_id": run["run_id"], "phase": phase}
            buf.add("fedcsl_round_current", progress.get("round"), labels)
            buf.add("fedcsl_round_total", progress.get("num_rounds"), labels)
            buf.add("fedcsl_round_test_accuracy", progress.get("test_acc"), labels)
            buf.add("fedcsl_round_train_accuracy", progress.get("train_acc"), labels)
            buf.add("fedcsl_round_best_accuracy", progress.get("best_acc"), labels)
            buf.add("fedcsl_round_avg_loss", progress.get("avg_loss"), labels)
            updated = progress.get("updated_at")
            if updated:
                buf.add("fedcsl_round_last_timestamp_seconds", updated, labels)
                # Freshness is only meaningful for active runs; completed runs
                # would otherwise look "stale" forever on the dashboard.
                if run["status"] == "running":
                    buf.add(
                        "fedcsl_run_progress_age_seconds",
                        max(0.0, epoch_now() - float(updated)),
                        {"run_id": run["run_id"], "phase": phase},
                    )
            for stage_name, seconds in (progress.get("timings") or {}).items():
                buf.add(
                    "fedcsl_round_stage_seconds",
                    seconds,
                    {"run_id": run["run_id"], "phase": phase, "stage": stage_name},
                )

        summary = run["summary"]
        if summary.get("status"):
            buf.add(
                "fedcsl_eval_passed",
                1 if summary["status"] == "passed" else 0,
                {"run_id": run["run_id"]},
            )
        finetune = summary.get("finetune") or {}
        for ratio_key, item in sorted((finetune.get("ratios") or {}).items()):
            ratio_label = "%.4f" % float(item.get("ratio", ratio_key))
            for seed_idx, accuracy in enumerate(item.get("per_seed_accuracy") or []):
                buf.add(
                    "fedcsl_finetune_test_accuracy",
                    accuracy,
                    {
                        "run_id": run["run_id"],
                        "ratio": ratio_label,
                        "seed": str(seed_idx),
                    },
                )
            buf.add(
                "fedcsl_finetune_mean_accuracy",
                item.get("mean_accuracy"),
                {"run_id": run["run_id"], "ratio": ratio_label},
            )
            buf.add(
                "fedcsl_finetune_std_accuracy",
                item.get("std_accuracy"),
                {"run_id": run["run_id"], "ratio": ratio_label},
            )

        artifacts = {
            "dataset_manifest": (run["path"] / "dataset.json").is_file(),
            "eval_summary": (run["path"] / "eval_summary.json").is_file(),
            "pretrain_log": bool((summary.get("pretrain") or {}).get("result_log")),
            "finetune_log": bool((summary.get("finetune") or {}).get("result_log")),
        }
        for kind, present in artifacts.items():
            if present:
                buf.add(
                    "fedcsl_artifact_present",
                    1,
                    {"run_id": run["run_id"], "kind": kind},
                )

    for status, count in sorted(status_counts.items()):
        buf.add("fedcsl_runs", count, {"status": status})

    for gpu in _gpu_rows():
        labels = {"gpu": str(gpu["index"])}
        buf.add(
            "fedcsl_gpu_memory_used_bytes",
            gpu["memory_used_mb"] * 1024 * 1024,
            labels,
        )
        buf.add(
            "fedcsl_gpu_memory_total_bytes",
            gpu["memory_total_mb"] * 1024 * 1024,
            labels,
        )
        buf.add(
            "fedcsl_gpu_utilization_ratio",
            gpu["utilization_pct"] / 100.0,
            labels,
        )
        if gpu.get("temperature_c") is not None:
            buf.add(
                "fedcsl_gpu_temperature_celsius",
                gpu["temperature_c"],
                labels,
            )
    return buf


def render_metrics(root, max_age_hours):
    runs = collect_runs(root, max_age_hours)
    return _metric_buffer(runs).render(), runs


class Handler(BaseHTTPRequestHandler):
    server_version = "fedcsl-exporter/0.1"

    def _root(self):
        return self.server.runs_root

    def _max_age(self):
        return self.server.max_age_hours

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/metrics", "/metrics/"):
            text, _ = render_metrics(self._root(), self._max_age())
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/healthz", "/health"):
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/runs"):
            _, runs = render_metrics(self._root(), self._max_age())
            payload = []
            for run in runs:
                payload.append(
                    {
                        "run_id": run["run_id"],
                        "dataset": run["dataset"],
                        "status": run["status"],
                        "stages": {
                            phase: stage.get("status")
                            for phase, stage in run["stages"].items()
                        },
                        "progress": run["progresses"][-1] if run["progresses"] else None,
                        "summary_status": (run["summary"] or {}).get("status"),
                        "path": relpath(run["path"]),
                    }
                )
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"not found\n")

    def log_message(self, fmt, *args):
        return


def serve(root, port, max_age_hours):
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
    server.runs_root = str(root)
    server.max_age_hours = max_age_hours
    print(
        "[exporter] serving %s on http://0.0.0.0:%s/metrics (runs=%s)"
        % (utc_now_iso(), port, root),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[exporter] shutting down", flush=True)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prometheus exporter for FedCSL runs")
    parser.add_argument(
        "--root",
        default=os.environ.get("FEDCSL_RUNS_DIR", str(DEFAULT_RUNS_DIR)),
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FEDCSL_EXPORTER_PORT", "9108")),
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=float(os.environ.get("FEDCSL_EXPORTER_MAX_AGE_HOURS", "168")),
    )
    parser.add_argument("--once", action="store_true", help="Print metrics and exit")
    args = parser.parse_args(argv)

    if args.once:
        text, _ = render_metrics(args.root, args.max_age_hours)
        sys.stdout.write(text)
        return 0
    serve(args.root, args.port, args.max_age_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
