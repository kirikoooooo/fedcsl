"""Per-round progress files consumed by the Prometheus exporter.

Training code calls :func:`write_progress` only when ``FEDCSL_RUN_DIR`` is set,
so normal research runs are unaffected.
"""

import json
import os
from pathlib import Path

from .common import atomic_write_json, epoch_now, json_safe, utc_now_iso


def _progress_name(phase):
    phase = (phase or "pretrain").strip().lower()
    if phase in ("pretrain", "fit", "train"):
        return "progress.json"
    return "progress_%s.json" % phase


def write_progress(run_dir, payload):
    """Atomically write the latest phase progress and append a JSONL history."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = dict(payload or {})
    phase = str(data.get("phase") or os.environ.get("FEDCSL_PHASE") or "pretrain")
    data["phase"] = phase
    data.setdefault("run_id", run_dir.name)
    data["updated_at"] = epoch_now()
    data["updated_at_iso"] = utc_now_iso()
    data = json_safe(data)

    target = run_dir / _progress_name(phase)
    atomic_write_json(target, data)
    with open(run_dir / "progress.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    return target


def read_progress(run_dir, phase="pretrain"):
    target = Path(run_dir) / _progress_name(phase)
    try:
        with open(target, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def iter_progress_files(run_dir):
    return sorted(Path(run_dir).glob("progress*.json"))
