"""Small shared helpers for the observability runtime."""

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "observability" / "runs"


def utc_now_iso():
    """Local-time ISO-8601 timestamp with timezone offset."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def epoch_now():
    return time.time()


def json_safe(value):
    """Recursively convert values to JSON-friendly ones (NaN/Inf -> None)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy / torch scalar-like objects
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def atomic_write_json(path, payload):
    """Write JSON atomically so readers never observe a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def parse_iso_epoch(value):
    """Parse an ISO-8601 timestamp into epoch seconds; return None on failure."""
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def relpath(path):
    """Repository-relative path when possible (metrics stay portable)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def float_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
