"""Data-preparation / validation stage.

For Epilepsy and LSST the tensors are already materialised, so this stage
validates them and writes a reproducible dataset manifest. For HAR it builds
``HAR/{train,val,test}.pt`` from ``UCI HAR Dataset`` when the raw data exists.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import REPO_ROOT, atomic_write_json, epoch_now, json_safe, relpath


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _torch():
    import torch  # imported lazily: only dataset stages need it

    return torch


def _inspect_pt_split(path):
    torch = _torch()
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    samples = payload["samples"]
    labels = payload["labels"]
    counts = {}
    if hasattr(labels, "numel") and labels.numel() > 0:
        values, frequencies = torch.unique(labels, return_counts=True)
        counts = {
            str(int(value.item())): int(frequency.item())
            for value, frequency in zip(values, frequencies)
        }
    return {
        "file": relpath(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "samples_shape": list(samples.shape),
        "labels_shape": list(labels.shape),
        "samples_dtype": str(samples.dtype),
        "labels_dtype": str(labels.dtype),
        "class_counts": counts,
    }


def _inspect_ts_file(path):
    """Lightweight UEA .ts inspection (header + row/class counts)."""
    classes = []
    counts = {}
    rows = 0
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("@"):
                lower = line.lower()
                if lower.startswith("@classlabel"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1].lower() == "true":
                        classes = parts[2:]
                in_data = lower.startswith("@data")
                continue
            if in_data:
                rows += 1
                label = line.split(",")[-1].strip()
                counts[label] = counts.get(label, 0) + 1
    return {
        "file": relpath(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "declared_classes": classes,
        "class_counts": counts,
    }


def _validate_epilepsy():
    root = REPO_ROOT / "Epilepsy"
    splits = {}
    for split in ("train", "val", "test"):
        path = root / ("%s.pt" % split)
        if not path.is_file():
            raise FileNotFoundError("missing %s" % path)
        splits[split] = _inspect_pt_split(path)
    train_shape = splits["train"]["samples_shape"]
    return {
        "kind": "preprocessed_pt",
        "splits": splits,
        "num_classes": len(splits["train"]["class_counts"]),
        "channels": train_shape[1] if len(train_shape) > 1 else None,
        "series_length": train_shape[2] if len(train_shape) > 2 else None,
    }


def _validate_har(build):
    processed = REPO_ROOT / "HAR"
    raw = REPO_ROOT / "UCI HAR Dataset"
    paths = {split: processed / ("%s.pt" % split) for split in ("train", "val", "test")}
    if not all(path.is_file() for path in paths.values()):
        raw_ready = (raw / "train").is_dir() and (raw / "test").is_dir()
        if not build:
            raise FileNotFoundError(
                "HAR tensors missing under %s; supply 'UCI HAR Dataset/' and rerun "
                "with --build-har (or run 'python process_har.py')" % processed
            )
        if not raw_ready:
            raise FileNotFoundError(
                "HAR tensors missing and 'UCI HAR Dataset/' is empty; "
                "download UCI HAR (dataset 240) first"
            )
        subprocess.run(
            [sys.executable, "process_har.py"],
            cwd=str(REPO_ROOT),
            check=True,
        )
    splits = {}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("missing %s after HAR preparation" % path)
        splits[split] = _inspect_pt_split(path)
    train_shape = splits["train"]["samples_shape"]
    return {
        "kind": "preprocessed_pt",
        "splits": splits,
        "num_classes": len(splits["train"]["class_counts"]),
        "channels": train_shape[1] if len(train_shape) > 1 else None,
        "series_length": train_shape[2] if len(train_shape) > 2 else None,
    }


def _validate_lsst():
    root = REPO_ROOT / "Multivariate_ts" / "LSST"
    files = {
        "train": root / "LSST_TRAIN.ts",
        "test": root / "LSST_TEST.ts",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing LSST files: %s" % ", ".join(missing))
    splits = {name: _inspect_ts_file(path) for name, path in files.items()}
    return {
        "kind": "uea_ts",
        "splits": splits,
        "num_classes": len(splits["train"]["class_counts"]),
        "channels": None,
        "series_length": None,
    }


def validate_dataset(dataset, build_har=False):
    if dataset == "Epilepsy-TSTCC":
        return _validate_epilepsy()
    if dataset == "HAR":
        return _validate_har(build_har)
    if dataset == "LSST":
        return _validate_lsst()
    raise ValueError(
        "unsupported dataset %r; supported: Epilepsy-TSTCC, HAR, LSST" % dataset
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate/prepare a FedCSL dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--build-har", action="store_true", help="Run process_har.py if needed")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "dataset.json"

    manifest = {
        "dataset": args.dataset,
        "status": "running",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generated_at_epoch": epoch_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repo_root": relpath(REPO_ROOT),
    }
    try:
        details = validate_dataset(args.dataset, build_har=args.build_har)
        manifest.update(details)
        manifest["status"] = "passed"
        atomic_write_json(manifest_path, manifest)
        print(
            "[prepare] %s validated: %s classes, splits=%s"
            % (
                args.dataset,
                manifest.get("num_classes"),
                ",".join(sorted(manifest.get("splits", {}).keys())),
            ),
            flush=True,
        )
        print("[prepare] manifest: %s" % manifest_path, flush=True)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = "%s: %s" % (type(exc).__name__, exc)
        atomic_write_json(manifest_path, manifest)
        sys.stderr.write("[prepare] FAILED: %s\n" % manifest["error"])
        return 2


if __name__ == "__main__":
    sys.exit(main())
