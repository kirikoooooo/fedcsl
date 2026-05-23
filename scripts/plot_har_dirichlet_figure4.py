#!/usr/bin/env python3
"""Multi-panel HAR convergence figure: one subplot per Dirichlet alpha."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAR_DIR = PROJECT_ROOT / "data" / "HAR"
DEFAULT_OUT = PROJECT_ROOT.parent / "figs" / "har_dirichlet_convergence.png"

ROUND_RE = re.compile(r"round:(\d+).*?testACC:([\d.]+)")
SPILTER_TOPM_RE = re.compile(r"local_top_m:\s*(\d+)")

METHOD_STYLE = {
    "FedCSL": {"color": "#F18F01", "ls": "-", "lw": 1.8},
    "Spilter-m1": {"color": "#7EB8DA", "ls": "-", "lw": 1.7},
    "Spilter-m2": {"color": "#2E86AB", "ls": "-", "lw": 1.9},
    "Spilter-m4": {"color": "#1B4965", "ls": "-", "lw": 2.0},
    "Orchestra": {"color": "#A23B72", "ls": "--", "lw": 1.5},
    "FedBYOL": {"color": "#6A994E", "ls": "-.", "lw": 1.5},
    "FedU2": {"color": "#3D5A80", "ls": ":", "lw": 1.6},
    "FedAvg": {"color": "#BC4A3C", "ls": "-", "lw": 1.5},
    "FedProx": {"color": "#7B2D8E", "ls": "--", "lw": 1.5},
}

METHOD_ORDER = [
    "FedCSL",
    "Spilter-m1",
    "Spilter-m2",
    "Spilter-m4",
    "Orchestra",
    "FedBYOL",
    "FedU2",
    "FedAvg",
    "FedProx",
]


def _parse_spilter_local_top_m(path: Path) -> int | None:
    head = path.read_text(encoding="utf-8", errors="ignore")[:8000]
    m = SPILTER_TOPM_RE.search(head)
    return int(m.group(1)) if m else None


def _method_label(path: Path) -> str | None:
    name = path.name.lower()
    if "fedavg" in name:
        return "FedAvg"
    if "fedprox" in name:
        return "FedProx"
    if "fedcsl" in name and "onehot" not in name:
        return "FedCSL"
    if "spilter" in name:
        m = _parse_spilter_local_top_m(path)
        if m is None:
            return None
        return f"Spilter-m{m}"
    if "orchestra" in name:
        return "Orchestra"
    if "byol" in name:
        return "FedBYOL"
    if "fedu2" in name:
        return "FedU2"
    return None


def _file_sort_key(path: Path) -> tuple:
    m = re.match(r"(\d{4}-\d{2}-\d{2}-\d{2})", path.name)
    ts = m.group(1) if m else "0000-00-00-00"
    return (ts, path.stat().st_mtime, path.name)


def parse_log(path: Path) -> tuple[list[int], list[float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rounds: list[int] = []
    accs: list[float] = []
    for m in ROUND_RE.finditer(text):
        rounds.append(int(m.group(1)))
        accs.append(float(m.group(2)))
    if not rounds:
        raise ValueError(f"no testACC rows in {path}")
    return rounds, accs


def _alpha_from_dir(path: Path) -> float | None:
    m = re.match(r"alpha_([\d.]+)$", path.name)
    return float(m.group(1)) if m else None


def discover_alpha_dirs(har_dir: Path) -> list[tuple[float, Path]]:
    dirs = [(a, p) for p in har_dir.iterdir() if p.is_dir() and (a := _alpha_from_dir(p)) is not None]
    dirs.sort(key=lambda x: x[0])
    if dirs:
        return dirs
    # Legacy flat layout: treat entire har_dir as alpha=0.1 if txt files exist.
    if list(har_dir.glob("*.txt")):
        return [(0.1, har_dir)]
    return []


def collect_series(alpha_dir: Path) -> dict[str, tuple[list[int], list[float], Path]]:
    best: dict[str, Path] = {}
    for path in alpha_dir.glob("*.txt"):
        label = _method_label(path)
        if label is None:
            continue
        prev = best.get(label)
        if prev is None or _file_sort_key(path) > _file_sort_key(prev):
            best[label] = path

    series: dict[str, tuple[list[int], list[float], Path]] = {}
    for algo, path in best.items():
        rounds, accs = parse_log(path)
        series[algo] = (rounds, accs, path)
    return series


def _zoom_ylim(accs: list[float], min_span: float = 0.012, pad_ratio: float = 0.12) -> tuple[float, float]:
    """Tight y-limits per subplot so curve differences are visible."""
    ymin = min(accs)
    ymax = max(accs)
    span = ymax - ymin
    if span < min_span:
        mid = 0.5 * (ymin + ymax)
        ymin = mid - min_span / 2
        ymax = mid + min_span / 2
        span = min_span
    pad = max(span * pad_ratio, 0.002)
    return ymin - pad, ymax + pad


def plot_figure(alpha_series: list[tuple[float, dict]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )

    n = len(alpha_series)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.2 * nrows), sharex=False)
    axes_flat = axes.flatten() if n > 1 else [axes]

    handles, labels = [], []

    for idx, (alpha, series) in enumerate(alpha_series):
        ax = axes_flat[idx]
        panel_accs: list[float] = []

        for algo in METHOD_ORDER:
            if algo not in series:
                continue
            rounds, accs, _ = series[algo]
            panel_accs.extend(accs)
            style = METHOD_STYLE.get(algo, {"color": "gray", "ls": "-", "lw": 1.5})
            line, = ax.plot(rounds, accs, label=algo, **style)
            if algo not in labels:
                handles.append(line)
                labels.append(algo)

        if panel_accs:
            lo, hi = _zoom_ylim(panel_accs)
            ax.set_ylim(lo, hi)

        ax.set_title(f"$\\alpha={alpha:g}$")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.28, linewidth=0.6)
        ax.tick_params(labelsize=9)

        if idx >= (nrows - 1) * ncols or idx >= n - ncols:
            ax.set_xlabel("Round")
        if idx % ncols == 0:
            ax.set_ylabel("Test acc.")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("HAR test accuracy vs. communication round (Dirichlet non-IID)", y=1.01, fontsize=13)
    fig.legend(handles, labels, loc="lower center", ncol=4, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har-dir", type=Path, default=HAR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    alpha_dirs = discover_alpha_dirs(args.har_dir)
    if not alpha_dirs:
        raise SystemExit(f"no alpha_* folders under {args.har_dir}")

    alpha_series: list[tuple[float, dict]] = []
    for alpha, path in alpha_dirs:
        series = collect_series(path)
        if not series:
            print(f"[warn] skip alpha={alpha}: no logs in {path}")
            continue
        alpha_series.append((alpha, series))

    if not alpha_series:
        raise SystemExit("no parsable logs found")

    plot_figure(alpha_series, args.output)

    print(f"[OK] wrote {args.output}  ({len(alpha_series)} panels)")
    for alpha, series in alpha_series:
        print(f"  alpha={alpha:g}:")
        for algo in METHOD_ORDER:
            if algo not in series:
                continue
            rounds, accs, path = series[algo]
            last10 = sum(accs[-10:]) / min(10, len(accs))
            lo, hi = _zoom_ylim(accs)
            print(
                f"    {algo:10s} n={len(rounds):3d}  "
                f"range=[{min(accs):.4f},{max(accs):.4f}]  "
                f"ylim=[{lo:.4f},{hi:.4f}]  last10={last10:.4f}  "
                f"({path.name[:48]}...)"
            )


if __name__ == "__main__":
    main()
