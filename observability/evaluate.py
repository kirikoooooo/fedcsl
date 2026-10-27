"""Evaluation gate: parse result logs into a machine-readable summary.

This is the final Argo DAG step. It does not re-run the model; it validates
that pretraining produced round metrics and that the fine-tuning block exists
when requested, then writes ``eval_summary.json`` for Grafana/Prometheus.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import REPO_ROOT, atomic_write_json, float_or_none, relpath

METRIC_RE = re.compile(
    r"round:(\d+)\s+server aggregation\s+testACC:(\S+)\s+"
    r"trainACC:(\S+)\s+avg_loss:(\S+)"
)
TIMING_RE = re.compile(
    r"\[round (\d+)\] timing train=([\d.]+)s distribution=([\d.]+)s "
    r"agg=([\d.]+)s eval=([\d.]+)s total=([\d.]+)s"
)
FT_LINE_RE = re.compile(
    r"^\s*([\d.]+)%\s*\|\s*(.*?)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*$"
)
FT_HEADER = "downstream fine-tuning ablation"
FT_ERROR_MARKERS = ("微调消融失败", "[ft-ablation] 微调消融失败", "Traceback (most recent call last)")


def find_result_log(dataset, description):
    folder = REPO_ROOT / "result" / dataset
    if not folder.is_dir():
        return None
    candidates = [
        path
        for path in folder.glob("*.txt")
        if description and description in path.name
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_pretrain_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    argv = ""
    for line in lines:
        if line.startswith("argv:"):
            argv = line[len("argv:"):].strip()
            break

    rounds = {}
    for line in lines:
        match = METRIC_RE.search(line)
        if not match:
            continue
        round_idx = int(match.group(1))
        entry = rounds.setdefault(round_idx, {"round": round_idx})
        entry["test_acc"] = float_or_none(match.group(2))
        entry["train_acc"] = float_or_none(match.group(3))
        entry["avg_loss"] = float_or_none(match.group(4))
        lower = line.lower()
        entry["eval_skipped"] = "eval_skipped" in lower

    for line in lines:
        match = TIMING_RE.search(line)
        if not match:
            continue
        round_idx = int(match.group(1))
        rounds.setdefault(round_idx, {"round": round_idx})["timings"] = {
            "train": float(match.group(2)),
            "distribution": float(match.group(3)),
            "agg": float(match.group(4)),
            "eval": float(match.group(5)),
            "total": float(match.group(6)),
        }

    ordered = [rounds[key] for key in sorted(rounds)]
    evaluated = [entry for entry in ordered if entry.get("test_acc") is not None]
    best = max(evaluated, key=lambda entry: entry["test_acc"]) if evaluated else None
    final = evaluated[-1] if evaluated else None
    return {
        "result_log": relpath(path),
        "argv": argv,
        "rounds": ordered,
        "rounds_total": len(ordered),
        "best_round": best.get("round") if best else None,
        "best_test_accuracy": best.get("test_acc") if best else None,
        "final_round": final.get("round") if final else None,
        "final_test_accuracy": final.get("test_acc") if final else None,
        "final_train_accuracy": final.get("train_acc") if final else None,
        "final_avg_loss": final.get("avg_loss") if final else None,
    }


def parse_finetune_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    has_header = any(FT_HEADER in line for line in lines)
    errors = [line for line in lines if any(marker in line for marker in FT_ERROR_MARKERS)]
    ratios = {}
    for line in lines:
        match = FT_LINE_RE.match(line)
        if not match:
            continue
        ratio_pct = float(match.group(1)) / 100.0
        seeds_text = match.group(2)
        accs = []
        for part in seeds_text.split(","):
            value = float_or_none(part.strip())
            if value is not None:
                accs.append(value)
        ratios["%.4f" % ratio_pct] = {
            "ratio": ratio_pct,
            "per_seed_accuracy": accs,
            "mean_accuracy": float(match.group(3)),
            "std_accuracy": float(match.group(4)),
        }
    return {
        "result_log": relpath(path),
        "has_finetune_block": has_header,
        "errors": errors[-5:],
        "ratios": ratios,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate FedCSL results and write a summary")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrain-desc", required=True)
    parser.add_argument("--finetune-desc", default=None)
    parser.add_argument("--expect-ft", action="store_true")
    parser.add_argument("--final-ckpt", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "eval_summary.json"

    checks = []

    def add_check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    summary = {
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "checks": checks,
    }

    pretrain_path = find_result_log(args.dataset, args.pretrain_desc)
    if pretrain_path is None:
        add_check("pretrain_result_exists", False, "no log for %r" % args.pretrain_desc)
        summary["status"] = "failed"
        atomic_write_json(summary_path, summary)
        sys.stderr.write("[evaluate] FAILED: pretrain result log not found\n")
        return 2

    pretrain = parse_pretrain_log(pretrain_path)
    summary["pretrain"] = pretrain
    add_check(
        "pretrain_has_round_metrics",
        pretrain["rounds_total"] >= 1,
        "%d round(s) parsed" % pretrain["rounds_total"],
    )
    add_check(
        "pretrain_has_test_accuracy",
        pretrain["best_test_accuracy"] is not None,
        "best test accuracy=%s" % pretrain["best_test_accuracy"],
    )

    if args.expect_ft:
        ft_path = find_result_log(args.dataset, args.finetune_desc or "")
        if ft_path is None:
            add_check("finetune_result_exists", False, "no log for %r" % args.finetune_desc)
            summary["finetune"] = None
        else:
            finetune = parse_finetune_log(ft_path)
            summary["finetune"] = finetune
            add_check(
                "finetune_block_present",
                finetune["has_finetune_block"],
                relpath(ft_path),
            )
            add_check(
                "finetune_has_ratios",
                bool(finetune["ratios"]),
                "ratios=%s" % ",".join(sorted(finetune["ratios"].keys())),
            )
            add_check(
                "finetune_no_errors",
                not finetune["errors"],
                "errors=%d" % len(finetune["errors"]),
            )

    if args.final_ckpt:
        ckpt = Path(args.final_ckpt)
        add_check(
            "final_checkpoint_exists",
            ckpt.is_file() and ckpt.stat().st_size > 0,
            "%s%s"
            % (
                relpath(ckpt),
                " (%d bytes)" % ckpt.stat().st_size if ckpt.is_file() else "",
            ),
        )

    passed = all(check["passed"] for check in checks)
    summary["status"] = "passed" if passed else "failed"
    atomic_write_json(summary_path, summary)

    print("[evaluate] summary: %s" % summary_path, flush=True)
    for check in checks:
        print(
            "[evaluate] %s %s - %s"
            % ("PASS" if check["passed"] else "FAIL", check["name"], check["detail"]),
            flush=True,
        )
    if pretrain.get("best_test_accuracy") is not None:
        print(
            "[evaluate] best round=%s test_acc=%.4f"
            % (pretrain.get("best_round"), pretrain["best_test_accuracy"]),
            flush=True,
        )
    if summary.get("finetune"):
        for ratio, item in sorted(summary["finetune"]["ratios"].items()):
            print(
                "[evaluate] finetune ratio=%.0f%% mean=%.4f std=%.4f"
                % (item["ratio"] * 100, item["mean_accuracy"], item["std_accuracy"]),
                flush=True,
            )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
