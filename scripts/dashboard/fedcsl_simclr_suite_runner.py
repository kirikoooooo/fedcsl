#!/usr/bin/env python3
"""Serial dashboard runner for FedCSL-SimCLR variant screening.

Features
--------
- Runs dozens of candidate plans sequentially.
- Stops a plan early if the first round/first global evaluation accuracy is below a threshold.
- Persists a markdown run history next to this script for later tuning.

Notes
-----
Current project logs stable round-level global evaluations rather than per-local-epoch
accuracies. Therefore the "first epoch" gate is implemented as "first global evaluation
after round 0". This works uniformly for FedCSL / baseline_runner / ssl_runner paths.
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
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = Path(__file__).resolve().parent
DEFAULT_PLANS = SUITE_DIR / "fedcsl_simclr_suite_plans.yml"
DEFAULT_HISTORY = SUITE_DIR / "RUN_HISTORY.md"
DEFAULT_LOG_DIR = SUITE_DIR / "logs"

ROUND_RE = re.compile(r"dataset:\s*(?P<dataset>.*?)round:(?P<round>\d+).*?testACC:(?P<test>[-+0-9.eE]+)")


@dataclass
class Plan:
    name: str
    config: str
    description: str
    dirichlet_alpha: float
    dataset: str
    seed: int
    num_round: Optional[int]
    extra_args: List[str]
    env: Dict[str, str]
    config_overrides: Dict[str, Any]
    gate_acc: Optional[float]
    notes: str


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid suite yaml: {path}")
    return data


def _load_plans(
    path: Path,
    cli_gate: Optional[float],
    dataset_override: Optional[str],
) -> tuple[Dict[str, Any], List[Plan]]:
    raw = _load_yaml(path)
    defaults = raw.get("defaults", {}) or {}
    raw_plans = raw.get("plans", []) or []
    if not isinstance(raw_plans, list):
        raise ValueError("plans must be a list")

    plans: List[Plan] = []
    for item in raw_plans:
        if not isinstance(item, dict):
            continue
        enabled = bool(item.get("enabled", True))
        if not enabled:
            continue
        name = str(item.get("name", "")).strip()
        config = str(item.get("config", "")).strip()
        if not name or not config:
            continue
        gate_acc = cli_gate if cli_gate is not None else item.get("gate_acc", defaults.get("gate_acc"))
        effective_dataset = str(
            dataset_override
            if dataset_override is not None
            else item.get("dataset", defaults.get("dataset", "HAR"))
        )
        plans.append(
            Plan(
                name=name,
                config=config,
                description=str(item.get("description", "")),
                dirichlet_alpha=float(item.get("dirichlet_alpha", defaults.get("dirichlet_alpha", 0.1))),
                dataset=effective_dataset,
                seed=int(item.get("seed", defaults.get("seed", 42))),
                num_round=int(item["num_round"]) if item.get("num_round") is not None else (
                    int(defaults["num_round"]) if defaults.get("num_round") is not None else None
                ),
                extra_args=[str(x) for x in (item.get("extra_args", []) or [])],
                env={str(k): str(v) for k, v in (item.get("env", {}) or {}).items()},
                config_overrides=dict(item.get("config_overrides", {}) or {}),
                gate_acc=float(gate_acc) if gate_acc is not None else None,
                notes=str(item.get("notes", "")),
            )
        )
    return defaults, plans


def _append_history(history_path: Path, text: str) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        history_path.write_text(
            "# FedCSL-SimCLR Suite History\n\n"
            "This file is appended automatically by the suite runner.\n\n",
            encoding="utf-8",
        )
    with history_path.open("a", encoding="utf-8") as f:
        f.write(text)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _parse_round_result(line: str) -> Optional[Dict[str, Any]]:
    m = ROUND_RE.search(line)
    if not m:
        return None
    try:
        return {
            "round": int(m.group("round")),
            "test_acc": float(m.group("test")),
            "raw": line.rstrip(),
        }
    except Exception:
        return None


def _build_command(plan: Plan) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "FedCSL_All.py",
        "-dataset",
        plan.dataset,
        "--config",
        plan.config,
        "--dirichlet-alpha",
        str(plan.dirichlet_alpha),
        "--seed",
        str(plan.seed),
        "--description",
        plan.name,
    ]
    if plan.num_round is not None:
        cmd += ["--config", plan.config]
    cmd += plan.extra_args
    return cmd


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _materialize_config(
    src_config: Path,
    *,
    num_round: Optional[int],
    config_overrides: Dict[str, Any],
    cache_dir: Path,
    plan_name: str,
) -> str:
    if num_round is None and not config_overrides:
        return str(src_config.relative_to(PROJECT_ROOT))
    data = _load_yaml(src_config)
    if num_round is not None:
        data.setdefault("federated", {})
        data["federated"]["numRound"] = int(num_round)
    if config_overrides:
        _deep_update(data, config_overrides)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", plan_name)
    out_path = cache_dir / f"{safe}.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return str(out_path.relative_to(PROJECT_ROOT))


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_plan(
    plan: Plan,
    *,
    gate_acc: Optional[float],
    log_dir: Path,
    history_path: Path,
    config_cache_dir: Path,
) -> Dict[str, Any]:
    src_cfg = (PROJECT_ROOT / plan.config).resolve()
    if not src_cfg.exists():
        raise FileNotFoundError(f"config not found: {plan.config}")

    effective_config = _materialize_config(
        src_cfg,
        num_round=plan.num_round,
        config_overrides=plan.config_overrides,
        cache_dir=config_cache_dir,
        plan_name=plan.name,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", plan.name)
    log_path = log_dir / f"{ts}_{safe_name}.log"

    cmd = [
        sys.executable,
        "-u",
        "FedCSL_All.py",
        "-dataset",
        plan.dataset,
        "--config",
        effective_config,
        "--dirichlet-alpha",
        str(plan.dirichlet_alpha),
        "--seed",
        str(plan.seed),
        "--description",
        plan.name,
    ] + plan.extra_args

    env = os.environ.copy()
    env.update(plan.env)

    started = time.time()
    first_round: Optional[Dict[str, Any]] = None
    status = "completed"
    reason = ""

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# started: {_now()}\n")
        log_file.write(f"# plan: {plan.name}\n")
        log_file.write(f"# config: {effective_config}\n")
        log_file.write(f"# cmd: {' '.join(shlex.quote(x) for x in cmd)}\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()

            parsed = _parse_round_result(line)
            if parsed is not None and first_round is None:
                first_round = parsed
                if gate_acc is not None and parsed["test_acc"] < gate_acc:
                    status = "early_stopped"
                    reason = (
                        f"first-round testACC={parsed['test_acc']:.4f} < gate={gate_acc:.4f}"
                    )
                    log_file.write(f"\n# early stop: {reason}\n")
                    log_file.flush()
                    _terminate_process_tree(proc)
                    break

        rc = proc.wait()

    elapsed = time.time() - started
    if status != "early_stopped":
        if rc == 0:
            status = "completed"
            reason = "finished normally"
        else:
            status = "failed"
            reason = f"exit code {rc}"

    result = {
        "name": plan.name,
        "config": effective_config,
        "dataset": plan.dataset,
        "dirichlet_alpha": plan.dirichlet_alpha,
        "gate_acc": gate_acc,
        "status": status,
        "reason": reason,
        "elapsed_sec": elapsed,
        "first_round": first_round,
        "log_path": _display_path(log_path),
    }

    history_block = [
        f"## {_now()} | {plan.name}",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- dataset: `{plan.dataset}`",
        f"- dirichlet_alpha: `{plan.dirichlet_alpha}`",
        f"- config: `{effective_config}`",
        f"- log: `{result['log_path']}`",
        f"- elapsed_sec: `{elapsed:.1f}`",
    ]
    if first_round is not None:
        history_block.append(
            f"- first_round: round={first_round['round']} testACC={first_round['test_acc']:.4f}"
        )
    if plan.notes:
        history_block.append(f"- notes: {plan.notes}")
    history_block.append("")
    history_block.append("```json")
    history_block.append(json.dumps(result, ensure_ascii=False, indent=2))
    history_block.append("```")
    history_block.append("")
    _append_history(history_path, "\n".join(history_block))

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="FedCSL-SimCLR suite runner")
    parser.add_argument("--plans", default=str(DEFAULT_PLANS), help="suite yaml path")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY), help="markdown history path")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="log dir")
    parser.add_argument("--dataset", default=None, help="override dataset for all enabled plans")
    parser.add_argument("--gate-acc", type=float, default=None, help="override first-round accuracy gate")
    parser.add_argument("--limit", type=int, default=None, help="run only first N enabled plans")
    args = parser.parse_args()

    plans_path = Path(args.plans).resolve()
    history_path = Path(args.history).resolve()
    log_dir = Path(args.log_dir).resolve()
    config_cache_dir = SUITE_DIR / ".generated_configs"

    dataset_override = args.dataset.strip() if isinstance(args.dataset, str) and args.dataset.strip() else None
    defaults, plans = _load_plans(plans_path, args.gate_acc, dataset_override)
    if args.limit is not None:
        plans = plans[: max(0, int(args.limit))]

    if not plans:
        print("No enabled plans found.")
        return 0

    suite_header = [
        f"## {_now()} | SUITE START",
        "",
        f"- plan_file: `{plans_path}`",
        f"- total_plans: `{len(plans)}`",
        f"- dataset_override: `{dataset_override}`",
        f"- default_gate_acc: `{defaults.get('gate_acc', args.gate_acc)}`",
        "",
    ]
    _append_history(history_path, "\n".join(suite_header))

    passed = 0
    failed = 0
    early_stopped = 0
    for idx, plan in enumerate(plans, start=1):
        gate_acc = plan.gate_acc
        print(
            f"[suite] ({idx}/{len(plans)}) start {plan.name} "
            f"config={plan.config} gate_acc={gate_acc}",
            flush=True,
        )
        try:
            result = _run_plan(
                plan,
                gate_acc=gate_acc,
                log_dir=log_dir,
                history_path=history_path,
                config_cache_dir=config_cache_dir,
            )
        except Exception as e:
            result = {
                "name": plan.name,
                "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
            }
            _append_history(
                history_path,
                "\n".join(
                    [
                        f"## {_now()} | {plan.name}",
                        "",
                        f"- status: `failed`",
                        f"- reason: {type(e).__name__}: {e}",
                        "",
                    ]
                ),
            )
        status = result["status"]
        if status == "completed":
            passed += 1
        elif status == "early_stopped":
            early_stopped += 1
        else:
            failed += 1

    summary = [
        f"## {_now()} | SUITE END",
        "",
        f"- completed: `{passed}`",
        f"- early_stopped: `{early_stopped}`",
        f"- failed: `{failed}`",
        "",
    ]
    _append_history(history_path, "\n".join(summary))
    print(
        f"[suite] done completed={passed} early_stopped={early_stopped} failed={failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
