"""Local executor for the same four-stage DAG used by Argo Workflows.

Stages: prepare -> pretrain -> finetune -> evaluate.

The Argo WorkflowTemplate (deploy/argo/workflowtemplate.yaml) runs the same
commands inside pods. This runner exists so the closed loop can be validated on
a workstation without a Kubernetes cluster.
"""

import argparse
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .common import (
    DEFAULT_RUNS_DIR,
    REPO_ROOT,
    atomic_write_json,
    read_json,
    relpath,
    utc_now_iso,
)


def _git(*args):
    try:
        return subprocess.check_output(
            ["git"] + list(args),
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _write_meta(run_dir, meta):
    meta["updated_at"] = utc_now_iso()
    atomic_write_json(run_dir / "run.json", meta)


def _run_stage(python, run_dir, phase, command, env):
    wrapper = [
        python,
        "-m",
        "observability.stage",
        "--run-dir",
        str(run_dir),
        "--phase",
        phase,
        "--",
    ] + command
    print("\n[pipeline] ===== %s =====" % phase, flush=True)
    print("[pipeline] %s" % subprocess.list2cmdline(wrapper), flush=True)
    started = time.time()
    completed = subprocess.run(wrapper, cwd=str(REPO_ROOT), env=env)
    stage = read_json(run_dir / "stages" / ("%s.json" % phase), {}) or {}
    print(
        "[pipeline] %s exit=%s duration=%.1fs"
        % (phase, completed.returncode, time.time() - started),
        flush=True,
    )
    return completed.returncode, stage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the FedCSL observability closed loop locally")
    parser.add_argument("--dataset", default="Epilepsy-TSTCC")
    parser.add_argument("--config", default="config/configACF.yml")
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--num-client", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ft-ratios", default="0.1,1.0")
    parser.add_argument("--ft-seeds", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or (
        "obs-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    )
    run_dir = DEFAULT_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pretrain_desc = "%s_pretrain" % run_id
    finetune_desc = "%s_finetune" % run_id
    ckpt_path = REPO_ROOT / "checkpoint" / args.dataset / ("%s_final_round.pt" % run_id)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    meta = {
        "run_id": run_id,
        "status": "running",
        "dataset": args.dataset,
        "config": args.config,
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "host": socket.gethostname(),
        "python": args.python,
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "parameters": {
            "num_rounds": args.num_rounds,
            "num_client": args.num_client,
            "dirichlet_alpha": args.alpha,
            "seed": args.seed,
            "ft_ratios": args.ft_ratios,
            "ft_seeds": args.ft_seeds,
        },
        "descriptions": {"pretrain": pretrain_desc, "finetune": finetune_desc},
        "artifacts": {
            "run_dir": relpath(run_dir),
            "final_checkpoint": relpath(ckpt_path),
        },
        "stages": {},
    }
    _write_meta(run_dir, meta)
    print("[pipeline] run_id=%s" % run_id, flush=True)
    print("[pipeline] run_dir=%s" % run_dir, flush=True)

    shared_train_args = [
        "-dataset",
        args.dataset,
        "--config",
        args.config,
        "--num-client",
        str(args.num_client),
        "--dirichlet-alpha",
        str(args.alpha),
        "--server-gpu",
        "0",
        "--client-gpus",
        "0",
        "--client-workers",
        "1",
        "--eval-protocol",
        "svm",
        "--eval-every-n-rounds",
        "1",
        "--seed",
        str(args.seed),
    ]

    if not args.skip_prepare:
        code, stage = _run_stage(
            args.python,
            run_dir,
            "prepare",
            [
                args.python,
                "-m",
                "observability.prepare",
                "--dataset",
                args.dataset,
                "--run-dir",
                str(run_dir),
            ],
            env,
        )
        meta["stages"]["prepare"] = stage
        _write_meta(run_dir, meta)
        if code != 0:
            meta["status"] = "failed"
            meta["ended_at"] = utc_now_iso()
            _write_meta(run_dir, meta)
            return code

    pretrain_cmd = [args.python, "-u", "FedCSL_All.py"] + shared_train_args + [
        "--num-rounds",
        str(args.num_rounds),
        "--description",
        pretrain_desc,
        "--save-final-checkpoint",
        "--final-checkpoint-path",
        str(ckpt_path),
    ]
    code, stage = _run_stage(args.python, run_dir, "pretrain", pretrain_cmd, env)
    meta["stages"]["pretrain"] = stage
    _write_meta(run_dir, meta)
    if code != 0:
        meta["status"] = "failed"
        meta["ended_at"] = utc_now_iso()
        _write_meta(run_dir, meta)
        return code
    if not ckpt_path.is_file():
        print("[pipeline] final checkpoint missing: %s" % ckpt_path, file=sys.stderr)
        meta["status"] = "failed"
        meta["ended_at"] = utc_now_iso()
        _write_meta(run_dir, meta)
        return 3

    finetune_cmd = [args.python, "-u", "FedCSL_All.py"] + shared_train_args + [
        "--num-rounds",
        "0",
        "--load-state",
        str(ckpt_path),
        "--ft-ratios",
        args.ft_ratios,
        "--ft-seeds",
        str(args.ft_seeds),
        "--description",
        finetune_desc,
    ]
    code, stage = _run_stage(args.python, run_dir, "finetune", finetune_cmd, env)
    meta["stages"]["finetune"] = stage
    _write_meta(run_dir, meta)
    if code != 0:
        meta["status"] = "failed"
        meta["ended_at"] = utc_now_iso()
        _write_meta(run_dir, meta)
        return code

    evaluate_cmd = [
        args.python,
        "-m",
        "observability.evaluate",
        "--run-dir",
        str(run_dir),
        "--dataset",
        args.dataset,
        "--pretrain-desc",
        pretrain_desc,
        "--finetune-desc",
        finetune_desc,
        "--expect-ft",
        "--final-ckpt",
        str(ckpt_path),
    ]
    code, stage = _run_stage(args.python, run_dir, "evaluate", evaluate_cmd, env)
    meta["stages"]["evaluate"] = stage
    meta["status"] = "succeeded" if code == 0 else "failed"
    meta["ended_at"] = utc_now_iso()
    _write_meta(run_dir, meta)

    summary = read_json(run_dir / "eval_summary.json", {}) or {}
    print("\n[pipeline] status=%s" % meta["status"], flush=True)
    print("[pipeline] summary=%s" % (run_dir / "eval_summary.json"), flush=True)
    if summary.get("status"):
        print("[pipeline] evaluation=%s" % summary["status"], flush=True)
    if (summary.get("pretrain") or {}).get("best_test_accuracy") is not None:
        print(
            "[pipeline] best round=%s test_acc=%.4f"
            % (
                summary["pretrain"]["best_round"],
                summary["pretrain"]["best_test_accuracy"],
            ),
            flush=True,
        )
    print("[pipeline] Prometheus http://localhost:9090 | Grafana http://localhost:3000", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
