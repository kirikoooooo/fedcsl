#!/usr/bin/env python3
"""Benchmark 8 parallel single-scale trainings vs one full-scale training.

Target question:
    On one GPU, is running N single-scale models in N processes faster than
    training one model that contains all N scales?

The benchmark excludes process startup from the main timing by warming up each
worker first, then releasing all workers with a multiprocessing Event.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
from queue import Empty

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F


def _make_lengths(seq_len: int, num_scales: int, min_ratio: float, max_ratio: float) -> list[int]:
    if num_scales <= 1:
        return [max(2, int(round(seq_len * min_ratio)))]
    raw = [
        max(2, int(round(seq_len * (min_ratio + i * (max_ratio - min_ratio) / (num_scales - 1)))))
        for i in range(num_scales)
    ]
    lengths = []
    last = 1
    for value in raw:
        value = max(value, last + 1)
        value = min(value, seq_len)
        lengths.append(value)
        last = value
    return lengths


def _make_model(args, shapelets_size_and_len: OrderedDict[int, int], device: torch.device):
    try:
        from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise
        # Some lightweight Python environments lack sklearn, while blocks.py
        # imports utils.py only for two helpers. Provide benchmark-only fallbacks.
        sys.modules.pop("blocks", None)
        fallback_utils = types.ModuleType("utils")

        def _compute_gap_scores(x):
            return torch.zeros(x.shape[-1], device=x.device), torch.tensor(0.0, device=x.device)

        def _generate_binomial_mask(shape, device=None):
            return torch.ones(shape, dtype=torch.bool, device=device)

        fallback_utils.compute_gap_scores = _compute_gap_scores
        fallback_utils.generate_binomial_mask = _generate_binomial_mask
        sys.modules["utils"] = fallback_utils
        from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances

    model_cls = LearningShapeletsModelMixDistances if args.dist_measure == "mix" else LearningShapeletsModel
    return model_cls(
        shapelets_size_and_len=shapelets_size_and_len,
        in_channels=args.channels,
        num_classes=args.num_classes,
        dist_measure=args.dist_measure,
        to_cuda=device.type == "cuda",
        device=device,
    ).to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _train_step(model, optimizer, x_q, x_k, temperature: float):
    q = model(x_q, optimize=None, masking=False)
    k = model(x_k, optimize=None, masking=False)
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = torch.einsum("nc,ck->nk", [q, k.t()]) / temperature
    labels = torch.arange(q.shape[0], dtype=torch.long, device=q.device)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def _run_steps(model, optimizer, x_q, x_k, *, steps: int, temperature: float, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        _train_step(model, optimizer, x_q, x_k, temperature)
    _sync(device)
    return time.perf_counter() - start


def _single_scale_worker(rank: int, args_dict: dict, start_event, ready_queue, result_queue) -> None:
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(int(args.seed) + rank)
    lengths = _make_lengths(args.seq_len, args.num_scales, args.min_ratio, args.max_ratio)
    scale_len = int(lengths[rank % len(lengths)])
    shapelets = OrderedDict([(scale_len, args.shapelets_per_scale)])
    model = _make_model(args, shapelets, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    x_q = torch.randn(args.batch_size, args.channels, args.seq_len, device=device)
    x_k = torch.randn(args.batch_size, args.channels, args.seq_len, device=device)

    for _ in range(args.warmup):
        _train_step(model, optimizer, x_q, x_k, args.temperature)
    _sync(device)

    ready_queue.put({"rank": rank, "scale_len": scale_len})
    start_event.wait()

    elapsed = _run_steps(
        model,
        optimizer,
        x_q,
        x_k,
        steps=args.steps,
        temperature=args.temperature,
        device=device,
    )
    result_queue.put(
        {
            "rank": rank,
            "scale_len": scale_len,
            "elapsed_sec": elapsed,
            "ms_per_step": elapsed * 1000.0 / float(args.steps),
        }
    )


def _benchmark_full(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    lengths = _make_lengths(args.seq_len, args.num_scales, args.min_ratio, args.max_ratio)
    shapelets = OrderedDict((int(length), args.shapelets_per_scale) for length in lengths)
    model = _make_model(args, shapelets, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    x_q = torch.randn(args.batch_size, args.channels, args.seq_len, device=device)
    x_k = torch.randn(args.batch_size, args.channels, args.seq_len, device=device)

    for _ in range(args.warmup):
        _train_step(model, optimizer, x_q, x_k, args.temperature)
    elapsed = _run_steps(
        model,
        optimizer,
        x_q,
        x_k,
        steps=args.steps,
        temperature=args.temperature,
        device=device,
    )
    return {
        "shapelets": dict(shapelets),
        "elapsed_sec": elapsed,
        "ms_per_step": elapsed * 1000.0 / float(args.steps),
    }


def _benchmark_parallel(args) -> dict:
    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    args_dict = vars(args).copy()

    processes = []
    launch_start = time.perf_counter()
    for rank in range(args.num_scales):
        proc = ctx.Process(
            target=_single_scale_worker,
            args=(rank, args_dict, start_event, ready_queue, result_queue),
        )
        proc.start()
        processes.append(proc)

    ready = []
    while len(ready) < args.num_scales:
        ready.append(ready_queue.get(timeout=args.startup_timeout))

    launch_and_warmup_sec = time.perf_counter() - launch_start
    measured_start = time.perf_counter()
    start_event.set()

    worker_results = []
    while len(worker_results) < args.num_scales:
        worker_results.append(result_queue.get(timeout=args.run_timeout))
    measured_wall_sec = time.perf_counter() - measured_start

    for proc in processes:
        proc.join(timeout=10)
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"worker pid={proc.pid} exited with code {proc.exitcode}")

    worker_results = sorted(worker_results, key=lambda item: item["rank"])
    return {
        "ready": sorted(ready, key=lambda item: item["rank"]),
        "workers": worker_results,
        "launch_and_warmup_sec": launch_and_warmup_sec,
        "measured_wall_sec": measured_wall_sec,
        "wall_ms_per_step": measured_wall_sec * 1000.0 / float(args.steps),
        "max_worker_ms_per_step": max(item["ms_per_step"] for item in worker_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="cuda:0 is recommended; cpu is allowed for debugging")
    parser.add_argument("--num-scales", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=9)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shapelets-per-scale", type=int, default=40)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--dist-measure", default="mix", choices=["mix", "euclidean", "cosine", "cross-correlation"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--min-ratio", type=float, default=0.1)
    parser.add_argument("--max-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=300.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    full = _benchmark_full(args, device)
    parallel = _benchmark_parallel(args)
    speedup = full["ms_per_step"] / parallel["wall_ms_per_step"]

    report = {
        "pid": os.getpid(),
        "device": args.device,
        "num_scales": args.num_scales,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "seq_len": args.seq_len,
        "shapelets_per_scale": args.shapelets_per_scale,
        "dist_measure": args.dist_measure,
        "steps": args.steps,
        "warmup": args.warmup,
        "full_model": full,
        "parallel_single_scale_models": parallel,
        "speedup_full_ms_over_parallel_wall_ms": speedup,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nSummary:")
    print(f"full model: {full['ms_per_step']:.3f} ms/step")
    print(f"{args.num_scales} parallel single-scale processes: {parallel['wall_ms_per_step']:.3f} ms/step wall-clock")
    print(f"speedup = {speedup:.2f}x")
    print(f"startup + warmup excluded from speedup, but measured separately: {parallel['launch_and_warmup_sec']:.3f}s")


if __name__ == "__main__":
    main()
