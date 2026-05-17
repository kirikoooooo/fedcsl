"""Per-client epoch timing on HAR — system efficiency micro-benchmark.

对单个联邦算法，串行（一次一个客户端独占 GPU）测量"完成 1 个本地 epoch"所需
wall-clock 时间，取最慢客户端的耗时作为该算法的系统效率指标。

设计目标：
  * 单 GPU 独占：脚本本身串行；用户应通过 CUDA_VISIBLE_DEVICES=K 锁卡，
    并保证同一时刻只有一个本进程运行；
  * 复用现有客户端类，不重写训练逻辑，确保耗时代表真实训练；
  * 不做联邦聚合 / 评估 / 多 round，只测 1 epoch 的客户端 fit；
  * 写一个 partial JSON 文件供 aggregate_results.py 汇总。

支持的算法（与 HAR_results.md 对齐）：
  fedavg, fedprox, fedcsl, spilter-m1, spilter-m2, spilter-m4,
  byol, fedu2, orchestra
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from dataset_utils import LoadDataset_HAR  # noqa: E402


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _shapelet_dict(len_ts: int) -> Dict[int, int]:
    return {
        int(i): 40
        for i in np.linspace(
            min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int
        )
    }


def _build_csl_like_config(algo_name: str) -> Dict[str, Any]:
    ablation = {
        "UseJointKD": algo_name in ("fedcsl", "spilter"),
        "UseJointCL": algo_name in ("fedcsl", "spilter"),
        "UseScaleKD": algo_name in ("fedcsl", "spilter"),
        "UseScaleCL": algo_name in ("fedcsl", "spilter"),
        "UseProdictor": False,
        "UseMTLHomo": False,
        "UseACF": algo_name in ("fedcsl", "spilter"),
        "UseDistribution": False,
    }
    spilter_cfg: Dict[str, Any] = {}
    if algo_name == "spilter":
        spilter_cfg = {
            "allocation_mode": "local_score_topm",
            "selected_scale_training": "stitched",
        }
    return {
        "algo": algo_name,
        "model": {"params": {"momentum": 0.9, "gamma": 0.5, "beta": 0.4}},
        "ablation": ablation,
        "spilter": spilter_cfg,
        "moon": {"mu": 1.0, "temperature": 0.5},
    }


def _ssl_args(algo_name: str, batch_size: int, lr: float, wd: float, num_epoch: int):
    from algo.flbench_compat import AttrDict

    return AttrDict({
        "common": AttrDict({
            "batch_size": int(batch_size),
            "local_epoch": int(num_epoch),
            "use_cuda": True,
            "seed": 42,
            "reset_optimizer_on_global_epoch": True,
            "buffers": "global",
            "join_ratio": 1.0,
            "verbose_gap": 1,
        }),
        "optimizer": AttrDict({
            "name": "sgd",
            "lr": float(lr),
            "weight_decay": float(wd),
            "momentum": 0.9,
        }),
        "dataset": AttrDict({"name": "HAR"}),
        "ssl": AttrDict({
            "method": algo_name,
            "temperature": 0.2,
            "projector_hidden_dim": 256,
            "projector_out_dim": 128,
            "predictor_hidden_dim": 256,
            "ema_tau": 0.99,
            "num_global_clusters": 32,
            "num_local_clusters": 8,
            "cluster_m_size": 128,
            "deg_num_classes": 5,
            "server_cluster_rounds": 80,
            "fur_weight": 0.1,
            "fur_tau_a": 0.8,
            "fur_tau_b": 0.8,
            "fur_num_steps": 5,
            "server_lr": 0.1,
            "sharpen_ratio": 0.1,
        }),
    })


def _time_csl_client_one_epoch(
    *,
    algo_alias: str,
    X_client: np.ndarray,
    shapelets_size_and_len: Dict[int, int],
    in_channels: int,
    num_classes: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
    teacher_state: Optional[Dict[str, torch.Tensor]] = None,
    warmup_batches: int = 1,
) -> float:
    from train import LearningShapeletsCL

    if algo_alias.startswith("spilter-m"):
        m = int(algo_alias.split("m")[-1])
        algo_name = "spilter"
        selected_scales = list(range(min(m, len(shapelets_size_and_len))))
    else:
        algo_name = algo_alias
        selected_scales = None

    config = _build_csl_like_config(algo_name)
    loss_func = nn.CrossEntropyLoss()

    client = LearningShapeletsCL(
        shapelets_size_and_len=shapelets_size_and_len,
        loss_func=loss_func,
        in_channels=in_channels,
        num_classes=num_classes,
        dist_measure="mix",
        verbose=0,
        to_cuda=(device.type == "cuda"),
        l3=0.0,
        l4=0.0,
        T=0.1,
        alpha=0.0,
        beta=0.4,
        seed=42,
        configDir=None,
        config=config,
        device=device,
    )
    client.set_optimizer(
        torch.optim.SGD(
            client.model.parameters(), lr=float(lr), weight_decay=float(wd), momentum=0.9
        )
    )

    teacher = None
    if algo_name in ("fedcsl", "spilter", "fedprox") and teacher_state is not None:
        teacher = LearningShapeletsCL(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            in_channels=in_channels,
            num_classes=num_classes,
            dist_measure="mix",
            verbose=0,
            to_cuda=(device.type == "cuda"),
            config=config,
            device=device,
        )
        teacher.model.load_state_dict(teacher_state, strict=False)
        teacher.model.eval()
        for p in teacher.model.parameters():
            p.requires_grad_(False)
        client.Global_Model = teacher.model

    if selected_scales is not None:
        client.Selected_Scales = selected_scales

    if warmup_batches > 0 and X_client.shape[0] >= max(batch_size, 1):
        warm_X = X_client[: max(batch_size, 1) * warmup_batches]
        try:
            client.train(warm_X, epochs=1, batch_size=batch_size, epoch_idx=-1, lr=lr)
        except Exception as e:
            print(f"  [warn] warmup failed for {algo_alias}: {e}", flush=True)

    _cuda_sync(device)
    t0 = time.perf_counter()
    try:
        client.train(X_client, epochs=1, batch_size=batch_size, epoch_idx=0, lr=lr)
    finally:
        _cuda_sync(device)
    dt = time.perf_counter() - t0

    del client
    if teacher is not None:
        del teacher
    gc.collect()
    _empty_cache(device)
    return float(dt)


def _time_ssl_client_one_epoch(
    *,
    algo_name: str,
    X_client: np.ndarray,
    y_client: np.ndarray,
    shapelets_size_and_len: Dict[int, int],
    in_channels: int,
    num_classes: int,
    batch_size: int,
    lr: float,
    wd: float,
    num_epoch: int,
    device: torch.device,
    warmup_batches: int = 1,
) -> float:
    from algo.flbench_compat import SequentialTrainer, TensorBaseDataset
    from algo.ssl_model import OrchestraShapeletModel, ShapeletSSLModel
    from algo.ssl_runner import BYOLClient, FedU2Client, OrchestraClient

    args = _ssl_args(algo_name, batch_size, lr, wd, num_epoch)

    def _mk_model():
        if algo_name == "orchestra":
            ssl_cfg = args.ssl
            return OrchestraShapeletModel(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure="mix",
                projector_hidden_dim=int(ssl_cfg.projector_hidden_dim),
                projector_out_dim=int(ssl_cfg.projector_out_dim),
                ema_tau=float(ssl_cfg.ema_tau),
                num_global_clusters=int(ssl_cfg.num_global_clusters),
                num_local_clusters=int(ssl_cfg.num_local_clusters),
                cluster_m_size=int(ssl_cfg.cluster_m_size),
                temperature=float(ssl_cfg.temperature),
                deg_num_classes=int(ssl_cfg.deg_num_classes),
                to_cuda=(device.type == "cuda"),
            )
        return ShapeletSSLModel(
            method=algo_name,
            shapelets_size_and_len=shapelets_size_and_len,
            in_channels=in_channels,
            num_classes=num_classes,
            dist_measure="mix",
            projector_hidden_dim=int(args.ssl.projector_hidden_dim),
            projector_out_dim=int(args.ssl.projector_out_dim),
            predictor_hidden_dim=int(args.ssl.predictor_hidden_dim),
            to_cuda=(device.type == "cuda"),
        )

    X_arr = np.asarray(X_client, dtype=np.float32)
    y_arr = np.asarray(y_client)
    dataset_obj = TensorBaseDataset(X_arr, y_arr)
    data_indices = [{"train": list(range(len(X_arr))), "val": [], "test": []}]

    def _make_optimizer_cls():
        def _cls(params):
            return torch.optim.SGD(
                params, lr=float(lr), weight_decay=float(wd), momentum=0.9
            )
        return _cls

    client_cls = {"byol": BYOLClient, "fedu2": FedU2Client, "orchestra": OrchestraClient}[algo_name]
    client = client_cls(
        model=_mk_model(),
        optimizer_cls=_make_optimizer_cls(),
        lr_scheduler_cls=None,
        args=args,
        dataset=dataset_obj,
        data_indices=data_indices,
        device=device,
        return_diff=False,
    )

    init_params = {k: v.detach().cpu().clone() for k, v in client.model.named_parameters()}
    server_package = {
        "client_id": 0,
        "local_epoch": int(num_epoch),
        "regular_model_params": init_params,
        "optimizer_state": {},
        "lr_scheduler_state": {},
        "current_round": 1,
    }
    client.set_parameters(server_package)

    if warmup_batches > 0:
        try:
            client.fit()
        except Exception as e:
            print(f"  [warn] ssl warmup failed: {e}", flush=True)

    _cuda_sync(device)
    t0 = time.perf_counter()
    try:
        client.fit()
    finally:
        _cuda_sync(device)
    dt = time.perf_counter() - t0

    del client
    gc.collect()
    _empty_cache(device)
    return float(dt)


_CSL_ALGOS = {"fedavg", "fedprox", "fedcsl", "spilter-m1", "spilter-m2", "spilter-m4"}
_SSL_ALGOS = {"byol", "fedu2", "orchestra"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo",
        required=True,
        choices=sorted(_CSL_ALGOS | _SSL_ALGOS),
    )
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num-epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="data/system_efficiency_HAR_partials/{algo}.json",
    )
    args = parser.parse_args()

    algo = args.algo.lower()
    output_path = Path(args.output.replace("{algo}", algo))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[warn] CUDA 不可用，将在 CPU 上测时", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[info] algo={algo} | num_clients={args.num_clients} | alpha={args.alpha}", flush=True)
    print(f"[info] device={device} | CUDA_VISIBLE_DEVICES={visible!r} | gpu={gpu_name}", flush=True)
    print(f"[info] batch_size={args.batch_size} | lr={args.lr} | epoch={args.num_epoch}", flush=True)

    t_load = time.perf_counter()
    X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(
        args.num_clients, args.alpha
    )
    dt_load = time.perf_counter() - t_load
    print(f"[info] LoadDataset_HAR done in {dt_load:.1f}s", flush=True)

    if isinstance(X_all, torch.Tensor):
        X_all = X_all.numpy()
    n_ts, in_channels, len_ts = X_all.shape
    num_classes = len(set(y_all.tolist() if hasattr(y_all, "tolist") else list(y_all)))
    shapelets_size_and_len = _shapelet_dict(len_ts)
    print(
        f"[info] shape: N={n_ts}, C={in_channels}, T={len_ts}, "
        f"num_classes={num_classes}, scales={list(shapelets_size_and_len.keys())}",
        flush=True,
    )

    teacher_state: Optional[Dict[str, torch.Tensor]] = None
    if algo in _CSL_ALGOS:
        from train import LearningShapeletsCL
        _seed_cfg = _build_csl_like_config("fedavg")
        seed_client = LearningShapeletsCL(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=nn.CrossEntropyLoss(),
            in_channels=in_channels,
            num_classes=num_classes,
            dist_measure="mix",
            verbose=0,
            to_cuda=(device.type == "cuda"),
            config=_seed_cfg,
            device=device,
        )
        teacher_state = {k: v.detach().cpu().clone() for k, v in seed_client.model.state_dict().items()}
        del seed_client
        gc.collect()
        _empty_cache(device)

    per_client_times: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    is_ssl = algo in _SSL_ALGOS

    for cid in range(args.num_clients):
        X_client = np.asarray(X_fed[cid]) if not isinstance(X_fed[cid], np.ndarray) else X_fed[cid]
        y_client = np.asarray(y_fed[cid]) if not isinstance(y_fed[cid], np.ndarray) else y_fed[cid]
        n_samples = int(X_client.shape[0]) if X_client.ndim >= 1 else 0
        if n_samples < max(2, args.batch_size):
            skipped.append({"client": cid, "samples": n_samples, "reason": "too_few_samples"})
            print(f"  [skip] client {cid} samples={n_samples}", flush=True)
            continue
        try:
            if is_ssl:
                dt = _time_ssl_client_one_epoch(
                    algo_name=algo,
                    X_client=X_client,
                    y_client=y_client,
                    shapelets_size_and_len=shapelets_size_and_len,
                    in_channels=in_channels,
                    num_classes=num_classes,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    num_epoch=args.num_epoch,
                    device=device,
                    warmup_batches=args.warmup_batches,
                )
            else:
                dt = _time_csl_client_one_epoch(
                    algo_alias=algo,
                    X_client=X_client,
                    shapelets_size_and_len=shapelets_size_and_len,
                    in_channels=in_channels,
                    num_classes=num_classes,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    device=device,
                    teacher_state=teacher_state,
                    warmup_batches=args.warmup_batches,
                )
        except Exception as exc:
            traceback.print_exc()
            skipped.append({"client": cid, "samples": n_samples, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"  [err]  client {cid} failed: {exc}", flush=True)
            continue

        per_client_times.append({"client": cid, "samples": n_samples, "epoch_sec": dt})
        print(f"  [ok]   client {cid:3d}  samples={n_samples:5d}  epoch={dt:.3f}s", flush=True)

    if not per_client_times:
        print(f"[err] {algo}: all clients failed/skipped", flush=True)
        return 1

    times = np.array([r["epoch_sec"] for r in per_client_times], dtype=np.float64)
    samples = np.array([r["samples"] for r in per_client_times], dtype=np.int64)
    slowest_idx_local = int(np.argmax(times))
    slowest = per_client_times[slowest_idx_local]

    summary = {
        "algo": algo,
        "num_clients_total": int(args.num_clients),
        "num_clients_timed": int(len(per_client_times)),
        "num_clients_skipped": int(len(skipped)),
        "alpha": float(args.alpha),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "num_epoch": int(args.num_epoch),
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_visible_devices": visible,
        "shape": {"N": int(n_ts), "C": int(in_channels), "T": int(len_ts)},
        "scales": list(shapelets_size_and_len.keys()),
        "slowest_client": slowest,
        "epoch_sec_max": float(times.max()),
        "epoch_sec_mean": float(times.mean()),
        "epoch_sec_median": float(np.median(times)),
        "epoch_sec_min": float(times.min()),
        "samples_at_max": int(samples[slowest_idx_local]),
        "per_client": per_client_times,
        "skipped": skipped,
        "timestamp": time.time(),
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n[done] {algo}: slowest #{slowest['client']} = {summary['epoch_sec_max']:.3f}s "
        f"| mean={summary['epoch_sec_mean']:.3f}s | median={summary['epoch_sec_median']:.3f}s\n"
        f"        -> {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
