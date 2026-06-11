"""Per-scale GPU peak-memory profiling on HAR — Spilter 显存代价标定.

为 §7.7 的显存约束尺度选择（背包 + DP）提供输入：逐个测量\textbf{单个尺度子模型}
的本地训练峰值显存，得到 $\{g_0, g_1, \dots, g_R\}$，再拟合一个可加/带修正的
组合显存预测函数 $\widehat{\mathrm{Mem}}(\mathcal{R})$，供任意尺度组合的 DP 选择使用。

为什么独立成一个脚本（不改 measure_har_epoch.py）：
  * measure_har_epoch.py 负责"每个算法跑 1 epoch 的耗时+峰值"，是 §7.6 已发表实验，
    不应改动其语义；
  * 本脚本只关心"单尺度 / 指定尺度子集"的峰值显存，目标不同；
  * 二者共享 LearningShapeletsCL 客户端与 reset_peak/max_memory 测量方法，
    确保 g_r 与 §7.6 的 m=1/2/4 实测在同一口径下可比。

测量口径（与 measure_har_epoch.py 完全一致）：
  * 单 GPU 独占，串行；用户用 CUDA_VISIBLE_DEVICES=K 锁卡；
  * 1 个 batch warmup 后再 reset_peak_memory_stats，避免 cudnn workspace 污染；
  * 取 max_memory_allocated() 为峰值；
  * batch_size 默认 32（与 config/configSpilter.yml 对齐）。

产物：data/scale_memory_HAR_partials/ 下
  * per_scale.json  —— 每个单尺度 r 的峰值显存（多客户端均值/最大）
  * 供 fit_scale_memory.py 拟合组合显存预测函数。

用法：
  CUDA_VISIBLE_DEVICES=0 python scripts/system_efficiency/measure_scale_memory.py \
      --num-clients 10 --alpha 0.1 --batch-size 32
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
from typing import Any, Dict, List, Optional, Sequence

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


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    return 0.0


def _shapelet_dict(len_ts: int) -> Dict[int, int]:
    """与 FedCSL_All.py:1339 及 measure_har_epoch.py 完全一致的尺度配置。"""
    return {
        int(i): 40
        for i in np.linspace(
            min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int
        )
    }


def _build_spilter_config(scale_aux: bool = True) -> Dict[str, Any]:
    """构造 Spilter 客户端 config（与 measure_har_epoch._build_csl_like_config 同源）。"""
    ablation = {
        "UseJointKD": True,
        "UseJointCL": True,
        "UseScaleKD": scale_aux,
        "UseScaleCL": scale_aux,
        "UseProdictor": False,
        "UseMTLHomo": False,
        "UseACF": True,
        "UseDistribution": False,
    }
    return {
        "algo": "spilter",
        "model": {"params": {"momentum": 0.9, "gamma": 0.5, "beta": 0.4}},
        "ablation": ablation,
        "spilter": {
            "allocation_mode": "local_score_topm",
            "selected_scale_training": "stitched",
            "stitched_feature_source": "selected_scales_only",
        },
        "moon": {"mu": 1.0, "temperature": 0.5},
    }


def _measure_subset_one_client(
    *,
    selected_scales: Sequence[int],
    X_client: np.ndarray,
    shapelets_size_and_len: Dict[int, int],
    in_channels: int,
    num_classes: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
    teacher_state: Optional[Dict[str, torch.Tensor]],
    warmup_batches: int,
    scale_aux: bool,
) -> float:
    """对单个客户端、给定尺度子集，训练 1 epoch 并返回峰值显存 (MB)。

    复用 train.LearningShapeletsCL，与 §7.6 measure_har_epoch 同口径。
    selected_scales 为任意尺度索引列表（不要求连续），通过 client.Selected_Scales
    交给 train.py 的 _normalize_scale_indices 处理。
    """
    from train import LearningShapeletsCL

    config = _build_spilter_config(scale_aux=scale_aux)
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
    if teacher_state is not None:
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

    client.Selected_Scales = list(int(s) for s in selected_scales)

    if warmup_batches > 0 and X_client.shape[0] >= max(batch_size, 1):
        warm_X = X_client[: max(batch_size, 1) * warmup_batches]
        try:
            client.train(warm_X, epochs=1, batch_size=batch_size, epoch_idx=-1, lr=lr)
        except Exception as e:
            print(f"  [warn] warmup failed for scales={list(selected_scales)}: {e}", flush=True)

    _cuda_sync(device)
    _reset_peak_mem(device)
    try:
        client.train(X_client, epochs=1, batch_size=batch_size, epoch_idx=0, lr=lr)
    finally:
        _cuda_sync(device)
    peak_mb = _peak_mem_mb(device)

    del client
    if teacher is not None:
        del teacher
    gc.collect()
    _empty_cache(device)
    return float(peak_mb)


def _aggregate(mems: List[float]) -> Dict[str, float]:
    arr = np.asarray(mems, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--max-clients-per-scale",
        type=int,
        default=0,
        help="每个尺度最多测多少客户端（0=全部）。标定 g_r 用少量客户端即可，"
             "默认全部以求稳。",
    )
    parser.add_argument(
        "--verify-subsets",
        type=str,
        default="",
        help="可选：分号分隔的尺度子集，逗号分隔索引，用于验证可加性，"
             "如 '0,1;0,3,7;2,4,6'。会额外测这些组合的真实峰值，"
             "与可加预测对比。",
    )
    parser.add_argument(
        "--no-scale-aux",
        action="store_true",
        help="关闭 per-scale CL/KD（与 measure_har_epoch 一致的开关）。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/scale_memory_HAR_partials/per_scale.json",
    )
    args = parser.parse_args()
    scale_aux = not args.no_scale_aux

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[warn] CUDA 不可用，将在 CPU 上测（峰值显存无意义，仅冒烟测试）", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[info] 逐尺度显存标定 | num_clients={args.num_clients} | alpha={args.alpha}", flush=True)
    print(f"[info] device={device} | CUDA_VISIBLE_DEVICES={visible!r} | gpu={gpu_name}", flush=True)
    print(f"[info] batch_size={args.batch_size} | scale_aux={scale_aux}", flush=True)

    X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(args.num_clients, args.alpha)
    if isinstance(X_all, torch.Tensor):
        X_all = X_all.numpy()
    n_ts, in_channels, len_ts = X_all.shape
    num_classes = len(set(y_all.tolist() if hasattr(y_all, "tolist") else list(y_all)))
    shapelets_size_and_len = _shapelet_dict(len_ts)
    scale_lengths = list(shapelets_size_and_len.keys())
    R = len(scale_lengths)
    print(
        f"[info] N={n_ts}, C={in_channels}, T={len_ts}, classes={num_classes}, "
        f"R={R}, scale_lengths={scale_lengths}",
        flush=True,
    )

    # teacher（Global_Model）—— 与 measure_har_epoch 一致，用 fedavg 种子初始化一份全模型
    from train import LearningShapeletsCL
    _seed_cfg = _build_spilter_config(scale_aux=scale_aux)
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

    # 选客户端：样本数 >= batch_size 的前若干个
    client_ids: List[int] = []
    for cid in range(args.num_clients):
        Xc = np.asarray(X_fed[cid]) if not isinstance(X_fed[cid], np.ndarray) else X_fed[cid]
        if int(Xc.shape[0]) >= max(2, args.batch_size):
            client_ids.append(cid)
    if args.max_clients_per_scale > 0:
        client_ids = client_ids[: args.max_clients_per_scale]
    print(f"[info] 参与标定的客户端: {client_ids}", flush=True)

    def _client_X(cid: int) -> np.ndarray:
        Xc = X_fed[cid]
        return np.asarray(Xc) if not isinstance(Xc, np.ndarray) else Xc

    def _measure_subset(scales: Sequence[int]) -> Dict[str, Any]:
        mems: List[float] = []
        for cid in client_ids:
            try:
                m = _measure_subset_one_client(
                    selected_scales=scales,
                    X_client=_client_X(cid),
                    shapelets_size_and_len=shapelets_size_and_len,
                    in_channels=in_channels,
                    num_classes=num_classes,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wd=args.wd,
                    device=device,
                    teacher_state=teacher_state,
                    warmup_batches=args.warmup_batches,
                    scale_aux=scale_aux,
                )
                mems.append(m)
            except Exception as exc:
                traceback.print_exc()
                print(f"  [err] scales={list(scales)} client {cid}: {exc}", flush=True)
        if not mems:
            return {"scales": list(scales), "error": "all clients failed"}
        agg = _aggregate(mems)
        return {"scales": list(int(s) for s in scales), "peak_mem_mb": agg}

    # ---- 1) 逐单尺度测 g_r ----
    per_scale: List[Dict[str, Any]] = []
    for r in range(R):
        res = _measure_subset([r])
        mb = res.get("peak_mem_mb", {})
        print(
            f"  [scale {r}] len={scale_lengths[r]:4d}  "
            f"mean={mb.get('mean', float('nan')):.1f}MB  max={mb.get('max', float('nan')):.1f}MB",
            flush=True,
        )
        per_scale.append(res)

    summary: Dict[str, Any] = {
        "task": "scale_memory_calibration",
        "dataset": "HAR",
        "num_clients_total": int(args.num_clients),
        "clients_used": client_ids,
        "alpha": float(args.alpha),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "gpu_name": gpu_name,
        "scale_aux": bool(scale_aux),
        "shape": {"N": int(n_ts), "C": int(in_channels), "T": int(len_ts)},
        "R": int(R),
        "scale_lengths": scale_lengths,
        "per_scale": per_scale,
        "timestamp": time.time(),
    }

    # ---- 2) 可选：验证可加性（实测组合 vs 单尺度之和）----
    verify_results: List[Dict[str, Any]] = []
    if args.verify_subsets.strip():
        single_means = [
            (s.get("peak_mem_mb", {}) or {}).get("mean") for s in per_scale
        ]
        for token in args.verify_subsets.split(";"):
            token = token.strip()
            if not token:
                continue
            try:
                scales = [int(x) for x in token.split(",") if x.strip() != ""]
            except ValueError:
                print(f"  [warn] 跳过非法子集 '{token}'", flush=True)
                continue
            res = _measure_subset(scales)
            measured = (res.get("peak_mem_mb", {}) or {}).get("mean")
            # 朴素可加预测（不含 g0 修正，仅看趋势；真正拟合在 fit_scale_memory.py）
            additive = None
            if all(single_means[s] is not None for s in scales):
                additive = float(sum(single_means[s] for s in scales))
            entry = {
                "scales": scales,
                "measured_mean_mb": measured,
                "naive_sum_of_singles_mb": additive,
            }
            if measured is not None and additive:
                entry["abs_err_mb"] = abs(measured - additive)
                entry["rel_err"] = abs(measured - additive) / measured if measured else None
            verify_results.append(entry)
            print(
                f"  [verify {scales}] measured={measured}  naive_sum={additive}",
                flush=True,
            )
        summary["verify_subsets"] = verify_results

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 逐尺度显存标定写入 -> {output_path}", flush=True)
    print("       下一步: python scripts/system_efficiency/fit_scale_memory.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
