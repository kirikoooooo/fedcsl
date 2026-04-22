"""FL-bench 风格基线（SCAFFOLD / FedProto）的入口函数。

职责：
1. 把 FedCSL 主流程（``FedCSL_All.train``）已经准备好的数据/配置转换成 FL-bench 风格的
   ``args`` / ``data_indices`` / ``dataset`` / ``model``；
2. 构造对应的 ``Server`` 与一组 ``Client`` 并跑 ``Server.train(numRound)``；
3. 每轮后在全局测试集上做与 FedCSL 相同的 SVC 下游评估，把结果写入 FedCSL 的结果文件，
   使得 baseline 曲线与 FedCSL 的日志格式、绘图脚本完全兼容；
4. **完全不碰** CSL 多尺度对比 / 联合蒸馏 / 结构对齐等逻辑；
5. 支持每 ``checkpoint_every`` 轮覆盖保存 checkpoint + 启动时 auto-resume
   （文件：``./checkpoint/resume/baseline_<algo>_<dataset>_<desc>.pt``）。
"""
from __future__ import annotations

import copy
import os
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .baseline_model import ShapeletClassifier
from .flbench_compat import AttrDict, SequentialTrainer, TensorBaseDataset
from .scaffold import SCAFFOLDClient, SCAFFOLDServer
from .fedproto import FedProtoClient, FedProtoServer


# -----------------------------------------------------------------------------
# 工具：数据索引构造 + FL-bench 风格 args 构造
# -----------------------------------------------------------------------------

def _build_data_indices(
    X_fed: List[np.ndarray], y_fed: List[np.ndarray]
) -> tuple[TensorBaseDataset, List[Dict[str, List[int]]]]:
    """把分好片的联邦数据合并为一个 ``TensorBaseDataset``，并返回每客户端的 indices。"""
    X_list, y_list = [], []
    data_indices: List[Dict[str, List[int]]] = []
    offset = 0
    for X_c, y_c in zip(X_fed, y_fed):
        X_c = np.asarray(X_c)
        y_c = np.asarray(y_c)
        n = int(X_c.shape[0])
        idx = list(range(offset, offset + n))
        data_indices.append({"train": idx if n > 0 else [0], "val": [], "test": []})
        X_list.append(X_c)
        y_list.append(y_c)
        offset += n
    if not X_list:
        raise ValueError("baseline_runner: 空联邦数据")
    X_cat = np.concatenate(X_list, axis=0)
    y_cat = np.concatenate(y_list, axis=0)
    dataset = TensorBaseDataset(X_cat, y_cat)
    return dataset, data_indices


def _build_args(
    *,
    algo: str,
    config: Dict[str, Any],
    num_classes: int,
    batch_size: int,
    lr: float,
    wd: float,
    num_epoch: int,
    seed: int,
    dataset_name: str,
) -> AttrDict:
    """构造 FL-bench 风格的 ``args``（仅包含移植代码实际访问的字段）。"""
    return AttrDict({
        "common": AttrDict({
            "batch_size": int(batch_size),
            "local_epoch": int(num_epoch),
            "use_cuda": True,
            "seed": int(seed),
            "reset_optimizer_on_global_epoch": True,
            "buffers": "global",  # 与 FL-bench 同义：BN buffer 参与聚合
            "join_ratio": 1.0,
            "verbose_gap": 1,
            "test": AttrDict({
                "client": AttrDict({"train": False, "val": False, "test": False, "interval": 0, "finetune_epoch": 0}),
                "server": AttrDict({"interval": 1}),
            }),
            "client_side_evaluation": False,
        }),
        "optimizer": AttrDict({
            "name": "sgd",
            "lr": float(lr),
            "weight_decay": float(wd),
            "momentum": float(config.get("model", {}).get("params", {}).get("momentum", 0.0)),
        }),
        "dataset": AttrDict({
            "name": dataset_name,
            "num_classes": int(num_classes),
        }),
        "scaffold": AttrDict({
            "global_lr": float(config.get("scaffold", {}).get("global_lr", 1.0)),
        }),
        "fedproto": AttrDict({
            "lamda": float(config.get("fedproto", {}).get("lamda", 1.0)),
        }),
    })


# -----------------------------------------------------------------------------
# Checkpoint：每 N round 覆盖保存 + 启动时 resume
# -----------------------------------------------------------------------------
_CKPT_DIR = "./checkpoint/resume"


def _ckpt_path(algo: str, dataset: str, formatted_date: str) -> str:
    # 同一算法+数据集+description 的多次实验共用一个 ckpt（覆盖式保存）
    os.makedirs(_CKPT_DIR, exist_ok=True)
    safe_date = (formatted_date or "default").replace("/", "_").replace("\\", "_")
    return os.path.join(_CKPT_DIR, f"baseline_{algo}_{dataset}_{safe_date}.pt")


def _save_baseline_checkpoint(
    *,
    algo: str,
    dataset: str,
    formatted_date: str,
    round_idx: int,
    server,
    server_model,
    clients,
    best: Dict[str, Any],
) -> None:
    """覆盖式保存：server / 每个 client 的 state_dict + 必要元信息。"""
    import torch

    path = _ckpt_path(algo, dataset, formatted_date)
    payload: Dict[str, Any] = {
        "algo": algo,
        "dataset": dataset,
        "formatted_date": formatted_date,
        "round_idx": int(round_idx),
        "best": dict(best),
        "server_model_state": {
            k: v.detach().cpu().clone() for k, v in server_model.state_dict().items()
        },
        "public_model_params": {
            k: v.detach().cpu().clone() for k, v in server.public_model_params.items()
        },
        "clients_state": [
            {k: v.detach().cpu().clone() for k, v in c.model.state_dict().items()}
            for c in clients
        ],
    }
    if algo == "scaffold":
        payload["c_global"] = [t.detach().cpu().clone() for t in server.c_global]
        payload["c_local"] = [
            [t.detach().cpu().clone() for t in vec] for vec in server.c_local
        ]
    elif algo == "fedproto":
        payload["global_prototypes"] = {
            int(k): v.detach().cpu().clone() for k, v in server.global_prototypes.items()
        }

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[{algo.upper()}] checkpoint saved → {path} (round={round_idx + 1})", flush=True)


def _try_load_baseline_checkpoint(
    *,
    algo: str,
    dataset: str,
    formatted_date: str,
    server,
    server_model,
    clients,
) -> Optional[Dict[str, Any]]:
    """尝试从 ckpt 恢复状态。返回 checkpoint payload（含 round_idx 与 best）或 None。"""
    import torch

    path = _ckpt_path(algo, dataset, formatted_date)
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[{algo.upper()}] 读取 checkpoint 失败（{path}）：{e}，从零开始", flush=True)
        return None

    try:
        server_model.load_state_dict(payload["server_model_state"], strict=False)
        server.public_model_params = OrderedDict(
            (k, v.clone()) for k, v in payload["public_model_params"].items()
        )
        for c, state in zip(clients, payload.get("clients_state", [])):
            c.model.load_state_dict(state, strict=False)
        if algo == "scaffold" and "c_global" in payload:
            server.c_global = [t.clone() for t in payload["c_global"]]
            server.c_local = [
                [t.clone() for t in vec] for vec in payload["c_local"]
            ]
        elif algo == "fedproto" and "global_prototypes" in payload:
            server.global_prototypes = {
                int(k): v.clone() for k, v in payload["global_prototypes"].items()
            }
    except Exception as e:  # pragma: no cover
        print(f"[{algo.upper()}] checkpoint 结构不兼容：{e}，从零开始", flush=True)
        return None

    resume_round = int(payload.get("round_idx", -1)) + 1
    print(
        f"[{algo.upper()}] checkpoint loaded ← {path}，将从 round {resume_round + 1} 继续",
        flush=True,
    )
    return payload


# -----------------------------------------------------------------------------
# Runner 主体
# -----------------------------------------------------------------------------

def run_baseline(
    *,
    algo: str,
    config: Dict[str, Any],
    seed: Optional[int],
    dataset: str,
    shapelets_size_and_len: dict,
    n_channels: int,
    num_classes: int,
    X_all,
    y_all,
    X_test,
    y_test,
    X_fed,
    y_fed,
    X_val=None,
    y_val=None,
    has_val: bool = False,
    num_rounds: int,
    num_epoch: int,
    batch_size: int,
    lr: float,
    wd: float,
    dist_measure: str,
    to_cuda: bool,
    logTxt: str,
    formatted_date: str,
    eval_train_test_fn: Callable,
    eval_tstcc_fn: Callable,
    save_model_fn: Callable,
) -> None:
    """主入口：准备一切并驱动 FL-bench 风格的训练。

    - ``eval_train_test_fn(transformation, transformation_test, y_train, y_test)``：
      FedCSL 中的 ``eval`` 函数；
    - ``eval_tstcc_fn(train, test, val, y_train, y_test, y_val)``：FedCSL 的 ``eval_TSTCC``；
    - ``save_model_fn(model, dataset, formatted_date)``：复用 FedCSL 的保存逻辑。
    """
    algo = algo.lower()
    if algo not in ("scaffold", "fedproto"):
        raise ValueError(f"run_baseline: 不支持的算法 {algo!r}（仅 scaffold / fedproto）")

    device = torch.device("cuda") if (to_cuda and torch.cuda.is_available()) else torch.device("cpu")
    num_clients = len(X_fed)

    # 1. 数据 + 索引
    dataset_obj, data_indices = _build_data_indices(X_fed, y_fed)

    # 2. args
    seed_i = int(seed) if seed is not None else 42
    args = _build_args(
        algo=algo,
        config=config,
        num_classes=num_classes,
        batch_size=batch_size,
        lr=lr,
        wd=wd,
        num_epoch=num_epoch,
        seed=seed_i,
        dataset_name=dataset,
    )

    # 3. 模型：服务端 + 每个客户端各自一份（FL-bench 架构里 client 有自己的 model 副本）
    def _mk_model() -> nn.Module:
        return ShapeletClassifier(
            shapelets_size_and_len=shapelets_size_and_len,
            in_channels=n_channels,
            num_classes=num_classes,
            dist_measure=dist_measure,
            to_cuda=to_cuda,
        )

    server_model = _mk_model()

    # 4. Server
    if algo == "scaffold":
        server = SCAFFOLDServer(args)
    else:
        server = FedProtoServer(args)
    server.model = server_model
    server.client_num = num_clients
    server.train_clients = list(range(num_clients))
    server.val_clients = list(range(num_clients))
    server.test_clients = list(range(num_clients))
    server.client_local_epoches = [int(num_epoch)] * num_clients
    # 与 FL-bench FedAvgServer.__init__ 一致：public_model_params 只纳入 named_parameters，
    # 不含 buffer（BN running stats 等按 strict=False 保留本地值）。
    server.public_model_params = OrderedDict(
        (k, p.detach().cpu().clone()) for k, p in server_model.named_parameters()
    )
    server.clients_personal_model_params = {i: OrderedDict() for i in range(num_clients)}
    server.client_optimizer_states = {i: {} for i in range(num_clients)}
    server.client_lr_scheduler_states = {i: {} for i in range(num_clients)}
    if algo == "scaffold":
        server.setup_control_variates()

    # 5. Clients
    common_lr = float(lr)
    common_wd = float(wd)
    common_mom = float(args.optimizer.momentum)

    def _make_optimizer_cls():
        def _cls(params):
            return torch.optim.SGD(params, lr=common_lr, weight_decay=common_wd, momentum=common_mom)
        return _cls

    ClientCls = SCAFFOLDClient if algo == "scaffold" else FedProtoClient
    clients = []
    for cid in range(num_clients):
        client_model = _mk_model()
        client = ClientCls(
            model=client_model,
            optimizer_cls=_make_optimizer_cls(),
            lr_scheduler_cls=None,
            args=args,
            dataset=dataset_obj,
            data_indices=data_indices,
            device=device,
            return_diff=server.return_diff,
        )
        clients.append(client)

    # 6. Trainer
    server.trainer = SequentialTrainer(server, clients)

    # 7. 注入 after_round hook：做全局 SVC 评估并写日志（与 FedCSL 的日志格式一致）
    best = {"acc": 0.0, "round": -1}
    y_all_np = y_all.cpu().numpy() if hasattr(y_all, "cpu") else np.asarray(y_all)
    y_test_np = y_test.cpu().numpy() if hasattr(y_test, "cpu") else np.asarray(y_test)
    y_val_np = None
    if has_val and y_val is not None:
        y_val_np = y_val.cpu().numpy() if hasattr(y_val, "cpu") else np.asarray(y_val)

    from sklearn.preprocessing import RobustScaler

    original_after = server.after_round
    original_train_one_round = server.train_one_round

    # 包一层：在每轮训练开始/结束时打印耗时与异常，定位卡顿和错误。
    def _wrapped_train_one_round():
        round_idx = int(server.current_epoch)
        t0 = time.time()
        print(
            f"[{algo.upper()}] round {round_idx + 1}/{int(num_rounds)} start  "
            f"selected={server.selected_clients}",
            flush=True,
        )
        try:
            packages = original_train_one_round()
        except Exception as e:  # pragma: no cover
            import traceback
            print(
                f"[{algo.upper()}] round {round_idx} TRAIN ERROR: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            raise
        dt = time.time() - t0
        print(
            f"[{algo.upper()}] round {round_idx + 1}/{int(num_rounds)} local_train done in {dt:.1f}s",
            flush=True,
        )
        return packages

    server.train_one_round = _wrapped_train_one_round

    def _after_round(client_packages):
        original_after(client_packages)
        round_idx = int(server.current_epoch)

        # 聚合各客户端本轮 loss（若 package 里有 eval_results 会为 dict；没有就跳过）
        client_losses = []
        for pkg in client_packages.values():
            er = pkg.get("eval_results", {}) or {}
            after = er.get("after", {}) if isinstance(er, dict) else {}
            tr = after.get("train", None) if isinstance(after, dict) else None
            if tr is not None and hasattr(tr, "loss"):
                client_losses.append(float(tr.loss))
        avg_loss = float(np.mean(client_losses)) if client_losses else float("nan")

        # 下游 SVC 评估用 try/except 包裹，单轮评估失败不应中断训练。
        try:
            t_eval = time.time()
            transformation = server_model.transform(X_all, batch_size=int(batch_size), normalize=True, result_type="numpy")
            transformation_test = server_model.transform(X_test, batch_size=int(batch_size), normalize=True, result_type="numpy")
            scaler = RobustScaler()
            transformation = scaler.fit_transform(transformation)
            transformation_test = scaler.transform(transformation_test)
            if has_val and X_val is not None and y_val_np is not None:
                transformation_val = server_model.transform(X_val, batch_size=int(batch_size), normalize=True, result_type="numpy")
                transformation_val = scaler.transform(transformation_val)
                train_acc, test_acc = eval_tstcc_fn(
                    transformation_train=transformation,
                    transformation_test=transformation_test,
                    transformation_val=transformation_val,
                    y_train=y_all_np, y_test=y_test_np, y_val=y_val_np,
                )
            else:
                train_acc, test_acc = eval_train_test_fn(transformation, transformation_test, y_train=y_all_np, y_test=y_test_np)
            eval_dt = time.time() - t_eval
        except Exception as e:  # pragma: no cover
            import traceback
            print(f"[{algo.upper()}] round {round_idx} EVAL ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            train_acc, test_acc, eval_dt = float("nan"), float("nan"), 0.0

        if np.isfinite(test_acc) and test_acc > best["acc"]:
            best["acc"] = float(test_acc)
            best["round"] = round_idx

        print(
            f"[{algo.upper()}] round {round_idx + 1}/{int(num_rounds)} done  "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"best={best['acc']:.4f}@r{best['round'] + 1 if best['round'] >= 0 else -1} "
            f"avg_loss={avg_loss} eval={eval_dt:.1f}s",
            flush=True,
        )
        try:
            with open(logTxt, mode="a+") as f:
                loss_str = str(avg_loss) if np.isfinite(avg_loss) else "nan"
                f.write(
                    f"dataset: {dataset}round:{round_idx} server aggregation  "
                    f"testACC:{test_acc} trainACC:{train_acc} avg_loss:{loss_str}\n"
                )
        except Exception as e:  # pragma: no cover - 日志失败不致命
            print(f"[{algo.upper()}] 写日志失败: {e}")

        # 断点续训：每 checkpoint_every 轮保存一次（0 表示关闭）
        ckpt_every = int(config.get("federated", {}).get("checkpoint_every", 0) or 0)
        if ckpt_every > 0 and (round_idx + 1) % ckpt_every == 0:
            try:
                _save_baseline_checkpoint(
                    algo=algo,
                    dataset=dataset,
                    formatted_date=formatted_date,
                    round_idx=round_idx,
                    server=server,
                    server_model=server_model,
                    clients=clients,
                    best=best,
                )
            except Exception as e:  # pragma: no cover
                print(f"[{algo.upper()}] 保存 checkpoint 失败: {e}")

    server.after_round = _after_round

    # 7.5. 尝试 auto-resume（若 config.federated.auto_resume != False 且存在 ckpt）
    auto_resume = bool(config.get("federated", {}).get("auto_resume", True))
    start_round = 0
    if auto_resume:
        payload = _try_load_baseline_checkpoint(
            algo=algo, dataset=dataset, formatted_date=formatted_date,
            server=server, server_model=server_model, clients=clients,
        )
        if payload is not None:
            start_round = int(payload.get("round_idx", -1)) + 1
            best["acc"] = float(payload.get("best", {}).get("acc", 0.0))
            best["round"] = int(payload.get("best", {}).get("round", -1))
            # 让 FL-bench 主循环从 start_round 开始
            server.current_epoch = start_round

    # 8. 正式训练
    t0 = time.time()
    remaining = max(0, int(num_rounds) - start_round)
    if remaining > 0:
        # server.train 里 for E in range(num_rounds)，需要改成 range(start_round, num_rounds)；
        # 此处通过简单 monkey-patch 实现，不动 FedAvgServer 基类。
        if start_round > 0:
            _orig_train = server.train

            def _resumable_train(num_rounds_: int) -> None:
                for E in range(start_round, int(num_rounds_)):
                    server.current_epoch = E
                    server.selected_clients = server.select_clients()
                    client_packages = server.train_one_round()
                    if server.model is not None:
                        server.model.load_state_dict(server.public_model_params, strict=False)
                    server.after_round(client_packages)

            server.train = _resumable_train  # type: ignore[assignment]

        try:
            server.train(int(num_rounds))
        except Exception as e:
            import traceback
            print(f"[{algo.upper()}] 训练异常终止：{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise
    else:
        print(f"[{algo.upper()}] checkpoint 已完成 {start_round}/{int(num_rounds)} 轮，无需继续训练。")

    dt = time.time() - t0
    print(f"[{algo.upper()}] training done in {dt:.1f}s, best round={best['round']} best_acc={best['acc']:.4f}")

    # 9. 保存最终全局模型（SCAFFOLD / FedProto 共用 shapelet backbone）。
    # FedCSL 主流程的 save_model 现在接受 state_dict（CPU 副本），保持接口一致。
    try:
        final_state = {
            k: v.detach().cpu().clone() for k, v in server_model.state_dict().items()
        }
        save_model_fn(final_state, dataset, formatted_date)
    except Exception as e:  # pragma: no cover
        print(f"[{algo.upper()}] 保存模型失败: {e}")
