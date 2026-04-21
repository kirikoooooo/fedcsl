"""FL-bench 风格基线（SCAFFOLD / FedProto）的入口函数。

职责：
1. 把 FedCSL 主流程（``FedCSL_All.train``）已经准备好的数据/配置转换成 FL-bench 风格的
   ``args`` / ``data_indices`` / ``dataset`` / ``model``；
2. 构造对应的 ``Server`` 与一组 ``Client`` 并跑 ``Server.train(numRound)``；
3. 每轮后在全局测试集上做与 FedCSL 相同的 SVC 下游评估，把结果写入 FedCSL 的结果文件，
   使得 baseline 曲线与 FedCSL 的日志格式、绘图脚本完全兼容；
4. **完全不碰** CSL 多尺度对比 / 联合蒸馏 / 结构对齐等逻辑。
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

        # 下游 SVC 评估（与 FedCSL 相同的流程）
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

        if test_acc > best["acc"]:
            best["acc"] = float(test_acc)
            best["round"] = round_idx
        print(f"[{algo.upper()}] round {round_idx} train_acc={train_acc:.4f} test_acc={test_acc:.4f} best={best['acc']:.4f} avg_loss={avg_loss}")
        try:
            with open(logTxt, mode="a+") as f:
                loss_str = str(avg_loss) if not (np.isnan(avg_loss) or np.isinf(avg_loss)) else "nan"
                f.write(
                    f"dataset: {dataset}round:{round_idx} server aggregation  "
                    f"testACC:{test_acc} trainACC:{train_acc} avg_loss:{loss_str}\n"
                )
        except Exception as e:  # pragma: no cover - 日志失败不致命
            print(f"[{algo.upper()}] 写日志失败: {e}")

    server.after_round = _after_round

    # 8. 正式训练
    t0 = time.time()
    server.train(int(num_rounds))
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
