"""Federated self-supervised baselines inspired by Orchestra.

当前先落地最小可跑版本：
- SimCLR
- SimSiam
- BYOL

共同特点：
- 复用当前项目的 shapelet encoder；
- 复用当前项目的 tsaug 双视图增强；
- 训练仍走 FedAvg 式联邦聚合；
- 评估仍使用全局 encoder + SVM / TSTCC 协议，便于与 FedCSL 直接对比。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
try:
    import tsaug  # type: ignore
except Exception:  # pragma: no cover
    tsaug = None

from .flbench_compat import AttrDict, FedAvgClient, FedAvgServer, SequentialTrainer, TensorBaseDataset
from .ssl_model import OrchestraShapeletModel, ShapeletSSLModel


def _build_data_indices(
    X_fed: List[np.ndarray], y_fed: List[np.ndarray]
) -> tuple[TensorBaseDataset, List[Dict[str, List[int]]]]:
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
        raise ValueError("ssl_runner: empty federated dataset")
    X_cat = np.concatenate(X_list, axis=0)
    y_cat = np.concatenate(y_list, axis=0)
    return TensorBaseDataset(X_cat, y_cat), data_indices


def _build_args(
    *,
    algo: str,
    config: Dict[str, Any],
    batch_size: int,
    lr: float,
    wd: float,
    num_epoch: int,
    seed: int,
    dataset_name: str,
) -> AttrDict:
    ssl_cfg = config.get("ssl", {})
    return AttrDict({
        "common": AttrDict({
            "batch_size": int(batch_size),
            "local_epoch": int(num_epoch),
            "use_cuda": True,
            "seed": int(seed),
            "reset_optimizer_on_global_epoch": True,
            "buffers": "global",
            "join_ratio": 1.0,
            "verbose_gap": 1,
        }),
        "optimizer": AttrDict({
            "name": "sgd",
            "lr": float(lr),
            "weight_decay": float(wd),
            "momentum": float(config.get("model", {}).get("params", {}).get("momentum", 0.9)),
        }),
        "dataset": AttrDict({
            "name": dataset_name,
        }),
        "ssl": AttrDict({
            "method": str(algo).lower(),
            "temperature": float(ssl_cfg.get("temperature", 0.2)),
            "projector_hidden_dim": int(ssl_cfg.get("projector_hidden_dim", 256)),
            "projector_out_dim": int(ssl_cfg.get("projector_out_dim", 128)),
            "predictor_hidden_dim": int(ssl_cfg.get("predictor_hidden_dim", 256)),
            "ema_tau": float(ssl_cfg.get("ema_tau", 0.99)),
            "num_global_clusters": int(ssl_cfg.get("num_global_clusters", 32)),
            "num_local_clusters": int(ssl_cfg.get("num_local_clusters", 8)),
            "cluster_m_size": int(ssl_cfg.get("cluster_m_size", 128)),
            "deg_num_classes": int(ssl_cfg.get("deg_num_classes", 5)),
            "server_cluster_rounds": int(ssl_cfg.get("server_cluster_rounds", 80)),
        }),
    })


def _make_aug(name: str, ts_l: int):
    if tsaug is None:
        return None
    seed = int(np.random.randint(2**31 - 1, dtype=np.int64))
    if name == "AddNoise":
        return tsaug.AddNoise(seed=seed)
    if name == "Crop":
        return tsaug.Crop(max(2, int(0.9 * ts_l)), seed=seed)
    if name == "Pool":
        return tsaug.Pool(seed=seed)
    if name == "Quantize":
        return tsaug.Quantize(seed=seed)
    if name == "TimeWarp":
        return tsaug.TimeWarp(seed=seed)
    raise ValueError(f"unknown augmentation: {name}")


def _resize_1d(vec: np.ndarray, out_len: int) -> np.ndarray:
    in_len = int(vec.shape[0])
    if in_len == out_len:
        return vec.copy()
    x_old = np.linspace(0.0, 1.0, in_len)
    x_new = np.linspace(0.0, 1.0, out_len)
    return np.interp(x_new, x_old, vec)


def _fallback_augment(name: str, arr: np.ndarray, ts_l: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    rng = np.random.default_rng(int(np.random.randint(2**31 - 1)))

    if name == "AddNoise":
        noise_scale = float(np.std(out) + 1e-6) * 0.03
        return out + rng.normal(0.0, noise_scale, size=out.shape).astype(np.float32)

    if name == "Crop":
        crop_len = max(2, int(0.9 * ts_l))
        for i in range(out.shape[0]):
            start = int(rng.integers(0, max(1, ts_l - crop_len + 1)))
            seg = out[i, start:start + crop_len, :]
            for c in range(out.shape[2]):
                out[i, :, c] = _resize_1d(seg[:, c], ts_l)
        return out

    if name == "Pool":
        if ts_l < 2:
            return out
        left = out[:, 0::2, :]
        right = out[:, 1::2, :]
        pair_len = min(left.shape[1], right.shape[1])
        pooled = 0.5 * (left[:, :pair_len, :] + right[:, :pair_len, :])
        for i in range(out.shape[0]):
            for c in range(out.shape[2]):
                out[i, :, c] = _resize_1d(pooled[i, :, c], ts_l)
        return out

    if name == "Quantize":
        levels = 16.0
        x_min = out.min(axis=1, keepdims=True)
        x_max = out.max(axis=1, keepdims=True)
        denom = np.maximum(x_max - x_min, 1e-6)
        norm = (out - x_min) / denom
        quant = np.round(norm * (levels - 1.0)) / (levels - 1.0)
        return quant * denom + x_min

    if name == "TimeWarp":
        scale = float(rng.uniform(0.8, 1.2))
        warped_len = max(2, int(ts_l * scale))
        for i in range(out.shape[0]):
            for c in range(out.shape[2]):
                warped = _resize_1d(out[i, :, c], warped_len)
                out[i, :, c] = _resize_1d(warped, ts_l)
        return out

    raise ValueError(f"unknown fallback augmentation: {name}")


def _ensure_ts_length(arr: np.ndarray, ts_l: int) -> np.ndarray:
    """Ensure augmented batch keeps the original temporal length ``ts_l``.

    tsaug 的某些增广（尤其是 Crop）会改变时间维长度；联邦训练与当前模型实现都要求
    输入保持固定长度，因此这里统一把结果 resize 回原始长度。
    输入/输出形状均约定为 ``(N, T, C)``。
    """
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3:
        raise ValueError(f"expected 3D array (N,T,C), got shape={out.shape}")
    if out.shape[1] == ts_l:
        return out

    resized = np.empty((out.shape[0], ts_l, out.shape[2]), dtype=np.float32)
    for i in range(out.shape[0]):
        for c in range(out.shape[2]):
            resized[i, :, c] = _resize_1d(out[i, :, c], ts_l)
    return resized


def make_two_views(x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ops = ["AddNoise", "Crop", "Pool", "Quantize", "TimeWarp"]
    ts_l = int(x.size(2))
    arr = x.transpose(1, 2).detach().cpu().numpy()

    aug1 = str(np.random.choice(ops))
    aug2 = str(np.random.choice(ops))
    while aug2 == aug1:
        aug2 = str(np.random.choice(ops))

    aug1_impl = _make_aug(aug1, ts_l)
    aug2_impl = _make_aug(aug2, ts_l)
    if aug1_impl is not None and aug2_impl is not None:
        x_q = aug1_impl.augment(arr)
        x_k = aug2_impl.augment(arr)
    else:
        x_q = _fallback_augment(aug1, arr, ts_l)
        x_k = _fallback_augment(aug2, arr, ts_l)

    x_q = _ensure_ts_length(x_q, ts_l)
    x_k = _ensure_ts_length(x_k, ts_l)

    x_q = torch.from_numpy(np.asarray(x_q)).float().transpose(1, 2).to(device)
    x_k = torch.from_numpy(np.asarray(x_k)).float().transpose(1, 2).to(device)
    return x_q, x_k


def make_labeled_view(
    x: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = ["AddNoise", "Crop", "Pool", "Quantize", "TimeWarp"]
    ts_l = int(x.size(2))
    arr = x.transpose(1, 2).detach().cpu().numpy()
    labels = np.random.randint(0, len(ops), size=arr.shape[0])
    out = np.asarray(arr, dtype=np.float32).copy()
    for idx, op_idx in enumerate(labels):
        op_name = ops[int(op_idx)]
        aug_impl = _make_aug(op_name, ts_l)
        if aug_impl is not None:
            aug_out = np.asarray(aug_impl.augment(arr[idx:idx + 1]), dtype=np.float32)
        else:
            aug_out = _fallback_augment(op_name, arr[idx:idx + 1], ts_l)
        out[idx:idx + 1] = _ensure_ts_length(aug_out, ts_l)
    x_deg = torch.from_numpy(out).float().transpose(1, 2).to(device)
    y_deg = torch.as_tensor(labels, dtype=torch.long, device=device)
    return x_deg, y_deg


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(reps, reps.T) / max(temperature, 1e-8)

    batch_size = z1.size(0)
    device = z1.device
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)

    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    return F.cross_entropy(logits, labels)


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


class SSLClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        agg_names = set(getattr(self.model, "aggregated_param_names", self.regular_params_name))
        self.regular_params_name = [name for name in self.regular_params_name if name in agg_names]
        self.personal_params_name = [
            name for name, _ in self.model.named_parameters() if name not in agg_names
        ]

    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self) -> None:
        self.model.train()
        self.dataset.train()
        losses = []
        for _ in range(self.local_epoch):
            for x, _y in self.trainloader:
                if len(x) <= 1:
                    continue
                x = x.to(self.device, non_blocking=True).float()
                loss = self._compute_loss(x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if getattr(self.model, "method", "") == "byol":
                    self.model.update_target_network(float(self.args.ssl.ema_tau))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        self.eval_results = {"after": {"train": AttrDict({"loss": avg_loss})}}


class SimCLRClient(SSLClient):
    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_k = make_two_views(x, self.device)
        _, z1 = self.model.online_project(x_q)
        _, z2 = self.model.online_project(x_k)
        return nt_xent_loss(z1, z2, float(self.args.ssl.temperature))


class SimSiamClient(SSLClient):
    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_k = make_two_views(x, self.device)
        _, z1 = self.model.online_project(x_q)
        _, z2 = self.model.online_project(x_k)
        p1 = self.model.predictor(z1)
        p2 = self.model.predictor(z2)
        return 0.5 * (
            negative_cosine_similarity(p1, z2.detach()) +
            negative_cosine_similarity(p2, z1.detach())
        )


class BYOLClient(SSLClient):
    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        self.model.reset_target_network()

    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_k = make_two_views(x, self.device)
        _, z1 = self.model.online_project(x_q)
        _, z2 = self.model.online_project(x_k)
        p1 = self.model.predictor(z1)
        p2 = self.model.predictor(z2)
        with torch.no_grad():
            t1 = self.model.target_project(x_q)
            t2 = self.model.target_project(x_k)
        return 0.5 * (
            (2.0 + 2.0 * negative_cosine_similarity(p1, t2)) +
            (2.0 + 2.0 * negative_cosine_similarity(p2, t1))
        )


class OrchestraClient(SSLClient):
    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        self.current_round = int(package.get("current_round", 0))
        self.model.reset_target_network()

    def fit(self) -> None:
        self.model.train()
        self.dataset.train()
        losses = []

        if self.current_round == 0:
            self.model.reset_memory(self.trainloader, self.device)
            self.model.local_clustering()
            self.eval_results = {"after": {"train": AttrDict({"loss": float("nan")})}}
            return

        for _ in range(self.local_epoch):
            for x, _y in self.trainloader:
                if len(x) <= 1:
                    continue
                x = x.to(self.device, non_blocking=True).float()
                x_q, x_k = make_two_views(x, self.device)
                x_deg, y_deg = make_labeled_view(x, self.device)
                loss = self.model.orchestra_loss(x_q, x_k, x_deg, y_deg)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
        self.model.local_clustering()
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        self.eval_results = {"after": {"train": AttrDict({"loss": avg_loss})}}

    def package(self) -> Dict[str, Any]:
        client_package = super().package()
        client_package["local_centroids"] = (
            self.model.local_centroids.weight.detach().cpu().clone()
        )
        return client_package


class OrchestraServer(FedAvgServer):
    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id)
        pkg["current_round"] = int(self.current_epoch)
        return pkg

    def train_one_round(self) -> "OrderedDict[int, Dict[str, Any]]":
        client_packages = self.trainer.train(self.selected_clients)
        self.aggregate_client_updates(client_packages)

        local_centroids = []
        for pkg in client_packages.values():
            lc = pkg.get("local_centroids", None)
            if lc is not None:
                local_centroids.append(lc.float())
        if local_centroids and self.model is not None:
            self.model.load_state_dict(self.public_model_params, strict=False)
            z = torch.cat(local_centroids, dim=0).to(self.device)
            self.model.global_clustering(
                z,
                total_rounds=int(getattr(self.args.ssl, "server_cluster_rounds", 80)),
            )
            agg_names = set(self.model.aggregated_param_names)
            self.public_model_params = OrderedDict(
                (k, p.detach().cpu().clone())
                for k, p in self.model.named_parameters()
                if k in agg_names
            )
        return client_packages


def run_ssl_baseline(
    *,
    algo: str,
    config: Dict[str, Any],
    seed: int | None,
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
    algo = str(algo).lower()
    if algo not in {"simclr", "simsiam", "byol", "orchestra"}:
        raise ValueError(f"unsupported ssl algo: {algo}")

    device = torch.device("cuda") if (to_cuda and torch.cuda.is_available()) else torch.device("cpu")
    num_clients = len(X_fed)
    dataset_obj, data_indices = _build_data_indices(X_fed, y_fed)
    seed_i = int(seed) if seed is not None else 42
    args = _build_args(
        algo=algo,
        config=config,
        batch_size=batch_size,
        lr=lr,
        wd=wd,
        num_epoch=num_epoch,
        seed=seed_i,
        dataset_name=dataset,
    )

    ssl_cfg = args.ssl

    def _mk_model():
        if algo == "orchestra":
            return OrchestraShapeletModel(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=n_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                projector_hidden_dim=int(ssl_cfg.projector_hidden_dim),
                projector_out_dim=int(ssl_cfg.projector_out_dim),
                ema_tau=float(ssl_cfg.ema_tau),
                num_global_clusters=int(ssl_cfg.num_global_clusters),
                num_local_clusters=int(ssl_cfg.num_local_clusters),
                cluster_m_size=int(ssl_cfg.cluster_m_size),
                temperature=float(ssl_cfg.temperature),
                deg_num_classes=int(ssl_cfg.deg_num_classes),
                to_cuda=to_cuda,
            )
        return ShapeletSSLModel(
            method=algo,
            shapelets_size_and_len=shapelets_size_and_len,
            in_channels=n_channels,
            num_classes=num_classes,
            dist_measure=dist_measure,
            projector_hidden_dim=int(ssl_cfg.projector_hidden_dim),
            projector_out_dim=int(ssl_cfg.projector_out_dim),
            predictor_hidden_dim=int(ssl_cfg.predictor_hidden_dim),
            to_cuda=to_cuda,
        )

    server_model = _mk_model()
    server = OrchestraServer(args) if algo == "orchestra" else FedAvgServer(args)
    server.model = server_model
    server.client_num = num_clients
    server.train_clients = list(range(num_clients))
    server.val_clients = list(range(num_clients))
    server.test_clients = list(range(num_clients))
    server.client_local_epoches = [int(num_epoch)] * num_clients

    agg_names = set(server_model.aggregated_param_names)
    server.public_model_params = OrderedDict(
        (k, p.detach().cpu().clone()) for k, p in server_model.named_parameters() if k in agg_names
    )
    server.clients_personal_model_params = {i: OrderedDict() for i in range(num_clients)}
    server.client_optimizer_states = {i: {} for i in range(num_clients)}
    server.client_lr_scheduler_states = {i: {} for i in range(num_clients)}

    def _make_optimizer_cls():
        def _cls(params):
            return torch.optim.SGD(
                params,
                lr=float(lr),
                weight_decay=float(wd),
                momentum=float(args.optimizer.momentum),
            )
        return _cls

    client_cls = {
        "simclr": SimCLRClient,
        "simsiam": SimSiamClient,
        "byol": BYOLClient,
        "orchestra": OrchestraClient,
    }[algo]
    clients = []
    for _cid in range(num_clients):
        client = client_cls(
            model=_mk_model(),
            optimizer_cls=_make_optimizer_cls(),
            lr_scheduler_cls=None,
            args=args,
            dataset=dataset_obj,
            data_indices=data_indices,
            device=device,
            return_diff=server.return_diff,
        )
        clients.append(client)

    server.trainer = SequentialTrainer(server, clients)
    best = {"acc": 0.0, "round": -1}
    y_all_np = y_all.cpu().numpy() if hasattr(y_all, "cpu") else np.asarray(y_all)
    y_test_np = y_test.cpu().numpy() if hasattr(y_test, "cpu") else np.asarray(y_test)
    y_val_np = y_val.cpu().numpy() if (has_val and y_val is not None and hasattr(y_val, "cpu")) else (
        np.asarray(y_val) if (has_val and y_val is not None) else None
    )

    from sklearn.preprocessing import RobustScaler

    original_after = server.after_round

    def _after_round(client_packages):
        original_after(client_packages)
        round_idx = int(server.current_epoch)
        client_losses = []
        for pkg in client_packages.values():
            er = pkg.get("eval_results", {}) or {}
            after = er.get("after", {}) if isinstance(er, dict) else {}
            tr = after.get("train", None) if isinstance(after, dict) else None
            if tr is not None and hasattr(tr, "loss"):
                client_losses.append(float(tr.loss))
        avg_loss = float(np.mean(client_losses)) if client_losses else float("nan")

        t_eval = time.time()
        try:
            transformation = server_model.transform(
                X_all, batch_size=int(batch_size), normalize=True, result_type="numpy"
            )
            transformation_test = server_model.transform(
                X_test, batch_size=int(batch_size), normalize=True, result_type="numpy"
            )
            scaler = RobustScaler()
            transformation = scaler.fit_transform(transformation)
            transformation_test = scaler.transform(transformation_test)
            if has_val and X_val is not None and y_val_np is not None:
                transformation_val = server_model.transform(
                    X_val, batch_size=int(batch_size), normalize=True, result_type="numpy"
                )
                transformation_val = scaler.transform(transformation_val)
                train_acc, test_acc = eval_tstcc_fn(
                    transformation_train=transformation,
                    transformation_test=transformation_test,
                    transformation_val=transformation_val,
                    y_train=y_all_np,
                    y_test=y_test_np,
                    y_val=y_val_np,
                )
            else:
                train_acc, test_acc = eval_train_test_fn(
                    transformation, transformation_test, y_train=y_all_np, y_test=y_test_np
                )
        except Exception as e:
            print(f"[{algo.upper()}] round {round_idx + 1} eval failed: {type(e).__name__}: {e}", flush=True)
            train_acc, test_acc = float("nan"), float("nan")
        eval_dt = time.time() - t_eval

        if np.isfinite(test_acc) and test_acc > best["acc"]:
            best["acc"] = float(test_acc)
            best["round"] = round_idx

        print(
            f"[{algo.upper()}] round {round_idx + 1}/{int(num_rounds)} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"best={best['acc']:.4f}@r{best['round'] + 1 if best['round'] >= 0 else -1} "
            f"avg_loss={avg_loss} eval={eval_dt:.1f}s",
            flush=True,
        )
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            loss_str = str(avg_loss) if np.isfinite(avg_loss) else "nan"
            f.write(
                f"dataset: {dataset}round:{round_idx} server aggregation  "
                f"testACC:{test_acc} trainACC:{train_acc} avg_loss:{loss_str}\n"
            )

    server.after_round = _after_round

    t0 = time.time()
    server.train(int(num_rounds))
    dt = time.time() - t0
    print(
        f"[{algo.upper()}] training done in {dt:.1f}s, "
        f"best round={best['round'] + 1 if best['round'] >= 0 else -1} best_acc={best['acc']:.4f}",
        flush=True,
    )

    final_state = {k: v.detach().cpu().clone() for k, v in server_model.state_dict().items()}
    save_model_fn(final_state, dataset, formatted_date)
