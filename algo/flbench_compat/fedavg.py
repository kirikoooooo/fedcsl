"""FL-bench ``FedAvgClient`` / ``FedAvgServer`` 的精简移植。

忠实还原 FL-bench ``src/client/fedavg.py`` 与 ``src/server/fedavg.py`` 中 SCAFFOLD /
FedProto 等方法实际依赖的接口与数据流；未用到的部分（Ray 并行、visdom/tb 监控、
DP、FedBabu 等）一概省略。

API 对照表（左：FL-bench 原版；右：此处实现）：

| FedAvgClient.__init__(**commons)                | FedAvgClient.__init__(**commons)              |
| FedAvgClient.set_parameters(package)            | 保持签名；加载 regular/personal 参数          |
| FedAvgClient.train(server_package)              | set_parameters + train_with_eval + package   |
| FedAvgClient.train_with_eval()                  | 调 fit()，不做子集评估（交给 runner 全局评估）|
| FedAvgClient.package()                          | 返回 weight / regular_model_params / ...      |
| FedAvgClient.fit()                              | 默认 FedAvg: SGD 遍历 trainloader              |
| FedAvgServer.__init__(args)                     | 由 runner 传入 model+data+clients 后续填充    |
| FedAvgServer.package(client_id)                 | 一致                                         |
| FedAvgServer.aggregate_client_updates(packages) | 按 ``package[\"weight\"]`` 加权聚合，支持 diff |
| FedAvgServer.train()                            | 主循环（无 Ray）；train_one_round 交由子类    |

一切与 FL-bench 逻辑等价，仅去掉对 Hydra/Ray/BaseDataset/DecoupledModel 的耦合。
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class AttrDict(dict):
    """支持点号访问的 dict，递归包装子 dict。

    与 OmegaConf DictConfig 的 ``cfg.foo.bar`` 访问兼容（SCAFFOLD/FedProto 原文件的使用面）。
    """

    __slots__ = ()

    def __getattr__(self, key: str):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            v = AttrDict(v)
            self[key] = v
        return v

    def __setattr__(self, key: str, value) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return AttrDict({k: deepcopy(v, memo) for k, v in self.items()})


class FedAvgClient:
    """精简版 ``FedAvgClient``：承载一个数据分片 + 一个本地模型副本。"""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer_cls,
        lr_scheduler_cls,
        args: AttrDict,
        dataset,
        data_indices: List[dict],
        device: Optional[torch.device] = None,
        return_diff: bool = False,
    ) -> None:
        self.client_id: Optional[int] = None
        self.args = args
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dataset = dataset
        self.model = model.to(self.device)

        # 与 FL-bench 一致：区分"regular"（参与聚合）与"personal"（不参与）参数名
        self.regular_params_name: List[str] = [k for k, _ in self.model.named_parameters()]
        self.personal_params_name: List[str] = []

        # 优化器 / lr_scheduler 的"原始"状态，用于每轮重置（除非 reset_optimizer_on_global_epoch=False）
        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.lr_scheduler = None
        self.init_lr_scheduler_state = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler = lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # data_indices: [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Subset 需要非空 indices 以兼容 shuffle=True（FL-bench 同样做法）
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        bs = int(args.common.batch_size)
        self.trainloader = DataLoader(
            self.trainset, batch_size=bs, shuffle=True, drop_last=False
        )
        self.valloader = DataLoader(self.valset, batch_size=bs)
        self.testloader = DataLoader(self.testset, batch_size=bs)

        self.local_epoch = int(args.common.local_epoch)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.return_diff = return_diff

        # 与 FL-bench 一致：return_diff 需要在 set_parameters 里记录"入站"参数快照
        self.regular_model_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.eval_results: dict = {}

    # ------------------------------------------------------------------
    # 与 FL-bench 同名的生命周期方法
    # ------------------------------------------------------------------
    def load_data_indices(self) -> None:
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def set_parameters(self, package: Dict[str, Any]) -> None:
        self.client_id = int(package["client_id"])
        self.local_epoch = int(package["local_epoch"])
        self.load_data_indices()

        if (
            package.get("optimizer_state")
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package.get("lr_scheduler_state"):
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        if package.get("regular_model_params"):
            self.model.load_state_dict(package["regular_model_params"], strict=False)
        if package.get("personal_model_params"):
            self.model.load_state_dict(package["personal_model_params"], strict=False)

        if self.return_diff:
            state = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (k, state[k].detach().cpu().clone()) for k in self.regular_params_name
            )

    def train_with_eval(self) -> None:
        """省略子集评估——SCAFFOLD/FedProto 只需要 fit()。"""
        if self.local_epoch > 0:
            self.fit()

    def train(self, server_package: Dict[str, Any]) -> Dict[str, Any]:
        self.set_parameters(server_package)
        self.train_with_eval()
        return self.package()

    def package(self) -> Dict[str, Any]:
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params=OrderedDict(
                (k, model_params[k].detach().cpu().clone()) for k in self.regular_params_name
            ),
            personal_model_params=OrderedDict(
                (k, model_params[k].detach().cpu().clone()) for k in self.personal_params_name
            ),
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {} if self.lr_scheduler is None else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                k: old - new
                for (k, new), old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package

    # ------------------------------------------------------------------
    # 默认 FedAvg 本地训练；子类（SCAFFOLDClient / FedProtoClient）会覆盖
    # ------------------------------------------------------------------
    def fit(self) -> None:
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).long()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


class FedAvgServer:
    """精简版 ``FedAvgServer``：对外提供 ``public_model_params`` 与主训练循环。"""

    algorithm_name: str = "FedAvg"
    all_model_params_personalized: bool = False
    return_diff: bool = False
    client_cls = FedAvgClient

    def __init__(self, args: AttrDict) -> None:
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.current_epoch: int = 0
        # 下面这些字段由 runner 在构造完 server 后显式注入：
        self.model: Optional[nn.Module] = None
        self.public_model_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.clients_personal_model_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_optimizer_states: Dict[int, dict] = {}
        self.client_lr_scheduler_states: Dict[int, dict] = {}
        self.client_local_epoches: List[int] = []
        self.client_num: int = 0
        self.train_clients: List[int] = []
        self.val_clients: List[int] = []
        self.test_clients: List[int] = []
        self.selected_clients: List[int] = []
        # 训练器（sequential）：由 runner 注入
        self.trainer = None

    # ------------------------------------------------------------------
    # 与 FL-bench FedAvgServer.package / get_client_model_params 对齐
    # ------------------------------------------------------------------
    def get_client_model_params(self, client_id: int) -> Dict[str, "OrderedDict[str, torch.Tensor]"]:
        regular_params = deepcopy(self.public_model_params)
        personal_params = self.clients_personal_model_params.get(client_id, OrderedDict())
        return dict(regular_model_params=regular_params, personal_model_params=personal_params)

    def package(self, client_id: int) -> Dict[str, Any]:
        local_epoch = (
            self.client_local_epoches[client_id]
            if client_id < len(self.client_local_epoches)
            else int(self.args.common.local_epoch)
        )
        return dict(
            client_id=client_id,
            local_epoch=local_epoch,
            **self.get_client_model_params(client_id),
            optimizer_state=self.client_optimizer_states.get(client_id, {}),
            lr_scheduler_state=self.client_lr_scheduler_states.get(client_id, {}),
            return_diff=self.return_diff,
        )

    # ------------------------------------------------------------------
    # FedAvg 默认聚合
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: "OrderedDict[int, Dict[str, Any]]") -> None:
        weights = [float(pkg["weight"]) for pkg in client_packages.values()]
        total = sum(weights) or 1.0
        w_t = torch.tensor([w / total for w in weights])

        if self.return_diff:
            for name in self.public_model_params.keys():
                diffs = torch.stack(
                    [pkg["model_params_diff"][name] for pkg in client_packages.values()],
                    dim=-1,
                ).float()
                aggregated = torch.sum(diffs * w_t, dim=-1)
                self.public_model_params[name].data -= aggregated.to(self.public_model_params[name].dtype)
        else:
            for name in self.public_model_params.keys():
                stacked = torch.stack(
                    [pkg["regular_model_params"][name] for pkg in client_packages.values()],
                    dim=-1,
                ).float()
                aggregated = torch.sum(stacked * w_t, dim=-1)
                self.public_model_params[name].data = aggregated.to(self.public_model_params[name].dtype)

        # 缓存每客户端 optimizer / lr 状态，以便下轮恢复
        for cid, pkg in client_packages.items():
            self.client_optimizer_states[cid] = pkg.get("optimizer_state", {})
            self.client_lr_scheduler_states[cid] = pkg.get("lr_scheduler_state", {})

    # ------------------------------------------------------------------
    # 训练主循环（简化版；不含 Ray / 监控 / straggler）
    # ------------------------------------------------------------------
    def train_one_round(self) -> "OrderedDict[int, Dict[str, Any]]":
        """默认行为：各客户端顺序 train，再用 ``aggregate_client_updates`` 聚合。

        子类可覆盖（如 FedProto 在此处替换为"原型聚合"）。
        """
        client_packages = self.trainer.train(self.selected_clients)
        self.aggregate_client_updates(client_packages)
        return client_packages

    def select_clients(self) -> List[int]:
        join_ratio = float(getattr(self.args.common, "join_ratio", 1.0) or 1.0)
        if join_ratio >= 1.0 - 1e-6:
            return list(self.train_clients)
        k = max(1, int(round(self.client_num * join_ratio)))
        rng = np.random.default_rng(int(self.args.common.seed) + self.current_epoch)
        picked = rng.choice(self.train_clients, size=k, replace=False)
        return sorted(int(i) for i in picked)

    def after_round(self, client_packages: "OrderedDict[int, Dict[str, Any]]") -> None:
        """Hook：运行完一轮之后做些事（runner 里覆盖来做全局 SVC 评估 + 日志）。"""

    def train(self, num_rounds: int) -> None:
        for E in range(num_rounds):
            self.current_epoch = E
            self.selected_clients = self.select_clients()
            client_packages = self.train_one_round()
            # 让 model 指向最新 public params（用于全局评估）
            if self.model is not None:
                self.model.load_state_dict(self.public_model_params, strict=False)
            self.after_round(client_packages)
