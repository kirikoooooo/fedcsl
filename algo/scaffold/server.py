"""SCAFFOLD 服务端：移植自 FL-bench ``src/server/scaffold.py``。

与原版的唯一差异：
- ``from src.client.scaffold import SCAFFOLDClient``  →  本地路径
- ``from src.server.fedavg import FedAvgServer``       →  ``flbench_compat``
- 类型注解/装饰器/算法步骤完全一致。
"""
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any

import torch

from ..flbench_compat import FedAvgServer, AttrDict
from .client import SCAFFOLDClient


class SCAFFOLDServer(FedAvgServer):
    algorithm_name: str = "SCAFFOLD"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = SCAFFOLDClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--global_lr", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(self, args: AttrDict):
        super().__init__(args)
        # c_global / c_local 在 runner 把 public_model_params 注入后才能初始化；
        # 这里推迟到 setup_control_variates() 由 runner 调用。
        self.c_global: list[torch.Tensor] = []
        self.c_local: list[list[torch.Tensor]] = []

    def setup_control_variates(self) -> None:
        """public_model_params / train_clients 注入完成后调用，初始化 c_global / c_local。"""
        self.c_global = [
            torch.zeros_like(param) for param in self.public_model_params.values()
        ]
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]
        return server_package

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: dict[int, dict[str, Any]]):
        c_delta_list = [package["c_delta"] for package in client_packages.values()]
        y_delta_list = [package["y_delta"] for package in client_packages.values()]
        weights = torch.ones(len(y_delta_list)) / len(y_delta_list)
        for param, y_delta in zip(
            self.public_model_params.values(), zip(*y_delta_list)
        ):
            param.data += self.args.scaffold.global_lr * torch.sum(
                torch.stack(y_delta, dim=-1) * weights, dim=-1
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_list)):
            c_global.data += torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num

        # 把每客户端 c_plus 写回 c_local（由 Client 在 package 里顺带返回）
        for cid, pkg in client_packages.items():
            if "c_plus" in pkg:
                self.c_local[cid] = pkg["c_plus"]

        # 保留 FedAvg 基类的 optimizer/lr 状态缓存逻辑
        for cid, pkg in client_packages.items():
            self.client_optimizer_states[cid] = pkg.get("optimizer_state", {})
            self.client_lr_scheduler_states[cid] = pkg.get("lr_scheduler_state", {})
