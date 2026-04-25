from __future__ import annotations

"""FedProto 服务端：移植自 FL-bench ``src/server/fedproto.py``。

相对 FL-bench 原版的改动：
- ``from src.utils.constants import NUM_CLASSES`` → 从 ``args.dataset.num_classes`` 动态读取；
- ``from src.server.fedavg import FedAvgServer`` → ``flbench_compat``；
- 其余逻辑一致：
    * ``return_diff = False``；模型参数按 FedAvg 聚合。
    * 每类原型按"贡献客户端数"求平均（未出现该类的客户端不计入）。
    * ``train_one_round`` 不调用 FedAvg 聚合，用自定义 ``aggregate_prototypes``
      同时保留了 FedAvg 的模型参数聚合（这里显式补上，以便全局模型可被评估）。
"""
from argparse import ArgumentParser, Namespace
from typing import Any

import torch

from ..flbench_compat import FedAvgServer, AttrDict
from .client import FedProtoClient


class FedProtoServer(FedAvgServer):
    algorithm_name: str = "FedProto"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False
    client_cls = FedProtoClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--lamda", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(self, args: AttrDict):
        super().__init__(args)
        self.global_prototypes: dict[int, torch.Tensor] = {}
        self.num_classes: int = int(args.dataset.num_classes)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["global_prototypes"] = self.global_prototypes
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train(self.selected_clients)
        # 原型聚合（FL-bench 原版仅做这一步；此处额外补上 FedAvg 式的模型参数聚合，
        # 以便外层 runner 能拿到一个可评估的全局模型）。
        self.aggregate_prototypes(
            [package["prototypes"] for package in client_packages.values()]
        )
        self.aggregate_client_updates(client_packages)
        return client_packages

    def aggregate_prototypes(
        self, client_prototypes_list: list[dict[int, torch.Tensor]]
    ):
        self.global_prototypes = {}
        # FL-bench 用 NUM_CLASSES[dataset_name]；此处改用 self.num_classes
        feat_dim = int(self.model.classifier.in_features) if self.model is not None else None
        for i in range(self.num_classes):
            size = 0
            if feat_dim is not None:
                prototypes = torch.zeros(feat_dim)
            else:
                prototypes = None
            for client_prototypes in client_prototypes_list:
                if i in client_prototypes.keys():
                    if prototypes is None:
                        prototypes = torch.zeros_like(client_prototypes[i])
                    prototypes += client_prototypes[i]
                    size += 1

            if size > 0 and prototypes is not None:
                self.global_prototypes[i] = prototypes / size
