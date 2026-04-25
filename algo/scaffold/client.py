from __future__ import annotations

"""SCAFFOLD 客户端：移植自 FL-bench ``src/client/scaffold.py``。

与原版的差异：
1. ``from src.client.fedavg import FedAvgClient`` → ``flbench_compat``；
2. ``fit()`` 按 "epoch" 语义训练（与本仓 FedCSL 对齐）：
   - 原 FL-bench ``fit()`` 以 ``local_epoch`` 为 **SGD 步数**（每轮一次 ``get_data_batch``）；
   - 本实现改为每个 epoch 遍历 ``trainloader`` 一遍，总步数 = ``local_epoch * num_batches``；
   - 因此 SCAFFOLD Option II 的系数 ``coef = 1 / (total_steps * lr)``（FL-bench 用的是
     ``1 / (local_epoch * lr)``，此处自洽修改，以匹配 epoch 语义下的真实 SGD 步数）。
3. 其它所有变量命名与更新规则完全对齐 FL-bench：
   - ``y_delta = y_i - x``
   - ``c_plus = c_i - c - coef * y_delta``
   - ``c_delta = c_plus - c_local``
"""
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from ..flbench_compat import FedAvgClient


class SCAFFOLDClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader: Iterator[DataLoader]
        self.c_local: list[torch.Tensor]
        self.c_global: list[torch.Tensor]
        self.y_delta: list[torch.Tensor]
        self.c_delta: list[torch.Tensor]
        self.c_plus: list[torch.Tensor] = []
        self._steps_taken: int = 0

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.iter_trainloader = iter(self.trainloader)
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]

    def train(self, server_package: dict[str, Any]):
        # 记录起点 x 供后面计算 y_delta；FL-bench 的 FedAvgClient.set_parameters
        # 在 return_diff=True 时自动维护 self.regular_model_params 做相同的事。
        self.set_parameters(server_package)
        self.train_with_eval()

        with torch.no_grad():
            self.y_delta = []
            self.c_plus = []
            self.c_delta = []

            model_params = self.model.state_dict()
            # 注意：FL-bench 原版假定 server_package["regular_model_params"] 不被 return_diff
            # 逻辑替换；本仓 FedAvgClient 在 return_diff=True 下会弹出 regular_model_params
            # 并塞入 model_params_diff。为保留对 x 的引用，SCAFFOLD 额外从 regular_model_params
            # （未弹出版本，即 set_parameters 中缓存的）读取。
            x_state = self.regular_model_params
            for key in x_state.keys():
                x, y_i = x_state[key], model_params[key]
                self.y_delta.append(y_i.detach().cpu() - x)

            # Option II: c_plus = c_i - c - (y_i - x) / (K * lr) = c_i - c - coef * y_delta
            total_steps = max(1, int(self._steps_taken))
            coef = 1.0 / (total_steps * float(self.args.optimizer.lr))
            for c, c_i, y_del in zip(self.c_global, self.c_local, self.y_delta):
                self.c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(self.c_plus, self.c_local):
                self.c_delta.append(c_p - c_l)

            # 持久化：server 端 aggregate_client_updates 会从 package 取回 c_plus 写入 c_local
            self.c_local = self.c_plus

        return self.package()

    def package(self):
        client_package = super().package()
        client_package["c_delta"] = self.c_delta
        client_package["y_delta"] = self.y_delta
        client_package["c_plus"] = self.c_plus
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        self._steps_taken = 0
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device).long()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                for param, c, c_i in zip(
                    self.model.parameters(), self.c_global, self.c_local
                ):
                    if param.requires_grad and param.grad is not None:
                        param.grad.data += (c - c_i).to(self.device)
                self.optimizer.step()
                self._steps_taken += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
