"""顺序训练器：对应 FL-bench ``src/utils/trainer.py`` 中非并行分支的行为。

FL-bench 的 ``FLbenchTrainer`` 把多个客户端按 Ray / 线程并行起来。本仓库不需要并行，
保留顺序行为即可；接口上暴露 ``train(selected_clients) -> OrderedDict[cid, package]``
与 ``test`` 占位方法，使 ``scaffold.py`` / ``fedproto.py`` 完全不修改仍可工作。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Sequence


class SequentialTrainer:
    """顺序执行客户端训练，保持与 FL-bench 相同的 ``client_packages`` 返回结构。"""

    def __init__(self, server, clients: Sequence) -> None:
        self.server = server
        self.clients = list(clients)

    def train(self, selected_clients: Iterable[int] | None = None) -> "OrderedDict[int, Dict[str, Any]]":
        if selected_clients is None:
            selected_clients = self.server.selected_clients
        out: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        for cid in selected_clients:
            server_package = self.server.package(cid)
            client_package = self.clients[cid].train(server_package)
            out[int(cid)] = client_package
            # 与 FL-bench FLbenchTrainer 同步行为：把 client 的 personal/optimizer/lr 状态
            # 回写 server，保证下一轮 server.package(cid) 能下发正确的个性化参数。
            if hasattr(self.server, "clients_personal_model_params"):
                self.server.clients_personal_model_params.setdefault(int(cid), OrderedDict()).update(
                    client_package.get("personal_model_params", {}) or {}
                )
            if hasattr(self.server, "client_optimizer_states"):
                self.server.client_optimizer_states[int(cid)] = client_package.get("optimizer_state", {})
            if hasattr(self.server, "client_lr_scheduler_states"):
                self.server.client_lr_scheduler_states[int(cid)] = client_package.get("lr_scheduler_state", {})
        return out

    def test(self, clients: Iterable[int] | None = None, results: dict | None = None) -> None:  # pragma: no cover
        # 未使用：FedCSL 的全局 SVC 评估由 runner 统一处理，不走 FL-bench 子集评估路径。
        return None
