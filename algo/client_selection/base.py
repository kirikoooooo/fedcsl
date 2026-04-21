# -*- coding: utf-8 -*-
"""
客户端选择策略统一接口。

所有具体策略（Uniform / OMP / Oort / FedCS）实现 ``ClientSelector`` 抽象基类，
由 ``FedCSL_All.py`` 的训练主循环通过统一协议调用：

    selector = make_selector(method, num_clients=..., sample_nums=..., ...)
    for round_idx in range(num_rounds):
        ...  # 各 client 本地训练
        select_mask = selector.on_round_start(round_idx, client_losses=..., y_fed=..., X_all_size=...)
        w_global = selector.aggregate(w_locals, y_fed, scores, select_mask)
        if w_global is None:                       # 未接管聚合，外部走默认 fedavg
            w_global = fedavg(w_locals, y_fed, [s*m for s,m in zip(scores, select_mask)])
        selector.on_round_end(round_idx,
                              w_locals=w_locals, w_global=w_global,
                              select_mask=select_mask, client_losses=client_losses)

基类对 "第一轮全选" 与 "无采样"（use_client_selection=False）都给出默认实现，
具体策略只需重写必要方法。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class ClientSelector:
    """客户端选择器基类。"""

    name: str = "base"

    def __init__(
        self,
        *,
        num_clients: int,
        sample_nums: int,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.num_clients = int(num_clients)
        self.sample_nums = max(1, int(sample_nums))
        self.seed = seed
        self.config = config or {}

    # ------------------------------------------------------------------
    # round 接口
    # ------------------------------------------------------------------
    def on_round_start(
        self,
        round_idx: int,
        *,
        client_losses: Optional[Sequence[float]] = None,
        **ctx: Any,
    ) -> List[float]:
        """返回长度为 num_clients 的 ``select_mask``（0/1 浮点）。

        默认行为：第 0 轮全选，其后全选——子类通常需要重写。
        """
        return [1.0] * self.num_clients

    def aggregate(
        self,
        w_locals: Sequence[Dict[str, Any]],
        y_fed: Sequence[Any],
        scores: Sequence[float],
        select_mask: Sequence[float],
        **ctx: Any,
    ) -> Optional[Dict[str, Any]]:
        """可选：策略自行聚合并返回 ``state_dict``；返回 ``None`` 表示让外部走默认 FedAvg。"""
        return None

    def on_round_end(
        self,
        round_idx: int,
        *,
        w_locals: Sequence[Dict[str, Any]],
        w_global: Dict[str, Any],
        select_mask: Sequence[float],
        client_losses: Sequence[float],
        **ctx: Any,
    ) -> None:
        """每轮结束后的状态更新（概率、reward、G/t 等），默认为空操作。"""
        return None

    # ------------------------------------------------------------------
    # 描述信息
    # ------------------------------------------------------------------
    def describe(self) -> str:
        return f"[{self.name}] num_clients={self.num_clients} sample_nums={self.sample_nums}"
