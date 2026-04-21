# -*- coding: utf-8 -*-
"""均匀采样客户端选择：所有客户端等概率被选中。"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from fedutil import sample_clients_mask_by_probability

from .base import ClientSelector


class UniformSelector(ClientSelector):
    name = "uniform"

    def on_round_start(
        self,
        round_idx: int,
        *,
        client_losses: Optional[Sequence[float]] = None,
        **ctx: Any,
    ) -> List[float]:
        if round_idx == 0:
            # 首轮全选，稳定初始状态
            return [1.0] * self.num_clients
        probs = [1.0 / self.num_clients] * self.num_clients
        mask = sample_clients_mask_by_probability(probs, self.sample_nums, seed=None)
        return [float(x) for x in mask]
