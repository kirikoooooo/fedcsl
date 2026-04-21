# -*- coding: utf-8 -*-
"""OMP 自适应客户端选择：用 OMP 稀疏系数向量更新各客户端采样概率。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from fedutil import (
    get_sampling_probs_from_omp,
    omp_from_state_dicts,
    sample_clients_mask_by_probability,
)

from .base import ClientSelector


class OMPSelector(ClientSelector):
    name = "omp"

    def __init__(
        self,
        *,
        num_clients: int,
        sample_nums: int,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        min_selection_prob: float = 0.01,
        ema_alpha: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_clients=num_clients,
            sample_nums=sample_nums,
            seed=seed,
            config=config,
            **kwargs,
        )
        self.min_selection_prob = float(min_selection_prob)
        self.ema_alpha = float(ema_alpha)
        self.probs: List[float] = [1.0 / num_clients] * num_clients
        self.last_sparse_vec = None

    def on_round_start(
        self,
        round_idx: int,
        *,
        client_losses: Optional[Sequence[float]] = None,
        **ctx: Any,
    ) -> List[float]:
        if round_idx == 0:
            self.probs = [1.0 / self.num_clients] * self.num_clients
            return [1.0] * self.num_clients
        mask = sample_clients_mask_by_probability(self.probs, self.sample_nums, seed=None)
        return [float(x) for x in mask]

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
        sparse_vec = omp_from_state_dicts(w_locals, w_global, self.sample_nums)
        self.last_sparse_vec = sparse_vec
        self.probs = list(
            get_sampling_probs_from_omp(
                sparse_vec,
                prev_probs=self.probs,
                selection_mask=select_mask,
                min_selection_prob=self.min_selection_prob,
                ema_alpha=self.ema_alpha,
            )
        )

    def get_probs(self) -> List[float]:
        return list(self.probs)
