# -*- coding: utf-8 -*-
"""FedCS 自适应客户端选择器 (Adaptive Client Sampling, Corollary 2)。

迁移自根目录 ``fedcs_selector.py``，根目录文件保留作为向后兼容 shim。
``FedCSSelector`` 是统一接口 ``ClientSelector`` 的适配器，内部复用
``FedCSClientSelector`` 的 q 更新与逆概率权重计算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fedavg import fedavg_fedcs
from fedutil import sample_clients_mask_by_probability

from .base import ClientSelector


class FedCSClientSelector:
    """闭式解：q_i ∝ p_i * G_i / sqrt(t_i)；聚合权重 p_i / (K*q_i)。"""

    def __init__(self, N, K, p, t=None, G=None):
        self.N = N
        self.K = K
        self.p = np.asarray(p, dtype=np.float64)
        self.p = self.p / self.p.sum()
        self.t = np.ones(N, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
        self.G = np.ones(N, dtype=np.float64) if G is None else np.asarray(G, dtype=np.float64)
        self._update_q()

    def _update_q(self):
        t_safe = np.maximum(self.t, 1e-8)
        raw = self.p * self.G / np.sqrt(t_safe)
        raw = np.maximum(raw, 1e-10)
        self.q = raw / raw.sum()

    def update_gradient_norm(self, client_id, g_norm, ema_alpha=0.3):
        g = max(float(g_norm), 1e-10)
        self.G[client_id] = (1 - ema_alpha) * self.G[client_id] + ema_alpha * g
        self._update_q()

    def update_client_time(self, client_id, duration, ema_alpha=0.3):
        d = max(float(duration), 1e-8)
        self.t[client_id] = (1 - ema_alpha) * self.t[client_id] + ema_alpha * d
        self._update_q()

    def get_sampling_probs(self):
        return self.q.copy()

    def get_aggregation_weights(self, select_mask):
        select_mask = np.asarray(select_mask, dtype=np.float64)
        n_selected = (select_mask != 0).sum()
        if n_selected >= self.N:
            return self.p.copy()
        q_safe = np.maximum(self.q, 1e-10)
        w = np.where(select_mask != 0, self.p / (self.K * q_safe), 0.0)
        s = w.sum()
        if s <= 0:
            return (self.p * select_mask).astype(np.float64)
        w = w / s
        return w


def create_fedcs_selector(
    num_clients, sample_nums, data_weights, client_times=None, gradient_norms=None, config=None
) -> FedCSClientSelector:
    N = num_clients
    K = max(1, sample_nums)
    if data_weights is None or len(data_weights) != N:
        p = np.ones(N) / N
    else:
        p = np.asarray(data_weights, dtype=np.float64)
        p = p / p.sum()
    return FedCSClientSelector(N=N, K=K, p=p, t=client_times, G=gradient_norms)


class FedCSSelector(ClientSelector):
    """FedCS 适配器：按 q 采样，并接管聚合（逆概率加权 FedAvg）。"""

    name = "fedcs"

    def __init__(
        self,
        *,
        num_clients: int,
        sample_nums: int,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        y_fed: Optional[Sequence[Any]] = None,
        y_all_size: Optional[int] = None,
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
        self.ema_alpha = float(ema_alpha)
        if y_fed is not None and y_all_size:
            data_weights = [len(y_fed[i]) / y_all_size for i in range(num_clients)]
        else:
            data_weights = None
        self.inner = create_fedcs_selector(
            num_clients=num_clients,
            sample_nums=sample_nums,
            data_weights=data_weights,
            config=config,
        )

    def on_round_start(
        self,
        round_idx: int,
        *,
        client_losses: Optional[Sequence[float]] = None,
        **ctx: Any,
    ) -> List[float]:
        if round_idx == 0:
            return [1.0] * self.num_clients
        q = self.inner.get_sampling_probs()
        q_list = q.tolist() if hasattr(q, "tolist") else list(q)
        mask = sample_clients_mask_by_probability(q_list, self.sample_nums, seed=None)
        return [float(x) for x in mask]

    def aggregate(
        self,
        w_locals: Sequence[Dict[str, Any]],
        y_fed: Sequence[Any],
        scores: Sequence[float],
        select_mask: Sequence[float],
        **ctx: Any,
    ):
        agg = self.inner.get_aggregation_weights(select_mask)
        agg_list = agg.tolist() if hasattr(agg, "tolist") else list(agg)
        return fedavg_fedcs(w_locals, y_fed, agg_list)

    def on_round_end(
        self,
        round_idx: int,
        *,
        w_locals,
        w_global,
        select_mask: Sequence[float],
        client_losses: Sequence[float],
        **ctx: Any,
    ) -> None:
        for idx in range(self.num_clients):
            loss = client_losses[idx] if idx < len(client_losses) else 0.0
            g_proxy = 1.0 / (1.0 + float(loss)) if loss else 1.0
            self.inner.update_gradient_norm(idx, g_proxy, ema_alpha=self.ema_alpha)
