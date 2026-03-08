# -*- coding: utf-8 -*-
"""
FedCS 自适应客户端选择器（基于论文 Adaptive Client Sampling）。
实现 Corollary 2 闭式解：q_i ∝ p_i * G_i / sqrt(t_i)，以及聚合权重 p_i / (K*q_i)。
"""
import numpy as np


class FedCSClientSelector:
    """
    FedCS 客户端选择器：按 q 采样，聚合时使用逆概率权重 p_i/(K*q_i) 保证无偏。
    输入：
        N: 客户端总数
        K: 每轮采样数
        p: 数据权重 [p1,...,pN]，p_i = n_i / n_tot，sum(p)=1
        t: 单轮耗时 [t1,...,tN]，可选，默认全 1
        G: 梯度范数界 [G1,...,GN]，可选，默认全 1，可随轮次更新
    """

    def __init__(self, N, K, p, t=None, G=None):
        self.N = N
        self.K = K
        self.p = np.asarray(p, dtype=np.float64)
        self.p = self.p / self.p.sum()
        self.t = np.ones(N, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
        self.G = np.ones(N, dtype=np.float64) if G is None else np.asarray(G, dtype=np.float64)
        self._update_q()

    def _update_q(self):
        """闭式解 (Corollary 2)：q_i ∝ p_i * G_i / sqrt(t_i)。"""
        t_safe = np.maximum(self.t, 1e-8)
        raw = self.p * self.G / np.sqrt(t_safe)
        raw = np.maximum(raw, 1e-10)
        self.q = raw / raw.sum()

    def update_gradient_norm(self, client_id, g_norm, ema_alpha=0.3):
        """用梯度范数（或 loss/score 代理）更新 G_i，可选 EMA 平滑。"""
        g = max(float(g_norm), 1e-10)
        self.G[client_id] = (1 - ema_alpha) * self.G[client_id] + ema_alpha * g
        self._update_q()

    def update_client_time(self, client_id, duration, ema_alpha=0.3):
        """更新客户端 i 的耗时估计 t_i。"""
        d = max(float(duration), 1e-8)
        self.t[client_id] = (1 - ema_alpha) * self.t[client_id] + ema_alpha * d
        self._update_q()

    def get_sampling_probs(self):
        """返回当前采样概率 q，sum(q)=1。"""
        return self.q.copy()

    def get_aggregation_weights(self, select_mask):
        """
        根据本轮选中结果返回聚合权重。被选中的客户端 i 权重为 p_i/(K*q_i)，再归一化使和为 1。
        select_mask: List[float] 或 ndarray，长度 N，选中为非 0，未选中为 0。
        """
        select_mask = np.asarray(select_mask, dtype=np.float64)
        q_safe = np.maximum(self.q, 1e-10)
        # 仅被选中的客户端有权重 p_i / (K * q_i)
        w = np.where(select_mask != 0, self.p / (self.K * q_safe), 0.0)
        s = w.sum()
        if s <= 0:
            return (self.p * select_mask).astype(np.float64)  # fallback: 按 p 与 mask
        w = w / s
        return w


def create_fedcs_selector(num_clients, sample_nums, data_weights, client_times=None,
                         gradient_norms=None, config=None):
    """
    工厂函数：从主代码传入的参数创建 FedCS 选择器。
    data_weights: list/array，各客户端数据量占比，若为 None 则用均匀权重。
    """
    N = num_clients
    K = max(1, sample_nums)
    if data_weights is None or len(data_weights) != N:
        p = np.ones(N) / N
    else:
        p = np.asarray(data_weights, dtype=np.float64)
        p = p / p.sum()
    t = client_times
    G = gradient_norms
    return FedCSClientSelector(N=N, K=K, p=p, t=t, G=G)
