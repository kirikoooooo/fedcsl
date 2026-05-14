"""Utilities for the FedU2 self-supervised baseline.

This module implements two lightweight pieces adapted for the current project:

- FUR: client-side flexible uniform regularization via projected-gradient UOT.
- EUA: server-side efficient unified aggregation via min-norm client deviation search.

The implementation is intentionally dependency-light so it can reuse the current
shapelet SSL pipeline without pulling in the original FedU2 toolbox stack.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class MinNormSolver:
    """Minimum-norm solver used by FedU2's EUA aggregation."""

    MAX_ITER = 20
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2.0 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        dmin = np.inf
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = float(np.dot(vecs[i], vecs[j]))
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = float(np.dot(vecs[i], vecs[i]))
                if (j, j) not in dps:
                    dps[(j, j)] = float(np.dot(vecs[j], vecs[j]))
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)],
                    dps[(i, j)],
                    dps[(j, j)],
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1.0) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape, dtype=y.dtype))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / proj_grad[proj_grad > 0]

        t = 1.0
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        return MinNormSolver._projection2simplex(next_point)

    @staticmethod
    def find_min_norm_element(vecs, sample_weights=None):
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n, dtype=np.float32)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1.0 - init_sol[1]
        if sample_weights is not None:
            sol_vec = np.asarray(sample_weights, dtype=np.float32)

        if n < 3:
            return sol_vec, init_sol[2]

        grad_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        for _ in range(MinNormSolver.MAX_ITER):
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1.0 - nc) * new_point
            if np.sum(np.abs(new_sol_vec - sol_vec)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

        return sol_vec, nd


def sharpen_weights(weights: torch.Tensor, temperature: float) -> torch.Tensor:
    weights = weights.float().clamp(min=0.0)
    denom = float(weights.sum())
    if denom <= 0:
        return torch.ones_like(weights) / max(1, weights.numel())
    weights = weights / denom
    temperature = float(temperature)
    if temperature <= 0:
        return weights
    sharp = weights.pow(1.0 / max(temperature, 1e-6))
    return sharp / sharp.sum().clamp(min=1e-12)


def _sample_spherical_gaussian(batch_size: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    samples = torch.randn(batch_size, dim, device=device, dtype=dtype)
    return F.normalize(samples, dim=1)


def fur_uot_loss(
    z: torch.Tensor,
    *,
    tau_a: float = 0.8,
    tau_b: float = 0.8,
    num_steps: int = 5,
) -> torch.Tensor:
    """Flexible uniform regularizer from FedU2.

    It approximately solves the paper's UOT objective with projected gradient
    descent on the transport plan. This keeps the implementation dependency-free
    and stable for the small batch sizes commonly used in this repository.
    """

    z = F.normalize(z, dim=1)
    batch_size, feat_dim = z.shape
    if batch_size <= 1:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    s = _sample_spherical_gaussian(batch_size, feat_dim, z.device, z.dtype)
    cost = torch.cdist(z, s, p=2).pow(2)

    a = torch.full((batch_size,), 1.0 / batch_size, device=z.device, dtype=z.dtype)
    b = torch.full((batch_size,), 1.0 / batch_size, device=z.device, dtype=z.dtype)
    pi = torch.outer(a, b)
    step = 1.0 / max(float(batch_size) * (float(tau_a) + float(tau_b)), 1.0)

    for _ in range(max(1, int(num_steps))):
        row_gap = pi.sum(dim=1) - a
        col_gap = pi.sum(dim=0) - b
        grad = cost + float(tau_a) * row_gap[:, None] + float(tau_b) * col_gap[None, :]
        pi = torch.clamp(pi - step * grad, min=0.0)

    row_gap = pi.sum(dim=1) - a
    col_gap = pi.sum(dim=0) - b
    loss = (
        (cost * pi).sum()
        + 0.5 * float(tau_a) * row_gap.square().sum()
        + 0.5 * float(tau_b) * col_gap.square().sum()
    )
    return loss / float(batch_size)


def weighted_average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: torch.Tensor,
) -> "OrderedDict[str, torch.Tensor]":
    weights = weights.float()
    weights = weights / weights.sum().clamp(min=1e-12)
    names = list(state_dicts[0].keys())
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name in names:
        ref = state_dicts[0][name]
        agg = None
        for idx, state in enumerate(state_dicts):
            value = state[name].float()
            weighted = value * float(weights[idx])
            agg = weighted if agg is None else agg + weighted
        out[name] = agg.to(ref.dtype)
    return out


def eua_aggregate_state_dicts(
    global_state: Mapping[str, torch.Tensor],
    client_states: Sequence[Mapping[str, torch.Tensor]],
    client_weights: Iterable[float] | torch.Tensor,
    *,
    server_lr: float = 0.1,
    sharpen_ratio: float = 0.1,
) -> "OrderedDict[str, torch.Tensor]":
    client_weights = torch.as_tensor(list(client_weights), dtype=torch.float32)
    client_weights = sharpen_weights(client_weights, float(sharpen_ratio))
    avg_state = weighted_average_state_dicts(client_states, client_weights)

    if len(client_states) <= 1 or float(server_lr) <= 0:
        return avg_state

    names = list(global_state.keys())
    vecs = []
    for state in client_states:
        diff = torch.cat(
            [
                (global_state[name].float() - state[name].float()).reshape(-1)
                for name in names
            ],
            dim=0,
        )
        norm = diff.norm().clamp(min=1e-12)
        vecs.append((2.0 * diff / norm).cpu().numpy())

    try:
        alpha, _ = MinNormSolver.find_min_norm_element(
            vecs,
            sample_weights=client_weights.cpu().numpy(),
        )
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
        if not torch.isfinite(alpha_t).all():
            return avg_state
        alpha_t = alpha_t / alpha_t.sum().clamp(min=1e-12)
    except Exception:
        return avg_state

    combined = None
    for idx, vec in enumerate(vecs):
        piece = torch.from_numpy(vec).float() * float(alpha_t[idx])
        combined = piece if combined is None else combined + piece

    updated: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for name in names:
        ref = global_state[name]
        numel = ref.numel()
        grad = combined[offset:offset + numel].view_as(ref.float())
        updated[name] = (ref.float() - float(server_lr) * grad).to(ref.dtype)
        offset += numel

    return updated


__all__ = [
    "MinNormSolver",
    "eua_aggregate_state_dicts",
    "fur_uot_loss",
    "sharpen_weights",
    "weighted_average_state_dicts",
]
