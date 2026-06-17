"""Lagrange-relaxation global coverage-aware scale allocation for Spilter.

Formulation
-----------
Given K clients, R scales:

  max  Σ_k Σ_r  v_{k,r} · x_{k,r}
  s.t.
    (1) g_0 + Σ_r g_r · x_{k,r} ≤ B_k   ∀k  (per-client memory / OOM)
    (2) Σ_k x_{k,r} ≥ C_min              ∀r  (global coverage floor)

where:
  v_{k,r} = period_score_{k,r}   (local periodicity, unsupervised)
  g_r     = marginal memory cost of scale r
  g_0     = fixed memory overhead
  B_k     = memory budget of client k
  C_min   = minimum number of clients that must cover each scale

Approach: Lagrange relaxation
-----------------------------
Relax constraint (2) with multipliers λ_r ≥ 0:

  L(x, λ) = Σ_k Σ_r (v_{k,r} + λ_r) · x_{k,r}  −  Σ_r λ_r · C_min

This decomposes into K **independent** 0-1 knapsack subproblems:

  For each client k:
    max  Σ_r (v_{k,r} + λ_r) · x_{k,r}
    s.t. g_0 + Σ_r g_r · x_{k,r} ≤ B_k

λ_r are updated via subgradient:
  λ_r ← max(0,  λ_r + η · (C_min − coverage_r))

Interpretation
--------------
λ_r > 0  →  scale r is under-covered  →  "shadow price" rises
             →  more clients are incentivised to pick it.
λ_r ≈ 0  →  scale r has enough coverage.

All values are unsupervised (period score, contrastive loss, gradient norm);
no label-dependent discriminative term is used.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# 0-1 knapsack DP (exact, for small R like 8)
# ---------------------------------------------------------------------------
def knapsack_dp_select(
    values: Sequence[float],
    weights: Sequence[float],
    budget: float,
    base_cost: float = 0.0,
    quantize: float = 0.5,
) -> List[int]:
    """Exact 0-1 knapsack via DP with weight quantisation.

    Maximises Σ v_r · x_r subject to base_cost + Σ w_r · x_r ≤ budget.

    Args:
        values:  per-scale value v_r, length R.
        weights: per-scale marginal memory g_r, length R.
        budget:  total memory budget B_k (includes base_cost).
        base_cost: fixed overhead g_0.
        quantize: weight quantisation granularity in MB.

    Returns:
        Sorted list of selected scale indices.  Empty list if nothing fits.
    """
    R = len(values)
    cap = budget - base_cost
    if cap <= 0:
        if R > 0 and weights:
            cheapest = int(np.argmin(weights))
            if base_cost + weights[cheapest] <= budget + 1e-6:
                return [cheapest]
        return []

    W = int(round(cap / quantize))
    w_int = [max(1, int(round(w / quantize))) for w in weights]

    NEG = float("-inf")
    dp_val = [NEG] * (W + 1)
    dp_sel: List[Tuple[int, ...]] = [tuple()] * (W + 1)
    dp_val[0] = 0.0

    for r in range(R):
        wr, vr = w_int[r], float(values[r])
        if vr <= 0 and wr > 0:
            continue
        for c in range(W, wr - 1, -1):
            if dp_val[c - wr] != NEG:
                new_val = dp_val[c - wr] + vr
                if new_val > dp_val[c]:
                    dp_val[c] = new_val
                    dp_sel[c] = dp_sel[c - wr] + (r,)

    best_c = max(
        range(W + 1), key=lambda c: dp_val[c] if dp_val[c] != NEG else NEG
    )
    selected = sorted(dp_sel[best_c])

    # Guard against quantisation error making selection infeasible
    if selected:
        actual = base_cost + sum(weights[s] for s in selected)
        if actual > budget + 1e-4:
            return _greedy_feasible_select(values, weights, budget, base_cost)

    return list(selected)


def _greedy_feasible_select(
    values: Sequence[float],
    weights: Sequence[float],
    budget: float,
    base_cost: float = 0.0,
) -> List[int]:
    """Greedy value-density fallback — strictly respects the budget."""
    R = len(values)
    remaining = budget - base_cost
    if remaining <= 0:
        return []
    order = sorted(
        range(R),
        key=lambda r: float(values[r]) / max(float(weights[r]), 1e-6),
        reverse=True,
    )
    selected: List[int] = []
    for r in order:
        if weights[r] <= remaining + 1e-6:
            selected.append(r)
            remaining -= weights[r]
    return sorted(selected)


# ---------------------------------------------------------------------------
# Lagrange relaxation
# ---------------------------------------------------------------------------
def _default_coverage_min(num_clients: int, num_scales: int) -> int:
    """Default C_min: ceil(K / R) — uniform spread."""
    if num_scales <= 0:
        return 0
    return max(1, int(np.ceil(num_clients / num_scales)))


def _compute_scale_coverage(
    client_selected: List[List[int]], num_scales: int
) -> np.ndarray:
    cov = np.zeros(num_scales, dtype=np.int64)
    for selected in client_selected:
        for s in selected:
            if 0 <= s < num_scales:
                cov[s] += 1
    return cov


def _update_lambdas(
    lambdas: np.ndarray,
    coverage: np.ndarray,
    c_min: Union[int, np.ndarray],
    learning_rate: float,
) -> np.ndarray:
    """Subgradient step: λ_r ← max(0, λ_r + η·(C_min − coverage_r))."""
    subgrad = np.asarray(c_min, dtype=np.float64) - coverage.astype(np.float64)
    return np.maximum(0.0, lambdas + learning_rate * subgrad)


def knapsack_lagrangian_assign(
    client_scores: Sequence[Sequence[float]],
    *,
    memory_budgets_mb: Optional[Union[float, Sequence[float]]] = None,
    scale_memory_costs_mb: Optional[Sequence[float]] = None,
    base_memory_mb: float = 0.0,
    coverage_min: Optional[Union[int, Sequence[int]]] = None,
    lambda_lr: float = 0.1,
    max_iter: int = 50,
    seed: Optional[int] = None,
) -> Tuple[List[List[int]], np.ndarray, Dict]:
    """Assign scales to clients via Lagrange relaxation.

    Each client independently solves a 0-1 knapsack:
        max  Σ_r (period_score_{k,r} + λ_r) · x_{k,r}
        s.t. g_0 + Σ_r g_r · x_{k,r} ≤ B_k

    λ_r are updated iteratively to push global coverage above C_min.

    Args:
        client_scores:  per-client period scores, shape [K, R].
        memory_budgets_mb:  per-client budget.
            * float → same budget for all clients
            * list [K] → per-client
            * None → unconstrained (all scales always fit)
        scale_memory_costs_mb:  per-scale marginal memory g_r, length R.
            None → equal unit cost (knapsack reduces to top-m).
        base_memory_mb:  fixed overhead g_0.
        coverage_min:  minimum clients per scale.
            * int → same for all scales
            * list [R] → per-scale
            * None → ceil(K/R)
        lambda_lr:  subgradient step size η.
        max_iter:  maximum Lagrange iterations.
        seed:  for reproducibility (currently unused; deterministic DP).

    Returns:
        client_selected:  list of per-client selected scale lists [K][m_k].
        scale_counts:  per-scale coverage count, shape [R].
        info:  diagnostics dict (λ history, coverage history, …).
    """
    K = len(client_scores)
    if K == 0:
        return [], np.zeros(0, dtype=np.int64), {"iterations": 0, "lambda_final": []}

    R = len(client_scores[0]) if K > 0 else 0
    if R == 0:
        return [[] for _ in range(K)], np.zeros(0, dtype=np.int64), {"iterations": 0, "lambda_final": []}

    # ---- normalise scores --------------------------------------------------
    scores = np.asarray(client_scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- memory costs ------------------------------------------------------
    if scale_memory_costs_mb is not None:
        g_r = np.asarray(scale_memory_costs_mb, dtype=np.float64)
        if len(g_r) != R:
            raise ValueError(f"scale_memory_costs_mb length {len(g_r)} != R={R}")
        g_r = np.maximum(g_r, 1e-6)
    else:
        g_r = np.ones(R, dtype=np.float64)

    g_0 = float(base_memory_mb)

    # ---- per-client budgets ------------------------------------------------
    if memory_budgets_mb is not None:
        if isinstance(memory_budgets_mb, (int, float)):
            B_k = np.full(K, float(memory_budgets_mb), dtype=np.float64)
        else:
            B_k = np.asarray(memory_budgets_mb, dtype=np.float64)
            if len(B_k) == 1:
                B_k = np.full(K, B_k[0], dtype=np.float64)
            elif len(B_k) != K:
                raise ValueError(f"memory_budgets_mb length {len(B_k)} != K={K}")
    else:
        # Unconstrained: budget large enough for all scales
        B_k = np.full(K, g_0 + float(np.sum(g_r)) + 1000.0, dtype=np.float64)

    # Each client must afford at least the cheapest scale
    min_possible = g_0 + float(np.min(g_r))
    B_k = np.maximum(B_k, min_possible)

    # ---- coverage minimum --------------------------------------------------
    if coverage_min is not None:
        if isinstance(coverage_min, (int, np.integer)):
            C_min = np.full(R, int(coverage_min), dtype=np.int64)
        else:
            C_min = np.asarray(coverage_min, dtype=np.int64)
            if len(C_min) != R:
                raise ValueError(f"coverage_min length {len(C_min)} != R={R}")
    else:
        C_min = np.full(R, _default_coverage_min(K, R), dtype=np.int64)

    # ---- Lagrange loop -----------------------------------------------------
    lambdas = np.zeros(R, dtype=np.float64)

    lambda_history: List[np.ndarray] = [lambdas.copy()]
    coverage_history: List[np.ndarray] = []
    selected_per_iter: List[List[List[int]]] = []

    # Track best solution: minimise sum of coverage shortfalls
    best_violation: float = float("inf")
    best_selected: Optional[List[List[int]]] = None
    best_coverage: Optional[np.ndarray] = None
    best_iter: int = 0
    plateau_count: int = 0
    max_plateau: int = max(10, max_iter // 5)

    for iteration in range(max_iter):
        client_selected: List[List[int]] = []

        for k in range(K):
            # adjusted value = local period score + global shadow price
            adj = np.maximum(scores[k] + lambdas, 0.0)

            selected = knapsack_dp_select(
                values=adj.tolist(),
                weights=g_r.tolist(),
                budget=float(B_k[k]),
                base_cost=g_0,
            )

            # Safety: if the DP returned nothing but at least one scale fits,
            # pick the best value/weight item.
            if not selected:
                feasible = [
                    r for r in range(R) if g_0 + g_r[r] <= B_k[k] + 1e-6
                ]
                if feasible:
                    best_r = max(feasible, key=lambda r: float(adj[r]))
                    selected = [best_r]

            client_selected.append(selected)

        coverage = _compute_scale_coverage(client_selected, R)
        coverage_history.append(coverage.copy())
        selected_per_iter.append(client_selected)

        # Evaluate this iteration: total shortfall across all scales
        shortfall = np.maximum(C_min.astype(np.float64) - coverage.astype(np.float64), 0.0)
        total_shortfall = float(np.sum(shortfall))

        if total_shortfall < best_violation:
            best_violation = total_shortfall
            best_selected = client_selected
            best_coverage = coverage.copy()
            best_iter = iteration
            plateau_count = 0
        else:
            plateau_count += 1

        # Early stop: converged (zero violation)
        if np.all(coverage >= C_min):
            break

        # Decaying learning rate (classic 1/√t schedule for subgradient)
        lr = lambda_lr / np.sqrt(1.0 + float(iteration))
        lambdas = _update_lambdas(lambdas, coverage, C_min, lr)
        lambda_history.append(lambdas.copy())

        # Plateau early stop: if no improvement for many iterations
        if plateau_count >= max_plateau:
            break

    # ---- return best solution found ----------------------------------------
    if best_selected is not None and best_coverage is not None:
        final_selected = best_selected
        final_coverage = best_coverage
    else:
        final_selected = selected_per_iter[-1] if selected_per_iter else []
        final_coverage = (
            coverage_history[-1]
            if coverage_history
            else _compute_scale_coverage(final_selected, R)
        )

    info: Dict = {
        "iterations": len(coverage_history),
        "best_iter": best_iter,
        "converged": bool(np.all(final_coverage >= C_min)),
        "total_shortfall": best_violation,
        "lambda_final": lambdas.tolist(),
        "lambda_history": [l.tolist() for l in lambda_history],
        "coverage_final": final_coverage.tolist(),
        "coverage_target": C_min.tolist(),
        "coverage_history": [c.tolist() for c in coverage_history],
        "num_clients": K,
        "num_scales": R,
    }

    return final_selected, final_coverage, info
