"""Fit additive scale-memory model + knapsack DP for Spilter scale selection.

读取 measure_scale_memory.py 产出的逐尺度峰值显存，完成 §7.7 背包方案的两件事：

  1. 拟合组合显存预测函数 $\widehat{\mathrm{Mem}}(\mathcal{R})$。
     可加模型：$\widehat{\mathrm{Mem}}(\mathcal{R}) = g_0 + \sum_{r\in\mathcal{R}} g_r$，
     其中 $g_r$ = 单尺度峰值 - $g_0$，$g_0$ 为固定开销（最小子模型基线）。
     若提供了 verify_subsets 实测，则用最小二乘在 {g0, 缩放系数} 上做一次校正，
     并报告可加性误差（MAE / 最大相对误差），作为可加假设是否成立的证据。

  2. 显存约束下的尺度选择（0-1 背包，DP 精确解）。
     对每个客户端给定显存预算 G_k，价值 = 周期感知评分 s_{k,r}（此处用占位/可注入），
     在 sum_{r in R} ghat_r <= G_k 约束下最大化 sum s_{k,r}。
     R=8 规模下用整数化 DP（O(R * G)）求全局最优，并与 top-m、min-cost-m 对照。

产物：data/scale_memory_HAR.{json,md}
  * 可加模型系数 g0、g_r；可加性误差；
  * 给定一组预算档位下，DP 最优组合 vs top-m 的对比表。

用法：
  python scripts/system_efficiency/fit_scale_memory.py \
      --partial data/scale_memory_HAR_partials/per_scale.json \
      --budgets 64,128,256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


# ---------------------------------------------------------------------------
# 1) 可加显存模型拟合
# ---------------------------------------------------------------------------
def fit_additive_model(
    single_means: Sequence[float],
    verify: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """拟合 Mem(R) ≈ g0 + sum_{r in R} g_r。

    单尺度实测 single_means[r] = g0 + g_r（即固定开销 + 该尺度边际）。
    若只有单尺度数据，无法分离 g0 与 g_r（欠定）；此时取
        g0 = min_r single_means[r] 的一个保守下界估计，
        g_r = single_means[r] - g0。
    若提供 verify（多尺度组合实测），用最小二乘联合估计 g0：
        measured(R) ≈ g0 + sum_{r in R} (single[r] - g0) = sum single[r] - (|R|-1) g0
        => measured(R) - sum_{r} single[r] ≈ -(|R|-1) g0
        对每个验证组合得到一个方程，最小二乘解 g0。
    """
    singles = np.asarray(single_means, dtype=np.float64)
    R = singles.size

    g0_est = float(singles.min())  # 初始保守估计：最便宜单尺度即 ~ g0 + min g_r
    method = "min_single_lower_bound"
    fit_diag: Dict[str, Any] = {}

    if verify:
        # 建立 (|R|-1) -> (sum_singles - measured) 的线性回归，斜率即 g0
        xs, ys = [], []
        for v in verify:
            scales = v.get("scales", [])
            measured = v.get("measured_mean_mb")
            if measured is None or len(scales) < 1:
                continue
            if any(s < 0 or s >= R for s in scales):
                continue
            sum_singles = float(sum(singles[s] for s in scales))
            xs.append(len(scales) - 1)
            ys.append(sum_singles - measured)
        if len(xs) >= 1 and any(x > 0 for x in xs):
            xs_a = np.asarray(xs, dtype=np.float64)
            ys_a = np.asarray(ys, dtype=np.float64)
            # ys ≈ g0 * xs  （过原点最小二乘）
            denom = float((xs_a * xs_a).sum())
            if denom > 0:
                g0_ls = float((xs_a * ys_a).sum() / denom)
                # g0 物理上应 >= 0 且 <= min single
                g0_est = max(0.0, min(g0_ls, float(singles.min())))
                method = "least_squares_on_verify"
                # 残差诊断
                pred = g0_est * xs_a
                fit_diag["ls_residual_mae"] = float(np.mean(np.abs(pred - ys_a)))
                fit_diag["ls_n_equations"] = int(xs_a.size)

    g_r = (singles - g0_est).tolist()
    return {
        "g0_mb": g0_est,
        "g_r_mb": g_r,
        "single_means_mb": singles.tolist(),
        "method": method,
        "diag": fit_diag,
    }


def predict_mem(model: Dict[str, Any], scales: Sequence[int]) -> float:
    g0 = float(model["g0_mb"])
    g_r = model["g_r_mb"]
    return g0 + float(sum(g_r[s] for s in scales))


def additivity_report(model: Dict[str, Any], verify: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对验证组合，比较 predict_mem 与实测，报告可加性误差。"""
    rows = []
    abs_errs, rel_errs = [], []
    for v in verify:
        scales = v.get("scales", [])
        measured = v.get("measured_mean_mb")
        if measured is None:
            continue
        pred = predict_mem(model, scales)
        ae = abs(pred - measured)
        re = ae / measured if measured else None
        rows.append({
            "scales": scales,
            "measured_mb": measured,
            "predicted_mb": pred,
            "abs_err_mb": ae,
            "rel_err": re,
        })
        abs_errs.append(ae)
        if re is not None:
            rel_errs.append(re)
    return {
        "rows": rows,
        "mae_mb": float(np.mean(abs_errs)) if abs_errs else None,
        "max_abs_err_mb": float(np.max(abs_errs)) if abs_errs else None,
        "max_rel_err": float(np.max(rel_errs)) if rel_errs else None,
    }


# ---------------------------------------------------------------------------
# 2) 0-1 背包 DP（显存约束下最大化评分）
# ---------------------------------------------------------------------------
def knapsack_select(
    values: Sequence[float],
    weights_mb: Sequence[float],
    budget_mb: float,
    g0_mb: float = 0.0,
    quantize_mb: float = 0.5,
) -> Dict[str, Any]:
    """0-1 背包：在 g0 + sum_{r in R} w_r <= budget 下最大化 sum_{r in R} v_r。

    R 很小（=8），用整数化 DP（按 quantize_mb 量化重量）求精确最优。
    返回最优尺度集合、总价值、预测显存。

    退化：当所有 w_r 相等时，等价于 top-m（m = floor((budget-g0)/w)）。
    """
    R = len(values)
    cap = budget_mb - g0_mb
    if cap < 0:
        return {"selected": [], "value": 0.0, "pred_mem_mb": g0_mb, "feasible": False}

    # 量化为整数背包
    W = int(round(cap / quantize_mb))
    w_int = [int(round(w / quantize_mb)) for w in weights_mb]

    NEG = float("-inf")
    # dp[c] = (best_value, selected_tuple)
    dp_val = [NEG] * (W + 1)
    dp_sel: List[Tuple[int, ...]] = [tuple()] * (W + 1)
    dp_val[0] = 0.0

    for r in range(R):
        wr, vr = w_int[r], float(values[r])
        # 0-1：逆序遍历容量
        for c in range(W, wr - 1, -1):
            if dp_val[c - wr] != NEG and dp_val[c - wr] + vr > dp_val[c]:
                dp_val[c] = dp_val[c - wr] + vr
                dp_sel[c] = dp_sel[c - wr] + (r,)

    best_c = max(range(W + 1), key=lambda c: (dp_val[c] if dp_val[c] != NEG else NEG))
    selected = sorted(dp_sel[best_c])
    used_mb = g0_mb + sum(weights_mb[s] for s in selected)
    return {
        "selected": list(selected),
        "value": float(dp_val[best_c]) if dp_val[best_c] != NEG else 0.0,
        "pred_mem_mb": float(used_mb),
        "feasible": len(selected) > 0,
    }


def topm_select(values: Sequence[float], m: int) -> List[int]:
    """纯 top-m：按价值降序取前 m 个（忽略显存）。"""
    order = sorted(range(len(values)), key=lambda r: values[r], reverse=True)
    return sorted(order[:m])


def topm_feasible_mem(
    selected: Sequence[int], weights_mb: Sequence[float], g0_mb: float
) -> float:
    return g0_mb + float(sum(weights_mb[s] for s in selected))


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _fmt(x: Optional[float], d: int = 1) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def write_markdown(
    out_md: Path,
    *,
    model: Dict[str, Any],
    scale_lengths: Sequence[int],
    add_report: Optional[Dict[str, Any]],
    budgets: Sequence[float],
    dp_rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    g0 = model["g0_mb"]
    g_r = model["g_r_mb"]
    R = len(g_r)
    lines: List[str] = []
    lines.append("# HAR 尺度显存标定与背包选择（§7.7 支撑）")
    lines.append("")
    lines.append("## 测量设置")
    lines.append("")
    lines.append(f"- 数据集 HAR，K={meta.get('num_clients_total','?')}，"
                 f"alpha={meta.get('alpha','?')}，batch={meta.get('batch_size','?')}")
    lines.append(f"- GPU: {meta.get('gpu_name') or meta.get('device','?')}")
    lines.append(f"- 尺度数 R={R}，scale_aux={meta.get('scale_aux','?')}")
    lines.append(f"- 标定客户端: {meta.get('clients_used', [])}")
    lines.append("")
    lines.append("## 可加显存模型")
    lines.append("")
    lines.append(r"$$\widehat{\mathrm{Mem}}(\mathcal{R}) = g_0 + \sum_{r\in\mathcal{R}} g_r$$")
    lines.append("")
    lines.append(f"- 固定开销 $g_0$ = **{_fmt(g0)} MB**（拟合方法: {model.get('method')}）")
    lines.append("")
    lines.append("| Scale $r$ | " + " | ".join(str(i) for i in range(R)) + " |")
    lines.append("|" + "---|" * (R + 1))
    lines.append("| $\\ell_r$ | " + " | ".join(str(scale_lengths[i]) for i in range(R)) + " |")
    lines.append("| 单尺度峰值 (MB) | " + " | ".join(_fmt(model["single_means_mb"][i]) for i in range(R)) + " |")
    lines.append("| 边际 $g_r$ (MB) | " + " | ".join(_fmt(g_r[i]) for i in range(R)) + " |")
    lines.append("")

    if add_report and add_report.get("rows"):
        lines.append("## 可加性验证（实测组合 vs 预测）")
        lines.append("")
        lines.append("| 尺度子集 | 实测 (MB) | 预测 (MB) | 绝对误差 (MB) | 相对误差 |")
        lines.append("|---|---|---|---|---|")
        for row in add_report["rows"]:
            lines.append(
                f"| {row['scales']} | {_fmt(row['measured_mb'])} | {_fmt(row['predicted_mb'])} | "
                f"{_fmt(row['abs_err_mb'])} | {_fmt(row.get('rel_err'), 3)} |"
            )
        lines.append("")
        lines.append(f"- 平均绝对误差 MAE = **{_fmt(add_report.get('mae_mb'))} MB**；"
                     f"最大相对误差 = **{_fmt(add_report.get('max_rel_err'), 3)}**。")
        lines.append("- 误差越小，可加模型越可信，背包 DP 的显存约束越精确。")
        lines.append("")

    lines.append("## 显存约束背包选择（DP 精确解）vs 纯 top-m")
    lines.append("")
    lines.append("> 价值 $s_r$ 此处用占位评分（实际应注入客户端本地 STFT/ACF 周期评分）。"
                 "DP 在 $g_0+\\sum_{r\\in\\mathcal{R}} g_r \\le G_k$ 下最大化 $\\sum_{r} s_r$。")
    lines.append("")
    lines.append("| 预算 $G_k$ (MB) | DP 选中尺度 | DP 价值 | DP 显存 (MB) | top-4 选中 | top-4 显存 (MB) | top-4 是否可行 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in dp_rows:
        lines.append(
            f"| {_fmt(row['budget'])} | {row['dp_selected']} | {_fmt(row['dp_value'], 3)} | "
            f"{_fmt(row['dp_mem'])} | {row['topm_selected']} | {_fmt(row['topm_mem'])} | "
            f"{'是' if row['topm_feasible'] else '否(OOM)'} |"
        )
    lines.append("")
    lines.append("## 解读")
    lines.append("")
    lines.append("- DP 在每档预算下给出可行且价值最大的尺度组合；当 top-4 超预算（OOM）时，"
                 "DP 仍能在预算内选出尽可能高价值的子集。")
    lines.append("- 若各 $g_r$ 接近相等，DP 解会退化为 top-$\\lfloor (G_k-g_0)/\\bar g\\rfloor$，与现有 top-m 一致。")
    lines.append("")
    lines.append(f"_最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial",
        type=str,
        default="data/scale_memory_HAR_partials/per_scale.json",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="64,128,256",
        help="逗号分隔的显存预算档位（MB），对应 §7.7 小/中/大显存设备。",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default="",
        help="可选：逗号分隔的 R 个占位评分 s_r；缺省用单尺度峰值的反序做演示评分。",
    )
    parser.add_argument("--topm", type=int, default=4, help="对照的固定 top-m 的 m。")
    parser.add_argument("--out-json", type=str, default="data/scale_memory_HAR.json")
    parser.add_argument("--out-md", type=str, default="data/scale_memory_HAR.md")
    args = parser.parse_args()

    partial_path = Path(args.partial)
    if not partial_path.is_file():
        print(f"[err] 找不到标定结果 {partial_path}；请先运行 measure_scale_memory.py", file=sys.stderr)
        return 1
    data = json.loads(partial_path.read_text(encoding="utf-8"))

    per_scale = data.get("per_scale", [])
    R = int(data.get("R", len(per_scale)))
    scale_lengths = data.get("scale_lengths", list(range(R)))
    single_means = []
    for s in per_scale:
        mb = (s.get("peak_mem_mb", {}) or {}).get("mean")
        single_means.append(float(mb) if mb is not None else float("nan"))
    if any(np.isnan(single_means)):
        print("[err] 存在缺失的单尺度峰值，无法拟合", file=sys.stderr)
        return 1

    verify = data.get("verify_subsets", [])
    model = fit_additive_model(single_means, verify=verify)
    add_report = additivity_report(model, verify) if verify else None

    # 价值：注入或演示（反序，让小尺度索引价值高，仅供管线演示）
    if args.scores.strip():
        scores = [float(x) for x in args.scores.split(",")]
        if len(scores) != R:
            print(f"[err] --scores 应有 {R} 个值", file=sys.stderr)
            return 1
    else:
        # 演示评分：与显存无关的一组单调值，确保非平凡
        scores = [float(R - r) for r in range(R)]

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    g_r = model["g_r_mb"]
    g0 = model["g0_mb"]

    dp_rows: List[Dict[str, Any]] = []
    topm_sel = topm_select(scores, args.topm)
    topm_mem = topm_feasible_mem(topm_sel, g_r, g0)
    for b in budgets:
        dp = knapsack_select(scores, g_r, b, g0_mb=g0)
        dp_rows.append({
            "budget": b,
            "dp_selected": dp["selected"],
            "dp_value": dp["value"],
            "dp_mem": dp["pred_mem_mb"],
            "topm_selected": topm_sel,
            "topm_mem": topm_mem,
            "topm_feasible": topm_mem <= b,
        })
        print(
            f"  [budget {b:.0f}MB] DP选中={dp['selected']} 价值={dp['value']:.2f} "
            f"显存={dp['pred_mem_mb']:.1f}MB | top{args.topm}={topm_sel} "
            f"显存={topm_mem:.1f}MB {'OK' if topm_mem<=b else 'OOM'}",
            flush=True,
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": model,
        "additivity_report": add_report,
        "scale_lengths": scale_lengths,
        "scores_used": scores,
        "topm": args.topm,
        "budgets": budgets,
        "dp_rows": dp_rows,
        "source_partial": str(partial_path),
        "generated_at": time.time(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    write_markdown(
        Path(args.out_md),
        model=model,
        scale_lengths=scale_lengths,
        add_report=add_report,
        budgets=budgets,
        dp_rows=dp_rows,
        meta=data,
    )

    print(f"\n[ok] 拟合 + 背包 DP 完成")
    print(f"     json -> {out_json}")
    print(f"     md   -> {args.out_md}")
    if add_report and add_report.get("mae_mb") is not None:
        print(f"     可加性 MAE = {add_report['mae_mb']:.2f} MB, "
              f"max rel err = {_fmt(add_report.get('max_rel_err'),3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
