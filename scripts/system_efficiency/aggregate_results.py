"""Aggregate per-algorithm partial JSON results into a single report.

读取 ``data/system_efficiency_HAR_partials/*.json`` 中每个算法的 partial
结果，按 HAR_results 表的算法顺序合并，产出：

  * data/system_efficiency_HAR.json  — 结构化机读
  * data/system_efficiency_HAR.csv   — 简表（算法 / 最慢客户端时间 / 样本数 / 均值 / 中位 / device）
  * data/system_efficiency_HAR.md    — 与 HAR_results.md 同风格的可读 markdown 报告
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PRESENTATION_ORDER = [
    "fedavg",
    "fedprox",
    "byol",
    "fedu2",
    "orchestra",
    "fedcsl",
    "spilter-m1",
    "spilter-m2",
    "spilter-m4",
]

DISPLAY_NAME = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "byol": "FedBYOL",
    "fedu2": "FedU2",
    "orchestra": "Orchestra",
    "fedcsl": "FedCSL",
    "spilter-m1": "Spilter-m1",
    "spilter-m2": "Spilter-m2",
    "spilter-m4": "Spilter-m4",
}


def _load_partials(partials_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not partials_dir.is_dir():
        return out
    for p in sorted(partials_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[warn] skip malformed {p}: {e}", file=sys.stderr)
            continue
        algo = str(obj.get("algo", p.stem)).lower()
        out[algo] = obj
    return out


def _fmt_float(value: Optional[float], digits: int = 3, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _ordered_algos(partials: Dict[str, Dict[str, Any]]) -> List[str]:
    primary = [a for a in PRESENTATION_ORDER if a in partials]
    others = sorted(a for a in partials if a not in PRESENTATION_ORDER)
    return primary + others


def write_csv(partials: Dict[str, Dict[str, Any]], path: Path) -> None:
    fields = [
        "algo",
        "epoch_sec_max",
        "samples_at_max",
        "epoch_sec_mean",
        "epoch_sec_median",
        "epoch_sec_min",
        "peak_mem_mb_mean",
        "peak_mem_mb_max",
        "peak_mem_mb_median",
        "peak_mem_mb_min",
        "num_clients_timed",
        "num_clients_skipped",
        "alpha",
        "batch_size",
        "gpu_name",
        "device",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for algo in _ordered_algos(partials):
            obj = partials[algo]
            row = {k: obj.get(k, "") for k in fields}
            row["algo"] = DISPLAY_NAME.get(algo, algo)
            for k in (
                "epoch_sec_max", "epoch_sec_mean", "epoch_sec_median", "epoch_sec_min",
                "peak_mem_mb_mean", "peak_mem_mb_max", "peak_mem_mb_median", "peak_mem_mb_min",
            ):
                if isinstance(row.get(k), (int, float)):
                    row[k] = round(float(row[k]), 3)
            w.writerow(row)


def write_markdown(partials: Dict[str, Dict[str, Any]], path: Path) -> None:
    ordered = _ordered_algos(partials)
    if not ordered:
        path.write_text("# HAR 系统效率实验\n\n暂无数据。\n", encoding="utf-8")
        return

    sample = partials[ordered[0]]
    alpha = sample.get("alpha", "?")
    bs = sample.get("batch_size", "?")
    num_clients = sample.get("num_clients_total", "?")
    gpu = sample.get("gpu_name") or sample.get("device", "?")
    epoch_n = sample.get("num_epoch", 1)
    shape = sample.get("shape", {})

    times = [partials[a].get("epoch_sec_max") for a in ordered if partials[a].get("epoch_sec_max") is not None]
    fastest = ordered[int(min(range(len(times)), key=lambda i: times[i]))] if times else None
    slowest = ordered[int(max(range(len(times)), key=lambda i: times[i]))] if times else None

    lines: List[str] = []
    lines.append("# HAR 数据集系统效率实验：最慢客户端 1-epoch 耗时")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append("- **数据集**: HAR (Human Activity Recognition)")
    lines.append(f"- **联邦客户端数**: K = {num_clients}, Dirichlet $\\alpha$ = {alpha}")
    lines.append(f"- **本地 epoch 数**: {epoch_n}（一次客户端调用 = 1 个 epoch）")
    lines.append(f"- **batch size**: {bs}")
    if shape:
        lines.append(f"- **样本形状**: N={shape.get('N','?')}, C={shape.get('C','?')}, T={shape.get('T','?')}")
    lines.append(f"- **GPU / 设备**: {gpu}")
    lines.append("- **测量方式**: 对每个算法、每个客户端**串行**独占 GPU 跑 1 epoch，"
                 "`torch.cuda.synchronize()` 前后用 `time.perf_counter()` 计时；"
                 "对所有客户端取最大值作为该算法的最慢客户端单 epoch 耗时。")
    lines.append("- **公平性**: 同时刻 GPU 上仅 1 个 client/算法在跑；含 1 个 batch warmup 避免首次 cudnn 算子选择影响。")
    lines.append("")
    lines.append("## 结果表格")
    lines.append("")
    lines.append(
        "| Method | Slowest Client Epoch (s) | #Samples@Slowest | Mean (s) | Median (s) | "
        "Min (s) | Mean Peak Mem (MB) | Max Peak Mem (MB) | Timed/Skipped |"
    )
    lines.append(
        "|--------|:------------------------:|:----------------:|:--------:|:----------:|"
        ":-------:|:------------------:|:-----------------:|:-------------:|"
    )
    for algo in ordered:
        obj = partials[algo]
        name = DISPLAY_NAME.get(algo, algo)
        slow = obj.get("epoch_sec_max")
        n_at_slow = obj.get("samples_at_max", "-")
        mean = obj.get("epoch_sec_mean")
        median = obj.get("epoch_sec_median")
        mn = obj.get("epoch_sec_min")
        mem_mean = obj.get("peak_mem_mb_mean")
        mem_max = obj.get("peak_mem_mb_max")
        timed = obj.get("num_clients_timed", "-")
        skipped = obj.get("num_clients_skipped", "-")
        bold_slow = f"**{_fmt_float(slow)}**" if algo == slowest else _fmt_float(slow)
        bold_fast = f"**{_fmt_float(slow)}**" if algo == fastest else bold_slow
        lines.append(
            f"| {name} | {bold_fast} | {n_at_slow} | "
            f"{_fmt_float(mean)} | {_fmt_float(median)} | {_fmt_float(mn)} | "
            f"{_fmt_float(mem_mean, digits=1)} | {_fmt_float(mem_max, digits=1)} | {timed}/{skipped} |"
        )
    lines.append("")
    if fastest and slowest and fastest != slowest:
        lines.append(f"> 最快算法（最慢客户端 epoch 最短）: **{DISPLAY_NAME.get(fastest, fastest)}**；"
                     f"最慢算法（最慢客户端 epoch 最长）: **{DISPLAY_NAME.get(slowest, slowest)}**。")
        lines.append("")
    lines.append("## 字段说明")
    lines.append("")
    lines.append("- **Slowest Client Epoch (s)**: $\\max_k T_k^{\\text{epoch}}$，"
                 "该算法所有客户端中跑完 1 epoch 用时最长的那个。"
                 "对应 round 同步联邦的 min--max 时延分析里的拖尾客户端。")
    lines.append("- **#Samples@Slowest**: 最慢客户端的本地样本数。"
                 "对照样本数与耗时可粗略反映"
                 "「耗时随样本量线性增长」/「客户端样本异质度」的影响。")
    lines.append("- **Mean / Median / Min**: 全部客户端 1-epoch 耗时的统计量。")
    lines.append("- **Timed/Skipped**: 实际计时的客户端数 / 因样本数 < batch_size 跳过的客户端数。")
    lines.append("")
    lines.append("## 解读建议")
    lines.append("")
    lines.append("- 同步 FedAvg 类联邦协议的每轮 wall-clock 至少为 `Slowest Client Epoch × local_epoch + 通信/聚合`。"
                 "因此该指标可作为 round-time 下界的代理。")
    lines.append("- Spilter (m=1/2/4) 与 FedCSL 的差距反映尺度切分对客户端本地算力开销的削减。")
    lines.append("- BYOL / FedU2 / Orchestra 走 SSL 路径，模型结构与 forward 不同，"
                 "时间不直接可比；但相对 FedCSL 仍可作为同 backbone 不同自监督方法的代价对比。")
    lines.append("")
    lines.append(f"_最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Spilter 显存节省专题报告（聚焦 topm 维度，vs FedCSL 全尺度 baseline）
# ---------------------------------------------------------------------------
SPILTER_TOPM_ALGOS = ["spilter-m1", "spilter-m2", "spilter-m4"]
SPILTER_TOPM_VALUES = {"spilter-m1": 1, "spilter-m2": 2, "spilter-m4": 4}


def write_memory_markdown(partials: Dict[str, Dict[str, Any]], path: Path) -> None:
    """聚焦 Spilter 不同 top-m 与 FedCSL 全尺度的客户端平均峰值显存对比。"""
    fedcsl = partials.get("fedcsl")
    spilter_present = [a for a in SPILTER_TOPM_ALGOS if a in partials]
    if not spilter_present and fedcsl is None:
        path.write_text("# Spilter 显存节省实验\n\n暂无 Spilter 或 FedCSL 数据。\n", encoding="utf-8")
        return

    sample = (fedcsl or partials[spilter_present[0]])
    alpha = sample.get("alpha", "?")
    bs = sample.get("batch_size", "?")
    K = sample.get("num_clients_total", "?")
    gpu = sample.get("gpu_name") or sample.get("device", "?")
    shape = sample.get("shape", {})
    scales = sample.get("scales", [])
    n_scales = len(scales) if scales else "?"
    baseline_mem = fedcsl.get("peak_mem_mb_mean") if fedcsl else None

    lines: List[str] = []
    lines.append("# Spilter 显存节省实验：客户端训练平均峰值显存随 top-m 变化")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append("- **数据集**: HAR (Human Activity Recognition)")
    lines.append(f"- **联邦客户端数**: K = {K}, Dirichlet $\\alpha$ = {alpha}, batch size = {bs}")
    if shape:
        lines.append(f"- **样本形状**: N={shape.get('N','?')}, C={shape.get('C','?')}, T={shape.get('T','?')}")
    lines.append(f"- **GPU**: {gpu}")
    lines.append(f"- **总尺度数**: {n_scales}（FedCSL baseline 等价于 $m = {n_scales}$；Spilter $m \\in \\{{1,2,4\\}}$ 拼接子模型）")
    lines.append("- **指标定义**: 对每个客户端独立训练 1 epoch，"
                 "用 `torch.cuda.reset_peak_memory_stats()` 清零后训练，结束时取 "
                 "`torch.cuda.max_memory_allocated()` 作为该客户端的峰值显存；"
                 "对所有客户端取均值得到 $\\bar M$。")
    lines.append("")
    lines.append("$$")
    lines.append("\\bar M(\\text{algo}) = \\frac{1}{K}\\sum_{k=1}^{K} \\max_{t} \\big\\| \\text{GPU mem}_k(t) \\big\\|, "
                 "\\quad \\text{Saving}(m) = 1 - \\frac{\\bar M(\\text{Spilter-}m)}{\\bar M(\\text{FedCSL})}")
    lines.append("$$")
    lines.append("")
    lines.append("- **公平性**: 同时刻 GPU 上仅 1 个客户端在训练；含 1 个 batch warmup 再 reset peak，"
                 "避免 cudnn workspace 探测污染统计。")
    lines.append("")

    lines.append("## 结果表格")
    lines.append("")
    lines.append("| Method | top-m | Mean Peak Mem (MB) | Max Peak Mem (MB) | Saving vs FedCSL | Compression Ratio |")
    lines.append("|--------|:-----:|:------------------:|:-----------------:|:----------------:|:-----------------:|")

    rows = []
    if fedcsl is not None:
        rows.append(("FedCSL (baseline)", n_scales, fedcsl))
    for a in spilter_present:
        rows.append((DISPLAY_NAME[a], SPILTER_TOPM_VALUES[a], partials[a]))

    for name, m_val, obj in rows:
        mem_mean = obj.get("peak_mem_mb_mean")
        mem_max = obj.get("peak_mem_mb_max")
        if baseline_mem and isinstance(mem_mean, (int, float)) and baseline_mem > 0:
            saving = 1.0 - float(mem_mean) / float(baseline_mem)
            ratio = float(baseline_mem) / float(mem_mean)
            saving_s = f"{saving * 100:.1f}%"
            ratio_s = f"{ratio:.2f}x"
        else:
            saving_s = "-"
            ratio_s = "-"
        # baseline 自己不显示节省
        if name.startswith("FedCSL"):
            saving_s = "—"
            ratio_s = "1.00x (ref)"
        lines.append(
            f"| {name} | {m_val} | {_fmt_float(mem_mean, digits=1)} | "
            f"{_fmt_float(mem_max, digits=1)} | {saving_s} | {ratio_s} |"
        )
    lines.append("")

    # 找最大节省
    if baseline_mem and spilter_present:
        best_alg = min(spilter_present, key=lambda a: partials[a].get("peak_mem_mb_mean", float("inf")))
        best_mem = partials[best_alg].get("peak_mem_mb_mean")
        if best_mem and best_mem > 0:
            best_saving = (1.0 - best_mem / baseline_mem) * 100
            best_ratio = baseline_mem / best_mem
            lines.append(
                f"> **最大节省**: {DISPLAY_NAME[best_alg]} 把客户端平均峰值显存从 "
                f"{baseline_mem:.1f} MB 降到 {best_mem:.1f} MB —— "
                f"节省 **{best_saving:.1f}%** ($\\approx${best_ratio:.2f}$\\times$ 显存压缩)。"
            )
            lines.append("")

    lines.append("## 字段说明")
    lines.append("")
    lines.append("- **top-m**: Spilter 在客户端本地实际激活并训练的尺度子集大小。"
                 "FedCSL 等价于 $m$ = 全部尺度（无切分）。")
    lines.append("- **Mean / Max Peak Mem (MB)**: 全部客户端 1-epoch 峰值显存的均值 / 最大值。"
                 "**均值是论文里最主要指标**（受样本量异质度影响小），"
                 "Max 用于说明最坏客户端仍能装入卡内。")
    lines.append("- **Saving vs FedCSL**: $1 - \\bar M(\\text{Spilter-}m) / \\bar M(\\text{FedCSL})$，"
                 "即 Spilter 相对全尺度 baseline 的相对显存节省。")
    lines.append("- **Compression Ratio**: $\\bar M(\\text{FedCSL}) / \\bar M(\\text{Spilter-}m)$，"
                 "即 baseline 显存是 Spilter 的多少倍，等价表达。")
    lines.append("")
    lines.append("## 解读建议")
    lines.append("")
    lines.append("- 显存近似随 top-m 线性下降（前向激活、后向缓存、teacher 的拼接子模型都按 m 缩放），"
                 "但**不会线性到 0**，因为 PyTorch context、cudnn workspace、"
                 "shapelet 参数本身（无论是否激活都常驻显存）有固定开销。")
    lines.append("- 在客户端显存受限场景（嵌入式 / 移动 GPU），"
                 "$m=1$ 的 Spilter 通常能让原本 OOM 的 FedCSL 设备重新可训练，"
                 "代价是精度（参见 `data/HAR_results.md`）。")
    lines.append("- 论文里建议把本表与 HAR_results 的精度表对照展示：横轴 $m$，"
                 "双纵轴分别为「精度」和「显存」，能直接画出 Spilter 的 Pareto 前沿。")
    lines.append("")
    lines.append(f"_最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partials",
        type=str,
        default="data/system_efficiency_HAR_partials",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="data/system_efficiency_HAR.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/system_efficiency_HAR.csv",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="data/system_efficiency_HAR.md",
    )
    parser.add_argument(
        "--out-mem-md",
        type=str,
        default="data/spilter_memory_HAR.md",
        help="Spilter 显存节省专题报告输出路径",
    )
    args = parser.parse_args()

    partials_dir = Path(args.partials)
    partials = _load_partials(partials_dir)
    if not partials:
        print(f"[err] no partial json found under {partials_dir}", file=sys.stderr)
        return 1

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "presentation_order": _ordered_algos(partials),
                "results": partials,
                "generated_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(partials, Path(args.out_csv))
    write_markdown(partials, Path(args.out_md))
    write_memory_markdown(partials, Path(args.out_mem_md))

    print(f"[ok] aggregated {len(partials)} algorithms")
    print(f"     json    -> {out_json}")
    print(f"     csv     -> {args.out_csv}")
    print(f"     md      -> {args.out_md}")
    print(f"     mem-md  -> {args.out_mem_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
