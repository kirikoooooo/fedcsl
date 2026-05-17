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
            for k in ("epoch_sec_max", "epoch_sec_mean", "epoch_sec_median", "epoch_sec_min"):
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
    lines.append("| Method | Slowest Client Epoch (s) | #Samples@Slowest | Mean (s) | Median (s) | Min (s) | Timed/Skipped |")
    lines.append("|--------|:------------------------:|:----------------:|:--------:|:----------:|:-------:|:-------------:|")
    for algo in ordered:
        obj = partials[algo]
        name = DISPLAY_NAME.get(algo, algo)
        slow = obj.get("epoch_sec_max")
        n_at_slow = obj.get("samples_at_max", "-")
        mean = obj.get("epoch_sec_mean")
        median = obj.get("epoch_sec_median")
        mn = obj.get("epoch_sec_min")
        timed = obj.get("num_clients_timed", "-")
        skipped = obj.get("num_clients_skipped", "-")
        bold_slow = f"**{_fmt_float(slow)}**" if algo == slowest else _fmt_float(slow)
        bold_fast = f"**{_fmt_float(slow)}**" if algo == fastest else bold_slow
        lines.append(
            f"| {name} | {bold_fast} | {n_at_slow} | "
            f"{_fmt_float(mean)} | {_fmt_float(median)} | {_fmt_float(mn)} | {timed}/{skipped} |"
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

    print(f"[ok] aggregated {len(partials)} algorithms")
    print(f"     json -> {out_json}")
    print(f"     csv  -> {args.out_csv}")
    print(f"     md   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
