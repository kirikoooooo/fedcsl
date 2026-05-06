#!/usr/bin/env python3
"""
academic-figures: Publication-quality academic figure generator.

Generates charts from JSON/CSV data with:
- Publication-grade aesthetics (Nature/Science/Lancet style)
- CJK (Chinese/Japanese/Korean) auto-detection, zero garbled text
- Bilingual labels (Chinese + English)
- Statistical annotations (error bars, significance markers)
- High-DPI output (PNG 300dpi + SVG vector)

Usage:
  python gen_figure.py --type bar --data data.json --out figure.png
  python gen_figure.py --type heatmap --data data.json --out figure.png --cjk
  python gen_figure.py --type scatter --data data.csv --out figure.svg

Data formats: JSON or CSV (first column = labels, rest = series)
"""
import argparse, json, csv, sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECT_SCRIPT = os.path.join(SCRIPT_DIR, "detect_cjk_font.py")

# ── Style presets ──────────────────────────────────────────────────────
THEMES = {
    "default": {
        "figsize": (10, 6),
        "dpi": 300,
        "font_size": 11,
        "colors": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
                   "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
        "grid_alpha": 0.3,
        "spines": ["top", "right"],  # spines to hide
    },
    "nature": {
        "figsize": (8, 5.5),
        "dpi": 300,
        "font_size": 9,
        "colors": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4",
                   "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
        "grid_alpha": 0.2,
        "spines": ["top", "right"],
    },
    "lancet": {
        "figsize": (9, 6),
        "dpi": 300,
        "font_size": 10,
        "colors": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#525252", "#7F6F00",
                   "#ED7D31", "#8B6914", "#4C0099", "#99CC00"],
        "grid_alpha": 0.25,
        "spines": ["top", "right"],
    },
    "conservative": {
        "figsize": (9, 5.5),
        "dpi": 300,
        "font_size": 10,
        "colors": ["#2E86C1", "#A0A0A0", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD",
                   "#1ABC9C", "#E67E22", "#34495E", "#16A085"],
        "grid_alpha": 0.3,
        "spines": ["top", "right"],
    },
}

# ── Font helpers ───────────────────────────────────────────────────────

def load_cjk_font(font_path=None):
    """Load CJK font. Returns (FontProperties, font_name) or (None, None)."""
    if font_path is None:
        # Auto-detect
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, DETECT_SCRIPT],
                capture_output=True, text=True, timeout=10
            )
            info = json.loads(result.stdout)
            font_path = info.get("path") if info.get("found") else None
        except Exception:
            font_path = None
    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        fp = fm.FontProperties(fname=font_path)
        # Extract family name for rcParams
        return fp, fp.get_name()
    return None, None


def has_cjk(text):
    """Check if text contains CJK characters."""
    if not text:
        return False
    return any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in str(text))


def safe_text(ax, text, fontprop=None, **kwargs):
    """Set text with CJK font if needed."""
    if fontprop and has_cjk(str(text)):
        return ax.set_text(text) if hasattr(ax, 'set_text') else ax.text(text, fontproperties=fontprop, **kwargs)
    return ax.set_text(text) if hasattr(ax, 'set_text') else ax.text(text, **kwargs)


# ── Data loading ───────────────────────────────────────────────────────

def _is_number(s):
    """Check if string represents a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def load_data(path, chart_type=None):
    """Load data from JSON or CSV. Returns dict with structure info.
    
    For CSV, chart_type is used to auto-convert long-format data:
    - scatter: first two numeric cols → {x, y}, third col → groups
    - box/violin: first col as groups, second numeric col as values → {series: {group: [values]}}
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext in ('.csv', '.tsv'):
        delimiter = '\t' if ext == '.tsv' else ','
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            headers = next(reader)
            rows = list(reader)
        # Build series, skipping non-numeric columns (C1 fix)
        # Identify numeric columns: >50% of values must be parseable as numbers
        numeric_col_indices = []
        for i, h in enumerate(headers):
            if i == 0:
                continue
            num_count = sum(1 for r in rows if i < len(r) and _is_number(r[i]))
            if num_count > len(rows) * 0.5:
                numeric_col_indices.append(i)

        # Collect all rows with non-numeric values across numeric columns (union)
        bad_rows = set()
        for i in numeric_col_indices:
            for ri, r in enumerate(rows):
                if ri < len(r) and not _is_number(r[i]):
                    bad_rows.add(ri)

        # Build series using only valid rows
        good_rows = [ri for ri in range(len(rows)) if ri not in bad_rows]
        series = {}
        for i in numeric_col_indices:
            vals = [float(rows[ri][i]) for ri in good_rows if ri < len(rows)]
            if vals:
                series[headers[i]] = vals

        # Sync labels with valid rows
        labels = [rows[ri][0] for ri in good_rows if ri < len(rows)]
        result = {"labels": labels, "series": series}

        # Long-format conversion for scatter (C1 real fix)
        if chart_type == 'scatter' and series:
            # Collect ALL numeric columns including index 0 if numeric header
            all_numeric = []
            for i, h in enumerate(headers):
                num_count = sum(1 for ri in good_rows if ri < len(rows) and _is_number(rows[ri][i]))
                if num_count > len(good_rows) * 0.5:
                    all_numeric.append((i, h))

            if len(all_numeric) >= 2:
                col_i, col_x_name = all_numeric[0]
                col_j, col_y_name = all_numeric[1]
                x_vals = [float(rows[ri][col_i]) for ri in good_rows if ri < len(rows) and col_i < len(rows[ri])]
                y_vals = [float(rows[ri][col_j]) for ri in good_rows if ri < len(rows) and col_j < len(rows[ri])]

                groups = None
                # Check remaining columns for groups (non-numeric with multiple unique values)
                if len(all_numeric) >= 3:
                    # Use third numeric column's position to find non-numeric columns
                    pass  # groups will be below
                # Try non-numeric columns for groups — prefer name hint, then fewest unique vals
                _group_hints = {'group', 'groups', 'category', 'categories', 'class', 'label', 'labels', 'type'}
                candidates = []
                for i, h in enumerate(headers):
                    if i in [idx for idx, _ in all_numeric]:
                        continue
                    vals = [rows[ri][i] for ri in good_rows if ri < len(rows)]
                    unique = set(vals)
                    if len(unique) > 1 and len(unique) <= 20:
                        score = (0, len(unique), i)  # (hint_match, unique_count, col_index)
                        if h.lower().strip() in _group_hints:
                            score = (-1, len(unique), i)
                        candidates.append((score, vals, h))
                if candidates:
                    candidates.sort()
                    groups = candidates[0][1]

                result = {"x": x_vals, "y": y_vals}
                if groups:
                    result["groups"] = groups
            elif len(all_numeric) == 1:
                # Only one numeric series from wide format, try using it as y with row index as x
                col_i, col_name = all_numeric[0]
                y_vals = [float(rows[ri][col_i]) for ri in good_rows if ri < len(rows) and col_i < len(rows[ri])]
                # Check if labels (first col) are numeric → use as x
                if labels and all(_is_number(l) for l in labels):
                    result = {"x": [float(l) for l in labels], "y": y_vals}
                else:
                    result = {"x": list(range(len(y_vals))), "y": y_vals, "groups": labels}

        # Long-format conversion for box/violin (C4 real fix)
        if chart_type in ('box', 'boxplot', 'violin') and series:
            first_col_vals = [rows[ri][0] for ri in good_rows if ri < len(rows)]
            unique_groups = list(dict.fromkeys(first_col_vals))  # preserve order
            # Check if first column looks like group labels (many repeats)
            if len(unique_groups) < len(first_col_vals) * 0.8 and len(unique_groups) >= 2:
                # Long format: group column + value column
                num_col_idx = numeric_col_indices[0] if numeric_col_indices else None
                if num_col_idx is not None:
                    grouped = {}
                    for g in unique_groups:
                        grouped[g] = []
                    for ri in good_rows:
                        if ri < len(rows) and num_col_idx < len(rows[ri]):
                            g = rows[ri][0]
                            v = rows[ri][num_col_idx]
                            if _is_number(v):
                                grouped[g].append(float(v))
                    # Only use if groups have multiple values
                    if any(len(v) >= 2 for v in grouped.values()):
                        result = {"labels": unique_groups, "series": grouped}

        return result
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .json or .csv")


# ── Figure generators ──────────────────────────────────────────────────

def apply_base_style(ax, theme):
    """Apply common styling to axes."""
    for spine in theme["spines"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, alpha=theme["grid_alpha"], linestyle='--')
    ax.tick_params(labelsize=theme["font_size"] - 1)


def gen_bar(data, ax, theme, cjk_fp, **kwargs):
    """Grouped bar chart with optional error bars and significance markers."""
    labels = data.get("labels", data.get("x", []))
    series = data.get("series", data.get("datasets", {}))

    n_groups = len(labels)
    n_series = len(series)
    x = np.arange(n_groups)
    w = 0.8 / max(n_series, 1)
    colors = theme["colors"]

    error_data = data.get("errors", {})
    significance = data.get("significance", {})

    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * w
        errors = error_data.get(name, None)
        if errors:
            yerr = [np.std(errors, ddof=1) if isinstance(errors, list) and len(errors) > 1 else errors]
            # Per-bar std
            yerr = None  # Handle below
        errs = None
        if name in error_data and error_data[name]:
            errs = error_data[name]
            if isinstance(errs, list) and all(isinstance(e, (int, float)) for e in errs):
                pass  # per-bar error
            elif isinstance(errs, (int, float)):
                errs = [errs] * n_groups

        bars = ax.bar(x + offset, values, w, yerr=errs,
                      color=colors[i % len(colors)], edgecolor='white', linewidth=0.5,
                      capsize=3, error_kw={'linewidth': 1},
                      label=name)

        # Value labels on bars
        if kwargs.get("show_values", False):
            for j, v in enumerate(values):
                err = errs[j] if errs and j < len(errs) else 0
                ax.text(x[j] + offset, v + err + 0.5, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=7, color=colors[i % len(colors)])

    # Significance brackets
    if significance:
        for key, label in significance.items():
            # key can be "series_name:group_idx" or just "group_idx"
            parts = key.split(":")
            if len(parts) == 2:
                grp_idx = int(parts[1])
            else:
                grp_idx = int(parts[0])
            y_top = ax.get_ylim()[1] * 0.95
            fc = '#C0392B' if label not in ('NS', 'ns') else 'gray'
            fw = 'bold' if label not in ('NS', 'ns') else 'normal'
            ax.annotate(label, (x[grp_idx], y_top), ha='center', va='bottom',
                        fontsize=8, fontweight=fw, color=fc)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=theme["font_size"] - 1,
                       fontproperties=cjk_fp if cjk_fp and any(has_cjk(l) for l in labels) else None)


def gen_heatmap(data, ax, theme, cjk_fp, **kwargs):
    """Heatmap with text annotations."""
    matrix = data.get("matrix", data.get("data", data.get("values", None)))
    if matrix is None:
        # Fallback: build matrix from series (C2 fix — CSV heatmap support)
        series = data.get("series", {})
        if series:
            matrix = np.array(list(series.values()))
        else:
            matrix = np.array([])
    else:
        matrix = np.array(matrix)
    row_labels = data.get("row_labels", data.get("y_labels", data.get("rows", data.get("labels", []))))
    col_labels = data.get("col_labels", data.get("x_labels", data.get("cols", list(data.get("series", {}).keys()))))
    cmap = kwargs.get("cmap", "RdBu_r")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    annot_fmt = kwargs.get("annot_format", "{:+.1f}")

    if vmin is None or vmax is None:
        abs_max = max(abs(matrix.min()), abs(matrix.max()))
        vmin = vmin if vmin is not None else -abs_max
        vmax = vmax if vmax is not None else abs_max

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = 'white' if abs(v) > (abs(vmin) + abs(vmax)) / 2 * 0.7 else 'black'
            ax.text(j, i, annot_fmt.format(v), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(len(col_labels)))
    use_cjk_cols = cjk_fp and any(has_cjk(l) for l in col_labels)
    ax.set_xticklabels(col_labels, fontsize=theme["font_size"] - 2,
                       fontproperties=cjk_fp if use_cjk_cols else None)
    ax.set_yticks(range(len(row_labels)))
    use_cjk_rows = cjk_fp and any(has_cjk(l) for l in row_labels)
    ax.set_yticklabels(row_labels, fontsize=theme["font_size"],
                       fontproperties=cjk_fp if use_cjk_rows else None)

    return im  # for colorbar


def gen_scatter(data, ax, theme, cjk_fp, **kwargs):
    """Scatter plot with optional trend line and grouping."""
    x_data = np.array(data.get("x", data.get("xs", [])))
    y_data = np.array(data.get("y", data.get("ys", [])))
    groups = data.get("groups", data.get("colors", None))

    if groups is None:
        ax.scatter(x_data, y_data, s=60, color=theme["colors"][0],
                   edgecolors='white', linewidth=0.5, alpha=0.7, zorder=3)
    else:
        unique_groups = list(dict.fromkeys(groups))  # preserve order
        for i, g in enumerate(unique_groups):
            mask = [j for j in range(len(groups)) if groups[j] == g]
            c = theme["colors"][i % len(theme["colors"])]
            ax.scatter(x_data[mask], y_data[mask], s=60, c=c,
                       edgecolors='white', linewidth=0.5, alpha=0.7,
                       label=g, zorder=3)

    # Trend line
    if kwargs.get("trend", True) and len(x_data) >= 3:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, linewidth=1.2, zorder=2)
        r = np.corrcoef(x_data, y_data)[0, 1]
        ax.text(0.95, 0.05, f'r = {r:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, fontstyle='italic', color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # Mean points
    if groups and kwargs.get("show_mean", False):
        unique_groups = list(dict.fromkeys(groups))
        for i, g in enumerate(unique_groups):
            mask = [j for j in range(len(groups)) if groups[j] == g]
            mx, my = np.mean(x_data[mask]), np.mean(y_data[mask])
            c = theme["colors"][i % len(theme["colors"])]
            ax.scatter(mx, my, s=150, c=c, edgecolors='black', linewidth=1.2,
                       marker='o', zorder=4)

    ax.xaxis.grid(True, alpha=theme["grid_alpha"], linestyle='--')


def gen_line(data, ax, theme, cjk_fp, **kwargs):
    """Line chart with optional error bands."""
    labels = data.get("labels", data.get("x", []))
    series = data.get("series", data.get("datasets", {}))
    error_data = data.get("errors", {})
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', '+', 'x']

    x = np.arange(len(labels)) if not any(isinstance(l, (int, float)) for l in labels) else np.array(labels)

    for i, (name, values) in enumerate(series.items()):
        c = theme["colors"][i % len(theme["colors"])]
        mk = markers[i % len(markers)]
        lw = kwargs.get("linewidth", 2)
        ax.plot(x, values, c=c, marker=mk, markersize=6, linewidth=lw, label=name, zorder=3)

        # Error band/fill
        if name in error_data and error_data[name]:
            errs = error_data[name]
            if isinstance(errs, list) and len(errs) == len(values):
                lower = [v - e for v, e in zip(values, errs)]
                upper = [v + e for v, e in zip(values, errs)]
                ax.fill_between(x, lower, upper, color=c, alpha=0.15, zorder=2)

    ax.set_xticks(x if not any(isinstance(l, (int, float)) for l in labels) else range(len(labels)))
    if not any(isinstance(l, (int, float)) for l in labels):
        use_cjk = cjk_fp and any(has_cjk(l) for l in labels)
        ax.set_xticklabels(labels, fontsize=theme["font_size"] - 1,
                           fontproperties=cjk_fp if use_cjk else None)


def gen_box(data, ax, theme, cjk_fp, **kwargs):
    """Box plot with optional jitter points."""
    labels = data.get("labels", data.get("x", []))
    series = data.get("series", data.get("datasets", {}))

    positions = list(range(len(series)))
    bp = ax.boxplot(list(series.values()), positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False)

    for i, (patch, name) in enumerate(zip(bp['boxes'], series.keys())):
        patch.set_facecolor(theme["colors"][i % len(theme["colors"])])
        patch.set_alpha(0.6)

    # Jitter points
    rng = np.random.default_rng(42)
    for i, (name, values) in enumerate(series.items()):
        vals = np.array(values, dtype=float)
        jitter_x = positions[i] + rng.normal(0, 0.06, len(vals))
        c = theme["colors"][i % len(theme["colors"])]
        ax.scatter(jitter_x, vals, s=18, c=c, alpha=0.5, zorder=3, edgecolors='none')

    ax.set_xticks(positions)
    use_cjk = cjk_fp and any(has_cjk(l) for l in series.keys())
    ax.set_xticklabels(list(series.keys()), fontsize=theme["font_size"] - 1,
                       fontproperties=cjk_fp if use_cjk else None)


def gen_forest(data, ax, theme, cjk_fp, **kwargs):
    """Forest plot for meta-analysis."""
    labels = data.get("labels", data.get("studies", []))
    estimates = data.get("estimates", data.get("values", []))
    ci_low = data.get("ci_low", data.get("lower", []))
    ci_high = data.get("ci_high", data.get("upper", []))
    overall = data.get("overall", None)
    ref_line = data.get("ref_line", 0)

    y_pos = range(len(labels))

    for i, (est, lo, hi) in enumerate(zip(estimates, ci_low, ci_high)):
        ax.plot([lo, hi], [i, i], '-', color=theme["colors"][0], linewidth=1.5, zorder=2)
        ax.scatter(est, i, c=theme["colors"][0], s=80, zorder=3, edgecolors='black', linewidth=0.5)
        # Label
        ax.text(hi + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, i,
                f'{est:.2f} [{lo:.2f}, {hi:.2f}]', va='center', fontsize=8)

    # Reference line
    ax.axvline(x=ref_line, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # Overall diamond
    if overall:
        oy = len(labels)
        est_o, lo_o, hi_o = overall["estimate"], overall["ci_low"], overall["ci_high"]
        diamond_x = [lo_o, est_o, hi_o, est_o, lo_o]
        diamond_y = [oy, oy - 0.2, oy, oy + 0.2, oy]
        ax.fill(diamond_x, diamond_y, color=theme["colors"][1], alpha=0.7, zorder=4)
        ax.text(hi_o + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, oy,
                f'Overall: {est_o:.2f} [{lo_o:.2f}, {hi_o:.2f}]', va='center',
                fontsize=9, fontweight='bold')

    ax.set_yticks(list(y_pos) + ([len(labels)] if overall else []))
    all_labels = list(labels) + (["Overall"] if overall else [])
    use_cjk = cjk_fp and any(has_cjk(l) for l in all_labels)
    ax.set_yticklabels(all_labels, fontsize=theme["font_size"],
                       fontproperties=cjk_fp if use_cjk else None)
    ax.invert_yaxis()


def gen_violin(data, ax, theme, cjk_fp, **kwargs):
    """Violin plot with optional inner box and jitter points."""
    labels = data.get("labels", data.get("x", []))
    series = data.get("series", data.get("datasets", {}))

    positions = list(range(len(series)))
    values_list = list(series.values())

    vp = ax.violinplot(values_list, positions=positions, widths=0.6,
                       showmeans=kwargs.get("show_means", True),
                       showmedians=kwargs.get("show_medians", True),
                       showextrema=kwargs.get("show_extrema", True))

    # Color the violin bodies
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(theme["colors"][i % len(theme["colors"])])
        body.set_alpha(0.6)
        body.set_edgecolor(theme["colors"][i % len(theme["colors"])])
        body.set_linewidth(1)

    # Style internal lines
    for part in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
        if part in vp:
            vp[part].set_color('#333333')
            vp[part].set_linewidth(1)

    ax.set_xticks(positions)
    use_cjk = cjk_fp and any(has_cjk(l) for l in series.keys())
    ax.set_xticklabels(list(series.keys()), fontsize=theme["font_size"] - 1,
                       fontproperties=cjk_fp if use_cjk else None)


# ── Registry ───────────────────────────────────────────────────────────

GENERATORS = {
    "bar": gen_bar,
    "grouped_bar": gen_bar,
    "heatmap": gen_heatmap,
    "scatter": gen_scatter,
    "line": gen_line,
    "box": gen_box,
    "boxplot": gen_box,
    "forest": gen_forest,
    "violin": gen_violin,
}

# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="academic-figures: Publication-quality figure generator")
    parser.add_argument("--type", "-t", required=True, choices=list(GENERATORS.keys()),
                        help="Figure type")
    parser.add_argument("--data", "-d", required=True, help="Input data file (JSON or CSV)")
    parser.add_argument("--out", "-o", required=True, help="Output file path (.png or .svg)")
    parser.add_argument("--title", default="", help="Figure title")
    parser.add_argument("--xlabel", default="", help="X-axis label")
    parser.add_argument("--ylabel", default="", help="Y-axis label")
    parser.add_argument("--theme", default="default", choices=list(THEMES.keys()),
                        help="Color theme")
    parser.add_argument("--cjk", action="store_true", help="Enable CJK font (auto-detect)")
    parser.add_argument("--cjk-font", default=None, help="Specific CJK font path")
    parser.add_argument("--width", type=float, default=None, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=None, help="Figure height in inches")
    parser.add_argument("--legend", action="store_true", default=True, help="Show legend")
    parser.add_argument("--no-legend", action="store_true", help="Hide legend")
    parser.add_argument("--show-values", action="store_true", help="Show value labels")
    parser.add_argument("--trend", action="store_true", default=True, help="Show trend line (scatter)")
    parser.add_argument("--no-trend", action="store_true", help="Hide trend line (scatter)")
    parser.add_argument("--cmap", default=None, help="Colormap for heatmap")
    parser.add_argument("--vmin", type=float, default=None, help="Heatmap vmin")
    parser.add_argument("--vmax", type=float, default=None, help="Heatmap vmax")

    args = parser.parse_args()

    theme = THEMES[args.theme]
    data = load_data(args.data, chart_type=args.type)

    # Validate minimal required fields
    if not data or not isinstance(data, dict):
        print("ERROR: Data file is empty or invalid. Must be a non-empty JSON object or CSV.", file=sys.stderr)
        sys.exit(1)
    # Check that at least one data field exists
    has_data = any(data.get(k) for k in ["series", "matrix", "x", "y", "estimates"])
    if not has_data:
        print("ERROR: No data fields found. Provide 'series', 'matrix', 'x'/'y', or 'estimates'.", file=sys.stderr)
        sys.exit(1)
    # Check that data fields are not empty (C3 fix)
    series = data.get("series", {})
    if series and all(len(v) == 0 for v in series.values()):
        print("ERROR: 'series' exists but all values are empty.", file=sys.stderr)
        sys.exit(1)
    matrix = data.get("matrix", data.get("data", data.get("values")))
    if matrix is not None and not isinstance(matrix, (int, float)):
        arr = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
        if arr.size == 0:
            print("ERROR: 'matrix' is empty.", file=sys.stderr)
            sys.exit(1)
    cjk_fp = None
    # Auto-detect: scan displayable text for CJK chars
    def _text_has_cjk():
        for t in [args.title, args.xlabel, args.ylabel]:
            if t and has_cjk(t):
                return True
        if data:
            for field in ["labels", "row_labels", "col_labels"]:
                for v in data.get(field, []):
                    if isinstance(v, str) and has_cjk(v):
                        return True
            for v in data.get("series", {}).keys():
                if isinstance(v, str) and has_cjk(v):
                    return True
            for v in data.get("groups", []):
                if isinstance(v, str) and has_cjk(v):
                    return True
        return False
    _auto_cjk = _text_has_cjk()
    if args.cjk or args.cjk_font or _auto_cjk:
        cjk_fp, cjk_name = load_cjk_font(args.cjk_font)
        if cjk_name:
            plt.rcParams['font.sans-serif'] = [cjk_name, 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        if cjk_fp:
            print(f"CJK font loaded: {cjk_name}", file=sys.stderr)
        else:
            print("WARNING: No CJK font found, Chinese characters may not render", file=sys.stderr)

    # Create figure
    width = args.width if args.width else theme["figsize"][0]
    height = args.height if args.height else theme["figsize"][1]
    fig, ax = plt.subplots(figsize=(width, height))

    # Generate
    gen_func = GENERATORS[args.type]
    kwargs = {
        "show_values": args.show_values,
        "trend": args.trend and not args.no_trend,
        "cmap": args.cmap,
        "vmin": args.vmin,
        "vmax": args.vmax,
    }
    extra = gen_func(data, ax, theme, cjk_fp, **kwargs)

    # Style
    apply_base_style(ax, theme)

    # Labels
    if args.title:
        title_text = args.title.replace('\\n', '\n')
        ax.set_title(title_text, fontsize=theme["font_size"] + 1, fontweight='bold', pad=12,
                     fontproperties=cjk_fp if cjk_fp and has_cjk(title_text) else None)
    if args.xlabel:
        ax.set_xlabel(args.xlabel, fontsize=theme["font_size"],
                      fontproperties=cjk_fp if cjk_fp and has_cjk(args.xlabel) else None)
    if args.ylabel:
        ax.set_ylabel(args.ylabel, fontsize=theme["font_size"],
                      fontproperties=cjk_fp if cjk_fp and has_cjk(args.ylabel) else None)

    # Colorbar (for heatmap)
    if extra is not None and hasattr(extra, 'get_cmap'):
        cbar = fig.colorbar(extra, ax=ax, shrink=0.8)
        cbar_label = data.get("cbar_label", args.ylabel)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=theme["font_size"] - 1,
                          fontproperties=cjk_fp if cjk_fp and has_cjk(cbar_label) else None)

    # Legend (skip if no labeled artists)
    if not args.no_legend and args.legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=theme["font_size"] - 1, loc='best', framealpha=0.9,
                  prop=cjk_fp if cjk_fp else None)

    plt.tight_layout()

    # Save
    fig.savefig(args.out, dpi=theme["dpi"], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    sz = os.path.getsize(args.out)
    print(f"Saved: {args.out} ({sz:,} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
