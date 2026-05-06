---
name: academic-figures
description: >
  一键生成出版级论文配图。支持柱状图、热力图、散点图、折线图、箱线图、森林图、小提琴图7种图表，
  4种配色主题（Nature/Lancet/保守/默认），中文零乱码自动检测，300DPI+SVG双格式输出。
  纯本地运行，数据不出本机。适用于学术论文、学位论文、会议海报等场景。
  触发词："论文配图"、"画图"、"柱状图"、"热力图"、"散点图"、"森林图"、"箱线图"、"折线图"、"小提琴图"、
  "make a figure"、"generate chart"、"plot data"、"create bar chart"、
  "heatmap"、"scatter plot"、"forest plot"、"box plot"、"line chart"、"violin plot"。
---

# Academic Figures — 学术论文配图生成器

一行命令从 JSON/CSV 数据生成出版级学术图表。纯本地运行，数据不出本机。

## 快速开始

```bash
# 基本用法
python3 scripts/gen_figure.py --type bar --data data.json --output figure.png

# 指定主题和中文支持
python3 scripts/gen_figure.py --type heatmap --data data.json --theme Nature --output heatmap.png

# 散点图含趋势线和置信区间
python3 scripts/gen_figure.py --type scatter --data data.json --trend --ci --output scatter.png
```

## 支持的图表类型（7种）

| 类型 | 说明 | 关键参数 |
|------|------|----------|
| `bar` | 柱状图（分组/堆叠） | `--groups`, `--stacked` |
| `heatmap` | 热力图（矩阵/带注释） | `--cmap`, `--annotate` |
| `scatter` | 散点图（趋势线/置信区间） | `--trend`, `--ci` |
| `line` | 折线图（误差带/数据标记） | `--ci`, `--markers` |
| `box` | 箱线图（含散点抖动） | `--jitter` |
| `forest` | 森林图（效应量+置信区间） | JSON 专用 |
| `violin` | 小提琴图 | `--groups` |

## 配色主题（4种）

- **Nature**：Nature 期刊风格，柔和专业
- **Lancet**：Lancet 期刊风格，鲜明高对比
- **Conservative**：保守风格，适合学位论文
- **Default**：默认 matplotlib 配色

## 中文支持

自动检测数据中的中文字符，无需手动指定 `--cjk` 参数即可正常显示中文标签。如需强制指定：

```bash
python3 scripts/gen_figure.py --type bar --data data.json --cjk
```

## 数据格式

支持 JSON 和 CSV 两种格式。详见 `references/data-formats.md`。

### JSON 示例（柱状图）

```json
{
  "labels": ["A组", "B组", "C组"],
  "series": {
    "基线": [85.2, 78.5, 92.1],
    "Auth RAG": [88.6, 85.3, 95.7]
  }
}
```

### CSV 示例（柱状图）

```csv
labels,基线,Auth RAG
A组,85.2,88.6
B组,78.5,85.3
C组,92.1,95.7
```

> CSV 格式适合柱状图和折线图。箱线图、森林图、散点图建议使用 JSON 格式。

## 统计标注

### 误差线

```bash
python3 scripts/gen_figure.py --type bar --data data.json --error-bars --error-value 5
```

### 显著性标记

```bash
python3 scripts/gen_figure.py --type bar --data data.json --significance "基线:0" --significance-level "0.01"
```

### 趋势线和置信区间（散点图）

```bash
python3 scripts/gen_figure.py --type scatter --data data.json --trend --ci
```

## 输出格式

- **PNG**：300 DPI，适合投稿和打印（默认）
- **SVG**：矢量格式，适合演示和编辑

```bash
# 同时输出两种格式
python3 scripts/gen_figure.py --type bar --data data.json --output figure.png --svg
```

## 完整参数列表

```bash
python3 scripts/gen_figure.py --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--type` | 图表类型 | 必填 |
| `--data` | 数据文件路径 | 必填 |
| `--output` | 输出文件路径 | 必填 |
| `--title` | 图表标题 | 无 |
| `--xlabel` | X轴标签 | 无 |
| `--ylabel` | Y轴标签 | 无 |
| `--theme` | 配色主题 | Default |
| `--cjk` | 强制启用中文字体 | 自动检测 |
| `--error-bars` | 显示误差线 | 关闭 |
| `--error-value` | 误差值 | 0 |
| `--significance` | 显著性标记 | 无 |
| `--significance-level` | 显著性水平标记 | 无 |
| `--stacked` | 堆叠柱状图 | 关闭 |
| `--trend` | 散点图趋势线 | 关闭 |
| `--ci` | 置信区间/误差带 | 关闭 |
| `--markers` | 折线图数据标记 | 关闭 |
| `--jitter` | 箱线图散点抖动 | 关闭 |
| `--svg` | 同时输出SVG | 关闭 |
| `--dpi` | 输出DPI | 300 |
| `--width` | 图表宽度(英寸) | 10 |
| `--height` | 图表高度(英寸) | 6 |
| `--cmap` | 热力图配色 | YlOrRd |

## 文件结构

```
academic-figures/
├── SKILL.md                    # 本文档
├── scripts/
│   ├── gen_figure.py           # 主生成引擎（支持7种图表）
│   └── detect_cjk_font.py      # CJK字体自动检测
└── references/
    └── data-formats.md         # 数据格式详细说明
```

## 常见问题

**Q: 中文标签显示为方块？**
A: 脚本会自动检测中文并加载字体。如果未检测到，手动添加 `--cjk` 参数。

**Q: CSV 格式支持哪些图表？**
A: 柱状图和折线图对 CSV 支持最好。箱线图、森林图、散点图建议使用 JSON 格式。

**Q: 如何添加多条显著性标记？**
A: 使用分号分隔：`--significance "基线:0;Auth RAG:1"`

**Q: 输出图片模糊？**
A: 默认 300 DPI 已满足投稿要求。如需更高分辨率，使用 `--dpi 600`。
