# R 科研作图 — STFT / ACF 周期感知分支

## 依赖

- R ≥ 4.6（本机：`brew install r`）
- R 包：`ggplot2 patchwork cowplot scales viridis RColorBrewer signal showtext sysfonts`

## 一键生成审阅图

```bash
cd fedcsl
Rscript scripts/figures_r/generate_review_batch.R "$(pwd)"
```

输出目录：`figs/review_r/`（PNG 300dpi + PDF 矢量）

## 数据

- HAR 样本 CSV 由 Python 从 `画图/data/HAR/train.pt` 导出至 `scripts/figures_r/data/`
- 计算逻辑对齐 `utils.py::period_score`

## Skill

Cursor 个人 skill：`~/.cursor/skills/r-scientific-figures/SKILL.md`
