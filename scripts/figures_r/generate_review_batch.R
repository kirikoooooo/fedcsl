#!/usr/bin/env Rscript
# Generate all STFT/ACF review figures for paper
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = TRUE) else normalizePath(getwd())
cat("Repo root:", root, "\n")

dir.create(file.path(root, "figs/review_r"), recursive = TRUE, showWarnings = FALSE)

scripts <- c("plot_stft_branch.R", "plot_acf_branch.R", "plot_fusion_panel.R", "plot_fusion_overview_variants.R")
for (s in scripts) {
  cat("\n=== Running", s, "===\n")
  status <- system2("Rscript", c(shQuote(file.path(root, "scripts/figures_r", s)), shQuote(root)))
  if (status != 0) stop("Failed: ", s)
}

out_dir <- file.path(root, "figs/review_r")
files <- sort(list.files(out_dir, pattern = "\\.(png|pdf)$"))
manifest <- file.path(out_dir, "REVIEW_MANIFEST.md")
lines <- c(
  "# STFT / ACF 审阅图清单",
  "",
  sprintf("生成时间: %s", Sys.time()),
  "",
  "| 文件 | 用途 | 建议 LaTeX 宽度 |",
  "|------|------|-----------------|",
  "| stft_v6_pub_twopanel | 论文 Fig STFT 双栏替换 v5 | `width=0.9\\\\linewidth` |",
  "| acf_v6_pub_twopanel | 论文 Fig ACF 双栏替换 v5 | `width=0.9\\\\linewidth` |",
  "| period_fusion_v6_overview_row | 三合一概览（STFT+ACF+π） | `width=\\\\linewidth` |",
  "| period_fusion_v6_bar | 单独融合权重柱状图 | `width=0.48\\\\linewidth` |",
  "| period_fusion_v7_refined | v7 精修：等高线+检索区+π 标注 | `width=\\\\linewidth` |",
  "| period_fusion_v7_annotated | v7 公式标注版 | `width=\\\\linewidth` |",
  "| period_fusion_v7_nature | v7 Nature 风格（inferno+灰底） | `width=\\\\linewidth` |",
  "| period_fusion_v7_dense | v7 信息密度版（STFT colorbar） | `width=\\\\linewidth` |",
  "| period_fusion_v7_tall | v7 加高 refined | `width=\\\\linewidth` |",
  "| period_fusion_v7_caption | v7 annotated + 底部 caption | `width=\\\\linewidth` |",
  "| stft_v6_compact / acf_v6_compact | 单栏排版 | `width=0.48\\\\linewidth` |",
  "",
  "## 文件列表",
  ""
)
for (f in files) lines <- c(lines, paste0("- `", f, "`"))
writeLines(lines, manifest)
cat("\nDone. Review folder:", out_dir, "\n")
