#!/usr/bin/env Rscript
# Enhanced period-fusion overview row variants (baseline: v6 overview)
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "."
source(file.path(root, "scripts/figures_r/theme_pub.R"))
source(file.path(root, "scripts/figures_r/compute_period_branch.R"))
library(ggplot2)
library(patchwork)
library(viridis)
library(scales)

data_dir <- file.path(root, "scripts/figures_r/data")
out_dir  <- file.path(root, "figs/review_r")

x <- read_har_channel(file.path(data_dir, "har_walk_2759_ch0.csv"))
L <- length(x)
st <- compute_stft_branch(x, L = L)
ac <- compute_acf_branch(x, L = L)
pi_vec <- fuse_period_scores(st, ac, alpha = 0.4)
tau <- st$tau

df_pi <- data.frame(
  r = seq_along(pi_vec),
  tau = tau,
  pi = pi_vec,
  branch = c(rep("STFT", 4), rep("ACF", 4))
)

power_db <- 10 * log10(st$power + 1e-12)
df_spec <- expand.grid(freq = st$f, time = st$t)
df_spec$power_db <- as.vector(power_db)
df_spec$high_energy <- as.vector(st$mask)

df_acf <- data.frame(lag = ac$lags, R = ac$acf_vals)
df_peaks <- if (length(ac$peaks) > 0) {
  data.frame(
    lag = ac$peaks,
    R = ac$acf_vals[match(ac$peaks, ac$lags) + 1]
  )
} else {
  data.frame(lag = numeric(0), R = numeric(0))
}

# ---- panel builders ----
build_stft <- function(style = "refined") {
  p <- ggplot(df_spec, aes(time, freq, fill = power_db)) +
    geom_raster(interpolate = TRUE)

  if (style %in% c("refined", "annotated", "nature")) {
    p <- p +
      stat_contour(aes(z = high_energy), breaks = 0.5, color = "white",
                   linewidth = 0.35, alpha = 0.85) +
      geom_hline(yintercept = st$f_bar, color = col_stft, linewidth = 0.55, linetype = "dashed")
  }

  if (style == "annotated") {
    p <- p + annotate("text", x = max(st$t) * 0.55, y = st$f_bar + 0.025,
                      label = sprintf("bar(f)[c]=%.3f", st$f_bar),
                      size = 2.6, color = col_stft, fontface = "italic")
  }

  pal <- if (style == "nature") "inferno" else "magma"
  p + scale_fill_viridis_c(
    option = pal, name = "Power (dB)",
    limits = quantile(df_spec$power_db, c(0.03, 0.97)),
    oob = squish
  ) +
    labs(
      title = if (style == "nature") "STFT branch" else "STFT spectrogram",
      subtitle = sprintf("HAR #2759  |  W=%d, hop=%d", st$window_size, st$hop_size),
      x = NULL, y = expression(italic(f))
    ) +
    theme_pub_sci(8.5, if (style == "nature") col_panel else "white") +
    theme(
      axis.text.x = element_blank(),
      legend.position = if (style == "dense") "bottom" else "none",
      legend.key.height = unit(0.25, "cm"),
      legend.title = element_text(size = 7)
    )
}

build_acf <- function(style = "refined") {
  p <- ggplot(df_acf, aes(lag, R)) +
    geom_hline(yintercept = 0, color = col_muted, linewidth = 0.25)

  if (style %in% c("refined", "annotated", "nature", "dense")) {
    p <- p +
      annotate("rect", xmin = ac$half_lag, xmax = max(ac$lags), ymin = -Inf, ymax = Inf,
               fill = col_acf, alpha = 0.07) +
      geom_vline(xintercept = ac$half_lag, color = col_acf,
                 linewidth = 0.45, linetype = "dashed", alpha = 0.8) +
      geom_hline(yintercept = acf_threshold, color = col_accent,
                 linewidth = 0.45, linetype = "dashed")
  }

  p <- p + geom_line(color = col_neutral, linewidth = 0.55)

  if (nrow(df_peaks) > 0 && style != "minimal") {
    p <- p + geom_point(data = df_peaks, aes(lag, R),
                        color = col_accent, size = 2.0, shape = 21, fill = "white", stroke = 0.45)
  }

  if (style == "annotated") {
    p <- p + annotate("text", x = ac$half_lag + 3, y = 0.92,
                      label = expression(ell >= floor(L/2)), size = 2.5, color = col_acf)
  }

  p + coord_cartesian(ylim = c(-0.2, 1.05)) +
    labs(
      title = "ACF branch",
      subtitle = if (length(ac$peaks) > 0) {
        sprintf("peaks: %s", paste(ac$peaks, collapse = ", "))
      } else {
        sprintf("L=%d,  theta[acf]=0.2", L)
      },
      x = expression(italic(l)), y = expression(italic(R)[c](italic(l)))
    ) +
    theme_pub_sci(8.5, if (style == "nature") col_panel else "white")
}

build_pi <- function(style = "refined") {
  divider <- mean(tau[4:5])
  p <- ggplot(df_pi, aes(tau, pi, fill = branch))

  if (style %in% c("refined", "annotated", "nature", "dense")) {
    p <- p +
      geom_col(width = 9, alpha = 0.92, color = "white", linewidth = 0.35) +
      geom_vline(xintercept = divider, color = col_muted, linewidth = 0.4, linetype = "dotted") +
      geom_text(aes(label = sprintf("%.2f", pi)), vjust = -0.35, size = 2.2, color = col_neutral) +
      annotate("text", x = mean(tau[1:4]), y = max(df_pi$pi) * 1.08, label = "S[ST]",
               parse = TRUE, size = 2.8, color = col_stft, fontface = "bold") +
      annotate("text", x = mean(tau[5:8]), y = max(df_pi$pi) * 1.08, label = "S[ACF]",
               parse = TRUE, size = 2.8, color = col_acf, fontface = "bold")
  } else {
    p <- p + geom_col(width = 8, alpha = 0.9, color = NA)
  }

  p + scale_fill_manual(
    values = c("STFT" = col_stft, "ACF" = col_acf),
    labels = c("STFT branch", "ACF branch"),
    name = NULL
  ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    scale_x_continuous(breaks = tau) +
    labs(
      title = expression("Fused weights " ~ pi),
      subtitle = sprintf("beta=0.4 STFT + 0.6 ACF  |  L=%d", L),
      x = expression(italic(tau)[r]), y = expression(italic(pi)[r])
    ) +
    theme_pub_sci(8.5, if (style == "nature") col_panel else "white") +
    theme(
      legend.position = c(0.82, 0.78),
      legend.background = element_rect(fill = scales::alpha("white", 0.92), color = "#E0E0E0", linewidth = 0.3)
    )
}

compose_overview <- function(style, tag = TRUE) {
  p_st <- build_stft(style)
  p_ac <- build_acf(style)
  p_pi <- build_pi(style)
  layout <- p_st + p_ac + p_pi + plot_layout(widths = c(1.05, 1, 1.12), ncol = 3)
  if (tag) {
    layout <- layout + plot_annotation(
      tag_levels = "a",
      theme = theme(plot.tag = element_text(face = "bold", size = 11, color = col_neutral))
    )
  }
  layout
}

# ---- variants ----
variants <- list(
  list(name = "period_fusion_v7_refined",     style = "refined",   w = 6.9, h = 2.35),
  list(name = "period_fusion_v7_annotated",   style = "annotated", w = 6.9, h = 2.45),
  list(name = "period_fusion_v7_nature",      style = "nature",    w = 6.9, h = 2.35),
  list(name = "period_fusion_v7_dense",       style = "dense",     w = 7.2, h = 2.55),
  list(name = "period_fusion_v7_tall",        style = "refined",   w = 6.9, h = 2.75)
)

for (v in variants) {
  fig <- compose_overview(v$style)
  save_pub(fig, file.path(out_dir, v$name), width = v$w, height = v$h)
}

# v7 with connecting flow annotation (subtitle row)
fig_flow <- compose_overview("annotated") +
  plot_annotation(
    caption = "Dual-branch period scoring: STFT covers short/mid scales (r=1..4); ACF covers mid/long scales (r=5..8).",
    theme = theme(
      plot.caption = element_text(size = 7.5, color = col_muted, hjust = 0, margin = margin(t = 4))
    )
  )
save_pub(fig_flow, file.path(out_dir, "period_fusion_v7_caption"), width = 6.9, height = 2.55)

message("Done: ", length(variants) + 1, " overview variants in ", out_dir)
