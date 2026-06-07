#!/usr/bin/env Rscript
# Fused period weights + overview row
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "."
source(file.path(root, "scripts/figures_r/theme_pub.R"))
source(file.path(root, "scripts/figures_r/compute_period_branch.R"))
library(ggplot2)
library(patchwork)
library(viridis)

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
  branch = rep(c("STFT", "STFT", "STFT", "STFT", "ACF", "ACF", "ACF", "ACF"), length.out = 8)
)

p_bar <- ggplot(df_pi, aes(tau, pi, fill = branch)) +
  geom_col(width = 8, alpha = 0.9, color = NA) +
  geom_text(aes(label = sprintf("r%d", r)), vjust = -0.4, size = 2.5, color = col_neutral) +
  scale_fill_manual(values = c("STFT" = col_stft, "ACF" = col_acf), name = "Branch") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = expression(Fused~period~weights~pi),
    subtitle = sprintf("beta=0.4 (STFT) + 0.6 (ACF)  |  L=%d", L),
    x = expression(tau[r]), y = expression(pi[r])
  ) +
  theme_pub(9) +
  tag_panel("(c)")

save_pub(p_bar, file.path(out_dir, "period_fusion_v6_bar"), width = 4.2, height = 2.4)

# Mini STFT heatmap for overview
power_db <- 10 * log10(st$power + 1e-12)
df_spec <- expand.grid(freq = st$f, time = st$t)
df_spec$power_db <- as.vector(power_db)

p_st <- ggplot(df_spec, aes(time, freq, fill = power_db)) +
  geom_raster() +
  scale_fill_viridis_c(option = "magma", guide = "none") +
  geom_hline(yintercept = st$f_bar, color = col_stft, linewidth = 0.4, linetype = "dashed") +
  labs(title = "STFT", x = NULL, y = "f") +
  theme_pub(8) + theme(axis.text.x = element_blank())

df_acf <- data.frame(lag = ac$lags, R = ac$acf_vals)
p_ac <- ggplot(df_acf, aes(lag, R)) +
  geom_line(color = col_neutral, linewidth = 0.45) +
  geom_hline(yintercept = acf_threshold, color = col_accent, linetype = "dashed", linewidth = 0.4) +
  labs(title = "ACF", x = expression(ell), y = expression(R[c])) +
  theme_pub(8)

overview <- p_st + p_ac + p_bar + plot_layout(widths = c(1, 1, 1.1), ncol = 3)
save_pub(overview, file.path(out_dir, "period_fusion_v6_overview_row"), width = 6.9, height = 2.2)
