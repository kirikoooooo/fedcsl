#!/usr/bin/env Rscript
# ACF branch figure — publication two-panel layout
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "."
source(file.path(root, "scripts/figures_r/theme_pub.R"))
source(file.path(root, "scripts/figures_r/compute_period_branch.R"))
library(ggplot2)
library(patchwork)
library(scales)

data_dir <- file.path(root, "scripts/figures_r/data")
out_dir  <- file.path(root, "figs/review_r")

# Concatenate two walking segments (as in paper caption)
x1 <- read_har_channel(file.path(data_dir, "har_walk_2759_ch0.csv"))
x2 <- read_har_channel(file.path(data_dir, "har_walk_2008_ch0.csv"))
x <- c(x1, x2)
L <- length(x)

ac <- compute_acf_branch(x, L = L)

df_acf <- data.frame(lag = ac$lags, R = ac$acf_vals)
df_peaks <- data.frame(lag = ac$peaks, R = ac$acf_vals[match(ac$peaks, ac$lags)])

tau <- ac$tau

p_a <- ggplot(df_acf, aes(lag, R)) +
  geom_hline(yintercept = 0, color = col_muted, linewidth = 0.3) +
  geom_vline(xintercept = ac$half_lag, color = col_acf, linewidth = 0.45, linetype = "dashed") +
  geom_hline(yintercept = acf_threshold, color = col_accent, linewidth = 0.45, linetype = "dashed") +
  annotate("rect", xmin = ac$half_lag, xmax = max(ac$lags), ymin = -Inf, ymax = Inf,
           fill = col_acf, alpha = 0.06) +
  geom_line(color = col_neutral, linewidth = 0.55) +
  geom_point(data = df_peaks, aes(lag, R), color = col_accent, size = 2.2) +
  annotate("text", x = ac$half_lag + 4, y = 0.92,
           label = expression(ell >= floor(L/2)), size = 2.8, color = col_acf) +
  annotate("text", x = max(ac$lags) * 0.55, y = acf_threshold + 0.06,
           label = expression(theta[acf]==0.2), size = 2.8, color = col_accent) +
  labs(
    title = expression(Autocorrelation~R[c](ell)),
    subtitle = sprintf("HAR Walking concat  |  L=%d  |  peaks: %s",
                       L, paste(ac$peaks, collapse = ", ")),
    x = expression(Lag~ell), y = expression(R[c](ell))
  ) +
  coord_cartesian(ylim = c(-0.15, 1.05)) +
  theme_pub(9) +
  tag_panel("(a)")

df_vote <- data.frame(
  tau = tau, r = seq_along(tau),
  score = ac$acf_scores,
  branch = ifelse(seq_along(tau) <= 4, "STFT branch", "ACF branch")
)

p_b <- ggplot(df_vote, aes(tau, score, fill = branch)) +
  geom_col(width = 9, alpha = 0.88, color = NA) +
  geom_point(data = df_peaks, aes(x = lag, y = 0.02), inherit.aes = FALSE,
             shape = 21, fill = col_accent, color = "white", size = 2.2, stroke = 0.3) +
  geom_segment(data = df_peaks, aes(x = lag, xend = lag, y = 0, yend = 0.04),
               inherit.aes = FALSE, color = col_accent, linewidth = 0.35, alpha = 0.7) +
  scale_fill_manual(values = c("STFT branch" = col_muted, "ACF branch" = col_acf), name = NULL) +
  scale_x_continuous(breaks = tau) +
  labs(
    title = "Peak lags -> tau[r] inverse-distance vote",
    subtitle = "Scores written to S^{ACF} (r=5..8)",
    x = expression(tau[r]), y = "Normalized vote"
  ) +
  theme_pub(9) +
  theme(legend.position = c(0.22, 0.82),
        legend.background = element_rect(fill = scales::alpha("white", 0.9))) +
  tag_panel("(b)")

fig <- p_a + p_b + plot_layout(widths = c(1.1, 1))
save_pub(fig, file.path(out_dir, "acf_v6_pub_twopanel"), width = 6.9, height = 2.65)
save_pub(fig, file.path(out_dir, "acf_v6_compact"), width = 3.45, height = 2.5)
