#!/usr/bin/env Rscript
# STFT branch figure — publication two-panel layout
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
csv_path <- file.path(data_dir, "har_walk_2759_ch0.csv")
x <- read_har_channel(csv_path)
L <- length(x)

st <- compute_stft_branch(x, L = L)
power_db <- 10 * log10(st$power + 1e-12)

df_spec <- expand.grid(freq = st$f, time = st$t)
df_spec$power_db <- as.vector(power_db)

p_a <- ggplot(df_spec, aes(time, freq, fill = power_db)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "magma", name = "Power (dB)",
                       limits = quantile(df_spec$power_db, c(0.02, 0.98))) +
  geom_hline(yintercept = st$f_bar, color = col_stft, linewidth = 0.55, linetype = "dashed") +
  annotate("text", x = max(st$t) * 0.72, y = st$f_bar + 0.015,
           label = sprintf("bar(f)[c]=%.4f", st$f_bar), size = 2.8, color = col_stft) +
  labs(
    title = "STFT spectrogram",
    subtitle = sprintf("HAR #2759 / total_acc_x  |  W=%d, Nfft=%d, hop=%d",
                       st$window_size, st$n_fft, st$hop_size),
    x = "Time frame", y = expression(Frequency~f)
  ) +
  theme_pub(9) +
  theme(legend.key.height = unit(0.35, "cm")) +
  tag_panel("(a)")

tau <- st$tau
df_vote <- data.frame(
  tau = tau,
  score = st$stft_scores,
  branch = ifelse(seq_along(tau) <= 4, "STFT branch", "ACF branch")
)
tau_seq <- seq(min(tau), max(tau), length.out = 200)
gauss_curve <- exp(-((st$inv_f - tau_seq)^2) / (2 * (L * 0.1)^2))
gauss_curve <- gauss_curve / sum(gauss_curve)
df_curve <- data.frame(tau = tau_seq, score = gauss_curve)

p_b <- ggplot(df_vote, aes(tau, score, fill = branch)) +
  geom_col(width = 8, alpha = 0.85, color = NA) +
  geom_line(data = df_curve, aes(x = tau, y = score), inherit.aes = FALSE,
            color = col_stft, linewidth = 0.7, linetype = "dashed") +
  geom_vline(xintercept = st$inv_f, color = col_stft, linewidth = 0.45, linetype = "dotted") +
  scale_fill_manual(values = c("STFT branch" = col_stft, "ACF branch" = col_muted), name = NULL) +
  scale_x_continuous(breaks = tau) +
  labs(
    title = "1/f[c] -> tau[r] Gaussian vote",
    subtitle = sprintf("1/f[c] = %.1f samples  |  S^ST (r=1..4)", st$inv_f),
    x = expression(tau[r]), y = "Normalized score"
  ) +
  theme_pub(9) +
  theme(legend.position = c(0.78, 0.82),
        legend.background = element_rect(fill = scales::alpha("white", 0.9))) +
  tag_panel("(b)")

fig <- p_a + p_b + plot_layout(widths = c(1.15, 1))
save_pub(fig, file.path(out_dir, "stft_v6_pub_twopanel"), width = 6.9, height = 2.65)
save_pub(fig, file.path(out_dir, "stft_v6_compact"), width = 3.45, height = 2.5)
