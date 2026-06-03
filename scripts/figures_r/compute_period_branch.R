# Mirror FedCSL utils.py period_score (STFT + ACF branches)
suppressPackageStartupMessages({
  library(signal)  # hamming window
})

list_points <- seq(0.1, 0.8, by = 0.1)
acf_threshold <- 0.2

stft_manual <- function(x, window_size, hop_size, n_fft, fs = 1) {
  L <- length(x)
  n_frames <- 1 + floor((L - window_size) / hop_size)
  n_freq <- n_fft %/% 2 + 1
  power <- matrix(0, nrow = n_freq, ncol = n_frames)
  win <- hamming(window_size)
  t <- numeric(n_frames)
  for (i in seq_len(n_frames)) {
    start <- (i - 1) * hop_size + 1
    seg <- x[start:(start + window_size - 1)] * win
    seg_pad <- c(seg, rep(0, n_fft - window_size))
    spec <- fft(seg_pad)
    power[, i] <- Mod(spec[1:n_freq])^2
    t[i] <- (start - 1 + window_size / 2) / fs
  }
  f <- (0:(n_freq - 1)) * fs / n_fft
  list(f = f, t = t, power = power)
}

compute_stft_branch <- function(x, L = length(x), fs = 1) {
  window_size <- max(4, L %/% 2)
  n_fft <- 8 * window_size
  hop_size <- window_size %/% 4

  sp <- stft_manual(x, window_size, hop_size, n_fft, fs)
  f <- sp$f
  t <- sp$t
  power <- sp$power

  thr <- quantile(power, 0.95, na.rm = TRUE)
  mask <- power >= thr
  denom <- sum(power * mask)
  if (denom > 1e-12) {
    freq_mat <- matrix(f, nrow = length(f), ncol = ncol(power))
    f_bar <- sum(freq_mat * power * mask) / denom
  } else {
    f_bar <- 0
  }

  inv_f <- if (f_bar > 1e-12) 1 / f_bar else 0
  tau <- as.integer(list_points * L)
  sigma <- L * 0.1
  gauss <- exp(-((inv_f - tau)^2) / (2 * sigma^2))
  gauss <- gauss / sum(gauss)

  list(
    f = f, t = t, power = power, mask = mask, thr = thr,
    f_bar = f_bar, inv_f = inv_f, tau = tau, stft_scores = gauss,
    window_size = window_size, n_fft = n_fft, hop_size = hop_size
  )
}

compute_acf_branch <- function(x, L = length(x)) {
  tau <- as.integer(list_points * L)
  ac <- acf(x, lag.max = L - 1, plot = FALSE, type = "correlation")
  lags <- as.integer(ac$lag)
  acf_vals <- as.numeric(ac$acf)

  half_lag <- L %/% 2
  search <- which(lags >= half_lag)
  peaks <- integer(0)
  if (length(search) > 2) {
    for (i in 2:(length(search) - 1)) {
      idx <- search[i]
      if (acf_vals[idx] > acf_threshold &&
          acf_vals[idx] >= acf_vals[idx - 1] &&
          acf_vals[idx] >= acf_vals[idx + 1]) {
        peaks <- c(peaks, lags[idx])
      }
    }
  }

  votes <- rep(0, length(tau))
  if (length(peaks) > 0) {
    for (ell in peaks) {
      votes <- votes + 1 / (abs(tau - ell) + 1e-6)
    }
    votes <- votes / sum(votes)
  }

  acf_scores_full <- rep(0, length(tau))
  acf_scores_full[5:8] <- if (sum(votes[5:8]) > 0) votes[5:8] / sum(votes[5:8]) else votes[5:8]

  list(
    lags = lags, acf_vals = acf_vals, peaks = peaks,
    half_lag = half_lag, tau = tau, acf_scores = votes,
    acf_scores_branch = acf_scores_full
  )
}

fuse_period_scores <- function(stft_obj, acf_obj, alpha = 0.4) {
  st_part <- stft_obj$stft_scores[1:4]
  ac_part <- acf_obj$acf_scores[5:8]
  st_part <- if (sum(st_part) > 0) st_part / sum(st_part) else st_part
  ac_part <- if (sum(ac_part) > 0) ac_part / sum(ac_part) else ac_part
  pi_vec <- c(st_part * alpha, ac_part * (1 - alpha))
  if (sum(pi_vec) > 0) pi_vec <- pi_vec / sum(pi_vec)
  pi_vec
}

read_har_channel <- function(path) {
  v <- read.csv(path, header = TRUE)
  as.numeric(v[[1]])
}
