# Publication theme for FedCSL / Spilter paper figures
suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

try({
  library(showtext)
  library(sysfonts)
  showtext_auto()
}, silent = TRUE)

# Palette
col_stft   <- "#C9A227"  # warm gold
col_stft2  <- "#E8C547"  # light gold highlight
col_acf    <- "#2A9D8F"  # teal
col_acf2   <- "#52B788"  # light teal
col_accent <- "#E76F51"  # coral
col_neutral<- "#3D405B"
col_muted  <- "#8D99AE"
col_mask   <- "#F4F1DE"
col_panel  <- "#FAFBFC"  # subtle panel fill

theme_pub_sci <- function(base_size = 9, panel_fill = "white") {
  theme_pub(base_size) +
    theme(
      panel.background = element_rect(fill = panel_fill, color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(face = "bold", size = base_size + 1, color = col_neutral),
      plot.tag.position = c(0.02, 0.98)
    )
}

theme_pub <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.title = element_text(size = base_size + 1, face = "bold", color = col_neutral,
                                margin = margin(b = 4)),
      plot.subtitle = element_text(size = base_size - 0.5, color = col_muted,
                                   margin = margin(b = 6)),
      axis.title = element_text(size = base_size, color = col_neutral),
      axis.text = element_text(size = base_size - 1, color = col_neutral),
      legend.title = element_text(size = base_size - 0.5, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.position = "bottom",
      legend.box.margin = margin(t = -2),
      panel.grid.major = element_line(color = "#EEF0F2", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "#DDE1E6", fill = NA, linewidth = 0.4),
      plot.margin = margin(6, 8, 6, 8),
      strip.text = element_text(face = "bold", size = base_size)
    )
}

tag_panel <- function(lbl) {
  annotate("text", x = -Inf, y = Inf, label = lbl, hjust = -0.08, vjust = 1.12,
           fontface = "bold", size = 3.8, color = col_neutral)
}

save_pub <- function(p, path_no_ext, width = 6.9, height = 2.8, dpi = 300) {
  dir.create(dirname(path_no_ext), recursive = TRUE, showWarnings = FALSE)
  ggsave(paste0(path_no_ext, ".pdf"), p, width = width, height = height,
         device = cairo_pdf, bg = "white")
  ggsave(paste0(path_no_ext, ".png"), p, width = width, height = height,
         dpi = dpi, bg = "white")
  message("[saved] ", path_no_ext, " (.pdf + .png)")
}
