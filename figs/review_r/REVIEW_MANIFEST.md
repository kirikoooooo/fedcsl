# STFT / ACF 审阅图清单

生成时间: 2026-05-21

## 推荐对比（period fusion 三栏概览）

| 文件 | 风格要点 | 建议 LaTeX 宽度 |
|------|----------|-----------------|
| period_fusion_v6_overview_row | 基准：简洁三栏 | `width=\linewidth` |
| **period_fusion_v7_refined** | **推荐**：95% 能量等高线、ACF 检索区阴影、π 分支标注 | `width=\linewidth` |
| period_fusion_v7_annotated | 在 refined 上增加 $\bar{f}[c]$、$\ell\ge\lfloor L/2\rfloor$ 等公式 | `width=\linewidth` |
| period_fusion_v7_nature | 浅灰 panel 底 + inferno 色图 | `width=\linewidth` |
| period_fusion_v7_dense | STFT 带 Power(dB) colorbar，略宽 | `width=\linewidth` |
| period_fusion_v7_tall | refined 加高版，单栏排版更舒展 | `width=\linewidth` |
| period_fusion_v7_caption | annotated + 底部双分支说明 caption | `width=\linewidth` |

## STFT / ACF 单图

| 文件 | 用途 | 建议 LaTeX 宽度 |
|------|------|-----------------|
| stft_v6_pub_twopanel | 论文 Fig STFT 双栏替换 v5 | `width=0.9\linewidth` |
| acf_v6_pub_twopanel | 论文 Fig ACF 双栏替换 v5 | `width=0.9\linewidth` |
| period_fusion_v6_bar | 单独融合权重柱状图 | `width=0.48\linewidth` |
| stft_v6_compact / acf_v6_compact | 单栏排版 | `width=0.48\linewidth` |

## 文件列表

- `acf_v6_compact.pdf` / `.png`
- `acf_v6_pub_twopanel.pdf` / `.png`
- `period_fusion_v6_bar.pdf` / `.png`
- `period_fusion_v6_overview_row.pdf` / `.png`
- `period_fusion_v7_annotated.pdf` / `.png`
- `period_fusion_v7_caption.pdf` / `.png`
- `period_fusion_v7_dense.pdf` / `.png`
- `period_fusion_v7_nature.pdf` / `.png`
- `period_fusion_v7_refined.pdf` / `.png`
- `period_fusion_v7_tall.pdf` / `.png`
- `stft_v6_compact.pdf` / `.png`
- `stft_v6_pub_twopanel.pdf` / `.png`

重新生成：

```bash
cd fedcsl
Rscript scripts/figures_r/generate_review_batch.R "$(pwd)"
```
