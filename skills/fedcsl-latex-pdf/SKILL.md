---
name: fedcsl-latex-pdf
description: Compile the FedCSL Chinese NeurIPS LaTeX paper under 论文撰写 to PDF with tectonic, including the proven command flow, output checks, and common local failure handling.
---

# FedCSL LaTeX PDF Export

Use this skill when the user wants to export the FedCSL paper to PDF from this project.

| File | Language |
|------|----------|
| `论文撰写/论文overleaf[不含omp，含spilter].tex` | Chinese (main) |
| `论文撰写/paper_fedcsl_spilter_en.tex` | English (full translation) |

## Proven Command Flow

Preferred: use the project export scripts under `论文撰写/`:

```powershell
# Windows PowerShell
cd 论文撰写
.\build_pdf.ps1
# or double-click build_pdf.cmd
# .\build_pdf.ps1 -Target "论文overleaf.tex"
```

```bash
# Git Bash / macOS / Linux
cd 论文撰写
./build_pdf.sh
# ./build_pdf.sh '论文overleaf.tex'
```

Scripts automatically:

- locate `tectonic` (Codex bundled path, `%LOCALAPPDATA%\Programs\tectonic`, or `PATH`)
- auto-pick `*spilter*.tex`, else `论文overleaf.tex`, when `-Target` is omitted
- sync `fedcsl/figs/har_mem_mean_epoch_bar.*` to `draw/figs/` before compile
- run `tectonic --keep-intermediates --keep-logs` on the target `.tex`

Manual equivalent:

```bash
cd 论文撰写
tectonic --keep-intermediates --keep-logs '论文overleaf[不含omp，含spilter].tex'
ls -lh '论文overleaf[不含omp，含spilter].pdf' '论文overleaf[不含omp，含spilter].log'
```

Expected PDF:

```text
/Users/lixiongfei/Nutstore Files/我的坚果云/Golang ReStudy/论文/draw/fedcsl/论文撰写/论文overleaf[不含omp，含spilter].pdf
```

On this machine, `tectonic` is the working compiler. `latexmk`, `xelatex`, and `bibtex`
may not be installed as standalone commands; `tectonic` handles the TeX and BibTeX passes
for this paper.

## Notes

- The paper contains Chinese text and is intended for XeLaTeX-compatible compilation.
- Keep `--keep-intermediates --keep-logs` so `.log`, `.aux`, `.bbl`, and related files are available for debugging.
- If `tectonic` panics inside a filesystem sandbox with a macOS system-configuration error, rerun the same command outside the sandbox with user approval. This was the successful path for the current environment.
- Successful compilation may still print layout warnings such as `Underfull \vbox`, `Overfull \hbox`, or duplicate PDF object warnings. Treat these as warnings unless `tectonic` exits nonzero or the PDF is missing.
- Verify success by checking that the PDF exists, is nonempty, and `file` reports `PDF document`.
