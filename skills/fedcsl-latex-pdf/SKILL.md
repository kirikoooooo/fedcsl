---
name: fedcsl-latex-pdf
description: Compile the FedCSL Chinese NeurIPS LaTeX paper under 论文撰写 to PDF with tectonic, including the proven command flow, output checks, and common local failure handling.
---

# FedCSL LaTeX PDF Export

Use this skill when the user wants to export the FedCSL paper
`论文撰写/论文overleaf[不含omp，含spilter].tex` to PDF from this project.

## Proven Command Flow

Run from the repository root:

```bash
cd "/Users/lixiongfei/Nutstore Files/我的坚果云/Golang ReStudy/论文/draw/fedcsl/论文撰写"
tectonic --keep-intermediates --keep-logs '论文overleaf[不含omp，含spilter].tex'
ls -lh '论文overleaf[不含omp，含spilter].pdf' '论文overleaf[不含omp，含spilter].log'
file '论文overleaf[不含omp，含spilter].pdf'
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
