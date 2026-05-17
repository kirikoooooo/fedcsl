#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 论文一键编译脚本（基于 Docker TeXLive，支持 xelatex + bibtex + 中文 ctex）
#
# 用法：
#   bash build_pdf.sh                                 # 默认编译"论文overleaf[不含omp，含spilter].tex"
#   bash build_pdf.sh "论文overleaf.tex"              # 编译指定 .tex 文件
#   CLEAN=1 bash build_pdf.sh                         # 强制清掉中间产物再编（含旧 .bbl）
#
# 关键设计：
#   1. 把宿主机的 论文/draw/ 整个挂进容器，使 .tex 里 "../../figs/" 能正确解析
#   2. 容器工作目录 = "论文撰写/"
#   3. 由于 bibtex 不支持含中括号 `[` `]` 与某些中文路径的文件名（会报
#      "Could not open bibtex log file"），编译时把 .tex 复制为 ASCII 临时名
#      paper_build.tex，编译完成后把 paper_build.pdf 重命名回原文件名 + .pdf。
#   4. latexmk -xelatex 自动跑 xelatex→bibtex→xelatex→xelatex 直到引用稳定。
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"
PAPER_DIR_HOST="$(pwd)"
ROOT_HOST="$(cd ../.. && pwd)"
PAPER_DIR_REL="$(python3 -c "import os; print(os.path.relpath('$PAPER_DIR_HOST', '$ROOT_HOST'))")"

TARGET="${1:-论文overleaf[不含omp，含spilter].tex}"

if [[ ! -f "$TARGET" ]]; then
  echo "[err] 找不到目标文件：$TARGET" >&2
  exit 1
fi

STEM="${TARGET%.tex}"
TMP_STEM="paper_build"
TMP_TEX="${TMP_STEM}.tex"
TMP_PDF="${TMP_STEM}.pdf"

IMAGE="texlive/texlive:latest"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[info] 镜像 $IMAGE 不存在，开始拉取（首次约 5-6 GB）..."
  docker pull "$IMAGE"
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[info] CLEAN=1：清掉旧中间产物（针对原名 + 临时名）"
  rm -f "${STEM}".{aux,bbl,blg,log,out,toc,xdv,fls,fdb_latexmk}
  rm -f "${TMP_STEM}".{aux,bbl,blg,log,out,toc,xdv,fls,fdb_latexmk,pdf,tex}
fi

# 把原 .tex 拷贝为 ASCII 临时名（cp 而非 mv，保留原文件）
cp "$TARGET" "$TMP_TEX"

echo "[info] 项目挂载根 (host)         : $ROOT_HOST"
echo "[info] 容器内 .tex 相对路径       : $PAPER_DIR_REL/$TMP_TEX"
echo "[info] 编译目标 (临时 ASCII 名)  : $TMP_TEX  (源: $TARGET)"

cleanup() {
  rm -f "$TMP_TEX"
}
trap cleanup EXIT

docker run --rm \
  -v "$ROOT_HOST:/work" \
  -w "/work/$PAPER_DIR_REL" \
  --user "$(id -u):$(id -g)" \
  "$IMAGE" \
  latexmk -xelatex -shell-escape -interaction=nonstopmode -halt-on-error -f \
  "$TMP_TEX"

if [[ -f "$TMP_PDF" ]]; then
  mv -f "$TMP_PDF" "${STEM}.pdf"
  echo "[ok] 编译完成 → ${STEM}.pdf"
  ls -lh "${STEM}.pdf"
else
  echo "[err] 没有生成 $TMP_PDF，请查 ${TMP_STEM}.log" >&2
  exit 1
fi
