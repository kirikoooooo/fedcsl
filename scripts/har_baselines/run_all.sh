#!/usr/bin/env bash
# HAR 数据集上，全部 dirichlet 值 × 全部 baseline（含 FedU2 / Spilter）的汇总入口。
# 依次串行调用 scripts/har_baselines/har_dir*.sh（按 α 数值从小到大）。
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/har_baselines/run_all.sh
#
# 并行方案（按不同 GPU 手动分配）示例:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/har_baselines/har_dir0.1.sh &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/har_baselines/har_dir1.0.sh &

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 按 α 数值升序排序（shell 通配符的字典序会把 '10.0' 排到 '2.0' 前，需显式数值排序）
SORTED=$(ls "${HERE}"/har_dir*.sh 2>/dev/null \
  | awk -F'har_dir|\\.sh' '{print $2"\t"$0}' \
  | sort -g -k1,1 \
  | cut -f2)

if [[ -z "${SORTED}" ]]; then
  echo "未找到 har_dir*.sh 脚本。" >&2
  exit 1
fi

echo "将依次执行:"
while IFS= read -r s; do
  echo "  - ${s}"
done <<< "${SORTED}"
echo ""

while IFS= read -r s; do
  bash "${s}"
done <<< "${SORTED}"

echo "全部脚本执行完毕。日志位于 result/har_baselines/"
