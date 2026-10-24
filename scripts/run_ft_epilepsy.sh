#!/usr/bin/env bash
# =============================================================================
# 下游微调消融（协议 A：冻结 encoder + 线性头）· Epilepsy 先行小数据集
#
# 用法（在服务器项目根目录执行）:
#   bash scripts/run_ft_epilepsy.sh          # 三个方法并行跑在 GPU 1/2/3
#   bash scripts/run_ft_epilepsy.sh 2 3 4    # 指定三张卡
#
# 每个方法：K=50 客户端、R=200 轮、Dirichlet α=0.1 预训练一次，
# 训练结束后基于最后一轮全局模型跑 5%/10%/20%/50%/100% × 3 采样种子的微调消融。
# 结果写入 result/Epilepsy/*.txt，final-round 模型存 checkpoint/Epilepsy/。
#
# 若与你的 Epilepsy 主实验口径（K/Round/α）不一致，请修改下方对应参数。
# =============================================================================
set -u
G_FEDCSL=${1:-1}
G_M4=${2:-2}
G_M2=${3:-3}

cd "$(dirname "$0")/.." || { echo "[ERROR] 无法进入项目根目录"; exit 1; }

# 前置检查
if ! grep -q "ft-ablation" FedCSL_All.py; then
    echo "[ERROR] FedCSL_All.py 不是新版本（缺少 ft-ablation 代码），请先同步代码"
    exit 1
fi
[ -f Epilepsy/train.pt ] && [ -f Epilepsy/test.pt ] || {
    echo "[ERROR] 缺少 ./Epilepsy/train.pt 或 ./Epilepsy/test.pt"
    exit 1
}

COMMON_ARGS="-dataset Epilepsy-TSTCC --num-client 50 --num-rounds 200 --dirichlet-alpha 0.1 \
  --server-gpu 0 --client-workers 1 \
  --ft-ratios 0.05,0.1,0.2,0.5,1.0 --ft-seeds 3"

# ① FedCSL
CUDA_VISIBLE_DEVICES=$G_FEDCSL nohup python -u FedCSL_All.py $COMMON_ARGS \
  --config configACF.yml --description "ft_epi_fedcsl" \
  > ft_epi_fedcsl.out 2>&1 &
echo "[1/3] FedCSL     -> GPU $G_FEDCSL  (PID $!)"

# ② Spilter-m4
CUDA_VISIBLE_DEVICES=$G_M4 nohup python -u FedCSL_All.py $COMMON_ARGS \
  --config configSpilter.yml --description "ft_epi_spilter_m4" \
  > ft_epi_spilter_m4.out 2>&1 &
echo "[2/3] Spilter-m4 -> GPU $G_M4  (PID $!)"

# ③ Spilter-m2
CUDA_VISIBLE_DEVICES=$G_M2 nohup python -u FedCSL_All.py $COMMON_ARGS \
  --config configSpilter.yml --spilter-top-m 2 --description "ft_epi_spilter_m2" \
  > ft_epi_spilter_m2.out 2>&1 &
echo "[3/3] Spilter-m2 -> GPU $G_M2  (PID $!)"

echo ""
echo "查看实时进度:  tail -f ft_epi_spilter_m2.out"
echo "查看微调结果:  tail -30 result/Epilepsy/*ft_epi_spilter_m2*.txt"
echo "停止:          kill <PID>"
