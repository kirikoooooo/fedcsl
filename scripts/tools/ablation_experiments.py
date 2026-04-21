#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客户端选择消融实验脚本（Python版本）
使用方法: python ablation_experiments.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 配置参数
BASE_CONFIG = "configEpilepsy.yml"
DATASET = "Epilepsy-TSTCC"
LOG_DIR = Path("logs")
MAX_PARALLEL = 4  # 最大并行数

# 创建日志目录
LOG_DIR.mkdir(exist_ok=True)


def run_command(cmd, log_file, background=False):
    """运行命令"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行: {cmd}")
    print(f"  日志文件: {log_file}")
    
    if background:
        # 后台运行
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        return process
    else:
        # 前台运行
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        return result


def build_command(ratio=None, min_prob=None, ema_alpha=None):
    """构建命令"""
    cmd_parts = [
        sys.executable,
        "FedCSL_Epilepsy.py",
        f"-dataset {DATASET}",
        f"--config {BASE_CONFIG}"
    ]
    
    if ratio is not None or min_prob is not None or ema_alpha is not None:
        cmd_parts.append("--use-client-selection")
    
    if ratio is not None:
        cmd_parts.append(f"--client-selection-ratio {ratio}")
    if min_prob is not None:
        cmd_parts.append(f"--min-selection-prob {min_prob}")
    if ema_alpha is not None:
        cmd_parts.append(f"--ema-alpha {ema_alpha}")
    
    return " ".join(cmd_parts)


def run_parallel_commands(commands_and_logs, max_parallel=MAX_PARALLEL):
    """并行运行多个命令"""
    processes = []
    
    for cmd, log_file in commands_and_logs:
        # 等待直到有可用槽位
        while len(processes) >= max_parallel:
            # 检查已完成的进程
            processes = [p for p in processes if p.poll() is None]
            if len(processes) >= max_parallel:
                import time
                time.sleep(1)
        
        # 启动新进程
        process = run_command(cmd, log_file, background=True)
        processes.append(process)
    
    # 等待所有进程完成
    print(f"\n等待所有任务完成...")
    for process in processes:
        process.wait()
    print("所有任务完成！\n")


def main():
    """主函数"""
    print("=" * 60)
    print("客户端选择消融实验")
    print("=" * 60)
    print(f"数据集: {DATASET}")
    print(f"配置文件: {BASE_CONFIG}")
    print(f"日志目录: {LOG_DIR}")
    print(f"最大并行数: {MAX_PARALLEL}")
    print("=" * 60 + "\n")
    
    # ============================================
    # Baseline: FedAvg without client selection
    # ============================================
    # print("\n=== Baseline (FedAvg without client selection) ===")
    # cmd = build_command()
    # run_command(cmd, LOG_DIR / "baseline_fedavg.log", background=False)
    
    # ============================================
    # Ablation 1: 不同采样比例 (ratio)
    # 固定: min_prob=0.01, ema_alpha=0.2
    # ============================================
    # print("\n=== Ablation 1: Different client selection ratios ===")
    # commands = []
    # for ratio in [0.5, 0.6, 0.7, 0.8]:
    #     cmd = build_command(ratio=ratio, min_prob=0.01, ema_alpha=0.2)
    #     log_file = LOG_DIR / f"ablation_ratio{ratio}.log"
    #     commands.append((cmd, log_file))
    # run_parallel_commands(commands, max_parallel=MAX_PARALLEL)
    
    # ============================================
    # Ablation 2: 不同最小选择概率 (min_prob)
    # 固定: ratio=0.7, ema_alpha=0.2
    # ============================================
    print("\n=== Ablation 2: Different minimum selection probabilities ===")
    commands = []
    for min_prob in [0.005, 0.01, 0.02, 0.05]:
        cmd = build_command(ratio=0.7, min_prob=min_prob, ema_alpha=0.2)
        log_file = LOG_DIR / f"ablation_minprob{min_prob}.log"
        commands.append((cmd, log_file))
    run_parallel_commands(commands, max_parallel=MAX_PARALLEL)
    
    # ============================================
    # Ablation 3: 不同EMA平滑系数 (ema_alpha)
    # 固定: ratio=0.7, min_prob=0.01
    # ============================================
    print("\n=== Ablation 3: Different EMA smoothing coefficients ===")
    commands = []
    for ema_alpha in [0.1, 0.2, 0.3, 0.5]:
        cmd = build_command(ratio=0.7, min_prob=0.01, ema_alpha=ema_alpha)
        log_file = LOG_DIR / f"ablation_ema{ema_alpha}.log"
        commands.append((cmd, log_file))
    run_parallel_commands(commands, max_parallel=MAX_PARALLEL)
    
    print("\n" + "=" * 60)
    print("所有消融实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

