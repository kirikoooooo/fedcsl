#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客户端选择超参数网格搜索脚本
批量并行训练，每次4个任务
"""

import os
import sys
import yaml
import subprocess
import multiprocessing
import copy
from datetime import datetime
from pathlib import Path
import shutil

# 设置基础参数
DATASET = "Epilepsy-TSTCC"
BASE_CONFIG = "configEpilepsy.yml"
NUM_ROUNDS = 50  # 50轮训练
MAX_PARALLEL = 4  # 最大并行数

# 超参数网格
CLIENT_SELECTION_RATIOS = [0.5, 0.6, 0.7, 0.8]  # 采样比例
MIN_SELECTION_PROBS = [0.005, 0.01, 0.02, 0.05]  # 最低选择概率
EMA_ALPHAS = [0.1, 0.2, 0.3, 0.5]  # EMA平滑系数

# 创建结果目录
RESULT_DIR = Path("./grid_search_results")
RESULT_DIR.mkdir(exist_ok=True)


def create_config(task_id, ratio, min_prob, ema_alpha):
    """创建配置文件（直接复制完整配置文件，只修改需要的部分）"""
    # 使用task_id确保文件名唯一，避免并行任务冲突
    config_name = f"config_grid_task{task_id}_ratio{ratio}_min{min_prob}_ema{ema_alpha}.yml"
    config_path = RESULT_DIR / config_name
    
    # 直接复制完整的基础配置文件
    shutil.copy2(BASE_CONFIG, config_path)
    
    # 读取完整配置（保留所有原有配置项）
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 只修改客户端选择相关的参数，其他配置保持不变
    if 'federated' not in config:
        config['federated'] = {}
    
    # 修改客户端选择参数
    config['federated']['use_client_selection'] = True
    config['federated']['client_selection_ratio'] = float(ratio)
    config['federated']['min_selection_prob'] = float(min_prob)
    config['federated']['ema_alpha'] = float(ema_alpha)
    config['federated']['numRound'] = NUM_ROUNDS
    
    # 只修改description，其他字段保持不变
    config['description'] = f"GridSearch_task{task_id}_ratio{ratio}_min{min_prob}_ema{ema_alpha}"
    
    # 保存配置（保留所有原有配置项）
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    return config_path


def extract_best_acc(log_file):
    """从日志文件中提取最佳准确率"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找 "best round is X, acc is Y"
            for line in content.split('\n'):
                if 'best round' in line and 'acc is' in line:
                    # 提取准确率
                    parts = line.split('acc is')
                    if len(parts) > 1:
                        acc_str = parts[1].strip().split()[0]
                        try:
                            return float(acc_str)
                        except ValueError:
                            pass
    except Exception as e:
        print(f"读取日志文件失败 {log_file}: {e}")
    return None


def run_task(args):
    """运行单个任务"""
    task_id, ratio, min_prob, ema_alpha = args
    config_path = None  # 初始化，确保在finally中能访问
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"开始任务 {task_id}: ratio={ratio}, min_prob={min_prob}, ema_alpha={ema_alpha}")
        
        # 创建配置文件（使用task_id确保唯一性）
        config_path = create_config(task_id, ratio, min_prob, ema_alpha)
        
        # 运行训练
        log_file = RESULT_DIR / f"task_{task_id}_ratio{ratio}_min{min_prob}_ema{ema_alpha}.log"
        
        cmd = [
            sys.executable,
            "FedCSL_Epilepsy.py",
            "-dataset", DATASET,
            "--config", str(config_path)
        ]
        
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                timeout=3600 * 2  # 2小时超时
            )
        
        # 提取最终准确率
        best_acc = extract_best_acc(log_file)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"任务 {task_id} 完成: best_acc={best_acc}")
        
        return {
            'task_id': task_id,
            'ratio': ratio,
            'min_prob': min_prob,
            'ema_alpha': ema_alpha,
            'best_acc': best_acc,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"任务 {task_id} 超时")
        return {
            'task_id': task_id,
            'ratio': ratio,
            'min_prob': min_prob,
            'ema_alpha': ema_alpha,
            'best_acc': None,
            'success': False
        }
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"任务 {task_id} 出错: {e}")
        return {
            'task_id': task_id,
            'ratio': ratio,
            'min_prob': min_prob,
            'ema_alpha': ema_alpha,
            'best_acc': None,
            'success': False
        }
    finally:
        # 无论成功或失败，都清理临时配置文件
        if config_path is not None:
            try:
                if config_path.exists():
                    config_path.unlink()
                    print(f"已清理临时配置文件: {config_path}")
            except Exception as e:
                print(f"清理配置文件失败 {config_path}: {e}")


def main():
    """主函数"""
    # 生成任务列表
    tasks = []
    task_id = 0
    for ratio in CLIENT_SELECTION_RATIOS:
        for min_prob in MIN_SELECTION_PROBS:
            for ema_alpha in EMA_ALPHAS:
                task_id += 1
                tasks.append((task_id, ratio, min_prob, ema_alpha))
    
    print(f"总共生成 {len(tasks)} 个任务")
    print(f"最大并行数: {MAX_PARALLEL}")
    print(f"预计总时间: 约 {len(tasks) / MAX_PARALLEL * 2:.1f} 小时（假设每个任务2小时）")
    
    # 初始化结果摘要文件
    summary_file = RESULT_DIR / "results_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("task_id|ratio|min_prob|ema_alpha|best_acc|success\n")
    
    # 并行执行任务
    print("\n开始执行任务...")
    with multiprocessing.Pool(processes=MAX_PARALLEL) as pool:
        results = pool.map(run_task, tasks)
    
    # 保存结果
    with open(summary_file, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['task_id']}|{result['ratio']}|{result['min_prob']}|"
                   f"{result['ema_alpha']}|{result['best_acc']}|{result['success']}\n")
    
    # 生成排序后的结果
    sorted_file = RESULT_DIR / "results_sorted.txt"
    valid_results = [r for r in results if r['best_acc'] is not None]
    valid_results.sort(key=lambda x: x['best_acc'] or 0, reverse=True)
    
    with open(sorted_file, 'w', encoding='utf-8') as f:
        f.write("task_id|ratio|min_prob|ema_alpha|best_acc|success\n")
        for result in valid_results:
            f.write(f"{result['task_id']}|{result['ratio']}|{result['min_prob']}|"
                   f"{result['ema_alpha']}|{result['best_acc']}|{result['success']}\n")
    
    print(f"\n所有任务完成！")
    print(f"结果摘要保存在: {summary_file}")
    print(f"排序后的结果保存在: {sorted_file}")
    
    # 打印最佳结果
    if valid_results:
        best = valid_results[0]
        print(f"\n最佳结果:")
        print(f"  ratio={best['ratio']}, min_prob={best['min_prob']}, "
              f"ema_alpha={best['ema_alpha']}, best_acc={best['best_acc']:.4f}")


if __name__ == "__main__":
    main()

