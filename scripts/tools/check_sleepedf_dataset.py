#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看数据集规模的脚本
支持：SleepEDF, HAR, Epilepsy, FD-A 等
"""

import torch
import numpy as np
import os
import argparse
from collections import Counter

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def check_dataset(dataset_name="SleepEDF"):
    """检查数据集规模
    
    Args:
        dataset_name: 数据集名称，如 'SleepEDF', 'HAR', 'Epilepsy', 'FD-A'
    """
    print("=" * 80)
    print(f"{dataset_name} 数据集规模信息")
    print("=" * 80)
    
    # 根据数据集名称确定路径
    dataset_paths = {
        "SleepEDF": "./sleepEDF",
        "HAR": "./HAR",
        "Epilepsy": "./Epilepsy",
        "FD-A": "./FD-A"
    }
    
    if dataset_name not in dataset_paths:
        print(f"❌ 错误：不支持的数据集 '{dataset_name}'")
        print(f"   支持的数据集: {', '.join(dataset_paths.keys())}")
        return
    
    base_path = dataset_paths[dataset_name]
    
    # 检查文件是否存在
    train_path = os.path.join(base_path, "train.pt")
    test_path = os.path.join(base_path, "test.pt")
    val_path = os.path.join(base_path, "val.pt")
    
    files_info = []
    if os.path.exists(train_path):
        files_info.append(("训练集", train_path))
    if os.path.exists(test_path):
        files_info.append(("测试集", test_path))
    if os.path.exists(val_path):
        files_info.append(("验证集", val_path))
    
    if not files_info:
        print(f"❌ 错误：未找到数据集文件！")
        print(f"   请确保以下文件存在：")
        print(f"   - {train_path}")
        print(f"   - {test_path}")
        print(f"   - {val_path} (可选)")
        return
    
    # 加载并分析每个数据集
    all_data = {}
    
    for name, path in files_info:
        print(f"\n📁 {name}: {path}")
        file_size = os.path.getsize(path)
        print(f"   文件大小: {format_size(file_size)}")
        
        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            
            if isinstance(data, dict):
                if 'samples' in data and 'labels' in data:
                    X = data['samples']
                    y = data['labels']
                    
                    # 转换为numpy进行分析
                    if isinstance(X, torch.Tensor):
                        X_np = X.numpy()
                    else:
                        X_np = np.array(X)
                    
                    if isinstance(y, torch.Tensor):
                        y_np = y.numpy()
                    else:
                        y_np = np.array(y)
                    
                    # 基本信息
                    print(f"   样本数量: {len(y_np):,}")
                    print(f"   数据形状: {X_np.shape}")
                    
                    if len(X_np.shape) == 3:
                        N, C, T = X_np.shape
                        print(f"   - 样本数 (N): {N:,}")
                        print(f"   - 通道数 (C): {C}")
                        print(f"   - 时间步长 (T): {T}")
                    elif len(X_np.shape) == 2:
                        N, T = X_np.shape
                        print(f"   - 样本数 (N): {N:,}")
                        print(f"   - 特征维度 (T): {T}")
                    
                    # 标签信息
                    unique_labels, counts = np.unique(y_np, return_counts=True)
                    num_classes = len(unique_labels)
                    print(f"   类别数量: {num_classes}")
                    print(f"   类别分布:")
                    
                    label_counter = Counter(y_np)
                    for label in sorted(unique_labels):
                        count = label_counter[int(label)]
                        percentage = count / len(y_np) * 100
                        print(f"     类别 {label}: {count:,} 个样本 ({percentage:.2f}%)")
                    
                    # 数据统计信息
                    print(f"   数据统计:")
                    print(f"     - 最小值: {X_np.min():.4f}")
                    print(f"     - 最大值: {X_np.max():.4f}")
                    print(f"     - 均值: {X_np.mean():.4f}")
                    print(f"     - 标准差: {X_np.std():.4f}")
                    
                    # 数据类型和内存占用
                    if isinstance(X, torch.Tensor):
                        memory_mb = X.element_size() * X.nelement() / (1024 ** 2)
                        print(f"   内存占用: {memory_mb:.2f} MB")
                        print(f"   数据类型: {X.dtype}")
                    
                    all_data[name] = {
                        'X': X_np,
                        'y': y_np,
                        'shape': X_np.shape,
                        'num_samples': len(y_np),
                        'num_classes': num_classes,
                        'class_distribution': label_counter
                    }
                else:
                    print(f"   ⚠️  警告：文件格式不符合预期，缺少'samples'或'labels'键")
                    print(f"   文件中的键: {list(data.keys())}")
            else:
                print(f"   ⚠️  警告：文件格式不符合预期，不是字典格式")
                print(f"   数据类型: {type(data)}")
                
        except Exception as e:
            print(f"   ❌ 加载文件时出错: {e}")
    
    # 汇总信息
    if len(all_data) > 0:
        print("\n" + "=" * 80)
        print("📊 数据集汇总")
        print("=" * 80)
        
        total_samples = sum([info['num_samples'] for info in all_data.values()])
        print(f"总样本数: {total_samples:,}")
        
        # 检查类别一致性
        all_classes = set()
        for name, info in all_data.items():
            all_classes.update(info['class_distribution'].keys())
        
        print(f"总类别数: {len(all_classes)}")
        print(f"类别列表: {sorted(all_classes)}")
        
        # 各数据集占比
        print(f"\n数据集分布:")
        for name, info in all_data.items():
            percentage = info['num_samples'] / total_samples * 100
            print(f"  {name}: {info['num_samples']:,} 个样本 ({percentage:.2f}%)")
        
        # 类别分布对比
        if len(all_data) > 1:
            print(f"\n各类别在不同数据集中的分布:")
            for label in sorted(all_classes):
                print(f"  类别 {label}:")
                for name, info in all_data.items():
                    count = info['class_distribution'].get(int(label), 0)
                    if info['num_samples'] > 0:
                        percentage = count / info['num_samples'] * 100
                        print(f"    {name}: {count:,} ({percentage:.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 数据集检查完成")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看数据集规模信息')
    parser.add_argument('--dataset', '-d', type=str, default='SleepEDF',
                       choices=['SleepEDF', 'HAR', 'Epilepsy', 'FD-A'],
                       help='要检查的数据集名称 (默认: SleepEDF)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='检查所有支持的数据集')
    
    args = parser.parse_args()
    
    if args.all:
        # 检查所有数据集
        datasets = ['SleepEDF', 'HAR', 'Epilepsy', 'FD-A']
        for dataset in datasets:
            check_dataset(dataset)
            print("\n" + "=" * 80 + "\n")
    else:
        check_dataset(args.dataset)

