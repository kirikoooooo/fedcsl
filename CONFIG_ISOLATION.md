# 配置文件隔离机制说明

## 问题

在并行运行多个网格搜索任务时，如果多个任务使用相同的配置文件名，会导致：
1. **配置文件冲突**：多个任务可能同时读写同一个配置文件
2. **配置混乱**：一个任务的配置可能被另一个任务覆盖
3. **原始配置被修改**：如果直接修改基础配置文件，可能破坏原始配置

## 解决方案

### 1. 唯一配置文件名

每个任务使用包含 `task_id` 的唯一配置文件名：
```python
config_name = f"config_grid_task{task_id}_ratio{ratio}_min{min_prob}_ema{ema_alpha}.yml"
```

**优点**：
- 每个任务有独立的配置文件
- 并行任务不会互相干扰
- 即使任务失败，配置文件也不会被其他任务覆盖

### 2. 配置文件隔离存储

所有临时配置文件保存在 `RESULT_DIR` 中：
```python
config_path = RESULT_DIR / config_name
```

**优点**：
- 不污染项目根目录
- 所有临时文件集中管理
- 便于清理和查找

### 3. 深拷贝保护原始配置

使用 `copy.deepcopy()` 确保不修改原始配置：
```python
config = yaml.load(f, Loader=yaml.FullLoader)
config = copy.deepcopy(config)  # 深拷贝，确保不修改原始配置
```

**优点**：
- 原始配置文件 `configEpilepsy.yml` 永远不会被修改
- 每个任务独立修改配置副本
- 即使脚本出错，原始配置也安全

### 4. 自动清理机制

使用 `finally` 块确保临时配置文件被清理：
```python
finally:
    # 无论成功或失败，都清理临时配置文件
    if config_path is not None:
        try:
            if config_path.exists():
                config_path.unlink()
        except Exception as e:
            print(f"清理配置文件失败 {config_path}: {e}")
```

**优点**：
- 即使任务失败或超时，也会清理临时文件
- 避免磁盘空间浪费
- 保持结果目录整洁

## 文件结构

```
项目根目录/
├── configEpilepsy.yml          # 原始配置文件（永远不会被修改）
├── grid_search_client_selection.py
├── grid_search_client_selection.sh
└── grid_search_results/       # 结果目录
    ├── config_grid_task1_ratio0.5_min0.005_ema0.1.yml  # 临时配置（会自动清理）
    ├── config_grid_task2_ratio0.5_min0.005_ema0.2.yml
    ├── task_1_ratio0.5_min0.005_ema0.1.log
    ├── task_2_ratio0.5_min0.005_ema0.2.log
    ├── results_summary.txt
    └── results_sorted.txt
```

## 安全性保证

1. **原始配置保护**：使用深拷贝，原始配置文件永远不会被修改
2. **并行安全**：每个任务使用唯一的配置文件名，不会冲突
3. **自动清理**：临时配置文件在使用后自动删除
4. **异常处理**：即使任务失败，也会尝试清理临时文件

## 验证方法

运行脚本后，检查：

1. **原始配置文件未被修改**：
   ```bash
   # 检查原始配置文件的时间戳
   ls -l configEpilepsy.yml
   ```

2. **临时配置文件已清理**：
   ```bash
   # 结果目录中不应该有config_grid_*.yml文件
   ls grid_search_results/config_grid_*.yml
   # 应该返回：No such file or directory
   ```

3. **配置文件唯一性**：
   ```bash
   # 检查是否有重复的配置文件名（不应该有）
   ls grid_search_results/config_grid_*.yml | sort | uniq -d
   ```

## 注意事项

1. **如果脚本被强制中断**（如Ctrl+C），可能留下临时配置文件
   - 解决方法：手动清理 `grid_search_results/config_grid_*.yml`

2. **如果磁盘空间不足**，配置文件可能无法创建
   - 解决方法：确保有足够的磁盘空间

3. **如果权限不足**，可能无法创建或删除配置文件
   - 解决方法：确保对结果目录有读写权限

