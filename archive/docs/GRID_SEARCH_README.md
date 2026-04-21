# 客户端选择超参数网格搜索使用说明

## 问题分析

### TestAcc波动大的可能原因：

1. **EMA平滑系数过大** (`ema_alpha > 0.3`)
   - 概率更新过于激进，变化剧烈
   - 建议：使用 0.1-0.2 的较小值

2. **采样比例过小** (`client_selection_ratio < 0.6`)
   - 参与聚合的客户端太少，模型代表性不足
   - 建议：使用 0.7-0.8 的较大值

3. **最小概率设置不当** (`min_selection_prob < 0.01`)
   - 某些客户端几乎不被选中，概率分布极不均匀
   - 建议：使用 0.01-0.05 的较大值

4. **OMP稀疏向量不稳定**
   - 训练初期模型参数变化大，导致稀疏向量波动
   - 影响：即使有EMA平滑，概率仍然波动

5. **采样随机性**
   - 随机采样导致每轮选择的客户端不同
   - 增加了额外的随机性

## 使用方法

### 方法1: Python脚本（推荐）

```bash
# 直接运行Python脚本
python grid_search_client_selection.py
```

**优点**：
- 跨平台（Windows/Linux/Mac）
- 更好的错误处理
- 自动提取结果

**参数调整**：
编辑 `grid_search_client_selection.py` 中的参数：
```python
CLIENT_SELECTION_RATIOS = [0.5, 0.6, 0.7, 0.8]  # 采样比例
MIN_SELECTION_PROBS = [0.005, 0.01, 0.02, 0.05]  # 最低选择概率
EMA_ALPHAS = [0.1, 0.2, 0.3, 0.5]  # EMA平滑系数
MAX_PARALLEL = 4  # 最大并行数
NUM_ROUNDS = 50   # 训练轮数
```

### 方法2: Shell脚本（Linux/Mac）

```bash
# 添加执行权限
chmod +x grid_search_client_selection.sh

# 运行脚本
./grid_search_client_selection.sh
```

**注意**：Windows用户需要使用Git Bash或WSL

## 输出结果

### 结果文件

1. **results_summary.txt**: 所有任务的原始结果
   ```
   task_id|ratio|min_prob|ema_alpha|best_acc|success
   1|0.5|0.005|0.1|0.8234|True
   ...
   ```

2. **results_sorted.txt**: 按准确率排序的结果（最佳在前）

3. **task_*.log**: 每个任务的详细日志

### 结果分析

运行完成后，查看 `results_sorted.txt` 找到最佳超参数组合：

```bash
# 查看前10个最佳结果
head -11 grid_search_results/results_sorted.txt
```

## 超参数建议

基于问题分析，推荐的超参数范围：

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `client_selection_ratio` | 0.7-0.8 | 更多客户端参与，更稳定 |
| `min_selection_prob` | 0.01-0.05 | 保证所有客户端都有机会 |
| `ema_alpha` | 0.1-0.2 | 更平滑的概率更新 |

## 预期结果

通过网格搜索，期望找到：
- **最佳超参数组合**: 在准确率和稳定性之间取得平衡
- **参数敏感性**: 了解哪些参数对结果影响最大
- **优化方向**: 为进一步优化提供指导

## 注意事项

1. **资源需求**: 每个任务需要2-4小时（取决于数据集和硬件）
   - 总任务数 = 4 × 4 × 4 = 64个
   - 4个并行：约32小时
   - 建议在服务器上运行

2. **磁盘空间**: 每个任务会生成日志文件，确保有足够空间

3. **中断恢复**: 如果中断，可以手动删除已完成的任务，重新运行

4. **监控**: 建议定期检查日志，确保任务正常进行

## 快速测试

如果想快速测试少量参数组合：

```python
# 修改 grid_search_client_selection.py
CLIENT_SELECTION_RATIOS = [0.7]  # 只测试一个值
MIN_SELECTION_PROBS = [0.01, 0.02]  # 测试两个值
EMA_ALPHAS = [0.1, 0.2]  # 测试两个值
# 总共 1 × 2 × 2 = 4 个任务
```

## 故障排除

### 问题1: 配置文件不存在
```
错误: 找不到 configEpilepsy.yml
解决: 确保在项目根目录运行脚本
```

### 问题2: Python模块缺失
```
错误: No module named 'yaml'
解决: pip install pyyaml
```

### 问题3: 任务超时
```
解决: 增加超时时间（在代码中修改 timeout 参数）
或减少训练轮数 NUM_ROUNDS
```

### 问题4: 内存不足
```
解决: 减少 MAX_PARALLEL（如改为2）
```

