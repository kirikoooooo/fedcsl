# Spilter 算法架构设计规范

> 本文档记录 Spilter 的设计原则与实现约束，供代码实现时参考，防止架构偏移。

---

## 核心设计原则

### 1. 客户端本地模型持续训练（不被服务端覆盖）

**原则**：客户端的本地模型（`c.model`）在整个联邦训练过程中保持连续更新，
服务端每轮下发的聚合模型**绝对不能**直接写入或覆盖客户端的本地模型参数。

**类比**：与 FedCSL（ACF）完全一致——FedCSL 的客户端也不会被服务端参数覆盖。

**代码约束（禁止行为）**：
- 禁止把 `server_state_cpu` 写入 `c.model`（无论全量还是部分尺度）
- 禁止任何形式的 `local_state.update(scale_state)` 再 load 进 client
- 禁止全量清除 `c.optimizer.state`

---

### 2. 服务端下发模型仅作 Teacher（蒸馏/对比参考）

**原则**：服务端下发的聚合模型通过 `c.Global_Model`（即 `teacher.model`）挂载，
仅用于对比损失和知识蒸馏，**梯度不回传给 teacher**（requires_grad=False）。

Teacher 参与的损失项：
- 跨模型对比损失（UseJointCL）
- 跨模型知识蒸馏（UseJointKD）
- 尺度级对比/蒸馏（UseScaleCL / UseScaleKD）

**Teacher 仅在 round >= 1 时创建**（round 0 无有效聚合模型，不创建 teacher）。

---

### 3. 全局聚合基于客户端自身训练结果

**原则**：全局聚合使用客户端自己训练后的选定尺度参数（`c.model.scale_state_dict`），
**不是**服务端上一轮下发的参数。

数据流：
```
c.model 训练 1 epoch（持续，不重置）
  -> result["scale_states"] = c.model.scale_state_dict(selected_scales)
  -> _aggregate_scale_updates(server_state_cpu, client_scale_states, y_fed)
  -> _load_state_to_model(server.model, w_global)
     （只更新 server.model，不写回任何 clientList[i].model）
```

---

## 与 FedCSL 的对比

| 维度 | FedCSL (ACF) | Spilter |
|------|-------------|---------|
| 本地训练尺度 | 全部 8 个尺度 | 选定的 top-m 个尺度（默认 4/8）|
| 特征编码 | model.forward() 全尺度拼接 | encode_scale * m，子模型拼接 |
| 全局聚合 | FedAvg 全模型 | 按尺度 FedAvg，只聚合上传尺度 |
| 客户端参数是否被覆盖 | 否 | **否（核心约束）** |
| Teacher 来源 | server.model（FedAvg 聚合） | server.model（尺度聚合）|
| Teacher 是否写回 client | 否，仅蒸馏 | **否，仅蒸馏** |

---

## 尺度分配机制（local_score_topm 模式）

- 每个客户端基于本地 ACF 周期评分，选取评分最高的 top-m 个尺度
- 计划在训练开始前一次性生成（`cached_client_scale_plans`），全程固定不变
- 强制覆盖检查：每个尺度至少被 1 个客户端覆盖
- `system_extra_scale_count` 在此模式下不参与（由 `local_top_m` 直接控制数量）

---

## 特征编码注意事项（Stitched 模式）

客户端本地训练使用「**FedCSL ``forward(optimize=None)`` 同源顺序**」，仅在所选尺度子集上计算，不全尺度过分支。

默认实现（`stitched_feature_source: selected_scales_only`）调用  
`LearningShapeletsModelMixDistances.encode_mix_forward_selected_scales`：

- 欧氏 / 余弦 / 互相关各 **`forward_subset`**（只跑所选尺度 block，等价于该分支上的 m 尺度子编码器一次 forward）；
- 与 **FedCSL 全模 forward** 一致：各分支先把所选尺度在通道维拼接成一向量，再对该向量做 **layer_norm**（结构上对应 ``ln1`` / ``ln2`` / ``ln3``；维度为「当前尺度集合在该分支上的总长」，使用 ``functional.layer_norm``，与固定宽度的 ``nn.LayerNorm`` 模块权重无关）；
- 按全局尺度边界拆回各行后，在同一尺度位置上 **拼接 eu∥co∥cc**，再按 `selected_scales` 顺序展平 —— 与 ``forward`` 里 ``reshape`` + ``torch.cat(outs, dim=2)`` + ``reshape`` 的布局一致（只是尺度维长度为 m）。
- **与 ``forward_slices`` 的差别**：后者 LN 统计量来自 **全部 8 个尺度**；默认路径 LN 仅基于所选 m 个尺度，数值可与切片全模输出不同。
- `UseScaleCL` / `UseScaleKD` 辅助项复用同一套 **按尺度 mix 向量**，不对同一尺度二次 forward。

**不应**再使用「整条表征单一 layer_norm」代替上述三分支结构；
欧氏距离值域远大于余弦/互相关，若违背 FedCSL 的三分支 LN，欧氏特征会在统计量上压制其余两支（见 ``LearningShapeletsModelMixDistances.forward``）。

对照模式（可选）：

```yaml
spilter:
  # 全 8 尺度 forward(optimize=None) 后切片 —— LN 统计量与全模一致，算力仍是全尺度
  stitched_feature_source: forward_slices
```

旧版复现（辅助项再跑一遍 `encode_scale`，计算翻倍；stitched 主项现为 FedCSL 同源子集编码）：

```yaml
spilter:
  stitched_feature_source: subset_ln_forward_scale
```

---

## 禁止操作清单（AI 编码助手必读）

1. 禁止在 `_train_client_worker` 中把任何服务端参数写入 `c.model`
2. 禁止全量或部分替换 `c.model` 的参数
3. 禁止全量清除 `c.optimizer.state`（只允许针对特定参数选择性清除）
4. 禁止对 teacher 模型的 requires_grad 设为 True
5. 禁止在聚合后把 `w_global` 写回各 `clientList[idx].model`

---

## 最后更新

2026-05-17：`selected_scales_only` 默认走 ``encode_mix_forward_selected_scales``（FedCSL ``forward`` 同源 LN→拼接顺序，仅所选尺度）；可选 ``forward_slices`` / ``subset_ln_forward_scale``。
