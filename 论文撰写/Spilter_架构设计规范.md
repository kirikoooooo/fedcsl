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

客户端本地训练使用"拼接子模型"（默认与 FedCSL 共用一次 `forward(optimize=None)`，再 `slice_scale_features` 拼接所选尺度）：

```
feat_q = model(x_q, optimize=None)
q_stitched = cat([slice_scale_features(feat_q, s) for s in selected_scales])
# 各尺度辅助损失：同样从 feat_q / feat_k / teacher 上切片，不再二次 forward_scale
```

**归一化应分距离类型**（euclidean / cosine / cross-corr）分别进行，
与 FedCSL `forward()` 中 ln1/ln2/ln3 的处理一致。

### 默认实现：`forward_slices`（与 FedCSL 同构、省显存）

配置项 `spilter.stitched_feature_source` 默认为 `forward_slices`：
对每个视图只做一次 `model(x, optimize=None)`，stitched 表征与各尺度辅助损失均通过对输出使用 **`slice_scale_features`** 切片得到，
不再对 stitched 与 auxiliary **重复**调用 `forward_scale` / `encode_scale`。这样 m=4 时的 autograd 峰值与 FedCSL「整模 forward + 循环切片」同量级，
避免出现「m4 显存反而大于 FedCSL」的反直觉现象。

若论文/对照实验需要恢复旧行为（子集上的 `_encode_stitched_scales` LN + 辅助项独立 forward），设置：

```yaml
spilter:
  stitched_feature_source: subset_ln_forward_scale
```

原因：欧氏距离值域远大于余弦相似度（[0,1]），若用单个 layer_norm
对所有特征一起归一化，欧氏距离会主导方差，余弦/互相关特征被压制，
导致梯度信号失效，这是 loss 不降的根本 bug（见 _encode_stitched_scales 修复）。

---

## 禁止操作清单（AI 编码助手必读）

1. 禁止在 `_train_client_worker` 中把任何服务端参数写入 `c.model`
2. 禁止全量或部分替换 `c.model` 的参数
3. 禁止全量清除 `c.optimizer.state`（只允许针对特定参数选择性清除）
4. 禁止对 teacher 模型的 requires_grad 设为 True
5. 禁止在聚合后把 `w_global` 写回各 `clientList[idx].model`

---

## 最后更新

2026-05-17：补充 stitched `forward_slices` 默认路径（显存与 FedCSL 同构）；保留 `subset_ln_forward_scale` legacy。
