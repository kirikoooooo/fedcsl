"""Baseline 用的 shapelet 特征 + 线性分类头包装，兼容 FL-bench 方法所需的
``classifier.in_features`` / ``get_last_features(x, detach=False)`` 接口。

设计说明：
- 复用 FedCSL 既有的 ``LearningShapeletsModel`` / ``LearningShapeletsModelMixDistances``
  作为编码器，保持 backbone 与 CSL 一致，公平对比；
- 外接一个 ``self.classifier = nn.Linear(num_shapelets, num_classes)``，与 FL-bench
  ``DecoupledModel`` 的 ``base + classifier`` 约定一致；
- 本模型只做 **分类**（CE/原型正则/SCAFFOLD 控制变量校正），不使用多尺度对比、联合
  蒸馏、结构对齐等 FedCSL 专属损失 —— 满足"用这两个方法时不做按尺度对比"的要求。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances


class ShapeletClassifier(nn.Module):
    def __init__(
        self,
        *,
        shapelets_size_and_len: dict,
        in_channels: int,
        num_classes: int,
        dist_measure: str = "mix",
        to_cuda: bool = True,
    ) -> None:
        super().__init__()
        if dist_measure == "mix":
            encoder = LearningShapeletsModelMixDistances(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )
        else:
            encoder = LearningShapeletsModel(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )
        self.encoder = encoder
        self.num_feat = int(encoder.num_shapelets)
        # 外部分类头——FL-bench ``DecoupledModel.classifier`` 约定。
        self.classifier = nn.Linear(self.num_feat, num_classes)
        self._to_cuda = bool(to_cuda) and torch.cuda.is_available()
        if self._to_cuda:
            self.cuda()

    # ------------------------------------------------------------------
    # FL-bench 侧需要的接口
    # ------------------------------------------------------------------
    def get_last_features(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """对应 ``DecoupledModel.get_last_features``：返回 classifier 之前的特征。"""
        z = self.encoder(x, optimize=None)
        if detach:
            z = z.detach()
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """默认 FedAvg/SCAFFOLD 用路径：直接给 logits。"""
        z = self.get_last_features(x)
        return self.classifier(z)

    # ------------------------------------------------------------------
    # 用于 runner 的全局下游 SVC 评估（与 ``LearningShapeletsCL.transform`` 接口对齐）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def transform(
        self,
        X,
        *,
        batch_size: int = 512,
        normalize: bool = True,
        result_type: str = "numpy",
    ):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float)
        self.eval()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=False
        )
        outs = []
        for (x,) in dl:
            if self._to_cuda:
                x = x.cuda()
            outs.append(self.get_last_features(x).cpu())
        z = torch.cat(outs, 0)
        if normalize:
            z = nn.functional.normalize(z, dim=1)
        if result_type == "tensor":
            return z
        return z.detach().numpy()
