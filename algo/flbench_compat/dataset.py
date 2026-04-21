"""最小 ``BaseDataset`` 兼容类，适配时序 (N, C, L) 输入。

FL-bench 的 ``FedAvgClient`` 期望：
- ``self.dataset`` 暴露 ``train()`` / ``eval()`` 做 transform 切换（本仓库里无增广，空实现即可）；
- 支持 ``Subset(self.dataset, indices=...)``，因此需实现 ``__len__`` 与 ``__getitem__``；
- ``__getitem__(i)`` 返回 ``(x_tensor, y_tensor)``。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TensorBaseDataset(Dataset):
    """把 (X, y) 两个 numpy / tensor 数组包装为类 FL-bench BaseDataset 的接口。"""

    def __init__(self, X, y) -> None:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.X = X.contiguous()
        self.y = y.long().contiguous()
        self._mode = "train"

    # FL-bench 语义：切换到训练/验证模式（本项目无 transform 差异，留空占位）。
    def train(self) -> None:
        self._mode = "train"

    def eval(self) -> None:
        self._mode = "eval"

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
