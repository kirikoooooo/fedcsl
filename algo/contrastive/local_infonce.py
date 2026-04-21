# -*- coding: utf-8 -*-
"""基础局部 InfoNCE：两次数据增强得到的 q / k 对比。

对应 FedCSL ``update_CL`` 中最外层的 ``loss = loss_func(logits, labels) * gamma``。
"""
from __future__ import annotations

import torch

from .primitives import infonce_labels, infonce_logits


def local_infonce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    loss_func,
    temperature: float,
) -> torch.Tensor:
    """局部 InfoNCE 损失（无权重，调用方负责乘 gamma）。"""
    logits = infonce_logits(q, k, temperature)
    labels = infonce_labels(q.shape[0], q.device)
    return loss_func(logits, labels)


__all__ = ["local_infonce_loss"]
