# -*- coding: utf-8 -*-
"""多尺度对比学习 (UseScaleCL)。

对应 FedCSL 原 ``update_CL`` 中 (对每个 scale 分片调用)::

    if config["ablation"]["UseScaleCL"]:
        loss_global_CLKD_mutiscale += self.loss_func(logits_g, labels_g) * precisions[length_i]

其中 ``logits_g`` 用全局分片 ``q_g[i]`` 与本地分片 ``k[i]`` 构造。
"""
from __future__ import annotations

import torch

from .primitives import infonce_labels, infonce_logits


def scale_contrastive_loss(
    qi_g: torch.Tensor,
    ki: torch.Tensor,
    loss_func,
    temperature: float,
    weight: float = 1.0,
) -> torch.Tensor:
    logits = infonce_logits(qi_g, ki, temperature)
    labels = infonce_labels(qi_g.shape[0], qi_g.device)
    return loss_func(logits, labels) * float(weight)


__all__ = ["scale_contrastive_loss"]
