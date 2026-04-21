# -*- coding: utf-8 -*-
"""Joint 全局-本地对比学习 (UseJointCL)。

将全局模型编码的 ``q_g`` 与本地编码的 ``k`` 做 InfoNCE，用于迫使本地表征
与全局对齐。对应 FedCSL 原 ``update_CL`` 中::

    if config["ablation"]["UseJointCL"]:
        loss_local_jointCLKD += self.loss_func(logits_g, labels_g)
"""
from __future__ import annotations

import torch

from .primitives import infonce_labels, infonce_logits


def joint_contrastive_loss(
    q_g: torch.Tensor,
    k: torch.Tensor,
    loss_func,
    temperature: float,
) -> torch.Tensor:
    logits = infonce_logits(q_g, k, temperature)
    labels = infonce_labels(q_g.shape[0], q_g.device)
    return loss_func(logits, labels)


__all__ = ["joint_contrastive_loss"]
