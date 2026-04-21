# -*- coding: utf-8 -*-
"""Joint 全局-本地 KL 蒸馏 (UseJointKD)。

对应 FedCSL 原 ``update_CL`` 中::

    if config["ablation"]["UseJointKD"]:
        loss_local_jointCLKD += direct_kl_loss(q, q_g) * zeta + direct_kl_loss(k, k_g)
"""
from __future__ import annotations

import torch

from .primitives import direct_kl_loss


def joint_distill_loss(
    q: torch.Tensor,
    q_g: torch.Tensor,
    k: torch.Tensor,
    k_g: torch.Tensor,
    zeta: float,
) -> torch.Tensor:
    return direct_kl_loss(q, q_g) * zeta + direct_kl_loss(k, k_g)


__all__ = ["joint_distill_loss"]
