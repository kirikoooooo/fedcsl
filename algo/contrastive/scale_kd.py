# -*- coding: utf-8 -*-
"""多尺度 KL 蒸馏 (UseScaleKD)。

对应 FedCSL 原 ``update_CL`` 中 (对每个 scale 分片调用)::

    if config["ablation"]["UseScaleKD"]:
        loss_global_CLKD_mutiscale += (
            direct_kl_loss(qi_g, qi) * precisions[length_i] * zeta
            + direct_kl_loss(ki_g, ki) * precisions[length_i]
        )
"""
from __future__ import annotations

import torch

from .primitives import direct_kl_loss


def scale_distill_loss(
    qi_g: torch.Tensor,
    qi: torch.Tensor,
    ki_g: torch.Tensor,
    ki: torch.Tensor,
    weight: float,
    zeta: float,
) -> torch.Tensor:
    w = float(weight)
    return direct_kl_loss(qi_g, qi) * w * zeta + direct_kl_loss(ki_g, ki) * w


__all__ = ["scale_distill_loss"]
