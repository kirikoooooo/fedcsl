# -*- coding: utf-8 -*-
"""MOON model-contrastive loss.

MOON contrasts the current local representation with the current global model
as the positive pair and the previous local model as the negative pair.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def moon_contrastive_loss(
    z_local: torch.Tensor,
    z_global: torch.Tensor,
    z_previous: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    z_local = F.normalize(z_local, dim=1)
    z_global = F.normalize(z_global, dim=1)
    z_previous = F.normalize(z_previous, dim=1)
    pos = F.cosine_similarity(z_local, z_global, dim=1).unsqueeze(1)
    neg = F.cosine_similarity(z_local, z_previous, dim=1).unsqueeze(1)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(z_local.shape[0], dtype=torch.long, device=z_local.device)
    return F.cross_entropy(logits, labels)


__all__ = ["moon_contrastive_loss"]
