# -*- coding: utf-8 -*-
"""对比学习通用原语：InfoNCE logits / labels 构造与 KL 散度等。"""
from __future__ import annotations

import torch
from torch import nn

# 从项目的 utils 重新导出，避免下游模块重复实现。
from utils import direct_kl_loss  # noqa: F401


def infonce_logits(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    """计算 ``q`` 与 ``k`` 的 InfoNCE logits（均为先 L2 归一化再内积）。

    输入两个 ``[B, D]`` 的特征，输出 ``[B, B]`` 的 logits（已除 ``temperature``）。
    """
    qn = nn.functional.normalize(q, dim=1)
    kn = nn.functional.normalize(k, dim=1)
    logits = torch.einsum("nc,ck->nk", [qn, kn.t()])
    return logits / temperature


def infonce_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    """返回与 ``infonce_logits`` 配套的对角标签 ``[0, 1, ..., B-1]``。"""
    return torch.arange(batch_size, dtype=torch.long, device=device)


__all__ = ["infonce_logits", "infonce_labels", "direct_kl_loss"]
