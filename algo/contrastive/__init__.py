# -*- coding: utf-8 -*-
"""FedCSL 对比/蒸馏损失组件。

FedCSL 的 ``update_CL`` 由 4 个可独立开关的 loss 项组合而成，本包把它们各自
抽成独立模块，便于消融实验与未来扩展新的对比方法：

- ``local_infonce``：局部数据增强 InfoNCE (基础项，不受消融开关控制)
- ``joint_cl``    : 全局↔本地 Joint InfoNCE (``ablation.UseJointCL``)
- ``joint_kd``    : 全局↔本地 Joint KL 蒸馏 (``ablation.UseJointKD``)
- ``scale_cl``    : 多尺度全局↔本地 InfoNCE (``ablation.UseScaleCL``)
- ``scale_kd``    : 多尺度全局↔本地 KL 蒸馏 (``ablation.UseScaleKD``)

以及通用原语 ``primitives``（``infonce_logits``, ``infonce_labels``, ``direct_kl_loss``）。
"""
from __future__ import annotations

from .joint_cl import joint_contrastive_loss
from .joint_kd import joint_distill_loss
from .local_infonce import local_infonce_loss
from .primitives import direct_kl_loss, infonce_labels, infonce_logits
from .scale_cl import scale_contrastive_loss
from .scale_kd import scale_distill_loss

__all__ = [
    "local_infonce_loss",
    "joint_contrastive_loss",
    "joint_distill_loss",
    "scale_contrastive_loss",
    "scale_distill_loss",
    "infonce_logits",
    "infonce_labels",
    "direct_kl_loss",
]
