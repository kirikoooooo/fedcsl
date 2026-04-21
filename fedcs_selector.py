# -*- coding: utf-8 -*-
"""向后兼容 shim：实际实现已迁移到 :mod:`algo.client_selection.fedcs`。"""
from algo.client_selection.fedcs import (  # noqa: F401
    FedCSClientSelector,
    create_fedcs_selector,
)

__all__ = ["FedCSClientSelector", "create_fedcs_selector"]
