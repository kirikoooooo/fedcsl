# -*- coding: utf-8 -*-
"""向后兼容 shim：实际实现已迁移到 :mod:`algo.client_selection.oort`。

保留此文件是为了让 ``from oort_selector import create_oort_selector`` 等旧式
导入仍然有效。
"""
from algo.client_selection.oort import (  # noqa: F401
    OortTrainingSelector,
    _make_oort_args,
    create_oort_selector,
)

__all__ = ["OortTrainingSelector", "create_oort_selector", "_make_oort_args"]
