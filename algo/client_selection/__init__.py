# -*- coding: utf-8 -*-
"""客户端选择策略包。

统一接口：``ClientSelector``。具体策略：
- ``uniform`` → :class:`UniformSelector`
- ``omp``     → :class:`OMPSelector`
- ``oort``    → :class:`OortSelector`
- ``fedcs``   → :class:`FedCSSelector`

使用示例::

    from algo.client_selection import make_selector
    selector = make_selector('omp', num_clients=N, sample_nums=K, config=cfg,
                             min_selection_prob=0.01, ema_alpha=0.3)
"""
from __future__ import annotations

from typing import Any

from .base import ClientSelector
from .fedcs import FedCSSelector
from .omp import OMPSelector
from .oort import OortSelector
from .uniform import UniformSelector


_REGISTRY = {
    "uniform": UniformSelector,
    "omp": OMPSelector,
    "oort": OortSelector,
    "fedcs": FedCSSelector,
}


def make_selector(method: str, **kwargs: Any) -> ClientSelector:
    """工厂函数：根据策略名返回对应的 ``ClientSelector`` 实例。"""
    key = (method or "").lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"未知客户端选择方法: {method!r}，可选项: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[key]
    return cls(**kwargs)


def available_methods() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "ClientSelector",
    "UniformSelector",
    "OMPSelector",
    "OortSelector",
    "FedCSSelector",
    "make_selector",
    "available_methods",
]
