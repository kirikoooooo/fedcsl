"""SCAFFOLD (Karimireddy et al., ICML 2020).

直接移植自 FL-bench ``src/server/scaffold.py`` 与 ``src/client/scaffold.py``，
仅替换 imports 以接入本仓库的 ``flbench_compat`` 基座；算法实现（控制变量 c、
delta_y/delta_c、服务端 global_lr 等）与 FL-bench 原版逐行一致。
"""
from .server import SCAFFOLDServer
from .client import SCAFFOLDClient

__all__ = ["SCAFFOLDServer", "SCAFFOLDClient"]
