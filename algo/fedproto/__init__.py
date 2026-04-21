"""FedProto (Tan et al., AAAI 2022)，移植自 FL-bench ``src/server/fedproto.py``
与 ``src/client/fedproto.py``。"""
from .server import FedProtoServer
from .client import FedProtoClient

__all__ = ["FedProtoServer", "FedProtoClient"]
