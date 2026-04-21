"""FL-bench 兼容层：提供 FedAvgClient / FedAvgServer / SequentialTrainer 的精简实现，
    足以直接承载从 FL-bench 原仓库搬运过来的 ``scaffold.py`` / ``fedproto.py`` 等方法文件。

精简原则：
- 只实现 SCAFFOLD / FedProto 真正调用到的属性与方法；
- 参数对象 ``args`` 用轻量 ``AttrDict``（与 OmegaConf DictConfig 的属性访问兼容）；
- 训练器为顺序（无 Ray）实现；
- 数据流：一个 ``BaseDataset`` 持有所有客户端样本，每个客户端通过 ``data_indices`` 划分 train 子集
  （val/test 为空；全局评估由 runner 层单独处理，避免与 FedCSL 现有评估流水线冲突）。

与 FL-bench 行为一致的关键点：
- ``FedAvgClient.fit()``、``set_parameters()``、``package()``、``train()`` 签名与 FL-bench 一致；
- ``FedAvgServer.package(client_id)`` 返回 ``client_id / local_epoch / regular_model_params /
    personal_model_params / optimizer_state / lr_scheduler_state / return_diff`` 这些键；
- ``aggregate_client_updates`` 支持 ``return_diff=True/False`` 两种模式；
- 参数聚合权重 = 每客户端 ``package["weight"]``；SCAFFOLD 会在其 server 端覆盖为等权。
"""
from .fedavg import FedAvgClient, FedAvgServer, AttrDict
from .trainer import SequentialTrainer
from .dataset import TensorBaseDataset

__all__ = [
    "FedAvgClient",
    "FedAvgServer",
    "AttrDict",
    "SequentialTrainer",
    "TensorBaseDataset",
]
