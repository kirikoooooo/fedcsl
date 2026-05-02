"""FedCSL 主入口：负责数据加载、客户端/服务器初始化、客户端选择与聚合、下游 SVC 评估。

SCAFFOLD / FedProto 等 FL-bench 原生算法会在识别到 ``algo`` 字段后路由到
``algo/baseline_runner.run_baseline``，跳过多尺度对比流程。
"""

import argparse
import copy
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from torch import nn, optim

from algo.client_selection import make_selector, available_methods
from dataset_utils import *   # noqa: F401,F403  (LoadDataset_* 全家桶)
from fedavg import fedavg
from fedutil import cal_score, normalize_to_near_one
from train import LearningShapeletsCL
from utils import period_score



parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='UEA dataset name')
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
parser.add_argument('-T', '--temperature', default=0.1, type=float, help='temperature')
parser.add_argument('-l', '--lmd', default=1e-2, type=float, help='multi-scale alignment weight')
parser.add_argument('-ls', '--lmd_s', default=1.0, type=float, help='SDL weight')
parser.add_argument('-a', '--alpha', default=0.5, type=float, help='covariance matrix decay')
parser.add_argument('-b', '--batch-size', type=int, default=None,
                    help='Batch size (None=use config file value)')
parser.add_argument('-g', '--to-cuda', default=True, type=bool)
parser.add_argument('-e', '--eval-per-x-epochs', default=10, type=int)
parser.add_argument('-d', '--dist-measure', default='mix', type=str)
#parser.add_argument('-r', '--rank', default=-1, type=int)
parser.add_argument('-w', '--world-size', default=-1, type=int)
parser.add_argument('-p', '--port', default=15535, type=int)
parser.add_argument('-r', '--resize', default=0, type=int)
parser.add_argument('-c', '--checkpoint', default=False, type=bool)
parser.add_argument('--task', default='classification', type=str)
parser.add_argument('--config', default='./config.yml', type=str, help='Path to the config file')
# 客户端选择超参数（命令行参数，会覆盖配置文件中的值）
parser.add_argument('--use-client-selection', action='store_true', help='Enable client selection')
parser.add_argument('--client-selection-ratio', type=float, default=None, help='Client selection ratio (0.0-1.0)')
parser.add_argument('--client-selection-method', type=str, default=None, choices=['uniform', 'omp', 'oort', 'fedcs'], help='Client selection: uniform, omp, oort, or fedcs (adaptive client sampling)')
parser.add_argument('--min-selection-prob', type=float, default=None, help='Minimum selection probability')
parser.add_argument('--ema-alpha', type=float, default=None, help='EMA smoothing coefficient (0.0-1.0)')
parser.add_argument('--description', type=str, default=None, help='Experiment description (overrides config file)')
parser.add_argument('--dirichlet-alpha', type=float, default=None, help='Dirichlet alpha for data heterogeneity (overrides config file)')
parser.add_argument('--server-gpu', type=int, default=None, help='GPU id used by the server/model evaluation thread')
parser.add_argument('--client-gpus', type=str, default=None,
                    help='Comma-separated GPU ids for client worker threads, e.g. "0,1,2"')
parser.add_argument('--client-workers', type=int, default=None,
                    help='Number of client training worker threads; default comes from config or 3')

args = parser.parse_args()
def _resolve_config_path(path):
    """支持三种写法：
    1. 绝对路径或存在的相对路径 → 直接使用；
    2. 纯文件名或 'config.yml' 等不存在时，自动在 ``config/`` 目录下查找；
    3. 老写法 './configXXX.yml' / 'configXXX.yml' 亦会被重定向到 ``config/configXXX.yml``。
    """
    if os.path.isfile(path):
        return path
    candidate = os.path.join('config', os.path.basename(path))
    if os.path.isfile(candidate):
        print(f"[config] '{path}' 未找到，重定向到 '{candidate}'")
        return candidate
    return path  # 让后续 open 抛出明确错误


_config_path = _resolve_config_path(args.config)
with open(_config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)  # yaml 已返回全新对象，无需再 deepcopy


def _sanitize_filename(s):
    """按当前操作系统替换非法文件名字符，保证 Win/Linux 兼容。"""
    if s is None:
        return ""
    s = str(s)
    if os.name == "nt":
        # Windows 非法: \ / : * ? " < > |
        invalid_chars = r'\/:*?"<>|'
    else:
        # Linux/macOS: 仅路径分隔符 / 和空字符非法
        invalid_chars = "/\x00"
    for c in invalid_chars:
        s = s.replace(c, "_")
    return s.strip() or "run"


def _is_splitteacher_algo(algo_name):
    return str(algo_name).lower() == "fedcsl-onehot-splitteacher"


def _is_teacher_scale_set_algo(algo_name):
    algo = str(algo_name).lower()
    return algo in ("fedcsl-onehot-fullteacher", "fedcsl-onehot-splitteacher")


def _uses_fedcsl_scale_scores(algo_name):
    algo = str(algo_name).lower()
    return algo in (
        "fedcsl",
        "fedcsl-onehot",
        "fedcsl-onehot-fullteacher",
        "fedcsl-onehot-splitteacher",
    )


def _parse_gpu_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    value = str(value).strip()
    if not value:
        return []
    gpu_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            gpu_ids.extend(range(start, end + step, step))
        else:
            gpu_ids.append(int(part))
    return gpu_ids


def _make_device(gpu_id, to_cuda=True):
    if to_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")


def _state_dict_to_cpu(state_dict, clone=True):
    out = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            out[key] = tensor.clone() if clone else tensor
        else:
            out[key] = copy.deepcopy(value) if clone else value
    return out


def _load_state_to_model(model, state_dict):
    model_device = next(model.parameters()).device
    device_state = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in state_dict.items()
    }
    model.load_state_dict(device_state, strict=False)


def _client_scale_score_batch(X_client, model, beta, batch_size=64, device=None):
    """用少量本地样本估计该 client 本轮应使用的尺度索引。"""
    if isinstance(X_client, list):
        if len(X_client) == 0:
            return 0
        X_arr = np.asarray(X_client[: batch_size], dtype=np.float32)
    elif isinstance(X_client, np.ndarray):
        if X_client.shape[0] == 0:
            return 0
        X_arr = X_client[: batch_size].astype(np.float32, copy=False)
    else:
        if len(X_client) == 0:
            return 0
        X_arr = np.asarray(X_client[: batch_size], dtype=np.float32)

    x = torch.from_numpy(X_arr).float()
    model_device = device if device is not None else next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        x = x.to(model_device)
    num_shapelet_lengths = len(model.shapelets_size_and_len)
    num_shapelet_per_length = model.num_shapelets // num_shapelet_lengths
    pscore = period_score(x, alpha=beta)
    if pscore is None or len(pscore) == 0:
        with torch.no_grad():
            feat = model(x, optimize=None, masking=False)
            feat = nn.functional.normalize(feat, dim=1)
            pscore = np.zeros(num_shapelet_lengths, dtype=np.float32)
            for i in range(num_shapelet_lengths):
                sl = feat[:, i * num_shapelet_per_length: (i + 1) * num_shapelet_per_length]
                pscore[i] = float(sl.pow(2).sum().item())
    pscore = np.asarray(pscore, dtype=np.float32).ravel()
    return int(np.argmax(pscore))


def _normalize_scale_scores(pscore, num_shapelet_lengths):
    pscore = np.asarray(pscore, dtype=np.float32).ravel()
    if pscore.size < num_shapelet_lengths:
        padded = np.zeros(num_shapelet_lengths, dtype=np.float32)
        padded[:pscore.size] = pscore
        pscore = padded
    elif pscore.size > num_shapelet_lengths:
        pscore = pscore[:num_shapelet_lengths]
    return np.nan_to_num(pscore, nan=0.0, posinf=0.0, neginf=0.0)


def _client_scale_scores_batch(X_client, model, beta, batch_size=64, device=None):
    """返回该 client 的整条尺度打分向量，供 server 做多尺度规划。"""
    if isinstance(X_client, list):
        if len(X_client) == 0:
            return np.zeros(len(model.shapelets_size_and_len), dtype=np.float32)
        X_arr = np.asarray(X_client[: batch_size], dtype=np.float32)
    elif isinstance(X_client, np.ndarray):
        if X_client.shape[0] == 0:
            return np.zeros(len(model.shapelets_size_and_len), dtype=np.float32)
        X_arr = X_client[: batch_size].astype(np.float32, copy=False)
    else:
        if len(X_client) == 0:
            return np.zeros(len(model.shapelets_size_and_len), dtype=np.float32)
        X_arr = np.asarray(X_client[: batch_size], dtype=np.float32)

    x = torch.from_numpy(X_arr).float()
    model_device = device if device is not None else next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        x = x.to(model_device)

    num_shapelet_lengths = len(model.shapelets_size_and_len)
    num_shapelet_per_length = model.num_shapelets // num_shapelet_lengths
    pscore = period_score(x, alpha=beta)
    if pscore is None or len(pscore) == 0:
        with torch.no_grad():
            feat = model(x, optimize=None, masking=False)
            feat = nn.functional.normalize(feat, dim=1)
            pscore = np.zeros(num_shapelet_lengths, dtype=np.float32)
            for i in range(num_shapelet_lengths):
                sl = feat[:, i * num_shapelet_per_length: (i + 1) * num_shapelet_per_length]
                pscore[i] = float(sl.pow(2).sum().item())
    return _normalize_scale_scores(pscore, num_shapelet_lengths)


def _pick_grouped_scales_from_scores(pscore):
    """按前半/后半尺度各选一个高分尺度，近似对应中短/中长各保留一个。"""
    num_shapelet_lengths = len(pscore)
    if num_shapelet_lengths <= 0:
        return []
    if num_shapelet_lengths == 1:
        return [0]

    split = max(1, num_shapelet_lengths // 2)
    groups = [list(range(0, split))]
    if split < num_shapelet_lengths:
        groups.append(list(range(split, num_shapelet_lengths)))

    selected = []
    for group in groups:
        if not group:
            continue
        group_scores = pscore[group]
        scale_idx = group[int(np.argmax(group_scores))]
        if scale_idx not in selected:
            selected.append(int(scale_idx))

    if not selected:
        selected.append(int(np.argmax(pscore)))
    return selected


def _precompute_client_scale_scores(X_fed, model, beta, device=None, batch_size=64):
    """仅基于原始客户端数据做一次尺度评分，供后续各轮复用。"""
    client_scores = []
    for X_client in X_fed:
        client_scores.append(
            _client_scale_scores_batch(
                X_client, model, beta=beta, batch_size=batch_size, device=device
            )
        )
    return client_scores


def _plan_balanced_client_scales_from_scores(client_scores, model, extra_scale_count=2):
    """每个 client 保留 2 个本地主尺度，再由 server 额外分配 2 个均衡尺度。"""
    num_clients = len(client_scores)
    num_scales = len(model.shapelets_size_and_len)
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64)

    client_selected = []
    scale_counts = np.zeros(num_scales, dtype=np.int64)

    for scores in client_scores:
        selected = _pick_grouped_scales_from_scores(scores)
        client_selected.append(list(selected))
        for scale_idx in selected:
            scale_counts[scale_idx] += 1

    for _ in range(max(0, int(extra_scale_count))):
        for cid in range(num_clients):
            available = [scale_idx for scale_idx in range(num_scales) if scale_idx not in client_selected[cid]]
            if not available:
                continue
            best_scale = min(
                available,
                key=lambda scale_idx: (
                    int(scale_counts[scale_idx]),
                    -float(client_scores[cid][scale_idx]),
                    int(scale_idx),
                ),
            )
            client_selected[cid].append(int(best_scale))
            scale_counts[best_scale] += 1

    return client_selected, scale_counts


def _aggregate_scale_updates(server_state, client_scale_states, y_fed):
    """仅聚合每个尺度被上传的参数，其它参数保持 server 原值。"""
    out = {k: v.detach().cpu().clone() for k, v in server_state.items()}
    scale_to_clients = {}
    for cid, payload in enumerate(client_scale_states):
        if not payload:
            continue
        states = payload.get("states", None)
        if not states:
            continue
        for scale_idx, state in states.items():
            if state:
                scale_to_clients.setdefault(int(scale_idx), []).append((cid, state))

    for _, client_items in scale_to_clients.items():
        total = sum(len(y_fed[cid]) for cid, _ in client_items)
        if total <= 0:
            continue
        keys = list(client_items[0][1].keys())
        for key in keys:
            agg = None
            for cid, state in client_items:
                weight = len(y_fed[cid]) / total
                value = state[key].detach().cpu()
                agg = value * weight if agg is None else agg + value * weight
            out[key] = agg
    return out


def _train_client_worker(
    worker_id,
    client_indices,
    device,
    clientList,
    X_fed,
    y_fed,
    total_samples,
    round_idx,
    numEpoch,
    batch_size,
    lr,
    wd,
    use_scale_split_comm,
    server_state_cpu,
    beta,
    shared_kwargs,
    client_scale_plans,
    client_scale_scores,
):
    if device.type == "cuda":
        torch.cuda.set_device(device)

    teacher = None
    if round_idx != 0 or use_scale_split_comm:
        teacher = LearningShapeletsCL(**{**shared_kwargs, "device": device})
        _load_state_to_model(teacher.model, server_state_cpu)
        teacher.model.eval()
        for p in teacher.model.parameters():
            p.requires_grad_(False)

    results = []
    for idx in client_indices:
        c = clientList[idx]
        c.set_device(device)
        if c.optimizer is None:
            c.set_optimizer(optim.SGD(c.model.parameters(), lr=lr, weight_decay=wd))
        else:
            for group in c.optimizer.param_groups:
                group["params"] = list(c.model.parameters())

        c.Q = len(y_fed[idx]) / total_samples if total_samples > 0 else 0.0
        c.Global_Model = teacher.model if round_idx != 0 and teacher is not None else None
        selected_scales = []
        if client_scale_plans is not None and idx < len(client_scale_plans):
            selected_scales = [int(scale_idx) for scale_idx in client_scale_plans[idx]]
        c.Selected_Scales = selected_scales if selected_scales else None
        if client_scale_scores is not None and idx < len(client_scale_scores):
            c.Cached_Scale_Scores = np.asarray(client_scale_scores[idx], dtype=np.float32).copy()
        else:
            c.Cached_Scale_Scores = None

        result = {
            "idx": idx,
            "loss": 0.0,
            "state": None,
            "scale_indices": list(selected_scales),
            "scale_states": None,
            "worker_id": worker_id,
            "device": str(device),
            "skipped": False,
        }

        if len(X_fed[idx]) == 0 or len(y_fed[idx]) == 0:
            print(f"[warn] client {idx} 数据为空，跳过训练")
            result["state"] = _state_dict_to_cpu(c.model.state_dict())
            result["skipped"] = True
            c.Global_Model = None
            c.Selected_Scales = None
            c.Cached_Scale_Scores = None
            results.append(result)
            continue

        if use_scale_split_comm:
            scale_state = {}
            selected_scale_indices = selected_scales or [
                _client_scale_score_batch(
                    X_fed[idx],
                    teacher.model if teacher is not None else c.model,
                    beta=beta,
                    device=device,
                )
            ]
            for scale_idx in selected_scale_indices:
                prefixes = c.model._scale_state_prefixes(scale_idx)
                for key, value in server_state_cpu.items():
                    if any(key.startswith(prefix) for prefix in prefixes):
                        scale_state[key] = value
            local_state = c.model.state_dict()
            local_state.update(scale_state)
            _load_state_to_model(c.model, local_state)
            result["scale_indices"] = [int(scale_idx) for scale_idx in selected_scale_indices]

        losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size,
                         epoch_idx=-1, lr=lr)
        if not losses:
            loss_all = 0.0
        else:
            loss_all = float(np.mean([loss[0] for loss in losses]))
            if not np.isfinite(loss_all):
                print(f"[warn] client {idx} loss NaN/Inf，置 0")
                loss_all = 0.0

        result["loss"] = loss_all
        result["state"] = _state_dict_to_cpu(c.model.state_dict())
        if use_scale_split_comm and result["scale_indices"]:
            result["scale_states"] = {
                int(scale_idx): c.model.scale_state_dict(scale_idx, clone=True, cpu=True)
                for scale_idx in result["scale_indices"]
            }
        c.Global_Model = None
        c.Selected_Scales = None
        c.Cached_Scale_Scores = None
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 下游 SVC 评估：在 [10^-4, 10^4] 的 9 个 C 里挑最优，再在测试集上评估
# ---------------------------------------------------------------------------
_SVC_C_GRID = [10 ** i for i in range(-4, 5)]


def eval(transformation, transformation_test, y_train, y_test):
    """训练集自评估选 C（不做 CV，更快），然后在测试集评估。"""
    best_acc, C_best = -1.0, _SVC_C_GRID[0]
    for C in _SVC_C_GRID:
        clf = SVC(C=C, random_state=42)
        clf.fit(transformation, y_train)
        acc = accuracy_score(clf.predict(transformation), y_train)
        if acc > best_acc:
            best_acc, C_best = acc, C
    clf = SVC(C=C_best, random_state=42)
    clf.fit(transformation, y_train)
    train_acc = accuracy_score(clf.predict(transformation), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc


def eval_TSTCC(transformation_train, transformation_test, transformation_val,
               y_train, y_test, y_val):
    """使用验证集选 C，再在测试集上评估（Epilepsy/SleepEDF/FD-A 等有 val.pt 的数据集）。"""
    best_val_acc, C_best = -1.0, _SVC_C_GRID[0]
    for C in _SVC_C_GRID:
        clf = SVC(C=C, random_state=42)
        clf.fit(transformation_train, y_train)
        acc_i = accuracy_score(clf.predict(transformation_val), y_val)
        if acc_i > best_val_acc:
            best_val_acc, C_best = acc_i, C
    clf = SVC(C=C_best, random_state=42)
    clf.fit(transformation_train, y_train)
    train_acc = accuracy_score(clf.predict(transformation_train), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc


def train(dataset="", seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True,
           eval_per_x_epochs=10, dist_measure='mix', rank=-1, world_size=-1, resize=0,
           checkpoint=False, task='classification'):
    # ----- DDP 初始化（可选；rank / world_size 默认 -1 表示单机）-----------------
    is_ddp = rank != -1 and world_size != -1
    if is_ddp:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # ----- seed：派生三个独立 seed 给 numpy / torch / cuda，避免相关性 -------------
    original_seed = seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

    # ----- 从 config + argv 派生超参（命令行优先）---------------------------------
    fed_cfg = config['federated']
    model_cfg = config['model']['params']

    numClient = fed_cfg['numClient']
    numRound = fed_cfg['numRound']
    numEpoch = fed_cfg['numEpoch']
    dirichlet_alpha = args.dirichlet_alpha if args.dirichlet_alpha is not None else fed_cfg['dirichlet_alpha']
    algo = config.get('algo', 'fedcsl')
    use_client_selection = args.use_client_selection if args.use_client_selection else fed_cfg.get('use_client_selection', False)
    client_selection_ratio = args.client_selection_ratio if args.client_selection_ratio is not None else fed_cfg.get('client_selection_ratio', 0.6)
    min_selection_prob = args.min_selection_prob if args.min_selection_prob is not None else fed_cfg.get('min_selection_prob', 0.01)
    ema_alpha = args.ema_alpha if args.ema_alpha is not None else fed_cfg.get('ema_alpha', 0.3)
    client_selection_method = args.client_selection_method if args.client_selection_method is not None else fed_cfg.get('client_selection_method', None)
    # 默认: fedavg→uniform，其余→omp
    if client_selection_method is None:
        client_selection_method = 'uniform' if algo == 'fedavg' else 'omp'

    dataset = args.dataset if args.dataset is not None else config.get('dataset', dataset)
    dist_measure = model_cfg['dist_measure']
    lr = model_cfg['lr']
    batch_size = args.batch_size if args.batch_size is not None else model_cfg['batch_size']
    wd = model_cfg['wd']
    ls = model_cfg['ls']
    l = model_cfg['l']
    beta = model_cfg.get('beta', 0.4)
    client_workers = int(
        args.client_workers if args.client_workers is not None
        else fed_cfg.get('client_workers', 3)
    )
    client_workers = max(1, client_workers)
    client_groups = [[] for _ in range(client_workers)]
    for idx in range(numClient):
        client_groups[idx % client_workers].append(idx)

    client_gpu_ids = (
        _parse_gpu_list(args.client_gpus)
        if args.client_gpus is not None
        else _parse_gpu_list(fed_cfg.get('client_gpus', None))
    )
    server_gpu = args.server_gpu if args.server_gpu is not None else fed_cfg.get('server_gpu', None)
    if not client_gpu_ids:
        if server_gpu is None:
            server_gpu = 0
        client_gpu_ids = [server_gpu]
    if server_gpu is None:
        server_gpu = client_gpu_ids[0]

    server_device = _make_device(server_gpu, to_cuda=to_cuda)
    client_devices = [
        _make_device(client_gpu_ids[worker_id % len(client_gpu_ids)], to_cuda=to_cuda)
        for worker_id in range(client_workers)
    ]
    client_device_by_idx = {
        idx: client_devices[worker_id]
        for worker_id, indices in enumerate(client_groups)
        for idx in indices
    }
    if server_device.type == "cuda":
        torch.cuda.set_device(server_device)

    shapelet_weight_X = np.load('./algoutils/shapelet_weight_All.npy')

    # ----- 加载数据集（统一入口：HAR / Epilepsy-TSTCC / SleepEDF / FD-A / 其他 UEA）
    has_val = False
    X_val, y_val = None, None

    if dataset == "HAR":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        if os.path.isfile("./HAR/val.pt"):
            val_data = torch.load("./HAR/val.pt")
            X_val = val_data["samples"].float()
            y_val = val_data["labels"].int()
            has_val = True
    elif dataset == "Epilepsy-TSTCC":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_Epilepsy(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./Epilepsy/val.pt")
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()
        has_val = True
    elif dataset == "SleepEDF":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_SleepEDF(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./sleepEDF/val.pt")
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()
        has_val = True
    elif dataset == "FD-A":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_FDA(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./FD-A/val.pt")
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()
        X_val = X_val.unsqueeze(1)
        has_val = True
    elif dataset != "":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_UEA(dataset, numClient, dirchlet_alpha=dirichlet_alpha,
                                                                     scoreX=shapelet_weight_X, scoreY=None)
    else:
        print("dataset not found")
        exit(0)
    # ------------------------------------------------------------------------------------------------------------

    n_ts, n_channels, len_ts = X_all.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_all))
    shapelets_size_and_len = {
        int(i): 40
        for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)
    }

    if args.description is not None:
        config['description'] = args.description

    # ----- 日志文件初始化 -------------------------------------------------------
    print("shapelet initialized!")
    desc_safe = _sanitize_filename(config.get('description', ''))
    formatted_date = datetime.now().strftime("%Y-%m-%d-%H") + desc_safe
    logTxt = f"./result/{dataset}/{formatted_date}_l={l}_lr={lr}_epoch{numEpoch}_alphadir{dirichlet_alpha}_{desc_safe}.txt"
    os.makedirs(os.path.dirname(logTxt), exist_ok=True)

    header_lines = [
        "Details of Training:-----------------------",
        f"dataset: {dataset}",
        f"local train epochs: {numEpoch}",
        f"round num: {numRound}",
        f"batch size: {batch_size}",
        f"lr: {lr}",
        f"use_client_selection: {use_client_selection}",
        f"server_device: {server_device}",
        f"client_workers: {client_workers}",
        f"client_worker_devices: {[str(d) for d in client_devices]}",
        f"client_groups: {client_groups}",
    ]
    if use_client_selection:
        header_lines += [
            f"client_selection_ratio: {client_selection_ratio}",
            f"client_selection_method: {client_selection_method}",
            f"min_selection_prob: {min_selection_prob}",
            f"ema_alpha: {ema_alpha}",
        ]
    header_lines += [
        "-------------------------------------------",
        str(config.get('description', '')),
        f"PID: {os.getpid()}",
        f"PPID: {os.getppid()}",
        yaml.dump(config).replace('\n', ''),
    ]
    with open(logTxt, mode="a+", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")

    # FL-bench 风格基线分发：SCAFFOLD / FedProto 不走 CSL 多尺度对比流程，直接接入
    # FL-bench 原实现（见 algo/scaffold/ 与 algo/fedproto/）。
    if algo.lower() in ('scaffold', 'fedproto'):
        from algo.baseline_runner import run_baseline
        run_baseline(
            algo=algo.lower(),
            config=config,
            seed=original_seed,
            dataset=dataset,
            shapelets_size_and_len=shapelets_size_and_len,
            n_channels=n_channels,
            num_classes=num_classes,
            X_all=X_all, y_all=y_all,
            X_test=X_test, y_test=y_test,
            X_fed=X_fed, y_fed=y_fed,
            X_val=X_val, y_val=y_val, has_val=has_val,
            num_rounds=numRound,
            num_epoch=numEpoch,
            batch_size=batch_size,
            lr=lr,
            wd=wd,
            dist_measure=dist_measure,
            to_cuda=to_cuda,
            logTxt=logTxt,
            formatted_date=formatted_date,
            eval_train_test_fn=eval,
            eval_tstcc_fn=eval_TSTCC,
            save_model_fn=save_model,
        )
        return

    # train----------------------------------------------------------------------------------------------------
    # 使用共享 kwargs 避免 server/client 两段近乎相同的构造参数重复
    shared_kwargs = dict(
        shapelets_size_and_len=shapelets_size_and_len,
        in_channels=n_channels,
        num_classes=num_classes,
        loss_func=loss_func,
        to_cuda=to_cuda,
        verbose=0,
        dist_measure=dist_measure,
        l3=l,
        l4=ls,
        T=T,
        alpha=alpha,
        is_ddp=is_ddp,
        checkpoint=checkpoint,
        seed=seed,
        shapelet_weight=shapelet_weight_X,
        configDir=args.config,
        config=config,
        beta=beta,
        device=server_device,
    )

    w_locals = []
    clientList = []
    server = LearningShapeletsCL(**shared_kwargs)
    # 服务器模型只做前向（作为 Global_Model 供客户端蒸馏/对比），
    # 关闭 requires_grad 可避免 client.train() 中 forward 时构建反向图，显著省显存。
    for p in server.model.parameters():
        p.requires_grad_(False)

    for idx in range(numClient):
        client = LearningShapeletsCL(**{**shared_kwargs, "device": client_device_by_idx[idx]})
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd)
        client.set_optimizer(optimizer)
        clientList.append(client)

    print(f"All {len(clientList)} clients initialized.")
    print(f"Server device: {server_device}")
    print(f"Client worker groups: {client_groups}")
    print(f"Client worker devices: {[str(d) for d in client_devices]}")

    best_acc = 0.0
    best_round = -1
    best_state_dict = None  # 只保存 state_dict 的 CPU 副本，避免频繁 deepcopy GPU 模型

    # ----- 断点续训（auto-resume） ----------------------------------------
    ckpt_every = int(fed_cfg.get('checkpoint_every', 30) or 0)
    auto_resume = bool(fed_cfg.get('auto_resume', True))
    resume_path = _resume_ckpt_path(dataset, formatted_date)
    start_round = 0
    if auto_resume:
        payload = _try_load_resume_ckpt(resume_path, server, clientList)
        if payload is not None:
            start_round = int(payload.get("round_idx", -1)) + 1
            best_acc = float(payload.get("best_acc", 0.0))
            best_round = int(payload.get("best_round", -1))
            best_state_dict = payload.get("best_state_dict", None)
            # w_locals 也恢复，保证首轮聚合时其它未被本轮训练的客户端权重不丢
            loaded_w_locals = payload.get("w_locals", [])
            if len(loaded_w_locals) == numClient:
                w_locals = [_state_dict_to_cpu(sd) for sd in loaded_w_locals]
                # 把恢复的权重同步给各 client.model（若 clients_state 字段缺失时兜底）
                for c, sd in zip(clientList, w_locals):
                    try:
                        _load_state_to_model(c.model, sd)
                    except Exception:
                        pass

    # ----- 客户端选择器：策略工厂见 algo/client_selection/ ----------------------
    selector = None
    if use_client_selection:
        if client_selection_method not in available_methods():
            raise ValueError(
                f"非法的 client_selection_method={client_selection_method!r}，"
                f"可选: {available_methods()}"
            )
        sample_nums_init = int(numClient * client_selection_ratio)
        selector = make_selector(
            client_selection_method,
            num_clients=numClient,
            sample_nums=sample_nums_init,
            seed=original_seed or 42,
            config=config,
            y_fed=y_fed,
            y_all_size=len(y_all),
            num_epoch=numEpoch,
            batch_size=batch_size,
            min_selection_prob=min_selection_prob,
            ema_alpha=ema_alpha,
        )
        print(f"客户端选择已启用，采样比例: {client_selection_ratio}, 选择方法: {client_selection_method}")
        print(f"最低选择概率: {min_selection_prob}, EMA平滑系数: {ema_alpha}")
    else:
        print("客户端选择未启用，所有客户端参与聚合")

    use_distribution = config['ablation'].get('UseDistribution', True)
    total_samples = len(X_all)
    use_scale_split_comm = _is_splitteacher_algo(algo)
    if not w_locals:
        w_locals = [_state_dict_to_cpu(c.model.state_dict()) for c in clientList]

    cached_client_scale_scores = None
    cached_client_scale_plans = None
    cached_scale_hist = None
    scale_score_prep_sec = 0.0
    if _uses_fedcsl_scale_scores(algo):
        scale_prep_t0 = time.perf_counter()
        cached_client_scale_scores = _precompute_client_scale_scores(
            X_fed,
            server.model,
            beta=beta,
            device=server_device,
            batch_size=batch_size,
        )
        if _is_teacher_scale_set_algo(algo):
            cached_client_scale_plans, cached_scale_hist = _plan_balanced_client_scales_from_scores(
                cached_client_scale_scores,
                server.model,
                extra_scale_count=2,
            )
        scale_score_prep_sec = time.perf_counter() - scale_prep_t0
        prep_msg = f"[fedcsl] precomputed client scale scores once in {scale_score_prep_sec:.3f}s"
        if cached_scale_hist is not None:
            prep_msg += f"; planned scale coverage: {cached_scale_hist.tolist()}"
        print(prep_msg, flush=True)
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            f.write(prep_msg + "\n")

    for round in range(start_round, numRound):
        round_t0 = time.perf_counter()
        avg_loss = 0.0
        client_losses = [0.0] * numClient  # 供 Oort 更新 reward
        client_scale_states = [None] * numClient
        client_scale_indices = [[] for _ in range(numClient)]

        server_state_cpu = _state_dict_to_cpu(server.model.state_dict())
        client_scale_plans = cached_client_scale_plans
        round_scale_hist = cached_scale_hist
        if round_scale_hist is not None:
            print(f"[round {round}] reused planned scale coverage: {round_scale_hist.tolist()}", flush=True)

        # ----- 本地训练阶段：多个 client worker 并行训练，client 按 round-robin 均分到 worker -----
        train_stage_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=client_workers) as executor:
            futures = []
            for worker_id, indices in enumerate(client_groups):
                if not indices:
                    continue
                futures.append(executor.submit(
                    _train_client_worker,
                    worker_id,
                    indices,
                    client_devices[worker_id],
                    clientList,
                    X_fed,
                    y_fed,
                    total_samples,
                    round,
                    numEpoch,
                    batch_size,
                    lr,
                    wd,
                    use_scale_split_comm,
                    server_state_cpu,
                    beta,
                    shared_kwargs,
                    client_scale_plans,
                    cached_client_scale_scores,
                ))
            for future in as_completed(futures):
                for result in future.result():
                    idx = result["idx"]
                    client_losses[idx] = result["loss"]
                    avg_loss += result["loss"] * len(y_fed[idx]) / total_samples
                    w_locals[idx] = result["state"]
                    if result.get("scale_indices"):
                        client_scale_indices[idx] = list(result["scale_indices"])
                    if use_scale_split_comm and result.get("scale_states"):
                        client_scale_states[idx] = {
                            "scale_indices": list(result.get("scale_indices", [])),
                            "states": result["scale_states"],
                        }
        train_stage_sec = time.perf_counter() - train_stage_t0

        # ----- 分布打分：cal_score(predict) + normalize（UseDistribution=False 时退化为全 1） -----
        dist_stage_t0 = time.perf_counter()
        scores = []
        for idx, c in enumerate(clientList):
            if len(X_fed[idx]) == 0:
                scores.append(1.0)
                continue
            features = c.predict(X_fed[idx])
            if features.size == 0 or (features.ndim > 0 and features.shape[0] == 0):
                scores.append(1.0)
                continue
            scores.append(cal_score(features))
        scores = normalize_to_near_one(scores)
        if not use_distribution:
            scores = [1] * numClient
        dist_stage_sec = time.perf_counter() - dist_stage_t0

        # ----- 客户端选择 + 聚合（策略见 algo/client_selection/）-----
        agg_stage_t0 = time.perf_counter()
        if use_scale_split_comm:
            if use_client_selection and selector is not None:
                select_mask = selector.on_round_start(round, client_losses=client_losses, y_fed=y_fed)
                print(f"[{selector.name}] 选择掩码: {select_mask}")
                filtered_payloads = []
                for i, payload in enumerate(client_scale_states):
                    if payload is not None and select_mask[i] != 0:
                        filtered_payloads.append(payload)
                    else:
                        filtered_payloads.append({})
                w_global = _aggregate_scale_updates(server_state_cpu, filtered_payloads, y_fed)
                _load_state_to_model(server.model, w_global)
                selector.on_round_end(
                    round,
                    w_locals=w_locals, w_global=w_global,
                    select_mask=select_mask, client_losses=client_losses,
                )
            else:
                filtered_payloads = [payload if payload is not None else {} for payload in client_scale_states]
                w_global = _aggregate_scale_updates(server_state_cpu, filtered_payloads, y_fed)
                _load_state_to_model(server.model, w_global)
        elif use_client_selection and selector is not None:
            select_mask = selector.on_round_start(round, client_losses=client_losses, y_fed=y_fed)
            print(f"[{selector.name}] 选择掩码: {select_mask}")
            w_global = selector.aggregate(w_locals, y_fed, scores, select_mask)
            if w_global is None:  # 策略未覆盖时回退默认 FedAvg
                combined_scores = [scores[i] * select_mask[i] for i in range(numClient)]
                w_global = fedavg(w_locals, y_fed, combined_scores)
            _load_state_to_model(server.model, w_global)
            selector.on_round_end(
                round,
                w_locals=w_locals, w_global=w_global,
                select_mask=select_mask, client_losses=client_losses,
            )
        else:
            w_global = fedavg(w_locals, y_fed, scores)
            _load_state_to_model(server.model, w_global)
        agg_stage_sec = time.perf_counter() - agg_stage_t0

        # ----- 下游 SVC 评估 -----
        eval_stage_t0 = time.perf_counter()
        transformation = server.transform(X_all, result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_test = server.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
        scaler = RobustScaler()
        transformation = scaler.fit_transform(transformation)
        transformation_test = scaler.transform(transformation_test)
        if has_val and X_val is not None and y_val is not None:
            transformation_val = server.transform(X_val, result_type='numpy', normalize=True, batch_size=batch_size)
            transformation_val = scaler.transform(transformation_val)
            y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else np.asarray(y_val)
            train_acc, test_acc = eval_TSTCC(
                transformation_train=transformation,
                transformation_test=transformation_test,
                transformation_val=transformation_val,
                y_train=y_all, y_test=y_test, y_val=y_val_np,
            )
        else:
            train_acc, test_acc = eval(transformation, transformation_test, y_train=y_all, y_test=y_test)
        eval_stage_sec = time.perf_counter() - eval_stage_t0
        round_total_sec = time.perf_counter() - round_t0

        if test_acc > best_acc:
            best_acc = test_acc
            best_round = round
            # 只 clone 到 CPU，避免 deepcopy 整个 GPU 模型（显著更快、更省显存）
            best_state_dict = _state_dict_to_cpu(server.model.state_dict())

        print(f"Classification: train={train_acc:.4f} test={test_acc:.4f} round={round}")
        print(
            f"[round {round}] timing train={train_stage_sec:.3f}s "
            f"distribution={dist_stage_sec:.3f}s agg={agg_stage_sec:.3f}s "
            f"eval={eval_stage_sec:.3f}s total={round_total_sec:.3f}s",
            flush=True,
        )

        avg_loss_str = str(avg_loss) if np.isfinite(avg_loss) else "nan"
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            f.write(
                f"dataset: {dataset}round:{round} server aggregation "
                f" testACC:{test_acc} trainACC:{train_acc} avg_loss:{avg_loss_str}\n"
            )
            f.write(
                f"[round {round}] timing train={train_stage_sec:.3f}s "
                f"distribution={dist_stage_sec:.3f}s agg={agg_stage_sec:.3f}s "
                f"eval={eval_stage_sec:.3f}s total={round_total_sec:.3f}s\n"
            )

        # 断点续训：每 ckpt_every 轮覆盖保存一次
        if ckpt_every > 0 and (round + 1) % ckpt_every == 0:
            try:
                _save_resume_ckpt(
                    resume_path, round, server, clientList,
                    w_locals, best_acc, best_round, best_state_dict,
                )
            except Exception as e:  # pragma: no cover
                print(f"[fedcsl] 保存 checkpoint 失败: {e}")

    print(f"best round is {best_round}, acc is {best_acc:.6f}")
    if best_state_dict is not None:
        save_model(best_state_dict, dataset, formatted_date)
    else:
        print("[warn] 没有任何一轮 test_acc > 0，跳过 best 模型保存")


def save_model(state_dict, dataset, formatted_date):
    """保存 best state_dict 到 ./checkpoint/<dataset>/<date>_<dataset>_model.pt。"""
    checkpoint_dir = f'./checkpoint/{dataset}'
    safe_date = _sanitize_filename(formatted_date) if formatted_date else formatted_date
    model_path = f'{checkpoint_dir}/{safe_date}_{dataset}_model.pt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")


# ---------------------------------------------------------------------------
# 断点续训：每 N round 覆盖保存 server+clients state_dict + 元信息
# 文件：./checkpoint/resume/fedcsl_<dataset>_<desc>.pt
# ---------------------------------------------------------------------------
_RESUME_DIR = "./checkpoint/resume"


def _resume_ckpt_path(dataset, formatted_date):
    os.makedirs(_RESUME_DIR, exist_ok=True)
    safe = _sanitize_filename(formatted_date) if formatted_date else "default"
    return os.path.join(_RESUME_DIR, f"fedcsl_{dataset}_{safe}.pt")


def _save_resume_ckpt(path, round_idx, server, clientList,
                       w_locals, best_acc, best_round, best_state_dict):
    payload = {
        "round_idx": int(round_idx),
        "server_state": _state_dict_to_cpu(server.model.state_dict()),
        "clients_state": [
            _state_dict_to_cpu(c.model.state_dict())
            for c in clientList
        ],
        "w_locals": [
            _state_dict_to_cpu(sd)
            for sd in w_locals
        ],
        "best_acc": float(best_acc),
        "best_round": int(best_round),
        "best_state_dict": best_state_dict,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[fedcsl] checkpoint saved → {path} (round={round_idx + 1})", flush=True)


def _try_load_resume_ckpt(path, server, clientList):
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[fedcsl] 读取 checkpoint 失败（{path}）：{e}，从零开始", flush=True)
        return None
    try:
        _load_state_to_model(server.model, payload["server_state"])
        for c, state in zip(clientList, payload.get("clients_state", [])):
            _load_state_to_model(c.model, state)
    except Exception as e:
        print(f"[fedcsl] checkpoint 结构不兼容：{e}，从零开始", flush=True)
        return None
    resume_round = int(payload.get("round_idx", -1)) + 1
    print(
        f"[fedcsl] checkpoint loaded ← {path}，将从 round {resume_round + 1} 继续",
        flush=True,
    )
    return payload


if __name__ == '__main__':
    train(dataset=args.dataset, seed=args.seed)
