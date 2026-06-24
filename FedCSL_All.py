"""FedCSL 主入口：负责数据加载、客户端/服务器初始化、客户端选择与聚合、下游 SVC 评估。

SCAFFOLD / FedProto 等 FL-bench 原生算法会在识别到 ``algo`` 字段后路由到
``algo/baseline_runner.run_baseline``，跳过多尺度对比流程。
"""

import argparse
import copy
import os
import random
import sys
import time
import traceback
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
from algo.scale_allocation import knapsack_lagrangian_assign
from dataset_utils import *   # noqa: F401,F403  (LoadDataset_* 全家桶)
from fedavg import fedavg
from fedutil import cal_score, normalize_to_near_one
from train import LearningShapeletsCL
from utils import period_score

# ---------------------------------------------------------------------------
# 诊断日志开关：export SPILTER_DEBUG=1 启用，默认关闭
# 输出以 [spilter_dbg] 为前缀，便于 grep 筛选
# ---------------------------------------------------------------------------
_SPILTER_DEBUG: bool = os.environ.get("SPILTER_DEBUG", "0").strip() == "1"


def _dbg_param_norm(model, prefixes) -> float:
    """计算 model 中所有匹配 prefixes 前缀的参数的总 L2 范数。"""
    total = 0.0
    for pname, p in model.named_parameters():
        if any(pname.startswith(pf) for pf in prefixes):
            total += float(p.detach().norm().item() ** 2)
    return total ** 0.5


def _dbg_grad_norm(model, prefixes) -> float:
    """计算 model 中匹配 prefixes 前缀的参数的梯度 L2 范数（无梯度时返回 0）。"""
    total = 0.0
    for pname, p in model.named_parameters():
        if any(pname.startswith(pf) for pf in prefixes) and p.grad is not None:
            total += float(p.grad.detach().norm().item() ** 2)
    return total ** 0.5



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
parser.add_argument('--num-client', type=int, default=None, help='Number of federated clients (overrides config file)')
parser.add_argument('--num-rounds', type=int, default=None, help='Number of federated rounds (overrides config file)')
parser.add_argument('--server-gpu', type=int, default=None, help='GPU id used by the server/model evaluation thread')
parser.add_argument('--client-gpus', type=str, default=None,
                    help='Comma-separated GPU ids for client worker threads, e.g. "0,1,2"')
parser.add_argument('--client-workers', type=int, default=None,
                    help='Number of client training worker threads; default comes from config or 3')
parser.add_argument('--eval-protocol', type=str, default=None, choices=['svm', 'linear_probe'],
                    help='Override downstream evaluation protocol: svm or linear_probe')
parser.add_argument('--eval-every-n-rounds', type=int, default=None,
                    help='Run downstream SVM/linear-probe eval every N federated rounds (default: evaluation.round_eval_interval or 1)')
parser.add_argument('--spilter-random', action='store_true', default=False,
                    help='Use random (non-topM) scale selection for spilter allocation (overrides config allocation_mode to local_score_random_topm)')
parser.add_argument('--spilter-knapsack', action='store_true', default=False,
                    help='Use knapsack-lagrangian global-coverage-aware scale allocation (overrides config allocation_mode to knapsack_lagrangian)')
parser.add_argument('--spilter-memory-budget', type=float, default=None,
                    help='Per-client memory budget in MB for knapsack_lagrangian mode (overrides config spilter.memory_budget_mb)')

args = parser.parse_args()

_ACTIVE_RESULT_LOG = None
_DEFAULT_EXCEPTHOOK = sys.excepthook
_LOSS_KEYS = (
    "total",
    "base",
    "structure",
    "joint",
    "scale",
    "scale_local",
    "scale_global",
    "cca",
    "sdl",
    "proximal",
    "moon",
)


def _append_text_to_log(log_path, text):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode="a+", encoding="utf-8") as f:
        f.write(text)


def _normalize_loss_breakdown(loss_entry):
    if isinstance(loss_entry, dict):
        normalized = {key: float(loss_entry.get(key, 0.0) or 0.0) for key in _LOSS_KEYS}
        normalized["primary_scale"] = int(loss_entry.get("primary_scale", -1))
        return normalized

    if isinstance(loss_entry, (list, tuple)):
        normalized = {key: 0.0 for key in _LOSS_KEYS}
        normalized["total"] = float(loss_entry[0]) if len(loss_entry) > 0 else 0.0
        normalized["cca"] = float(loss_entry[2]) if len(loss_entry) > 2 else 0.0
        normalized["sdl"] = float(loss_entry[3]) if len(loss_entry) > 3 else 0.0
        normalized["primary_scale"] = int(loss_entry[4]) if len(loss_entry) > 4 else -1
        return normalized

    normalized = {key: 0.0 for key in _LOSS_KEYS}
    normalized["total"] = float(loss_entry or 0.0)
    normalized["primary_scale"] = -1
    return normalized


def _mean_loss_breakdown(losses):
    if not losses:
        return {**{key: 0.0 for key in _LOSS_KEYS}, "primary_scale": -1}

    breakdowns = [_normalize_loss_breakdown(loss) for loss in losses]
    means = {}
    for key in _LOSS_KEYS:
        value = float(np.mean([loss[key] for loss in breakdowns]))
        means[key] = value if np.isfinite(value) else 0.0
    primary_scales = [loss["primary_scale"] for loss in breakdowns if loss.get("primary_scale", -1) >= 0]
    means["primary_scale"] = int(primary_scales[0]) if primary_scales else -1
    return means


def _format_exception_block(exc_type, exc_value, exc_traceback):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    return (
        "\n"
        "[ERROR] Unhandled exception -----------------------\n"
        f"time: {ts}\n"
        f"type: {getattr(exc_type, '__name__', str(exc_type))}\n"
        f"message: {exc_value}\n"
        f"{tb}"
        "-------------------------------------------\n"
    )


def _fmt_knap_budget(ki):
    """Format knapsack budget info for log messages.

    Handles both uniform (scalar) and per-client (list) budgets.
    """
    budget_val = ki.get('_budget_mb')
    if budget_val is None:
        return 'unconstrained'
    bmin = ki.get('_budget_mb_min')
    bmax = ki.get('_budget_mb_max')
    if bmin is not None and bmax is not None and bmin != bmax:
        return f"{budget_val:.1f}MB [{bmin:.0f}..{bmax:.0f}]"
    return f"{budget_val:.1f}MB"


def _result_log_excepthook(exc_type, exc_value, exc_traceback):
    if _ACTIVE_RESULT_LOG and not issubclass(exc_type, KeyboardInterrupt):
        try:
            _append_text_to_log(
                _ACTIVE_RESULT_LOG,
                _format_exception_block(exc_type, exc_value, exc_traceback),
            )
        except Exception:
            pass
    _DEFAULT_EXCEPTHOOK(exc_type, exc_value, exc_traceback)


sys.excepthook = _result_log_excepthook


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


def _is_spilter_algo(algo_name):
    """Spilter keeps only selected scale slices on each client."""
    return str(algo_name).lower() in (
        "spilter",
        "fedcsl-spilter",
    )


def _uses_scale_split_algo(algo_name):
    return _is_spilter_algo(algo_name)


def _uses_fedcsl_scale_scores(algo_name):
    algo = str(algo_name).lower()
    return algo in (
        "fedcsl",
        "spilter",
        "fedcsl-spilter",
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


def _normalize_positive_costs(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(values[values > 0])) if np.any(values > 0) else 1.0
    return np.maximum(values / max(mean, 1e-8), 1e-6)


def _scale_system_costs(model):
    """估计每个尺度的系统代价：参数通信量 + sliding-window 计算量代理。"""
    lengths = np.asarray(list(model.shapelets_size_and_len.keys()), dtype=np.float32)
    widths = np.asarray(list(model.shapelets_size_and_len.values()), dtype=np.float32)
    if lengths.size == 0:
        return np.zeros(0, dtype=np.float32)

    seq_len_proxy = max(float(np.max(lengths)) / 0.8, float(np.max(lengths)))
    window_counts = np.maximum(seq_len_proxy - lengths + 1.0, 1.0)
    compute_costs = lengths * np.maximum(widths, 1.0) * window_counts

    comm_costs = []
    for scale_idx, (length, width) in enumerate(zip(lengths, widths)):
        try:
            state = model.scale_state_dict(scale_idx, clone=False, cpu=False)
            comm = sum(
                int(value.numel()) for value in state.values()
                if hasattr(value, "numel")
            )
        except Exception:
            comm = int(length * max(width, 1.0))
        comm_costs.append(max(float(comm), 1.0))

    compute_norm = _normalize_positive_costs(compute_costs)
    comm_norm = _normalize_positive_costs(comm_costs)
    return np.maximum(0.5 * compute_norm + 0.5 * comm_norm, 1e-6)


def _plan_efficiency_aware_client_scales_from_scores(
    client_scores,
    model,
    extra_scale_count=2,
    coverage_weight=0.35,
):
    """每个 client 保留 2 个本地主尺度，再按系统效率补若干个额外尺度。

    额外尺度的选择同时考虑三项：尺度覆盖均衡、该客户端的周期/能量得分、
    以及尺度参数量和滑窗计算量导致的系统成本。这样可避免额外分发总是偏向
    长尺度大模型，同时仍尽量维持全局尺度覆盖。
    """
    num_clients = len(client_scores)
    num_scales = len(model.shapelets_size_and_len)
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64)

    client_selected = []
    scale_counts = np.zeros(num_scales, dtype=np.int64)
    scale_costs = _scale_system_costs(model)
    target_count = max(
        1.0,
        float(num_clients * min(num_scales, 2 + max(0, int(extra_scale_count)))) / num_scales,
    )

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
            scores = _normalize_scale_scores(client_scores[cid], num_scales)
            max_score = float(np.max(scores)) if scores.size else 0.0
            if max_score > 0:
                scores = scores / max_score
            else:
                scores = np.ones(num_scales, dtype=np.float32)

            def _assignment_score(scale_idx):
                coverage_penalty = float(scale_counts[scale_idx]) / target_count
                efficiency_gain = float(scores[scale_idx]) / float(scale_costs[scale_idx])
                return efficiency_gain - coverage_weight * coverage_penalty

            best_scale = max(
                available,
                key=lambda scale_idx: (
                    _assignment_score(scale_idx),
                    -float(scale_costs[scale_idx]),
                    -int(scale_counts[scale_idx]),
                    -int(scale_idx),
                ),
            )
            client_selected[cid].append(int(best_scale))
            scale_counts[best_scale] += 1

    return client_selected, scale_counts


def _plan_uniform_single_client_scales(num_clients, model):
    """每个 client 只训练一个尺度，并按 client id 轮转均匀覆盖全部尺度。"""
    num_scales = len(model.shapelets_size_and_len)
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64)

    client_selected = []
    scale_counts = np.zeros(num_scales, dtype=np.int64)
    for cid in range(num_clients):
        scale_idx = int(cid % num_scales)
        client_selected.append([scale_idx])
        scale_counts[scale_idx] += 1
    return client_selected, scale_counts


def _plan_global_score_random_single_client_scales(client_scores, y_fed, seed=None):
    """按全局周期评分分布随机给每个 client 分配 1 个尺度。"""
    num_clients = len(client_scores)
    num_scales = len(client_scores[0]) if num_clients > 0 else 0
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    weights = np.asarray([len(y) for y in y_fed], dtype=np.float32)
    if weights.size != num_clients or float(weights.sum()) <= 0:
        weights = np.ones(num_clients, dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)

    global_scores = np.zeros(num_scales, dtype=np.float32)
    for cid, scores in enumerate(client_scores):
        global_scores += weights[cid] * _normalize_scale_scores(scores, num_scales)
    global_scores = np.nan_to_num(global_scores, nan=0.0, posinf=0.0, neginf=0.0)
    if float(global_scores.sum()) <= 0:
        probs = np.ones(num_scales, dtype=np.float32) / float(num_scales)
    else:
        probs = global_scores / float(global_scores.sum())

    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.arange(num_scales), size=num_clients, replace=True, p=probs)
    scale_counts = np.bincount(sampled, minlength=num_scales).astype(np.int64)
    client_selected = [[int(scale_idx)] for scale_idx in sampled]
    return client_selected, scale_counts, probs


def _plan_local_score_topm_client_scales(client_scores, top_m=4):
    """每个 client 按自己的周期评分选择 top-m 尺度，作为拼接子模型训练。

    严格按 top-m 选择，不做未覆盖尺度的兜底补分配。

    top_m 可以是 int（所有 client 相同）或 list（per-client m 值）。
    """
    num_clients = len(client_scores)
    num_scales = len(client_scores[0]) if num_clients > 0 else 0
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64)

    if isinstance(top_m, (list, tuple)):
        top_ms = [max(1, min(int(m), num_scales)) for m in top_m]
        # Pad or truncate to match num_clients
        if len(top_ms) < num_clients:
            top_ms.extend([max(1, min(int(top_m[0]) if len(top_m) > 0 else 4, num_scales))]
                          * (num_clients - len(top_ms)))
        top_ms = top_ms[:num_clients]
    else:
        m_val = max(1, min(int(top_m), num_scales))
        top_ms = [m_val] * num_clients

    client_selected = []
    scale_counts = np.zeros(num_scales, dtype=np.int64)
    normalized_scores = []

    for cid, scores in enumerate(client_scores):
        scores = _normalize_scale_scores(scores, num_scales)
        normalized_scores.append(scores)
        m_c = top_ms[cid]
        if np.any(scores > 0):
            # Stable tie-breaker: higher score first, then lower scale index.
            order = np.lexsort((np.arange(num_scales), -scores))
        else:
            order = np.arange(num_scales)
        selected = [int(scale_idx) for scale_idx in order[:m_c]]
        client_selected.append(selected)
        for scale_idx in selected:
            scale_counts[scale_idx] += 1

    return client_selected, scale_counts


def _plan_local_score_random_topm_client_scales(client_scores, top_m=4, seed=None):
    """每个 client 随机选 m 个尺度（非 top-m，均匀分布），不按周期评分排序。

    top_m 可以是 int（所有 client 相同）或 list（per-client m 值）。
    """
    num_clients = len(client_scores)
    num_scales = len(client_scores[0]) if num_clients > 0 else 0
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64)

    if isinstance(top_m, (list, tuple)):
        top_ms = [max(1, min(int(m), num_scales)) for m in top_m]
        if len(top_ms) < num_clients:
            top_ms.extend([max(1, min(int(top_m[0]) if len(top_m) > 0 else 4, num_scales))]
                          * (num_clients - len(top_ms)))
        top_ms = top_ms[:num_clients]
    else:
        m_val = max(1, min(int(top_m), num_scales))
        top_ms = [m_val] * num_clients

    rng = np.random.default_rng(seed)
    client_selected = []
    scale_counts = np.zeros(num_scales, dtype=np.int64)
    all_scales = np.arange(num_scales)

    for cid in range(num_clients):
        m_c = top_ms[cid]
        selected = [int(s) for s in rng.choice(all_scales, size=m_c, replace=False)]
        client_selected.append(selected)
        for s in selected:
            scale_counts[s] += 1

    return client_selected, scale_counts


def _spilter_knapsack_lagrangian_params(config, override_memory_budget=None):
    """Extract knapsack-lagrangian parameters from config.spilter.

    Resolution order for memory budgets:
      1. CLI arg ``--spilter-memory-budget`` (single float)
      2. Env var ``SPILTER_MEMORY_BUDGETS`` (comma-separated per-client list)
      3. Config file ``spilter.memory_budget_mb``

    When ``SPILTER_MEMORY_BUDGETS`` is set, it overrides the single-value budget
    with a per-client list (used by the dashboard's spilter-memory-budget experiment).
    """
    spilter_cfg = config.get("spilter", {}) or {}
    knapsack_cfg = spilter_cfg.get("knapsack_lagrangian", {}) or {}

    # Resolve budget: CLI > env var > config
    budget = override_memory_budget if override_memory_budget is not None else spilter_cfg.get("memory_budget_mb", None)

    # Per-client budgets via env var (dashboard spilter-memory-budget experiment)
    env_budgets = os.environ.get("SPILTER_MEMORY_BUDGETS", "").strip()
    if env_budgets:
        try:
            per_client = [float(v.strip()) for v in env_budgets.split(",") if v.strip()]
            if per_client:
                budget = per_client
                print(
                    f"[spilter] memory budgets from SPILTER_MEMORY_BUDGETS: "
                    f"{len(per_client)} clients, "
                    f"range=[{min(per_client):.0f}, {max(per_client):.0f}] MB, "
                    f"mean={np.mean(per_client):.1f} MB",
                    flush=True,
                )
        except (ValueError, TypeError):
            pass

    return {
        "memory_budgets_mb": budget,
        "scale_memory_costs_mb": spilter_cfg.get("scale_memory_costs_mb", None),
        "base_memory_mb": float(spilter_cfg.get("base_memory_mb", 0.0)),
        "coverage_min": knapsack_cfg.get("coverage_min", None),
        "lambda_lr": float(knapsack_cfg.get("lambda_lr", 0.1)),
        "max_iter": int(knapsack_cfg.get("max_iter", 50)),
    }


# ---------------------------------------------------------------------------
# Per-scale GPU memory calibration (same approach as measure_scale_memory.py)
# ---------------------------------------------------------------------------
_KNAPSACK_CALIBRATION_DIR = os.path.join("data", "knapsack_calibration")
os.makedirs(_KNAPSACK_CALIBRATION_DIR, exist_ok=True)


def _calibrate_per_scale_memory_mb(X_fed, model, device, batch_size,
                                     in_channels, num_classes, dist_measure,
                                     lr, wd, seed, dataset_tag):
    """Measure per-scale GPU peak memory by training one scale at a time.

    Replicates the measurement protocol of
    ``scripts/system_efficiency/measure_scale_memory.py``:
      1. Create a fresh LearningShapeletsCL client per scale
      2. Set Selected_Scales = [s], SGD+momentum optimizer
      3. Warmup 1 batch, reset peak stats, train 1 epoch, record peak
      4. Cache results to data/knapsack_calibration/<dataset>.json

    Returns (g_r: np.ndarray, g_0: float) in MB, or (None, 0) on CPU.
    """
    import gc as _gc
    import json as _json

    if device is None or device.type != "cuda":
        return None, 0.0

    cache_path = os.path.join(
        _KNAPSACK_CALIBRATION_DIR, f"{dataset_tag}_scale_memory.json"
    )
    if os.path.isfile(cache_path):
        try:
            cached = _json.loads(open(cache_path, encoding="utf-8").read())
            g_r = np.asarray(cached["g_r_mb"], dtype=np.float64)
            g_0 = float(cached.get("g_0_mb", 0.0))
            print(
                f"[spilter] knapsack_lagrangian: loaded cached calibration "
                f"({cache_path}): g_0={g_0:.1f}MB, g_r={[f'{v:.1f}' for v in g_r]} MB",
                flush=True,
            )
            return g_r, g_0
        except Exception:
            pass  # Re-calibrate on cache corruption

    print(
        "[spilter] knapsack_lagrangian: no cached calibration; "
        "measuring per-scale GPU memory (round-0, ~30s) ...",
        flush=True,
    )
    from train import LearningShapeletsCL as _CL

    num_scales = len(model.shapelets_size_and_len)
    shapelets_size_and_len = dict(model.shapelets_size_and_len)

    # Sample data from first non-empty client
    sample_data = None
    for client_x in X_fed:
        if len(client_x) >= batch_size:
            sample_data = client_x[:batch_size * 2]
            break
    if sample_data is None:
        return None, 0.0

    # Build a minimal spilter config for the calibration client
    calib_config = {
        "algo": "spilter",
        "model": {"params": {"momentum": 0.9, "gamma": 0.5, "beta": 0.4}},
        "ablation": {
            "UseJointKD": True, "UseJointCL": True,
            "UseScaleKD": True, "UseScaleCL": True,
            "UseProdictor": False, "UseMTLHomo": False,
            "UseACF": True, "UseDistribution": False,
        },
        "spilter": {
            "allocation_mode": "local_score_topm",
            "selected_scale_training": "stitched",
            "stitched_feature_source": "selected_scales_only",
        },
    }
    loss_func = torch.nn.CrossEntropyLoss()

    per_scale_mb = []
    t0 = time.perf_counter()
    for s in range(num_scales):
        client = _CL(
            shapelets_size_and_len=shapelets_size_and_len,
            loss_func=loss_func,
            in_channels=in_channels,
            num_classes=num_classes,
            dist_measure=dist_measure,
            verbose=0,
            to_cuda=True,
            l3=0.0, l4=0.0, T=0.1, alpha=0.0, beta=0.4,
            seed=seed,
            configDir=None,
            config=calib_config,
            device=device,
        )
        client.set_optimizer(
            torch.optim.SGD(
                client.model.parameters(), lr=float(lr),
                weight_decay=float(wd), momentum=0.9,
            )
        )
        client.Selected_Scales = [int(s)]

        # Warmup
        try:
            client.train(sample_data, epochs=1, batch_size=batch_size,
                         epoch_idx=-1, lr=lr)
        except Exception:
            pass

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            client.train(sample_data, epochs=1, batch_size=batch_size,
                         epoch_idx=0, lr=lr)
        except Exception:
            pass
        torch.cuda.synchronize(device)
        peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        per_scale_mb.append(peak_mb)

        del client
        _gc.collect()
        torch.cuda.empty_cache()

    g_r = np.asarray(per_scale_mb, dtype=np.float64)
    g_0 = float(np.min(g_r)) * 0.5  # Conservative base overhead estimate
    g_r_marginal = np.maximum(g_r - g_0, 1.0)

    # Persist
    try:
        open(cache_path, "w", encoding="utf-8").write(_json.dumps({
            "dataset": dataset_tag,
            "g_0_mb": g_0,
            "g_r_mb": g_r_marginal.tolist(),
            "g_r_raw_mb": g_r.tolist(),
            "scale_lengths": list(shapelets_size_and_len.keys()),
            "calibrated_at": time.time(),
            "device": str(device),
            "batch_size": batch_size,
        }, indent=2, ensure_ascii=False))
        print(f"[spilter] knapsack_lagrangian: calibration cached → {cache_path}",
              flush=True)
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    print(
        f"[spilter] knapsack_lagrangian: calibration done in {elapsed:.1f}s: "
        f"g_0={g_0:.1f}MB, g_r={[f'{v:.1f}' for v in g_r_marginal]} MB",
        flush=True,
    )
    return g_r_marginal, g_0


def _measure_server_per_scale_memory(server_model, X_fed, device, batch_size):
    """Measure per-scale GPU memory by forwarding EACH scale individually.

    A full forward (all 8 scales × 3 branches = 24 blocks) allows PyTorch's
    caching allocator to reuse memory across sequential blocks, producing
    unrealistically low per-block peak deltas.  Clients only train a subset
    of scales, so we measure each scale in isolation to get costs that match
    real training memory.

    Returns a dict {scale_length: total_mem_mb} suitable for knapsack g_r.
    """
    if device is None or device.type != "cuda":
        return None
    import gc as _gc
    sample_data = None
    for client_x in X_fed:
        if len(client_x) >= batch_size:
            sample_data = client_x[:batch_size]
            break
    if sample_data is None:
        return None

    def _all_blocks(model):
        """Yield all block modules reachable from the model."""
        if hasattr(model, "shapelets_blocks"):
            yield from model.shapelets_blocks.blocks
        if hasattr(model, "shapelets_euclidean"):
            yield from model.shapelets_euclidean.blocks
            yield from model.shapelets_cosine.blocks
            yield from model.shapelets_cross_correlation.blocks

    # Enable autograd so forward includes saved-activation memory
    saved_grad_flags = {}
    for pname, p in server_model.named_parameters():
        saved_grad_flags[pname] = p.requires_grad
        p.requires_grad_(True)

    num_scales = len(server_model.shapelets_size_and_len)
    per_scale_total_mb = {}
    x = torch.from_numpy(np.asarray(sample_data, dtype=np.float32)).float().to(device)

    try:
        for scale_idx in range(num_scales):
            # Reset ALL blocks before measuring just this scale
            for blk in _all_blocks(server_model):
                blk._peak_mem_measured = False
                blk._peak_mem_delta_bytes = 0

            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            # Forward only this scale (like a client that only trains this scale)
            try:
                enc = getattr(server_model, "encode_scale", None)
                if enc is not None:
                    _ = enc(x, scale_idx, masking=False, normalize=False)
                else:
                    _ = server_model.shapelets_blocks.forward_scale(x, scale_idx, masking=False)
            except Exception:
                # Fallback: forward_subset
                try:
                    _ = server_model.shapelets_blocks.forward_subset(x, [scale_idx], masking=False)
                except Exception:
                    continue
            torch.cuda.synchronize(device)

            # Collect measured total for this scale
            lengths = list(server_model.shapelets_size_and_len.keys())
            L = lengths[scale_idx]
            scale_total = 0.0
            # Sum peak+param from blocks belonging to this scale
            for blk_type, branch_blocks in [
                ("euclidean", getattr(server_model, "shapelets_euclidean", None)),
                ("cosine", getattr(server_model, "shapelets_cosine", None)),
                ("cross_corr", getattr(server_model, "shapelets_cross_correlation", None)),
            ]:
                if branch_blocks is not None and scale_idx < len(branch_blocks.blocks):
                    blk = branch_blocks.blocks[scale_idx]
                    scale_total += blk.peak_mem_mb + float(blk.param_mem_bytes) / (1024.0 * 1024.0)
            if scale_total <= 0 and hasattr(server_model, "shapelets_blocks"):
                sbd = server_model.shapelets_blocks
                if sbd.dist_measure == "mix":
                    for offset in range(3):
                        blk = sbd.blocks[scale_idx * 3 + offset]
                        scale_total += blk.peak_mem_mb + float(blk.param_mem_bytes) / (1024.0 * 1024.0)
                else:
                    blk = sbd.blocks[scale_idx]
                    scale_total += blk.peak_mem_mb + float(blk.param_mem_bytes) / (1024.0 * 1024.0)
            per_scale_total_mb[L] = max(scale_total, 0.1)  # floor to avoid zero
    finally:
        for pname, p in server_model.named_parameters():
            p.requires_grad_(saved_grad_flags.get(pname, False))

    _gc.collect()
    torch.cuda.empty_cache()

    if not per_scale_total_mb:
        return None
    lengths_sorted = sorted(per_scale_total_mb.keys())
    print(
        f"[spilter] per-scale memory (isolated, with autograd): "
        f"g_r={[f'{per_scale_total_mb[L]:.1f}' for L in lengths_sorted]} MB",
        flush=True,
    )
    return {"total_mem_mb": per_scale_total_mb}


def _plan_topm_then_local_knapsack_client_scales(
    client_scores,
    server_model,
    *,
    top_m=4,
    memory_budgets_mb=None,
    scale_memory_costs_mb=None,
    base_memory_mb=0.0,
    X_fed=None,
    device=None,
    batch_size=32,
    seed=None,
    # Calibration extras (passed through to _calibrate_per_scale_memory_mb)
    in_channels=1,
    num_classes=2,
    dist_measure="mix",
    lr=0.01,
    wd=0.0001,
    dataset_tag="unknown",
    # Absorb legacy knapsack_lagrangian params (ignored in local mode)
    coverage_min=None,
    lambda_lr=0.1,
    max_iter=50,
    **kwargs,
):
    """Two-stage spilter-memory-budget allocation.

    Stage 1 (server): top-M per client by local period score (same as local_score_topm).
    Stage 2 (local): each client runs a 0-1 knapsack on its top-M candidates,
    constrained by its per-client memory budget.

    Returns (client_selected, scale_counts, info_dict).
    """
    from algo.scale_allocation.lagrangian import knapsack_dp_select

    num_clients = len(client_scores)
    num_scales = len(client_scores[0]) if num_clients > 0 else 0
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64), {}

    # ---- Stage 1: top-M per client (server side) ----
    topm_selected, _ = _plan_local_score_topm_client_scales(
        client_scores, top_m=top_m
    )

    # ---- Resolve scale memory costs for knapsack internal use only ----
    # Use lightweight system proxy; real per-scale memory will be measured
    # from client training in round 0 and replace these values afterward.
    if scale_memory_costs_mb is None:
        scale_memory_costs_mb = _scale_system_costs(server_model)
        costs_source = "system_proxy"
    else:
        costs_source = "config"

    g_r = np.asarray(scale_memory_costs_mb, dtype=np.float64)
    g_r = np.maximum(g_r, 1e-6)
    g_0 = 0.0

    # ---- Per-client budgets ----
    if memory_budgets_mb is not None:
        if isinstance(memory_budgets_mb, (int, float)):
            B_k = np.full(num_clients, float(memory_budgets_mb), dtype=np.float64)
        else:
            B_k = np.asarray(memory_budgets_mb, dtype=np.float64)
            if len(B_k) == 1:
                B_k = np.full(num_clients, B_k[0], dtype=np.float64)
    else:
        B_k = np.full(num_clients, g_0 + float(np.sum(g_r)) + 1000.0, dtype=np.float64)

    # ---- Stage 2: per-client local knapsack on top-M candidates ----
    client_selected = []
    for cid in range(num_clients):
        candidates = topm_selected[cid]
        if not candidates:
            candidates = [int(np.argmax(client_scores[cid]))]
        candidate_values = [max(float(client_scores[cid][s]), 0.0) for s in candidates]
        candidate_costs = [float(g_r[s]) for s in candidates]
        budget = float(B_k[cid])

        selected_indices = knapsack_dp_select(
            values=candidate_values,
            weights=candidate_costs,
            budget=budget,
            base_cost=g_0,
        )
        client_selected.append([candidates[i] for i in selected_indices])

    # ---- Coverage stats ----
    scale_counts = np.zeros(num_scales, dtype=np.int64)
    for sel in client_selected:
        for s in sel:
            if 0 <= s < num_scales:
                scale_counts[s] += 1

    info = {
        "iterations": 1,
        "best_iter": 0,
        "converged": True,
        "coverage_final": scale_counts.tolist(),
        "coverage_target": "n/a (local knapsack on top-M)",
        "total_shortfall": 0.0,
        "lambda_final": [],
        "_costs_source": costs_source,
        "_budgets_source": os.environ.get("SPILTER_MEMORY_BUDGETS", "") and "env_per_client" or "config",
        "_budget_mb": float(np.mean(B_k)) if len(B_k) > 0 else None,
        "_budget_mb_min": float(np.min(B_k)) if len(B_k) > 0 else None,
        "_budget_mb_max": float(np.max(B_k)) if len(B_k) > 0 else None,
        "_budget_mb_per_client": B_k.tolist() if len(B_k) > 0 else None,
        "_base_memory_mb": g_0,
        "_scale_memory_costs_mb": scale_memory_costs_mb,
        "_per_client_scale_counts": [len(s) for s in client_selected],
        "_min_scales_per_client": min(len(s) for s in client_selected) if client_selected else 0,
        "_max_scales_per_client": max(len(s) for s in client_selected) if client_selected else 0,
        "_avg_scales_per_client": (
            sum(len(s) for s in client_selected) / len(client_selected) if client_selected else 0.0
        ),
        "_top_m": top_m,
        "_topm_candidates": topm_selected,  # per-client top-M lists (before knapsack)
    }
    return client_selected, scale_counts, info


def _plan_knapsack_lagrangian_client_scales(
    client_scores,
    model=None,
    *,
    memory_budgets_mb=None,
    scale_memory_costs_mb=None,
    base_memory_mb=0.0,
    coverage_min=None,
    lambda_lr=0.1,
    max_iter=50,
    seed=None,
    # Calibration extras
    X_fed=None,
    device=None,
    batch_size=32,
    in_channels=1,
    num_classes=2,
    dist_measure="mix",
    lr=0.01,
    wd=0.0001,
    dataset_tag="unknown",
):
    """Per-client knapsack under memory budget with global-coverage Lagrangian.

    Memory costs (g_r) resolution order:
      1. ``scale_memory_costs_mb`` from config (real GPU-profiled MB)
      2. Round-0 GPU calibration via LearningShapeletsCL (cached to disk)
      3. ``_scale_system_costs`` from model (unitless proxy)

    Returns (client_selected, scale_counts, info_dict).
    """
    num_clients = len(client_scores)
    num_scales = len(client_scores[0]) if num_clients > 0 else 0
    if num_scales <= 0:
        return [[] for _ in range(num_clients)], np.zeros(0, dtype=np.int64), {}

    # ---- Resolve scale memory costs ----------------------------------------
    costs_source = "config"
    if scale_memory_costs_mb is None and model is not None:
        # Try GPU calibration first
        g_r, g_0 = _calibrate_per_scale_memory_mb(
            X_fed, model, device, batch_size,
            in_channels, num_classes, dist_measure,
            lr, wd, seed, dataset_tag,
        )
        if g_r is not None:
            scale_memory_costs_mb = g_r
            base_memory_mb = g_0 if base_memory_mb <= 0 else base_memory_mb
            costs_source = "round0_calibrated"
        else:
            scale_memory_costs_mb = _scale_system_costs(model)
            costs_source = "system_proxy"

    # ---- Auto budget if not set --------------------------------------------
    if memory_budgets_mb is not None:
        budgets_source = os.environ.get("SPILTER_MEMORY_BUDGETS", "") and "env_per_client" or "config"
    elif scale_memory_costs_mb is not None:
        if costs_source in ("config", "round0_calibrated"):
            # Real MB: budget = 60% of total → ~5-6 scales per client
            memory_budgets_mb = float(np.sum(scale_memory_costs_mb)) * 0.6 + base_memory_mb
            budgets_source = "auto_60pct"
        else:
            # System proxy (unitless)
            memory_budgets_mb = float(np.sum(scale_memory_costs_mb)) * 0.5
            budgets_source = "auto_50pct"
        print(
            f"[spilter] knapsack_lagrangian: auto budget = {memory_budgets_mb:.1f} "
            f"({'MB' if costs_source != 'system_proxy' else 'units'}) "
            f"(source={budgets_source}, costs={costs_source})",
            flush=True,
        )
    else:
        budgets_source = "unconstrained"

    selected, counts, info = knapsack_lagrangian_assign(
        client_scores,
        memory_budgets_mb=memory_budgets_mb,
        scale_memory_costs_mb=scale_memory_costs_mb,
        base_memory_mb=base_memory_mb,
        coverage_min=coverage_min,
        lambda_lr=lambda_lr,
        max_iter=max_iter,
        seed=seed,
    )

    # Log diagnostics (stdout + returned info dict for log file enrichment)
    print(
        f"[spilter] knapsack_lagrangian: {info['iterations']} iters, "
        f"converged={info['converged']}, "
        f"coverage={info['coverage_final']} "
        f"(target={info['coverage_target']}), "
        f"λ_final={[round(v, 4) for v in info['lambda_final']]}",
        flush=True,
    )

    # Extra convenience fields for log messages
    info["_costs_source"] = costs_source
    info["_budgets_source"] = budgets_source
    # Store budget summary: if per-client list, store mean; else store the scalar.
    if isinstance(memory_budgets_mb, (list, tuple)):
        mb_arr = np.asarray(memory_budgets_mb, dtype=np.float64)
        info["_budget_mb"] = float(np.mean(mb_arr))
        info["_budget_mb_min"] = float(np.min(mb_arr))
        info["_budget_mb_max"] = float(np.max(mb_arr))
        info["_budget_mb_per_client"] = mb_arr.tolist()
    else:
        info["_budget_mb"] = float(memory_budgets_mb) if memory_budgets_mb is not None else None
    info["_base_memory_mb"] = base_memory_mb
    info["_scale_memory_costs_mb"] = (
        scale_memory_costs_mb.tolist() if hasattr(scale_memory_costs_mb, 'tolist')
        else list(scale_memory_costs_mb) if scale_memory_costs_mb is not None else None
    )
    info["_per_client_scale_counts"] = [len(s) for s in selected]
    info["_min_scales_per_client"] = min(len(s) for s in selected) if selected else 0
    info["_max_scales_per_client"] = max(len(s) for s in selected) if selected else 0
    info["_avg_scales_per_client"] = (
        sum(len(s) for s in selected) / len(selected) if selected else 0.0
    )

    return selected, counts, info


def _spilter_allocation_mode(config):
    spilter_cfg = config.get("spilter", {}) or {}
    mode = str(spilter_cfg.get("allocation_mode", "efficiency_aware")).strip().lower()
    aliases = {
        "uniform": "uniform_single",
        "single": "uniform_single",
        "uniform-single": "uniform_single",
        "uniform_single_scale": "uniform_single",
        "global": "global_score_random_single",
        "global-score": "global_score_random_single",
        "global_score": "global_score_random_single",
        "score_random": "global_score_random_single",
        "period_random": "global_score_random_single",
        "local": "local_score_topm",
        "local_topm": "local_score_topm",
        "local-topm": "local_score_topm",
        "local-score-topm": "local_score_topm",
        "local_period_topm": "local_score_topm",
        "period_topm": "local_score_topm",
        "period-aware-topm": "local_score_topm",
        "local_random_topm": "local_score_random_topm",
        "random_topm": "local_score_random_topm",
        "local-random-topm": "local_score_random_topm",
        "random-local-topm": "local_score_random_topm",
        "efficiency": "efficiency_aware",
        "efficiency-aware": "efficiency_aware",
        "knapsack": "knapsack_lagrangian",
        "knapsack-lagrangian": "knapsack_lagrangian",
        "knapsack_lagrangian": "knapsack_lagrangian",
        "lagrangian": "knapsack_lagrangian",
        "lagrange": "knapsack_lagrangian",
    }
    return aliases.get(mode, mode)


def _spilter_local_top_m(config, default=4):
    """读取 Spilter 的 local_top_m 参数，支持 int 或 list（per-client m 值）。"""
    spilter_cfg = config.get("spilter", {}) or {}
    for key in ("local_top_m", "top_m", "num_selected_scales"):
        if key in spilter_cfg:
            val = spilter_cfg[key]
            if isinstance(val, (list, tuple)):
                return [max(1, int(v)) for v in val]
            return max(1, int(val))
    return max(1, int(default))


def _spilter_system_extra_scale_count(config, default=2):
    """读取 Spilter 系统效率补发尺度数。"""
    spilter_cfg = config.get("spilter", {}) or {}
    if "system_extra_scale_count" in spilter_cfg:
        return max(0, int(spilter_cfg["system_extra_scale_count"]))
    return max(0, int(default))


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
    momentum,
    use_scale_split_comm,
    server_state_cpu,
    previous_client_states,
    beta,
    shared_kwargs,
    client_scale_plans,
    client_scale_scores,
):
    if device.type == "cuda":
        torch.cuda.set_device(device)

    teacher = None
    algo_name = str(shared_kwargs.get("config", {}).get("algo", "fedcsl")).lower()
    need_teacher = algo_name in ("fedcsl", "spilter", "fedcsl-spilter", "fedprox", "moon")
    if need_teacher and round_idx != 0:
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
            c.set_optimizer(optim.SGD(c.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum))
        else:
            for group in c.optimizer.param_groups:
                group["params"] = list(c.model.parameters())

        c.Q = len(y_fed[idx]) / total_samples if total_samples > 0 else 0.0
        c.Global_Model = teacher.model if round_idx != 0 and teacher is not None else None
        previous_model = None
        if algo_name == "moon" and round_idx != 0 and previous_client_states is not None:
            previous_state = previous_client_states[idx] if idx < len(previous_client_states) else None
            if previous_state is not None:
                previous_model = LearningShapeletsCL(**{**shared_kwargs, "device": device})
                _load_state_to_model(previous_model.model, previous_state)
                previous_model.model.eval()
                for p in previous_model.model.parameters():
                    p.requires_grad_(False)
        c.Previous_Model = previous_model.model if previous_model is not None else None
        if algo_name == "moon":
            _load_state_to_model(c.model, server_state_cpu)
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
            "loss_breakdown": {key: 0.0 for key in _LOSS_KEYS},
            "state": None,
            "scale_indices": list(selected_scales),
            "scale_states": None,
            "memory_summary": None,
            "worker_id": worker_id,
            "device": str(device),
            "skipped": False,
        }

        if len(X_fed[idx]) == 0 or len(y_fed[idx]) == 0:
            print(f"[warn] client {idx} 数据为空，跳过训练")
            result["state"] = _state_dict_to_cpu(c.model.state_dict())
            result["skipped"] = True
            c.Global_Model = None
            c.Previous_Model = None
            c.Selected_Scales = None
            c.Cached_Scale_Scores = None
            results.append(result)
            continue

        if use_scale_split_comm:
            # Spilter 架构：客户端本地模型持续训练，不被服务端参数覆盖。
            # 服务端聚合模型（Global_Model / teacher）仅作为对比蒸馏参考，
            # 梯度不回传给 teacher，全局聚合使用客户端自身训练后的尺度参数。
            selected_scale_indices = selected_scales or [
                _client_scale_score_batch(
                    X_fed[idx],
                    teacher.model if teacher is not None else c.model,
                    beta=beta,
                    device=device,
                )
            ]
            if _SPILTER_DEBUG:
                _pre_norms = {
                    si: _dbg_param_norm(c.model, c.model._scale_state_prefixes(si))
                    for si in selected_scale_indices
                }
                print(
                    f"[spilter_dbg] round={round_idx} client={idx} "
                    f"selected_scales={selected_scale_indices} "
                    f"pre_train_norms={ {si: f'{_pre_norms[si]:.3f}' for si in selected_scale_indices} }",
                    flush=True,
                )
            result["scale_indices"] = [int(scale_idx) for scale_idx in selected_scale_indices]

        if _SPILTER_DEBUG and use_scale_split_comm:
            _pre_train_norms = {
                si: _dbg_param_norm(c.model, c.model._scale_state_prefixes(si))
                for si in (result.get("scale_indices") or [])
            }

        losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size,
                         epoch_idx=-1, lr=lr)
        if not losses:
            loss_all = 0.0
            loss_breakdown = {key: 0.0 for key in _LOSS_KEYS}
        else:
            loss_breakdown = _mean_loss_breakdown(losses)
            loss_all = loss_breakdown["total"]
            if not np.isfinite(loss_all):
                print(f"[warn] client {idx} loss NaN/Inf，置 0")
                loss_all = 0.0
                loss_breakdown["total"] = 0.0

        # ---- round-0 per-scale memory collection (MAST-style) ----
        memory_summary = None
        if round_idx == 0 and use_scale_split_comm:
            try:
                memory_summary = c.model.get_per_scale_memory_summary()
            except Exception:
                memory_summary = None

        if _SPILTER_DEBUG and use_scale_split_comm:
            scale_ids = result.get("scale_indices") or []
            all_prefixes = []
            for si in scale_ids:
                all_prefixes.extend(c.model._scale_state_prefixes(si))

            # ① param norm change after local training
            post_norms = {
                si: _dbg_param_norm(c.model, c.model._scale_state_prefixes(si))
                for si in scale_ids
            }
            norm_delta = " ".join(
                f"s{si}:{_pre_train_norms.get(si, 0.):.3f}->{post_norms.get(si, 0.):.3f}"
                for si in scale_ids
            )

            # ② gradient norm from the last batch (grad still in param.grad after step)
            grad_norm = _dbg_grad_norm(c.model, all_prefixes)

            # ③ within-epoch loss trend: first 3 vs last 3 batches
            if losses:
                n = len(losses)
                K = min(3, n)
                def _fmt(bd):
                    return f"{bd['total']:.4f}(b:{bd['base']:.4f})"
                first_k = [_fmt(bd) for bd in losses[:K]]
                last_k  = [_fmt(bd) for bd in losses[-K:]]
                avg_base = float(np.mean([bd['base'] for bd in losses]))
                trend_str = (
                    f"batches={n} first{K}={first_k} last{K}={last_k} "
                    f"avg_base={avg_base:.4f}"
                )
            else:
                trend_str = "batches=0"

            print(
                f"[spilter_dbg] round={round_idx} client={idx} "
                f"{trend_str} "
                f"grad_norm(last_batch)={grad_norm:.4f} "
                f"param_norm_delta({norm_delta})",
                flush=True,
            )

        result["loss"] = loss_all
        result["loss_breakdown"] = loss_breakdown
        result["state"] = _state_dict_to_cpu(c.model.state_dict())
        result["memory_summary"] = memory_summary
        if use_scale_split_comm and result["scale_indices"]:
            result["scale_states"] = {
                int(scale_idx): c.model.scale_state_dict(scale_idx, clone=True, cpu=True)
                for scale_idx in result["scale_indices"]
            }
        c.Global_Model = None
        c.Previous_Model = None
        c.Selected_Scales = None
        c.Cached_Scale_Scores = None
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 下游评估：SVM / Linear Probe（二选一，可由 config.evaluation.protocol 控制）
# ---------------------------------------------------------------------------
_SVC_C_GRID = [10 ** i for i in range(-4, 5)]


def _to_numpy_labels(y):
    if hasattr(y, "cpu"):
        y = y.cpu().numpy()
    return np.asarray(y)


def _round_eval_interval(config, args=None):
    """联邦通信轮下游评估间隔（默认每轮评估一次）。"""
    if args is not None and getattr(args, "eval_every_n_rounds", None) is not None:
        return max(1, int(args.eval_every_n_rounds))
    env_val = os.environ.get("EVAL_EVERY_N_ROUNDS")
    if env_val is not None and str(env_val).strip():
        return max(1, int(env_val))
    cfg = (config or {}).get("evaluation", {}) or {}
    for key in ("round_eval_interval", "eval_every_n_rounds"):
        if key in cfg and cfg[key] is not None:
            return max(1, int(cfg[key]))
    return 1


def _should_eval_at_round(round_idx, num_rounds, interval):
    interval = max(1, int(interval))
    if round_idx == 0:
        return True
    if round_idx + 1 >= int(num_rounds):
        return True
    return (round_idx + 1) % interval == 0


def _evaluation_config(config, protocol_override=None):
    cfg = (config or {}).get("evaluation", {}) or {}
    protocol = protocol_override or os.environ.get("EVAL_PROTOCOL") or cfg.get("protocol", cfg.get("method", "svm"))
    protocol = str(protocol).strip().lower()
    aliases = {
        "svc": "svm",
        "linear": "linear_probe",
        "linear-probe": "linear_probe",
        "probe": "linear_probe",
        "lp": "linear_probe",
    }
    protocol = aliases.get(protocol, protocol)
    if protocol not in {"svm", "linear_probe"}:
        protocol = "svm"

    c_grid = cfg.get("svm_c_grid", _SVC_C_GRID)
    if not isinstance(c_grid, (list, tuple)) or not c_grid:
        c_grid = _SVC_C_GRID
    c_grid = [float(v) for v in c_grid]

    lp = cfg.get("linear_probe", {}) or {}
    return {
        "protocol": protocol,
        "svm_c_grid": c_grid,
        "linear_probe": {
            "lr": float(lp.get("lr", 1e-3)),
            "wd": float(lp.get("wd", 1e-4)),
            "batch_size": int(lp.get("batch_size", 256)),
            "max_epoch": int(lp.get("max_epoch", 200)),
            "eval_interval": int(lp.get("eval_interval", 1)),
            "seed": int(lp.get("seed", 42)),
        },
    }


def _eval_svm_train_test(transformation, transformation_test, y_train, y_test, *, c_grid=None):
    y_train = _to_numpy_labels(y_train)
    y_test = _to_numpy_labels(y_test)
    grid = c_grid or _SVC_C_GRID
    best_acc, c_best = -1.0, grid[0]
    for c in grid:
        clf = SVC(C=float(c), random_state=42)
        clf.fit(transformation, y_train)
        acc = accuracy_score(clf.predict(transformation), y_train)
        if acc > best_acc:
            best_acc, c_best = acc, float(c)
    clf = SVC(C=c_best, random_state=42)
    clf.fit(transformation, y_train)
    train_acc = accuracy_score(clf.predict(transformation), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc


def _eval_svm_with_val(transformation_train, transformation_test, transformation_val, y_train, y_test, y_val, *, c_grid=None):
    y_train = _to_numpy_labels(y_train)
    y_test = _to_numpy_labels(y_test)
    y_val = _to_numpy_labels(y_val)
    grid = c_grid or _SVC_C_GRID
    best_val_acc, c_best = -1.0, grid[0]
    for c in grid:
        clf = SVC(C=float(c), random_state=42)
        clf.fit(transformation_train, y_train)
        acc_i = accuracy_score(clf.predict(transformation_val), y_val)
        if acc_i > best_val_acc:
            best_val_acc, c_best = acc_i, float(c)
    clf = SVC(C=c_best, random_state=42)
    clf.fit(transformation_train, y_train)
    train_acc = accuracy_score(clf.predict(transformation_train), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc


def _linear_probe_eval(
    transformation_train,
    transformation_test,
    y_train,
    y_test,
    *,
    transformation_val=None,
    y_val=None,
    params=None,
):
    params = params or {}
    lr = float(params.get("lr", 1e-3))
    wd = float(params.get("wd", 1e-4))
    batch_size = max(4, int(params.get("batch_size", 256)))
    max_epoch = max(1, int(params.get("max_epoch", 200)))
    eval_interval = max(1, int(params.get("eval_interval", 1)))
    seed = int(params.get("seed", 42))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_train = torch.as_tensor(np.asarray(transformation_train), dtype=torch.float32)
    x_test = torch.as_tensor(np.asarray(transformation_test), dtype=torch.float32)
    y_train_np = _to_numpy_labels(y_train)
    y_test_np = _to_numpy_labels(y_test)

    classes = np.unique(y_train_np)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    y_train_enc = torch.as_tensor([class_to_idx[c] for c in y_train_np.tolist()], dtype=torch.long)
    y_test_enc = torch.as_tensor([class_to_idx[c] for c in y_test_np.tolist()], dtype=torch.long)

    has_val = transformation_val is not None and y_val is not None
    if has_val:
        x_val = torch.as_tensor(np.asarray(transformation_val), dtype=torch.float32)
        y_val_np = _to_numpy_labels(y_val)
        y_val_enc = torch.as_tensor([class_to_idx[c] for c in y_val_np.tolist()], dtype=torch.long)
    else:
        x_val = None
        y_val_enc = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = nn.Linear(x_train.shape[1], len(classes)).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(10, max_epoch // 10), min_lr=1e-4
    )

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train_enc),
        batch_size=max(4, min(batch_size, int(x_train.shape[0]))),
        shuffle=True,
        drop_last=False,
    )

    best_metric = -1.0
    best_state = copy.deepcopy(probe.state_dict())

    def _accuracy(x_tensor, y_tensor):
        probe.eval()
        preds = []
        labels = []
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=max(4, min(batch_size, int(x_tensor.shape[0]))),
            shuffle=False,
            drop_last=False,
        )
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                logits = probe(xb)
                preds.append(torch.argmax(logits, dim=1).cpu())
                labels.append(yb.cpu())
        pred = torch.cat(preds).numpy()
        label = torch.cat(labels).numpy()
        return accuracy_score(label, pred)

    for epoch in range(max_epoch):
        probe.train()
        epoch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        if (epoch + 1) % eval_interval != 0 and epoch + 1 != max_epoch:
            continue
        metric = _accuracy(x_val, y_val_enc) if has_val else _accuracy(x_train, y_train_enc)
        if metric >= best_metric:
            best_metric = float(metric)
            best_state = copy.deepcopy(probe.state_dict())

    probe.load_state_dict(best_state)
    train_acc = _accuracy(x_train, y_train_enc)
    test_acc = _accuracy(x_test, y_test_enc)
    return train_acc, test_acc


def _make_downstream_eval_fns(config, protocol_override=None):
    eval_cfg = _evaluation_config(config, protocol_override=protocol_override)
    protocol = eval_cfg["protocol"]

    if protocol == "linear_probe":
        lp_params = eval_cfg["linear_probe"]

        def eval_train_test_fn(transformation, transformation_test, y_train, y_test):
            return _linear_probe_eval(
                transformation,
                transformation_test,
                y_train,
                y_test,
                params=lp_params,
            )

        def eval_tstcc_fn(transformation_train, transformation_test, transformation_val, y_train, y_test, y_val):
            return _linear_probe_eval(
                transformation_train,
                transformation_test,
                y_train,
                y_test,
                transformation_val=transformation_val,
                y_val=y_val,
                params=lp_params,
            )

        desc = (
            f"linear_probe(lr={lp_params['lr']}, wd={lp_params['wd']}, "
            f"batch={lp_params['batch_size']}, epoch={lp_params['max_epoch']})"
        )
        return eval_train_test_fn, eval_tstcc_fn, desc

    svm_grid = eval_cfg["svm_c_grid"]

    def eval_train_test_fn(transformation, transformation_test, y_train, y_test):
        return _eval_svm_train_test(
            transformation,
            transformation_test,
            y_train,
            y_test,
            c_grid=svm_grid,
        )

    def eval_tstcc_fn(transformation_train, transformation_test, transformation_val, y_train, y_test, y_val):
        return _eval_svm_with_val(
            transformation_train,
            transformation_test,
            transformation_val,
            y_train,
            y_test,
            y_val,
            c_grid=svm_grid,
        )

    return eval_train_test_fn, eval_tstcc_fn, f"svm(C_grid={svm_grid})"


def eval(transformation, transformation_test, y_train, y_test, evaluation_cfg=None):
    cfg = evaluation_cfg or {"protocol": "svm", "svm_c_grid": _SVC_C_GRID, "linear_probe": {}}
    if cfg.get("protocol") == "linear_probe":
        return _linear_probe_eval(transformation, transformation_test, y_train, y_test, params=cfg.get("linear_probe"))
    return _eval_svm_train_test(transformation, transformation_test, y_train, y_test, c_grid=cfg.get("svm_c_grid"))


def eval_TSTCC(transformation_train, transformation_test, transformation_val,
               y_train, y_test, y_val, evaluation_cfg=None):
    cfg = evaluation_cfg or {"protocol": "svm", "svm_c_grid": _SVC_C_GRID, "linear_probe": {}}
    if cfg.get("protocol") == "linear_probe":
        return _linear_probe_eval(
            transformation_train,
            transformation_test,
            y_train,
            y_test,
            transformation_val=transformation_val,
            y_val=y_val,
            params=cfg.get("linear_probe"),
        )
    return _eval_svm_with_val(
        transformation_train,
        transformation_test,
        transformation_val,
        y_train,
        y_test,
        y_val,
        c_grid=cfg.get("svm_c_grid"),
    )


def train(dataset="", seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True,
           eval_per_x_epochs=10, dist_measure='mix', rank=-1, world_size=-1, resize=0,
           checkpoint=False, task='classification'):
    global _ACTIVE_RESULT_LOG
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

    numClient = args.num_client if args.num_client is not None else fed_cfg['numClient']
    numClient = max(1, int(numClient))
    numRound = args.num_rounds if args.num_rounds is not None else fed_cfg['numRound']
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
    eval_protocol_override = args.eval_protocol if args.eval_protocol is not None else None
    eval_train_test_fn, eval_tstcc_fn, eval_protocol_desc = _make_downstream_eval_fns(
        config, protocol_override=eval_protocol_override
    )
    round_eval_interval = _round_eval_interval(config, args)
    print(f"downstream evaluation: {eval_protocol_desc}; every {round_eval_interval} round(s)")
    lr = model_cfg['lr']
    batch_size = args.batch_size if args.batch_size is not None else model_cfg['batch_size']
    wd = model_cfg.get('wd', 0.0001)
    momentum = model_cfg.get('momentum', 0.9)
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

    if args.description is not None:
        config['description'] = args.description
    config['dataset'] = dataset
    config.setdefault('federated', {})
    config.setdefault('model', {}).setdefault('params', {})
    config['federated']['dirichlet_alpha'] = dirichlet_alpha
    config['federated']['numClient'] = numClient
    config['federated']['use_client_selection'] = use_client_selection
    config['federated']['client_selection_ratio'] = client_selection_ratio
    config['federated']['min_selection_prob'] = min_selection_prob
    config['federated']['ema_alpha'] = ema_alpha
    config['federated']['client_selection_method'] = client_selection_method
    config['federated']['client_workers'] = client_workers
    config['federated']['client_gpus'] = client_gpu_ids
    config['federated']['server_gpu'] = server_gpu
    config['model']['params']['batch_size'] = batch_size

    desc_safe = _sanitize_filename(config.get('description', ''))
    formatted_date = datetime.now().strftime("%Y-%m-%d-%H") + desc_safe
    logTxt = f"./result/{dataset}/{formatted_date}_l={l}_lr={lr}_epoch{numEpoch}_alphadir{dirichlet_alpha}_{desc_safe}.txt"
    _ACTIVE_RESULT_LOG = logTxt
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
        f"argv: {' '.join(sys.argv)}",
        f"effective_dataset: {dataset}",
        f"effective_dirichlet_alpha: {dirichlet_alpha}",
        f"effective_eval_every_n_rounds: {round_eval_interval}",
        yaml.dump(config).replace('\n', ''),
    ]
    _append_text_to_log(logTxt, "\n".join(header_lines) + "\n")

    shapelet_weight_X = np.load('./algoutils/shapelet_weight_All.npy')

    # ----- 加载数据集（统一入口：HAR / Epilepsy-TSTCC / SleepEDF / FD-A / 其他 UEA）
    has_val = False
    X_val, y_val = None, None

    if dataset == "HAR":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        if os.path.isfile("./HAR/val.pt"):
            val_data = torch.load("./HAR/val.pt", weights_only=True)
            X_val = val_data["samples"].float()
            y_val = val_data["labels"].int()
            has_val = True
    elif dataset == "Epilepsy-TSTCC":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_Epilepsy(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./Epilepsy/val.pt", weights_only=True)
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()
        has_val = True
    elif dataset == "SleepEDF":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_SleepEDF(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./sleepEDF/val.pt", weights_only=True)
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()
        has_val = True
    elif dataset == "FD-A":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_FDA(numClient, dirichlet_alpha, scoreX=shapelet_weight_X, scoreY=None)
        val_data = torch.load("./FD-A/val.pt", weights_only=True)
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
    print("shapelet initialized!")

    # 有监督 FL-bench 风格基线：SCAFFOLD / FedProto 不走 CSL 多尺度对比流程，直接接入。
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
            eval_train_test_fn=eval_train_test_fn,
            eval_tstcc_fn=eval_tstcc_fn,
            save_model_fn=save_model,
            round_eval_interval=round_eval_interval,
        )
        return

    # 无监督联邦表征基线：BYOL / Orchestra / FedU2 / PatchTST 复用 SVM 评估协议。
    if algo.lower() in ('byol', 'orchestra', 'fedu2', 'fedu2-byol', 'patchtst', 'fedpatchtst'):
        from algo.ssl_runner import run_ssl_baseline
        run_ssl_baseline(
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
            eval_train_test_fn=eval_train_test_fn,
            eval_tstcc_fn=eval_tstcc_fn,
            save_model_fn=save_model,
            round_eval_interval=round_eval_interval,
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
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
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
    ckpt_every = int(fed_cfg.get('checkpoint_every', 0) or 0)
    auto_resume = bool(fed_cfg.get('auto_resume', False))
    resume_path = _resume_ckpt_path(dataset, formatted_date)
    resume_ckpt_override = os.environ.get("RESUME_CKPT", "").strip()
    if resume_ckpt_override:
        override_path = resume_ckpt_override
        if not os.path.isabs(override_path):
            override_path = os.path.join(".", override_path)
        if os.path.isfile(override_path):
            resume_path = override_path
            print(f"[fedcsl] using resume checkpoint override: {resume_path}", flush=True)
        else:
            print(f"[fedcsl] resume checkpoint override not found: {resume_ckpt_override}", flush=True)
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
    use_scale_split_comm = _uses_scale_split_algo(algo)
    if not w_locals:
        w_locals = [_state_dict_to_cpu(c.model.state_dict()) for c in clientList]

    cached_client_scale_scores = None
    cached_client_scale_plans = None
    cached_scale_hist = None
    cached_global_scale_probs = None
    cached_knapsack_info = None
    scale_score_prep_sec = 0.0
    spilter_allocation_mode = _spilter_allocation_mode(config) if _uses_scale_split_algo(algo) else None
    if spilter_allocation_mode and args.spilter_random:
        spilter_allocation_mode = "local_score_random_topm"
        if args.description:
            config['description'] = args.description + "_random"
    # knapsack-lagrangian override: CLI flag OR env var
    _spilter_knapsack_env = os.environ.get("SPILTER_KNAPSACK", "").strip()
    if spilter_allocation_mode and (args.spilter_knapsack or _spilter_knapsack_env == "1"):
        spilter_allocation_mode = "knapsack_lagrangian"
        if args.description:
            config['description'] = args.description + "_knapsack"
    if _uses_fedcsl_scale_scores(algo):
        scale_prep_t0 = time.perf_counter()
        if _uses_scale_split_algo(algo) and spilter_allocation_mode == "uniform_single":
            cached_client_scale_plans, cached_scale_hist = _plan_uniform_single_client_scales(
                len(X_fed),
                server.model,
            )
        else:
            cached_client_scale_scores = _precompute_client_scale_scores(
                X_fed,
                server.model,
                beta=beta,
                device=server_device,
                batch_size=batch_size,
            )
            if _uses_scale_split_algo(algo) and spilter_allocation_mode == "global_score_random_single":
                cached_client_scale_plans, cached_scale_hist, cached_global_scale_probs = (
                    _plan_global_score_random_single_client_scales(
                        cached_client_scale_scores,
                        y_fed,
                        seed=original_seed or 42,
                    )
                )
            elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "local_score_topm":
                local_top_m = _spilter_local_top_m(config, default=4)
                cached_client_scale_plans, cached_scale_hist = _plan_local_score_topm_client_scales(
                    cached_client_scale_scores,
                    top_m=local_top_m,
                )
            elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "local_score_random_topm":
                local_top_m = _spilter_local_top_m(config, default=4)
                cached_client_scale_plans, cached_scale_hist = _plan_local_score_random_topm_client_scales(
                    cached_client_scale_scores,
                    top_m=local_top_m,
                    seed=original_seed or 42,
                )
            elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "knapsack_lagrangian":
                knap_params = _spilter_knapsack_lagrangian_params(config, override_memory_budget=args.spilter_memory_budget)
                # Force top-4 for spilter-memory-budget experiment (config may
                # set per-client values for other experiments; knapsack always uses 4)
                cached_client_scale_plans, cached_scale_hist, cached_knapsack_info = (
                    _plan_topm_then_local_knapsack_client_scales(
                        cached_client_scale_scores,
                        server_model=server.model,
                        top_m=4,
                        X_fed=X_fed,
                        device=server_device,
                        batch_size=batch_size,
                        seed=original_seed or 42,
                        in_channels=n_channels,
                        num_classes=num_classes,
                        dist_measure=dist_measure,
                        lr=lr,
                        wd=wd,
                        dataset_tag=dataset,
                        **knap_params,
                    )
                )
        if _uses_scale_split_algo(algo) and spilter_allocation_mode not in (
            "uniform_single",
            "global_score_random_single",
            "local_score_topm",
            "local_score_random_topm",
            "knapsack_lagrangian",
        ):
            system_extra_scale_count = _spilter_system_extra_scale_count(config, default=2)
            cached_client_scale_plans, cached_scale_hist = _plan_efficiency_aware_client_scales_from_scores(
                cached_client_scale_scores,
                server.model,
                extra_scale_count=system_extra_scale_count,
            )
        scale_score_prep_sec = time.perf_counter() - scale_prep_t0
        if _uses_scale_split_algo(algo) and spilter_allocation_mode == "uniform_single":
            prep_msg = f"[spilter] planned uniform single-scale clients in {scale_score_prep_sec:.3f}s"
        elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "global_score_random_single":
            prep_msg = f"[spilter] planned global-score random single-scale clients in {scale_score_prep_sec:.3f}s"
        elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "local_score_topm":
            prep_msg = f"[spilter] planned local-score top-m stitched clients in {scale_score_prep_sec:.3f}s"
        elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "local_score_random_topm":
            prep_msg = f"[spilter] planned local-score RANDOM top-m stitched clients in {scale_score_prep_sec:.3f}s"
        elif _uses_scale_split_algo(algo) and spilter_allocation_mode == "knapsack_lagrangian":
            prep_msg = f"[spilter] planned knapsack-lagrangian global-coverage scales in {scale_score_prep_sec:.3f}s"
        else:
            prep_msg = f"[fedcsl] precomputed client scale scores once in {scale_score_prep_sec:.3f}s"
        if cached_scale_hist is not None:
            prep_msg += f"; spilter_allocation_mode={spilter_allocation_mode}"
            if spilter_allocation_mode not in (
                "uniform_single",
                "global_score_random_single",
                "local_score_topm",
                "local_score_random_topm",
                "knapsack_lagrangian",
            ):
                prep_msg += f"; system_extra_scale_count={system_extra_scale_count}"
            if spilter_allocation_mode in ("local_score_topm", "local_score_random_topm"):
                prep_msg += f"; local_top_m={local_top_m}"
            if cached_global_scale_probs is not None:
                prep_msg += f"; global_scale_probs={np.round(cached_global_scale_probs, 4).tolist()}"
            prep_msg += f"; planned scale coverage: {cached_scale_hist.tolist()}"
            if cached_knapsack_info:
                ki = cached_knapsack_info
                g_r = ki.get('_scale_memory_costs_mb')
                g_r_str = f"[{', '.join(f'{v:.1f}' for v in g_r)}]" if g_r is not None and len(g_r) > 0 else "?"
                prep_msg += (
                    f"; top_m={ki.get('_top_m', '?')}"
                    f" costs={ki.get('_costs_source', '?')}"
                    f" g_r={g_r_str}{'MB' if ki.get('_costs_source') in ('config', 'server_forward') else ''}"
                    f" g_0={ki.get('_base_memory_mb', 0):.1f}MB"
                    f" budget={_fmt_knap_budget(ki)}"
                    f" budgets_src={ki.get('_budgets_source', '?')}"
                    f" scales_per_client=[{ki.get('_min_scales_per_client', '?')}"
                    f",{ki.get('_avg_scales_per_client', 0):.1f}"
                    f",{ki.get('_max_scales_per_client', '?')}]"
                )
        print(prep_msg, flush=True)
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            f.write(prep_msg + "\n")

    for round in range(start_round, numRound):
        round_t0 = time.perf_counter()
        avg_loss = 0.0
        avg_loss_terms = {key: 0.0 for key in _LOSS_KEYS}
        client_losses = [0.0] * numClient  # 供 Oort 更新 reward
        client_scale_states = [None] * numClient
        client_scale_indices = [[] for _ in range(numClient)]

        server_state_cpu = _state_dict_to_cpu(server.model.state_dict())
        client_scale_plans = cached_client_scale_plans
        round_scale_hist = cached_scale_hist
        if round_scale_hist is not None:
            mem_info = ""
            if cached_knapsack_info:
                ki = cached_knapsack_info
                budget_val = ki.get('_budget_mb')
                if budget_val is not None:
                    bmin = ki.get('_budget_mb_min')
                    bmax = ki.get('_budget_mb_max')
                    if bmin is not None and bmax is not None and bmin != bmax:
                        budget_str = f"{budget_val:.0f}MB [{bmin:.0f}..{bmax:.0f}]"
                    else:
                        budget_str = f"{budget_val:.0f}MB"
                else:
                    budget_str = "unconstrained"
                mem_info = (
                    f" | knapsack: budget={budget_str}"
                    f" g_0={ki.get('_base_memory_mb',0):.0f}MB"
                    f" scales/client=[{ki.get('_min_scales_per_client','?')}"
                    f"..{ki.get('_max_scales_per_client','?')}]"
                    f" costs={ki.get('_costs_source','?')}"
                )
            print(f"[round {round}] planned scale coverage: {round_scale_hist.tolist()}{mem_info}", flush=True)
            if client_scale_plans is not None:
                g_r_list = cached_knapsack_info.get("_scale_memory_costs_mb") if cached_knapsack_info else None
                costs_src = cached_knapsack_info.get("_costs_source", "") if cached_knapsack_info else ""
                per_client_budgets = cached_knapsack_info.get("_budget_mb_per_client") if cached_knapsack_info else None
                topm_candidates = cached_knapsack_info.get("_topm_candidates") if cached_knapsack_info else None
                # Only show memory when we have real measurements (not system proxy)
                show_mem = g_r_list is not None and costs_src not in ("system_proxy", "unknown", "config")
                plan_lines = []
                for cid, scales in enumerate(client_scale_plans):
                    topm = topm_candidates[cid] if topm_candidates and cid < len(topm_candidates) else []
                    if show_mem:
                        g_r = np.asarray(g_r_list, dtype=np.float64)
                        used = sum(g_r[s] for s in scales)
                        c_budget = per_client_budgets[cid] if per_client_budgets and cid < len(per_client_budgets) else None
                        budget_str = f" {used:.0f}/{c_budget:.0f}MB" if c_budget is not None else f" {used:.0f}MB"
                    else:
                        budget_str = ""
                    topm_str = f"top{len(topm)}={sorted(topm)}" if topm else ""
                    sel_str = f"→{sorted(scales)}" if sorted(scales) != sorted(topm) else ""
                    plan_lines.append(f"c{cid}:{topm_str}{sel_str}{budget_str}")
                print(f"[round {round}] per-client scales: {' | '.join(plan_lines)}", flush=True)

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
                    momentum,
                    use_scale_split_comm,
                    server_state_cpu,
                    w_locals,
                    beta,
                    shared_kwargs,
                    client_scale_plans,
                    cached_client_scale_scores,
                ))
            for future in as_completed(futures):
                for result in future.result():
                    idx = result["idx"]
                    sample_weight = len(y_fed[idx]) / total_samples
                    client_losses[idx] = result["loss"]
                    avg_loss += result["loss"] * sample_weight
                    breakdown = result.get("loss_breakdown") or {key: 0.0 for key in _LOSS_KEYS}
                    for key in _LOSS_KEYS:
                        avg_loss_terms[key] += float(breakdown.get(key, 0.0) or 0.0) * sample_weight
                    w_locals[idx] = result["state"]
                    if result.get("scale_indices"):
                        client_scale_indices[idx] = list(result["scale_indices"])
                    if use_scale_split_comm and result.get("scale_states"):
                        client_scale_states[idx] = {
                            "scale_indices": list(result.get("scale_indices", [])),
                            "states": result["scale_states"],
                        }
        train_stage_sec = time.perf_counter() - train_stage_t0

        # ----- round-0 per-scale memory report (client-level, pick strongest client) -----
        if round == 0 and use_scale_split_comm:
            # Collect all client memory summaries, pick the "strongest" one
            # (most measured scales) for display.  Differences between clients
            # are expected because each client only trains its assigned scales.
            all_mem_results = []
            for result_list in [future.result() for future in futures]:
                for result in result_list:
                    if result.get("memory_summary"):
                        all_mem_results.append(result)

            if all_mem_results:
                # Pick the client with the most measured scales as the "strong" reference
                def _num_measured(mem):
                    return sum(
                        1 for v in mem.get("per_scale_measured", {}).values() if v
                    )
                best = max(all_mem_results, key=lambda r: _num_measured(r["memory_summary"]))
                best_mem = best["memory_summary"]
                best_idx = best["idx"]
                best_scales = best.get("scale_indices", [])
                n_measured = _num_measured(best_mem)
                per_branch = best_mem.get("per_branch")
                has_branches = bool(per_branch)

                # Accumulated module memory for this client (sum of measured total_mem)
                acc_module_mb = sum(
                    best_mem.get("total_mem_mb", {}).get(L, 0.0)
                    for L, measured in best_mem.get("per_scale_measured", {}).items()
                    if measured
                )
                # Client budget
                client_budget_mb = None
                if cached_knapsack_info:
                    per_client_budgets = cached_knapsack_info.get("_budget_mb_per_client")
                    if per_client_budgets and best_idx < len(per_client_budgets):
                        client_budget_mb = per_client_budgets[best_idx]
                    elif cached_knapsack_info.get("_budget_mb") is not None:
                        client_budget_mb = cached_knapsack_info["_budget_mb"]

                sep = "=" * 94 if has_branches else "=" * 72
                dash = "-" * 94 if has_branches else "-" * 72
                mem_report_lines = [
                    "",
                    sep,
                    "[spilter-memory-budget] Round-0 per-scale GPU memory "
                    "(client {} — {} of {} scales trained)".format(
                        best_idx, n_measured, len(best_mem.get("total_mem_mb", {}))
                    ),
                    "  selected scales: {} | accumulated module mem: {:.1f} MB{}".format(
                        sorted(best_scales) if best_scales else "?",
                        acc_module_mb,
                        " / budget: {:.1f} MB (used {:.0f}%)".format(
                            client_budget_mb,
                            acc_module_mb / client_budget_mb * 100,
                        ) if client_budget_mb is not None and client_budget_mb > 0
                        else "",
                    ),
                ]
                if has_branches:
                    mem_report_lines.append("  → mix-distance: 三个子模块分别显示")
                mem_report_lines.append(dash)

                if has_branches:
                    sorted_lengths = sorted(best_mem.get("total_mem_mb", {}).keys())
                    mem_report_lines.append(
                        f"{'Scale':>6s}  {'Eu(MB)':>8s}  {'Co(MB)':>8s}  {'CC(MB)':>8s}  "
                        f"{'SumPeak':>9s}  {'Param':>8s}  {'Total':>8s}"
                    )
                    eu_data = per_branch.get("euclidean", {})
                    co_data = per_branch.get("cosine", {})
                    cc_data = per_branch.get("cross_corr", {})
                    sum_eu = 0.0
                    sum_co = 0.0
                    sum_cc = 0.0
                    sum_param = 0.0
                    for L in sorted_lengths:
                        eu_mb = eu_data.get("peak", {}).get(L, 0.0)
                        co_mb = co_data.get("peak", {}).get(L, 0.0)
                        cc_mb = cc_data.get("peak", {}).get(L, 0.0)
                        peak_sum = eu_mb + co_mb + cc_mb
                        param_mb = best_mem.get("param_mem_mb", {}).get(L, 0.0)
                        total_mb = peak_sum + param_mb
                        sum_eu += eu_mb
                        sum_co += co_mb
                        sum_cc += cc_mb
                        sum_param += param_mb
                        mem_report_lines.append(
                            f"{L:>6d}  {eu_mb:>8.2f}  {co_mb:>8.2f}  {cc_mb:>8.2f}  "
                            f"{peak_sum:>9.2f}  {param_mb:>8.2f}  {total_mb:>8.2f}"
                        )
                    sum_peak = sum_eu + sum_co + sum_cc
                    sum_total = sum_peak + sum_param
                    mem_report_lines.append(dash)
                    mem_report_lines.append(
                        f"{'SUM':>6s}  {sum_eu:>8.2f}  {sum_co:>8.2f}  {sum_cc:>8.2f}  "
                        f"{sum_peak:>9.2f}  {sum_param:>8.2f}  {sum_total:>8.2f}"
                    )
                    if sum_peak > 0:
                        mem_report_lines.append(
                            f"{'%':>6s}  {sum_eu/sum_peak*100:>7.1f}%  {sum_co/sum_peak*100:>7.1f}%  "
                            f"{sum_cc/sum_peak*100:>7.1f}%"
                        )
                    total_all = sum_total
                else:
                    sorted_lengths = sorted(best_mem.get("total_mem_mb", {}).keys())
                    mem_report_lines.append(
                        f"{'Scale Length':>14s}  {'Peak(MB)':>10s}  {'Param(MB)':>10s}  {'Total(MB)':>10s}"
                    )
                    total_peak = 0.0
                    total_param = 0.0
                    for L in sorted_lengths:
                        peak_mb = best_mem.get("peak_mem_mb", {}).get(L, 0.0)
                        param_mb = best_mem.get("param_mem_mb", {}).get(L, 0.0)
                        total_mb = best_mem.get("total_mem_mb", {}).get(L, 0.0)
                        total_peak += peak_mb
                        total_param += param_mb
                        mem_report_lines.append(
                            f"{L:>14d}  {peak_mb:>10.2f}  {param_mb:>10.2f}  {total_mb:>10.2f}"
                        )
                    total_all = total_peak + total_param
                    mem_report_lines.append(dash)
                    mem_report_lines.append(
                        f"{'SUM':>14s}  {total_peak:>10.2f}  {total_param:>10.2f}  {total_all:>10.2f}"
                    )

                mem_report_lines.append(sep)
                # Also show scale coverage across all clients for reference
                scale_cov = {}
                for r in all_mem_results:
                    for L, measured in r["memory_summary"].get("per_scale_measured", {}).items():
                        if measured:
                            scale_cov[L] = scale_cov.get(L, 0) + 1
                cov_str = " ".join(
                    f"L{L}:{scale_cov.get(L, 0)}/{len(all_mem_results)}"
                    for L in sorted(scale_cov.keys())
                )
                mem_report_lines.append(
                    f"[spilter-memory-budget] scale coverage across {len(all_mem_results)} clients: {cov_str}"
                )
                mem_report_lines.append("")

                mem_report_str = "\n".join(mem_report_lines)
                print(mem_report_str, flush=True)
                with open(logTxt, mode="a+", encoding="utf-8") as f:
                    f.write(mem_report_str + "\n")

                # Store per-scale memory from strongest client for reference
                sorted_lengths_store = sorted(best_mem.get("total_mem_mb", {}).keys())
                _scale_mem_mb_list = [
                    best_mem.get("total_mem_mb", {}).get(L, 0.0)
                    for L in sorted_lengths_store
                ]
                if cached_knapsack_info is None:
                    cached_knapsack_info = {}
                cached_knapsack_info["_round0_scale_mem_mb"] = _scale_mem_mb_list
                cached_knapsack_info["_round0_scale_peak_mb"] = [
                    best_mem.get("peak_mem_mb", {}).get(L, 0.0)
                    for L in sorted_lengths_store
                ]
                cached_knapsack_info["_round0_scale_param_mb"] = [
                    best_mem.get("param_mem_mb", {}).get(L, 0.0)
                    for L in sorted_lengths_store
                ]
                cached_knapsack_info["_round0_total_mb"] = total_all

                # ---- Replace server-estimated g_r with actual client-measured per-scale memory ----
                agg_total = {}
                agg_count = {}
                for r in all_mem_results:
                    mem = r["memory_summary"]
                    for L, measured in mem.get("per_scale_measured", {}).items():
                        if measured:
                            agg_total[L] = agg_total.get(L, 0.0) + mem.get("total_mem_mb", {}).get(L, 0.0)
                            agg_count[L] = agg_count.get(L, 0) + 1
                if agg_total:
                    lengths_sorted = sorted(agg_total.keys())
                    measured_g_r = [agg_total[L] / max(agg_count[L], 1) for L in lengths_sorted]
                    cached_knapsack_info["_scale_memory_costs_mb"] = measured_g_r
                    cached_knapsack_info["_costs_source"] = "round0_measured"
                    print(
                        f"[spilter-memory-budget] updated g_r from round-0 client measurements: "
                        f"g_r={[f'{v:.1f}' for v in measured_g_r]} MB",
                        flush=True,
                    )
                    # Re-print per-client scales with corrected memory values
                    corrected_lines = []
                    per_client_budgets = cached_knapsack_info.get("_budget_mb_per_client")
                    for cid, scales in enumerate(client_scale_plans):
                        c_budget = per_client_budgets[cid] if per_client_budgets and cid < len(per_client_budgets) else None
                        used = sum(measured_g_r[s] for s in scales if s < len(measured_g_r))
                        budget_str = f" {used:.0f}/{c_budget:.0f}MB" if c_budget is not None else f" {used:.0f}MB"
                        corrected_lines.append(f"c{cid}:{sorted(scales)}{budget_str}")
                    print(
                        f"[round {round}] per-client scales (corrected): {' | '.join(corrected_lines)}",
                        flush=True,
                    )

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

            if _SPILTER_DEBUG:
                # 计算本轮聚合后服务端各尺度参数相对上轮的平均变化幅度
                delta_norms = []
                for key in w_global:
                    if key in server_state_cpu:
                        d = float((w_global[key].float() - server_state_cpu[key].float()).norm().item())
                        delta_norms.append(d)
                avg_delta = float(np.mean(delta_norms)) if delta_norms else 0.0
                max_delta = float(np.max(delta_norms)) if delta_norms else 0.0
                n_uploaded = sum(
                    1 for p in filtered_payloads
                    if p and p.get("states")
                )
                print(
                    f"[spilter_dbg] round={round} AGG "
                    f"clients_uploaded={n_uploaded} "
                    f"server_param_delta avg={avg_delta:.5f} max={max_delta:.5f}",
                    flush=True,
                )

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
        do_eval = _should_eval_at_round(round, numRound, round_eval_interval)
        if do_eval:
            transformation = server.transform(X_all, result_type='numpy', normalize=True, batch_size=batch_size)
            transformation_test = server.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
            scaler = RobustScaler()
            transformation = scaler.fit_transform(transformation)
            transformation_test = scaler.transform(transformation_test)
            if has_val and X_val is not None and y_val is not None:
                transformation_val = server.transform(X_val, result_type='numpy', normalize=True, batch_size=batch_size)
                transformation_val = scaler.transform(transformation_val)
                y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else np.asarray(y_val)
                train_acc, test_acc = eval_tstcc_fn(
                    transformation_train=transformation,
                    transformation_test=transformation_test,
                    transformation_val=transformation_val,
                    y_train=y_all, y_test=y_test, y_val=y_val_np,
                )
            else:
                train_acc, test_acc = eval_train_test_fn(transformation, transformation_test, y_train=y_all, y_test=y_test)
            eval_stage_sec = time.perf_counter() - eval_stage_t0

            if test_acc > best_acc:
                best_acc = test_acc
                best_round = round
                # 只 clone 到 CPU，避免 deepcopy 整个 GPU 模型（显著更快、更省显存）
                best_state_dict = _state_dict_to_cpu(server.model.state_dict())

            print(f"Classification: train={train_acc:.4f} test={test_acc:.4f} round={round}")
        else:
            train_acc, test_acc = float("nan"), float("nan")
            eval_stage_sec = 0.0
            print(
                f"[round {round}] downstream eval skipped (every {round_eval_interval} rounds)",
                flush=True,
            )
        round_total_sec = time.perf_counter() - round_t0

        print(
            f"[round {round}] timing train={train_stage_sec:.3f}s "
            f"distribution={dist_stage_sec:.3f}s agg={agg_stage_sec:.3f}s "
            f"eval={eval_stage_sec:.3f}s total={round_total_sec:.3f}s",
            flush=True,
        )

        avg_loss_str = str(avg_loss) if np.isfinite(avg_loss) else "nan"
        avg_loss_terms_str = " ".join(
            f"{key}:{avg_loss_terms[key]:.6f}" if np.isfinite(avg_loss_terms[key]) else f"{key}:nan"
            for key in _LOSS_KEYS
        )
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            if do_eval:
                f.write(
                    f"dataset: {dataset}round:{round} server aggregation "
                    f" testACC:{test_acc} trainACC:{train_acc} avg_loss:{avg_loss_str} {avg_loss_terms_str}\n"
                )
            else:
                f.write(
                    f"dataset: {dataset}round:{round} server aggregation "
                    f" avg_loss:{avg_loss_str} {avg_loss_terms_str} eval_skipped:interval={round_eval_interval}\n"
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
        payload = torch.load(path, map_location="cpu", weights_only=True)
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
