"""FedCSL 主入口：负责数据加载、客户端/服务器初始化、客户端选择与聚合、下游 SVC 评估。

SCAFFOLD / FedProto 等 FL-bench 原生算法会在识别到 ``algo`` 字段后路由到
``algo/baseline_runner.run_baseline``，跳过多尺度对比流程。
"""

import argparse
import os
import random
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
    )

    w_locals = []
    clientList = []
    server = LearningShapeletsCL(**shared_kwargs)
    # 服务器模型只做前向（作为 Global_Model 供客户端蒸馏/对比），
    # 关闭 requires_grad 可避免 client.train() 中 forward 时构建反向图，显著省显存。
    for p in server.model.parameters():
        p.requires_grad_(False)

    for idx in range(numClient):
        client = LearningShapeletsCL(**shared_kwargs)
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd)
        client.set_optimizer(optimizer)
        clientList.append(client)

    print(f"All {len(clientList)} clients initialized.")

    best_acc = 0.0
    best_round = -1
    best_state_dict = None  # 只保存 state_dict 的 CPU 副本，避免频繁 deepcopy GPU 模型

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

    for round in range(numRound):
        avg_loss = 0.0
        client_losses = [0.0] * numClient  # 供 Oort 更新 reward

        # ----- 本地训练阶段：每个客户端在本地数据上训练 numEpoch -----
        for idx, c in enumerate(clientList):
            c.Q = len(y_fed[idx]) / total_samples
            # 客户端对比学习/FedProx 需要参考 Global_Model：直接共享 server.model 引用即可
            # （只做 forward；server.model 已冻结 requires_grad，无副作用且节省一次 deepcopy）
            if round != 0:
                c.Global_Model = server.model

            # 空客户端兜底：保留上一轮权重
            if len(X_fed[idx]) == 0 or len(y_fed[idx]) == 0:
                print(f"[warn] client {idx} 数据为空，跳过训练")
                if round == 0:
                    w_locals.append(c.model.state_dict())
                else:
                    w_locals[idx] = c.model.state_dict()
                continue

            losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size,
                             epoch_idx=-1, lr=lr)
            if not losses:
                loss_all = 0.0
            else:
                loss_all = float(np.mean([loss[0] for loss in losses]))
                if not np.isfinite(loss_all):
                    print(f"[warn] client {idx} loss NaN/Inf，置 0")
                    loss_all = 0.0
            client_losses[idx] = loss_all
            avg_loss += loss_all * len(y_fed[idx]) / total_samples

            if round == 0:
                w_locals.append(c.model.state_dict())
            else:
                w_locals[idx] = c.model.state_dict()

        # ----- 分布打分：cal_score(predict) + normalize（UseDistribution=False 时退化为全 1） -----
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

        # ----- 客户端选择 + 聚合（策略见 algo/client_selection/）-----
        if use_client_selection and selector is not None:
            select_mask = selector.on_round_start(round, client_losses=client_losses, y_fed=y_fed)
            print(f"[{selector.name}] 选择掩码: {select_mask}")
            w_global = selector.aggregate(w_locals, y_fed, scores, select_mask)
            if w_global is None:  # 策略未覆盖时回退默认 FedAvg
                combined_scores = [scores[i] * select_mask[i] for i in range(numClient)]
                w_global = fedavg(w_locals, y_fed, combined_scores)
            server.model.load_state_dict(w_global)
            selector.on_round_end(
                round,
                w_locals=w_locals, w_global=w_global,
                select_mask=select_mask, client_losses=client_losses,
            )
        else:
            w_global = fedavg(w_locals, y_fed, scores)
            server.model.load_state_dict(w_global)

        # ----- 下游 SVC 评估 -----
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

        if test_acc > best_acc:
            best_acc = test_acc
            best_round = round
            # 只 clone 到 CPU，避免 deepcopy 整个 GPU 模型（显著更快、更省显存）
            best_state_dict = {k: v.detach().cpu().clone() for k, v in server.model.state_dict().items()}

        print(f"Classification: train={train_acc:.4f} test={test_acc:.4f} round={round}")

        avg_loss_str = str(avg_loss) if np.isfinite(avg_loss) else "nan"
        with open(logTxt, mode="a+", encoding="utf-8") as f:
            f.write(
                f"dataset: {dataset}round:{round} server aggregation "
                f" testACC:{test_acc} trainACC:{train_acc} avg_loss:{avg_loss_str}\n"
            )

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


if __name__ == '__main__':
    train(dataset=args.dataset, seed=args.seed)


