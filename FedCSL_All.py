from seaborn import dark_palette
from train import LearningShapeletsCL
#from train_origin import LearningShapeletsCL

from fedutil import *
from fedavg import fedavg,fedavg2,FedAvg3
import torch.distributed as dist
import torch
from torch import nn, optim
import random
import numpy as np
from utils import z_normalize,TSC_multivariate_data_loader
import os
import tsaug
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sktime.datasets import load_from_tsfile_to_dataframe,load_UCR_UEA_dataset
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from model.ema import EMA ,update_moving_average,calculate_divergence
import copy
from dataset_utils import *
import yaml
from datetime import datetime

import time



parser = argparse.ArgumentParser()
parser.add_argument('-dataset', help='UEA dataset name')
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
parser.add_argument('-T', '--temperature', default=0.1, type=float, help='temperature')
parser.add_argument('-l', '--lmd', default=1e-2, type=float, help='multi-scale alignment weight')
parser.add_argument('-ls', '--lmd_s', default=1.0, type=float, help='SDL weight')
parser.add_argument('-a', '--alpha', default=0.5, type=float, help='covariance matrix decay')
parser.add_argument('-b', '--batch-size', default=8, type=int)
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
parser.add_argument('--client-selection-method', type=str, default=None, choices=['uniform', 'omp'], help='Client selection method: uniform (uniform probability) or omp (OMP-based adaptive)')
parser.add_argument('--min-selection-prob', type=float, default=None, help='Minimum selection probability')
parser.add_argument('--ema-alpha', type=float, default=None, help='EMA smoothing coefficient (0.0-1.0)')
parser.add_argument('--description', type=str, default=None, help='Experiment description (overrides config file)')

args = parser.parse_args()
with open(args.config, 'r',encoding='utf-8') as f:
    config = copy.deepcopy(yaml.load(f, Loader=yaml.FullLoader))  # 使用深拷贝保护原始配置


# 训练一个SVC 分类器 进行下游分类评估
def eval(transformation,transformation_test,y_train,y_test):
    # 评估模块
    acc_val = -1
    C_best = None
    best_acc =0
    for C in [10 ** i for i in range(-4, 5)]:
        # clf = SVC(C=C, random_state=42) # 原本的交叉验证
        # acc_i = cross_val_score(clf, transformation, y_train, cv=5)
        # if acc_i.mean() > acc_val:
        #     C_best = C
        clf = SVC(C=C, random_state=42)
        clf.fit(transformation, y_train)
        acc = accuracy_score(clf.predict(transformation), y_train)
        if acc > best_acc:
            best_acc, C_best = acc, C
    clf = SVC(C=C_best, random_state=42)
    clf.fit(transformation, y_train)
    # 作图

    #draw_scatter_plot(transformation_test,y_test)

    train_acc = accuracy_score(clf.predict(transformation), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc
# 训练一个SVC 分类器 进行下游分类评估
def evalwithCV(transformation,transformation_test,y_train,y_test):
    # 评估模块
    acc_val = -1
    C_best = None
    for C in [10 ** i for i in range(-4, 5)]:
        clf = SVC(C=C, random_state=42)
        acc_i = cross_val_score(clf, transformation, y_train, cv=5)
        if acc_i.mean() > acc_val:
            C_best = C
    clf = SVC(C=C_best, random_state=42)
    clf.fit(transformation, y_train)
    train_acc = accuracy_score(clf.predict(transformation), y_train)
    test_acc = accuracy_score(clf.predict(transformation_test), y_test)
    return train_acc, test_acc

def train(dataset="", seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True,
           eval_per_x_epochs=10, dist_measure='mix', rank=-1, world_size=-1, resize=0,
           checkpoint=False, task='classification'):
    # init data--------------------------------------------------------------------------------------
    is_ddp = False
    if rank != -1 and world_size != -1:
        is_ddp = True
    if is_ddp:
        # initialize the process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # 保存原始seed值用于客户端选择
    original_seed = seed
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        seed += 1
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    numClient= config['federated']['numClient']
    numRound = config['federated']['numRound']
    numEpoch = config['federated']['numEpoch']
    dirichlet_alpha = config['federated']['dirichlet_alpha']
    # 读取算法类型
    algo = config.get('algo', 'fedcsl')
    # 命令行参数优先，如果没有则使用配置文件中的值
    use_client_selection = args.use_client_selection if args.use_client_selection else config['federated'].get('use_client_selection', False)
    client_selection_ratio = args.client_selection_ratio if args.client_selection_ratio is not None else config['federated'].get('client_selection_ratio', 0.6)
    min_selection_prob = args.min_selection_prob if args.min_selection_prob is not None else config['federated'].get('min_selection_prob', 0.01)
    ema_alpha = args.ema_alpha if args.ema_alpha is not None else config['federated'].get('ema_alpha', 0.3)
    # 客户端选择方法：uniform（均匀概率）或omp（OMP自适应），默认根据算法类型自动选择
    client_selection_method = args.client_selection_method if args.client_selection_method is not None else config['federated'].get('client_selection_method', None)
    # 如果没有指定方法，fedavg默认使用uniform，其他算法默认使用omp
    if client_selection_method is None:
        client_selection_method = 'uniform' if algo == 'fedavg' else 'omp'
    if args.dataset is not None:
        dataset = args.dataset
    else:
        dataset = config['dataset']
    #dataset = args.dataset
    dist_measure = config['model']['params']['dist_measure']
    lr = config['model']['params']['lr']
    batch_size = config['model']['params']['batch_size']
    wd = config['model']['params']['wd']
    ls = config['model']['params']['ls']
    l = config['model']['params']['l']
    beta = config['model']['params'].get('beta', 0.4)  # 默认值0.4，参考configAVG.yml
    gamma = config['model']['params'].get('gamma', 0.5)  # 默认值0.5，参考configAVG.yml

    # 加载shapelets weight权重
    shapelet_weight_X = np.load('./algoutils/shapelet_weight_All.npy')

    print(shapelet_weight_X)


    # 加载数据集
    if dataset == "HAR":
         X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(numClient,dirichlet_alpha,scoreX=shapelet_weight_X,scoreY=None)
    elif dataset != "":
         X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_UEA(dataset, numClient,dirchlet_alpha=dirichlet_alpha,
                                                                                            scoreX=shapelet_weight_X,scoreY=None)
    else:
        print("dataset not found")
        exit(0)
    # ------------------------------------------------------------------------------------------------------------

    n_ts, n_channels, len_ts = X_all.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_all))
    isAllocateMat = False
    isEMA = False
    shapelets_size_and_len = {int(i): 40 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}

    # 命令行参数优先，如果没有则使用配置文件中的值
    if args.description is not None:
        config['description'] = args.description

    #Print logs------------------------------------------------------------------------------------------------------------
    print("shapelet initialized! \n")
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H")+str(config['description'])
    #logTxt = "./result/"+dataset+"l=1e-2lr=0.01epoch3 contrastive.txt"
    logTxt = f"./result/{dataset}/{formatted_date}_l={l}_lr={lr}_epoch{numEpoch}_alphadir{dirichlet_alpha}_{config['description']}.txt"

    f = open(logTxt, mode="a+")
    f.writelines("Details of Training:-----------------------\n")
    f.writelines("dataset: "+dataset+"\n")
    f.writelines("local train epochs:"+str(numEpoch)+"\n")
    f.writelines("round num:"+str(numRound)+"\n")
    f.writelines("batch size:"+str(batch_size)+"\n")
    f.writelines("lr:"+str(lr)+"\n")
    f.writelines("isAllocateMat:"+str(isAllocateMat)+"\n")
    f.writelines("isEMA:"+str(isEMA)+"\n")
    f.writelines("use_client_selection:"+str(use_client_selection)+"\n")
    if use_client_selection:
        f.writelines("client_selection_ratio:"+str(client_selection_ratio)+"\n")
        f.writelines("client_selection_method:"+str(client_selection_method)+"\n")
        f.writelines("min_selection_prob:"+str(min_selection_prob)+"\n")
        f.writelines("ema_alpha:"+str(ema_alpha)+"\n")
    f.writelines("-------------------------------------------"+"\n")
    f.writelines(config['description']+"\n")
    f.writelines("PID:"+str(os.getpid())+"\n")
    f.writelines("PPID:"+str(os.getppid())+"\n")

    yaml_str = yaml.dump(config)

    # 去掉换行符
    yaml_str_no_newline = yaml_str.replace('\n', '')
    f.write(yaml_str_no_newline+"\n")
    f.close()
    # train----------------------------------------------------------------------------------------------------
    w_locals = []
    clientList = []
    server = LearningShapeletsCL(
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
    for idx in range(numClient):
        client = LearningShapeletsCL(
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
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd)
        # optimizer = optim.SGD([
        #     {'params': client.model.parameters()},
        #     {'params': [client.log_vars]}
        # ], lr=lr, weight_decay=wd)
        client.set_optimizer(optimizer)
        clientList.append(client)

    print("All %d clinet initialized! \n" % len(clientList))

    # 先验计算每个客户端的数据分布情况
    # score_list = []
    # for idx,c in enumerate(clientList):
    #     data = torch.tensor(y_fed[idx])
    #     #print(data.shape)
    #     score =map_to_near_one(data,num_classes)
    #     score_list.append(score)
    #     print("clinet %d score: %f", (idx, score))

    C_accu_server = None
    scalers = []
    best_acc = 0
    
    # 初始化客户端选择相关变量
    probs = None
    if use_client_selection:
        probs = [1.0/numClient] * numClient  # 初始化采样概率
        method_name = "均匀采样" if client_selection_method == 'uniform' else "OMP自适应采样"
        print(f"客户端选择已启用，采样比例: {client_selection_ratio}, 选择方法: {method_name}")
        print(f"最低选择概率: {min_selection_prob}, EMA平滑系数: {ema_alpha}")
    else:
        print("客户端选择未启用，所有客户端参与聚合")

    for round in range(numRound):
        avg_loss = 0
        if round == 1:
            one_round_time_start = time.time()
        for idx,c in enumerate(clientList):
            # 比例系数Q
            c.Q = len(y_fed[idx]) / len(y_all)
            if isEMA:
                if round == 0:
                    #分发模型
                    c.model.load_state_dict(server.model.state_dict())
                else:
                    #分发模型EMA
                    weight_scaler = min(scalers[idx] * calculate_divergence(c.model, server.model),1)
                    ema = EMA(weight_scaler)
                    update_moving_average(ema,c.model, server.model)
            else:
                #分发模型
                #c.model.load_state_dict(server.model.state_dict())
                # 不直接加载全局模型
                if round!=0:
                    c.Global_Model =copy.deepcopy(server.model)
            #分发矩阵
            if isAllocateMat and round != 0:
                c.C_accu_Server = C_accu_server


            # print(len(X_fed[idx]))
            # print(X_fed[idx][0].shape)
            # print("以下来自第{idx}个客户端")
            
            # 检查客户端数据是否为空
            if len(X_fed[idx]) == 0 or len(y_fed[idx]) == 0:
                print(f"警告：客户端 {idx} 的数据为空，跳过训练")
                # 如果数据为空，使用当前模型状态作为本地模型
                if round == 0:
                    w_locals.append(c.model.state_dict())
                else:
                    w_locals[idx] = c.model.state_dict()
                continue
            
            losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size, epoch_idx=-1,lr=lr)
            # 检查losses是否为空或包含NaN
            if len(losses) == 0:
                print(f"警告：客户端 {idx} 的训练损失为空，跳过损失计算")
                loss_all = 0.0
            else:
                loss_all = np.mean([loss[0] for loss in losses])
                # 检查loss_all是否为NaN
                if np.isnan(loss_all) or np.isinf(loss_all):
                    print(f"警告：客户端 {idx} 的训练损失为 NaN/Inf，使用0.0")
                    loss_all = 0.0
            # loss_align = np.mean([loss[2] for loss in losses])
            # loss_sdl = np.mean([loss[3] for loss in losses])
            avg_loss+=(loss_all) * len(y_fed[idx]) / len(X_all)



            if round==0 :
                w_locals.append(c.model.state_dict())
            else:
                w_locals[idx] = c.model.state_dict()

        scores = []
        #before aggregation
        for idx,c in enumerate(clientList):
            # 检查数据是否为空
            if len(X_fed[idx]) == 0:
                print(f"警告：客户端 {idx} 的数据为空，跳过预测，使用默认分数")
                scores.append(1.0)  # 使用默认分数
                continue
            features = c.predict(X_fed[idx])
            # 检查预测结果是否为空（处理一维或二维数组）
            if features.size == 0 or (len(features.shape) > 0 and features.shape[0] == 0):
                print(f"警告：客户端 {idx} 的预测结果为空，使用默认分数")
                scores.append(1.0)  # 使用默认分数
                continue
            print(features.shape)
            scores.append(cal_score(features))
        scores = normalize_to_near_one(scores)
        print(scores)

        # 分布打分
        if config['ablation']['UseDistribution']==False:
            scores = [1] * numClient  # 修改为动态长度
        # scores = [1,1,1]
        
        # 根据配置决定是否使用客户端选择
        if use_client_selection:
            # 启用客户端选择
            sample_nums = int(numClient * client_selection_ratio)  # 采样数
            
            # 根据选择方法进行客户端选择
            if client_selection_method == 'uniform':
                # 均匀采样：所有客户端概率相等
                probs = [1.0/numClient] * numClient
                if round == 0:
                    # 第一轮全选所有客户端
                    select_mask = [1.0] * numClient  # 使用浮点数
                    print(f"第一轮：全选所有客户端（均匀采样模式）")
                else:
                    # 从第二轮开始按均匀概率选择
                    print(f"均匀采样模式，概率: {probs}")
                    select_mask = sample_clients_mask_by_probability(probs, sample_nums, seed=original_seed)
                    # 转换为浮点数类型
                    select_mask = [float(x) for x in select_mask]
                    print(f"客户端选择掩码: {select_mask}")
            else:
                # OMP自适应采样
                if round == 0:
                    # 第一轮全选所有客户端
                    select_mask = [1.0] * numClient  # 使用浮点数
                    probs = [1.0/numClient] * numClient
                    print(f"第一轮：全选所有客户端（OMP自适应采样模式）")
                else:
                    # 从第二轮开始按采样概率选择
                    print(f"本轮采样概率阵: {probs}")
                    select_mask = sample_clients_mask_by_probability(probs, sample_nums, seed=original_seed)
                    # 转换为浮点数类型
                    select_mask = [float(x) for x in select_mask]
                    print(f"客户端选择掩码: {select_mask}")
            
            # 使用select_mask与scores结合进行聚合：如果客户端未被选中，则score为0
            combined_scores = [scores[i] * select_mask[i] for i in range(numClient)]
            w_global = fedavg(w_locals, y_fed, combined_scores)
            server.model.load_state_dict(w_global)
            
            # 更新概率：只有使用OMP自适应采样时才更新
            if client_selection_method == 'omp':
                sparse_vec = omp_from_state_dicts(w_locals, w_global, sample_nums)
                probs = get_sampling_probs_from_omp(
                    sparse_vec,
                    prev_probs=probs,
                    selection_mask=select_mask,
                    min_selection_prob=min_selection_prob,
                    ema_alpha=ema_alpha
                )
                print(f"稀疏向量: {sparse_vec}")
                print(f"更新后概率: {probs}")
            else:
                # 均匀采样模式保持均匀概率，不更新
                print(f"均匀采样模式：保持均匀采样概率，不更新")
        else:
            # 不使用客户端选择，所有客户端都参与聚合
            print("所有客户端参与聚合（客户端选择未启用）")
            # 使用原有的scores进行聚合
            w_global = fedavg(w_locals, y_fed, scores)
            server.model.load_state_dict(w_global)


        # 下游分类器
        transformation = server.transform(X_all, result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_test = server.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
        scaler = RobustScaler()
        transformation = scaler.fit_transform(transformation)
        transformation_test = scaler.transform(transformation_test)
        train_acc, test_acc =  eval(transformation,transformation_test,y_train=y_all,y_test=y_test) #验证训练集是全局训练集， 测试集为全集测试集
        if best_acc< test_acc:
            best_acc = test_acc
            best_round = round
            best_model = copy.deepcopy(server.model)
        print('Classification:', train_acc, test_acc, round)
        #print("round %d Server %d trained.\n"%(round,idx))

        f = open(logTxt, mode="a+")
        # 检查avg_loss是否为NaN，如果是则显示为字符串
        avg_loss_str = str(avg_loss) if not (np.isnan(avg_loss) or np.isinf(avg_loss)) else "nan"
        f.writelines("dataset: "+dataset+"round:"+str(round)+" server aggregation "+" testACC:"+str(test_acc)+" trainACC:"+str(train_acc)+" avg_loss:"+avg_loss_str+"\n")
        f.close()

        # # 太大的模型早退
        # if round == 1 and dataset!="HAR":
        #     time_round = time.time() - one_round_time_start
        #     if time_round*numRound > 40000:
        #         return

    print("best round is %d, acc is %f"%(best_round,best_acc))
    # 画一下散点图
    # vectors = server.predict(X_all)
    # draw_scatter_plot(vectors,y_all)


    #torch.save(best_model.state_dict(), f'./checkpoint/{dataset}/{formatted_date}_{dataset}_model.pt')
    save_model(best_model, dataset, formatted_date)


    return
def save_model(model, dataset, formatted_date):
    # 定义保存路径
    checkpoint_dir = f'./checkpoint/{dataset}'
    model_path = f'{checkpoint_dir}/{formatted_date}_{dataset}_model.pt'

    # 检查并创建目录（如果不存在）
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
if __name__ == '__main__':

    train(dataset=args.dataset, seed=args.seed)


