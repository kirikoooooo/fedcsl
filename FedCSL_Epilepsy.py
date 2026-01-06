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


# 没毛病
def eval_TSTCC(transformation_train, transformation_test, transformation_val, y_train, y_test, y_val):
    # 如果你的验证集和测试集是从 .pt 文件加载的，你可能需要在这里加载它们
    # 例如：
    # transformation_val = torch.load('path/to/your/val_features.pt')
    # y_val = torch.load('path/to/your/val_labels.pt')
    # transformation_test = torch.load('path/to/your/test_features.pt')
    # y_test = torch.load('path/to/your/test_labels.pt')

    acc_val_best = -1
    C_best = None

    # 使用提供的验证集来选择最佳的 C 值
    for C in [10 ** i for i in range(-4, 5)]:
        clf = SVC(C=C, random_state=42)
        # 在训练集上训练模型
        clf.fit(transformation_train, y_train)
        # 在验证集上评估模型
        acc_i = accuracy_score(clf.predict(transformation_val), y_val)

        if acc_i > acc_val_best:
            acc_val_best = acc_i
            C_best = C

    # 使用最佳 C 值在全部训练数据上重新训练最终模型
    clf = SVC(C=C_best, random_state=42)
    clf.fit(transformation_train, y_train)

    # 在训练集和测试集上进行最终评估
    train_acc = accuracy_score(clf.predict(transformation_train), y_train)
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
    use_client_selection = config['federated'].get('use_client_selection', False)  # 默认不启用
    client_selection_ratio = config['federated'].get('client_selection_ratio', 0.6)  # 默认采样60%
    min_selection_prob = config['federated'].get('min_selection_prob', 0.01)  # 最低选择概率，默认1%
    ema_alpha = config['federated'].get('ema_alpha', 0.3)  # 指数移动平均平滑系数，默认0.3
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
    beta = config['model']['params']['beta']

    # 加载shapelets weight权重
    shapelet_weight_X = np.load('./algoutils/shapelet_weight_All.npy')

    print(shapelet_weight_X)


    # 加载数据集
    if dataset == "HAR":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_HAR(numClient,dirichlet_alpha,scoreX=shapelet_weight_X,scoreY=None)
        val_data = torch.load("./HAR/val.pt")
        X_val = val_data["samples"].float()
        y_val = val_data["labels"].int()

    elif dataset == "Epilepsy-TSTCC":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_Epilepsy(numClient,dirichlet_alpha,scoreX=shapelet_weight_X,scoreY=None)
        val_data = torch.load("./Epilepsy/val.pt")
        X_val = val_data["samples"].float()

        y_val = val_data["labels"].int()
    elif dataset == "SleepEDF":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_SleepEDF(numClient,dirichlet_alpha,scoreX=shapelet_weight_X,scoreY=None)
        val_data= torch.load("./sleepEDF/val.pt")
        X_val= val_data["samples"].float()
        y_val = val_data["labels"].int()
        print(X_all.shape)
        print(X_val.shape)

    elif dataset == "FD-A":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_FDA(numClient,dirichlet_alpha,scoreX=shapelet_weight_X,scoreY=None)
        val_data= torch.load("./FD-A/val.pt")
        X_val= val_data["samples"].float()
        y_val = val_data["labels"].int()
        X_val = X_val.unsqueeze(1)

        # print(X_all.shape)
        # print(X_val.shape)
        # exit(0)
    elif dataset != "":
        X_all, y_all, X_test, y_test, X_fed, y_fed = LoadDataset_UEA(dataset, numClient,dirchlet_alpha=dirichlet_alpha,
                                                                                            scoreX=shapelet_weight_X,scoreY=None)
    else:
        print("dataset not found")
        exit(0)



    print(X_all.shape)
    X_train_linear_1per,y_train_linear_1per = sample_with_min_per_class(X_all,y_all)
    # ------------------------------------------------------------------------------------------------------------
    print(X_train_linear_1per.shape)
    print("1% 样本数量:", len(y_train_linear_1per))
    print("各类别出现次数:\n", np.bincount(y_train_linear_1per))
    # exit(0)

    n_ts, n_channels, len_ts = X_all.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_all))
    isAllocateMat = False
    isEMA = False
    shapelets_size_and_len = {int(i): 40 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}

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
        beta = beta,
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
            beta = beta
        )
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd)
        # optimizer = optim.SGD([
        #     {'params': client.model.parameters()},
        #     {'params': [client.log_vars]}
        # ], lr=lr, weight_decay=wd)
        client.set_optimizer(optimizer)
        clientList.append(client)

    print("All %d clinet initialized! \n" % len(clientList))

    #先验计算每个客户端的数据分布情况
    score_list = []
    for idx,c in enumerate(clientList):
        data = torch.tensor(y_fed[idx])
        #print(data.shape)
        score =map_to_near_one(data,num_classes)
        score_list.append(score)
        print("clinet %d score: %f", (idx, score))

    C_accu_server = None
    scalers = []
    best_acc = 0
    
    # 初始化客户端选择相关变量
    probs = None
    if use_client_selection:
        probs = [1.0/numClient] * numClient  # 初始化采样概率
        print(f"客户端选择已启用，采样比例: {client_selection_ratio}")
        print(f"最低选择概率: {min_selection_prob}, EMA平滑系数: {ema_alpha}")
    else:
        print("客户端选择未启用，所有客户端参与聚合")

    for round in range(numRound):
        avg_loss = 0
        client_select_scores = []
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
                if round != 0:
                    c.model.load_state_dict(server.model.state_dict())
                    # 不直接加载全局模型
                    c.Global_Model =copy.deepcopy(server.model)
            #分发矩阵
            if isAllocateMat and round != 0:
                c.C_accu_Server = C_accu_server


            # print(len(X_fed[idx]))
            # print(X_fed[idx][0].shape)
            # print("以下来自第{idx}个客户端")
            losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size, epoch_idx=-1,lr=lr)
            loss_all = np.mean([loss[0] for loss in losses])
            # loss_align = np.mean([loss[2] for loss in losses])
            # loss_sdl = np.mean([loss[3] for loss in losses])
            avg_loss+=(loss_all) * len(y_fed[idx]) / len(X_all)



            if round==0 :
                w_locals.append(c.model.state_dict())
            else:
                w_locals[idx] = c.model.state_dict()

            transformation_local = c.transform(X_fed[idx], result_type='numpy', normalize=True, batch_size=batch_size)
            # mean_vector = np.mean(np.abs(transformation), axis=0) # 沿着行方向（纵向）求均值
            # std_vector = np.std(np.abs(transformation), axis=0)    # 同样沿着行方向求标准差
            # #mean_vector = np.mean(transformation_local, axis=0) # 沿着行方向（纵向）求均值
            # #std_vector = np.std(transformation_local, axis=0)    # 同样沿着行方向求标准差
            # print(mean_vector)
            # print(std_vector)
            gap_score_vec = compute_gap_scores_from_transformation(transformation_local)
            sum_score = np.sum(gap_score_vec)
            client_select_scores.append(sum_score)
            #print(gap_score_vec)
            print("ROUND:",round,"SUM",sum_score)





        scores = [1] * numClient
        #aggregation: model
        # if config['algo'] == 'fedcsl' or config['algo']=='fedprox' or config['algo'] == 'fedavg':
        #     scores = sparse_mask
        #     print("CLIENT SELECTION:",sparse_mask)

        # 根据配置决定是否使用客户端选择
        if use_client_selection:
            # 启用客户端选择
            sample_nums = int(numClient * client_selection_ratio)  # 采样数
            if round == 0:
                # 第一轮全选所有客户端
                select_mask = [1] * numClient
                probs = [1.0/numClient] * numClient
                print(f"第一轮：全选所有客户端")
            else:
                # 从第二轮开始按采样概率选择
                print(f"本轮采样概率阵: {probs}")
                select_mask = sample_clients_mask_by_probability(probs, sample_nums)
                print(f"客户端选择掩码: {select_mask}")
            
            # 使用select_mask进行聚合
            w_global = fedavg(w_locals, y_fed, select_mask)
            server.model.load_state_dict(w_global)
            
            # omp计算重新分配概率（在聚合后执行，第一轮也需要更新概率）
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
            # 不使用客户端选择，所有客户端都参与聚合
            select_mask = [1] * numClient
            print("所有客户端参与聚合（客户端选择未启用）")
            # 使用select_mask进行聚合
            w_global = fedavg(w_locals, y_fed, select_mask)
            server.model.load_state_dict(w_global)

        # 下游分类器，这里normalize好像有问题
        transformation = server.transform(X_all, result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_test = server.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_val = server.transform(X_val,result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_X_train_linear_1per = server.transform(X_train_linear_1per,result_type='numpy', normalize=True, batch_size=batch_size)
        scaler = RobustScaler()
        transformation = scaler.fit_transform(transformation)
        transformation_test = scaler.transform(transformation_test)
        transformation_val =scaler.transform(transformation_val)
        transformation_X_train_linear_1per = scaler.transform(transformation_X_train_linear_1per)

        #train_acc, test_acc =  eval(transformation,transformation_test,y_train=y_all,y_test=y_test) #验证训练集是全局训练集， 测试集为全集测试集
        # train_acc, test_acc = eval_TSTCC(
        #     transformation_train=transformation_X_train_linear_1per, # linear 训练1% 数据
        #     transformation_test=transformation_test,
        #     transformation_val=transformation_val,
        #     y_train= y_train_linear_1per,
        #     y_test=y_test,
        #     y_val = y_val,
        # )
        # 根据对x_train的transformation 进行客户端选择, gapscore 原文使用的是欧式距离
        #print(type(np.abs(transformation)))


        train_acc, test_acc = eval_TSTCC(
            transformation_train=transformation, # linear 训练1% 数据
            transformation_test=transformation_test,
            transformation_val=transformation_val,
            y_train= y_all,
            y_test=y_test,
            y_val = y_val,
        )
        if best_acc< test_acc:
            best_acc = test_acc
            best_round = round
            best_model = copy.deepcopy(server.model)
        print('Classification:', train_acc, test_acc, round)
        #print("round %d Server %d trained.\n"%(round,idx))

        f = open(logTxt, mode="a+")
        f.writelines("dataset: "+dataset+"round:"+str(round)+" server aggregation "+" testACC:"+str(test_acc)+" trainACC:"+str(train_acc)+" avg_loss:"+str(avg_loss)+"\n")
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

    train(dataset=args.dataset)


