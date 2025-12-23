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
UEA_path = './Multivariate_ts'
UEA_datasets = os.listdir(UEA_path)
UEA_datasets.sort()




parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='UEA dataset name')
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


# 训练一个SVC 分类器 进行下游分类评估
def eval(transformation,transformation_test,y_train,y_test):
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


def train(dataset="Epilepsy", seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True,
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
              
        
    numClient= 3
    numRound = 1000
    numEpoch = 3 
    alpha_dir = 0.1
    # UEA 数据集测试   
    # N x D x T
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(UEA_path, dataset)
    #print(X_train.shape) #(40, 6, 100)npArray

    # 归一化训练集、测试集
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)

    #allocate sample-------------------------------------------------------------------------------------------
    DATA_PATH = "Multivariate_ts/"
    X = load_from_tsfile_to_dataframe(
        os.path.join(DATA_PATH, f"{dataset}/{dataset}_TRAIN.ts"), return_separate_X_and_y=False,
        replace_missing_vals_with='NaN'
    )
    X2 = load_from_tsfile_to_dataframe(
        os.path.join(DATA_PATH, f"{dataset}/{dataset}_TEST.ts"), return_separate_X_and_y=False,
        replace_missing_vals_with='NaN'
    )
    print(len(X))
    print(X.info())
    X_test = X2.iloc[:, :-1]
    y_test = X2.iloc[:, -1].to_numpy()
    lengths = check_series_lengths(X_test)
    print(lengths)
    X_test = reshape_dataframe_to_tensor(X_test)
    X_test = z_normalize(X_test)
    #(40,7)
    print(X.info())
    X_all = X.iloc[:, :-1]  # 特征转换为DataFrame
    y_all = X.iloc[:, -1].to_numpy()  # 标签转换为numpy数组
    X_all = reshape_dataframe_to_tensor(X_all)
    X_all = z_normalize(X_all)
    # print(len(y_all)) #检查sktime 读取是否正确
    # print(len(y_train))
    # #exit(0)
    if dataset =="HAR":
        X = torch.load("./HAR/train.pt")
        X_all = X["samples"].float()
        y_all = X["labels"].float()
        X2= torch.load("./HAR/test.pt")
        X_test= X2["samples"].float()
        y_test = X2["labels"].float()
        
    #X_fed,y_fed = federated_data_split_XD_non_iid(X_all,y_all,num_parties=numClient,ratio=[1,2,7]) # 分出来得到二维dataframe ， 每一个表项 里面有N列，T个时间戳，由逗号分隔 , 需要拿出来作为新的维度，譬如每个客户端拥有的dataframe（12 ，4）需要变成（12，4，100）的nparray
    #non iid UEA split, input is X_train,y_train
    # output is X_fed,y_fed
    X_fed = []
    y_fed = []
    print(y_train)
    #exit(0)
    client_idc = dirichlet_split_noniid(train_labels=y_train,alpha=alpha_dir,n_clients=numClient)
    print(client_idc[0])
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_train[client_idc[i][j]])
            tmpLabel.append(y_train[client_idc[i][j]])
        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
    print(y_fed[0])
    print(y_fed[1])
    print(y_fed[2])
    # exit()
    #X_fed,y_fed = federated_data_split_XD(X,num_parties=numClient) # 分出来得到二维dataframe ， 每一个表项 里面有N列，T个时间戳，由逗号分隔 , 需要拿出来作为新的维度，譬如每个客户端拥有的dataframe（12 ，4）需要变成（12，4，100）的nparray
    # print("len Xfed :%d "%len(X_fed[0][0]))
    # print(X_fed[0].shape)
    # for i in range(len(X_fed)):
    #     X_fed[i] = reshape_dataframe_to_tensor(X_fed[i])
    #     X_fed[i] = z_normalize(X_fed[i])


    # ------------------------------------------------------------------------------------------------------------
    dist_measure = "cosine"
    lr = 0.01
    batch_size = 8
    wd = 0
    ls = 1
    l = 1e-2
    momentum = 0.9
    n_ts, n_channels, len_ts = X_all.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_all))
    isAllocateMat = False
    isEMA = False
    shapelets_size_and_len = {int(i): 40 for i in np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}
    print("shapelet initialized! \n")
    #logTxt = "./result/"+dataset+"l=1e-2lr=0.01epoch3 contrastive.txt"
    logTxt = f"./result/{dataset}_l={l}_lr={lr}_epoch{numEpoch}_alphadir{alpha_dir}_contrastive_distil.txt"
    
    f = open(logTxt, mode="a+")
    f.writelines("Details of Training:-----------------------\n")
    f.writelines("dataset: "+dataset+"\n")
    f.writelines("local train epochs:"+str(numEpoch)+"\n")
    f.writelines("round num:"+str(numRound)+"\n")
    f.writelines("batch size:"+str(batch_size)+"\n")
    f.writelines("lr:"+str(lr)+"\n")
    f.writelines("isAllocateMat:"+str(isAllocateMat)+"\n")
    f.writelines("isEMA:"+str(isEMA)+"\n")
    f.writelines("-------------------------------------------"+"\n")
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
        seed=seed

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
            seed=seed        
        )
        optimizer = optim.SGD(client.model.parameters(), lr=lr, weight_decay=wd)
        client.set_optimizer(optimizer)
        clientList.append(client)
    
    print("All %d clinet initialized! \n" % len(clientList))

    C_accu_server = None
    scalers = []
    for round in range(numRound):    
        avg_loss = 0
        for idx,c in enumerate(clientList):
            # 比例系数Q
            c.Q = len(y_fed[idx]) / len(X)
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
            #for epoch in range(numEpoch):
            # print(X_fed[idx][0])
            # exit(0)
            losses = c.train(X_fed[idx], epochs=numEpoch, batch_size=batch_size, epoch_idx=-1,lr=lr)
            loss_all = np.mean([loss[0] for loss in losses])
            # loss_align = np.mean([loss[2] for loss in losses])
            # loss_sdl = np.mean([loss[3] for loss in losses])
            avg_loss+=(loss_all) * len(y_fed[idx]) / len(X)
            
            

            if round==0 :
                w_locals.append(c.model.state_dict())
            else:
                w_locals[idx] = c.model.state_dict()


        # aggregation: model
        w_global = fedavg(w_locals,y_fed)
        server.model.load_state_dict(w_global) 
        #print(C_accu_server)
        if isAllocateMat:
            # aggregation: covariance matrix
            C_accu_server = [torch.zeros_like(tensor) for tensor in clientList[0].C_accu_trans]
            for idx, c in enumerate(clientList):
                # 按比例将每个 client 的 C_accu_trans 累加到 C_accu_server 的对应位置
                weight = len(y_fed[idx]) / len(X)
                for i in range(len(C_accu_server)):
                    C_accu_server[i] += c.C_accu_trans[i] * weight
        #print(C_accu_server)
        if isEMA:
            auto_scaler = 0.7
            # 分配lambda 计算每个client的scaler            
            for idx, c in enumerate(clientList):
                # 计算lambda
                weight_scaler = auto_scaler / calculate_divergence(c.model, server.model)
                scalers.append(weight_scaler)
        
        transformation = server.transform(X_all, result_type='numpy', normalize=True, batch_size=batch_size)
        transformation_test = server.transform(X_test, result_type='numpy', normalize=True, batch_size=batch_size)
        scaler = RobustScaler()
        transformation = scaler.fit_transform(transformation)
        transformation_test = scaler.transform(transformation_test)
        train_acc, test_acc =  eval(transformation,transformation_test,y_train=y_all,y_test=y_test) #验证训练集是全局训练集， 测试集为全集测试集
        #print('Classification:', train_acc, test_acc, round)
        #print("round %d Server %d trained.\n"%(round,idx))
        
        f = open(logTxt, mode="a+")
        f.writelines("dataset: "+dataset+"round:"+str(round)+" server aggregation "+" acc:"+str(test_acc)+" avg_loss:"+str(avg_loss)+"\n")
        f.close()

        



    return

if __name__ == '__main__':
    args = parser.parse_args()
    train(dataset=args.dataset)


