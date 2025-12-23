from utils import *
from fedutil import *
import os
from utils import *
from sktime.datasets import load_from_tsfile_to_dataframe,load_UCR_UEA_dataset

def LoadDataset_SleepEDF(numClient,dirchlet_alpha=0.1,scoreX=None,scoreY=None):
    # 加载HAR数据集
    X = torch.load("./sleepEDF/train.pt")
    X_all = X["samples"].float()
    y_all = X["labels"].int()
    X2= torch.load("./sleepEDF/test.pt")
    X_test= X2["samples"].float()
    y_test = X2["labels"].int()

    X_fed = []
    y_fed = []
    score_fedX = []
    #score_fedY = []
    X_all =X_all.numpy()
    y_all = y_all.numpy()

    print(y_all)

    client_idc = dirichlet_split_noniid(train_labels=y_all,alpha=dirchlet_alpha,n_clients=numClient)
    print(client_idc[0])
    # client_idc 表示每个客户端的样本索引，待后续使用score_fed[client_idc[i][j]]来获取每个客户端的样本weight
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        #tmpScoreX = []
        #tmpScoreY = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_all[client_idc[i][j]])
            tmpLabel.append(y_all[client_idc[i][j]])
            #tmpScoreX.append(scoreX[client_idc[i][j]])
            #tmpScoreY.append(scoreY[client_idc[i][j]])
        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
        #score_fedX.append(tmpScoreX)
        #score_fedY.append(tmpScoreY)
    print(y_fed[0])
    print(y_fed[1])
    print(y_fed[2])
    return X_all, y_all, X_test, y_test, X_fed, y_fed #score_fedX #,score_fedY

def LoadDataset_FDA(numClient,dirchlet_alpha=0.1,scoreX=None,scoreY=None):
    # 加载HAR数据集
    X = torch.load("./FD-A/train.pt")
    X_all = X["samples"].float()
    y_all = X["labels"].int()

    X_all = X_all[:, np.newaxis, :]
    X2= torch.load("./FD-A/test.pt")
    X_test= X2["samples"].float()
    y_test = X2["labels"].int()

    X_test = X_test[:, np.newaxis, :]
    X_fed = []
    y_fed = []
    score_fedX = []
    #score_fedY = []
    X_all =X_all.numpy()
    y_all = y_all.numpy()

    print(y_all)

    client_idc = dirichlet_split_noniid(train_labels=y_all,alpha=dirchlet_alpha,n_clients=numClient)
    print(client_idc[0])
    # client_idc 表示每个客户端的样本索引，待后续使用score_fed[client_idc[i][j]]来获取每个客户端的样本weight
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        #tmpScoreX = []
        #tmpScoreY = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_all[client_idc[i][j]])
            tmpLabel.append(y_all[client_idc[i][j]])
            #tmpScoreX.append(scoreX[client_idc[i][j]])
            #tmpScoreY.append(scoreY[client_idc[i][j]])
        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
        #score_fedX.append(tmpScoreX)
        #score_fedY.append(tmpScoreY)
    print(y_fed[0])
    print(y_fed[1])
    print(y_fed[2])
    return X_all, y_all, X_test, y_test, X_fed, y_fed #score_fedX #,score_fedY

def LoadDataset_Epilepsy(numClient,dirchlet_alpha=0.1,scoreX=None,scoreY=None):
    # 加载HAR数据集
    X = torch.load("./Epilepsy/train.pt")
    X_all = X["samples"].float()
    y_all = X["labels"].int()
    X2= torch.load("./Epilepsy/test.pt")
    X_test= X2["samples"].float()
    y_test = X2["labels"].int()

    X_fed = []
    y_fed = []
    score_fedX = []
    #score_fedY = []
    X_all =X_all.numpy()
    y_all = y_all.numpy()

    print(y_all)

    client_idc = dirichlet_split_noniid(train_labels=y_all,alpha=dirchlet_alpha,n_clients=numClient)
    print(client_idc[0])
    # client_idc 表示每个客户端的样本索引，待后续使用score_fed[client_idc[i][j]]来获取每个客户端的样本weight
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        #tmpScoreX = []
        #tmpScoreY = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_all[client_idc[i][j]])
            tmpLabel.append(y_all[client_idc[i][j]])
            #tmpScoreX.append(scoreX[client_idc[i][j]])
            #tmpScoreY.append(scoreY[client_idc[i][j]])
        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
        #score_fedX.append(tmpScoreX)
        #score_fedY.append(tmpScoreY)

    for i in range(numClient):
        print(y_fed[i])
    return X_all, y_all, X_test, y_test, X_fed, y_fed #score_fedX #,score_fedY


def LoadDataset_HAR(numClient,dirchlet_alpha=0.1,scoreX=None,scoreY=None):
    # 加载HAR数据集
    X = torch.load("./HAR/train.pt")
    X_all = X["samples"].float()
    y_all = X["labels"].int()
    X2= torch.load("./HAR/test.pt")
    X_test= X2["samples"].float()
    y_test = X2["labels"].int()

    X_fed = []
    y_fed = []
    score_fedX = []
    #score_fedY = []
    X_all =X_all.numpy()
    y_all = y_all.numpy()

    print(y_all)

    client_idc = dirichlet_split_noniid(train_labels=y_all,alpha=dirchlet_alpha,n_clients=numClient)
    print(client_idc[0])
    # client_idc 表示每个客户端的样本索引，待后续使用score_fed[client_idc[i][j]]来获取每个客户端的样本weight
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        #tmpScoreX = []
        #tmpScoreY = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_all[client_idc[i][j]])
            tmpLabel.append(y_all[client_idc[i][j]])
            #tmpScoreX.append(scoreX[client_idc[i][j]])
            #tmpScoreY.append(scoreY[client_idc[i][j]])
        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
        #score_fedX.append(tmpScoreX)
        #score_fedY.append(tmpScoreY)
    print(y_fed[0])
    print(y_fed[1])
    print(y_fed[2])
    return X_all, y_all, X_test, y_test, X_fed, y_fed #score_fedX #,score_fedY


def LoadDataset_UEA(dataset, numClient,dirchlet_alpha=0.1,scoreX=None,scoreY=None):
    # 加载UEA数据集
    # UEA 数据集测试
    # N x D x T
    if dataset == "":
        print("No dataset is given!\n")
    UEA_path = './Multivariate_ts'
    UEA_datasets = os.listdir(UEA_path)
    UEA_datasets.sort()

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

    X_fed = []
    y_fed = []
    score_fedX = []
    score_fedY = []

    print(y_train)
    #exit(0)
    client_idc = dirichlet_split_noniid(train_labels=y_train,alpha=dirchlet_alpha,n_clients=numClient)
    print(client_idc[0])
    for i in range(numClient):
        tmpData = []
        tmpLabel = []
        tmpScoreX = []
        tmpScoreY = []
        for j in range(len(client_idc[i])):
            tmpData.append(X_train[client_idc[i][j]])
            tmpLabel.append(y_train[client_idc[i][j]])
            #tmpScoreX.append(scoreX[client_idc[i][j]])
            #tmpScoreY.append(scoreY[client_idc[i][j]])

        X_fed.append(tmpData)
        y_fed.append(tmpLabel)
        #score_fedX.append(tmpScoreX)
        #score_fedY.append(tmpScoreY)
    print(y_fed[0])
    print(y_fed[1])
    print(y_fed[2])


    return X_all, y_all, X_test, y_test, X_fed, y_fed#, score_fedX#, score_fedY