import random
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import wandb
from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder

from ..commons.optim import cosine_learning_rates
from ..commons.evaluate import knn_evaluate
from .client import FedEmaClient
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler


class FedEmaServer:
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        autoscaler_tau: float = 0.7,
        checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedEmaClient] = []
        self.online_encoder = encoder
        self.online_projector = projector
        self.predictor = predictor
        # self.test_data = test_data
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.autoscaler_tau = autoscaler_tau
        self.checkpoint_path = checkpoint_path
        
        self.test_data = test_data
        self.test_labels = test_labels

    # server.fit 和 client.fit 都得修改，尤其是 client.fit
    def fit(self):
        # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        print(f'knn without training: {self.knn_test()}')
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            responses = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                response = client.fit( #传入global的encoder和projector
                    self.online_encoder,
                    self.online_projector,
                    self.predictor,
                    lr=None,
                    current_round=self.current_round,
                )
                responses.append(response)
            self.online_encoder.load_state_dict(model_average([response['online_encoder'] for response in responses], client_weights))
            self.online_projector.load_state_dict(model_average([response['online_projector'] for response in responses], client_weights))
            self.predictor.load_state_dict(model_average([response['predictor'] for response in responses], client_weights))
            train_loss = np.mean([response['train_loss'] for response in responses])
            divergence = np.mean([response['divergence'] for response in responses])
            for client in selected_clients:
                if np.isnan(client.scaler):
                    client.calculate_scaler(self.autoscaler_tau, self.online_encoder)
                    
            # evaluate
            server = self.online_encoder
            transformation_test = self.transform(self.test_data, result_type='numpy', normalize=True, batch_size=256)
            scaler = RobustScaler()
            transformation_test = scaler.transform(transformation_test)
            _, acc =  self.eval(transformation_test,y_test=self.test_labels)
            
            # acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, svc acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}, divergence: {divergence:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'divergence': divergence,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            }, step=self.current_round)
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)
                
    #svc evaluate
    def eval(self,transformation_test,y_test):
        # 初始化最佳 C 值
        C_best = None
        acc_val = -1
        # 网格搜索最佳 C 值
        for C in [10 ** i for i in range(-4, 5)]:
            clf = SVC(C=C, random_state=42)
            clf.fit(transformation_test, y_test)
            acc_i = accuracy_score(clf.predict(transformation_test), y_test)
            if acc_i > acc_val:
                acc_val = acc_i
                C_best = C
        # 使用最佳 C 值训练最终模型
        clf = SVC(C=C_best, random_state=42)
        clf.fit(transformation_test, y_test)
        # 计算测试数据上的准确率
        test_acc = accuracy_score(clf.predict(transformation_test), y_test)
        return test_acc
    
    def transform(self, X, *, batch_size=512, result_type='tensor', normalize=False):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        
        self.online_encoder.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shapelet_transform = []
        for (x, ) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
            #shapelet_transform = self.model.transform(X)
                shapelet_transform.append(self.online_encoder(x, optimize=None,isEncoder=True).cpu())
        shapelet_transform = torch.cat(shapelet_transform, 0)
        if normalize:
            shapelet_transform = nn.functional.normalize(shapelet_transform, dim=1)
        if result_type == 'tensor':
            return shapelet_transform
        return shapelet_transform.detach().numpy()
        
    #knn evaluate
    def knn_test(self) -> float:
        train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.online_encoder,
            train_set=train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round'] + 1
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)