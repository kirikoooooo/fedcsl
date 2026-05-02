import os
import warnings

import numpy as np
import torch

import tsaug
from torch import nn,optim

from tqdm import tqdm

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances

import copy

from fedutil import *
from utils import *
import yaml

# 对比学习/蒸馏损失模块（见 algo/contrastive/）。每项对应一个消融开关。
from algo.contrastive import (
    local_infonce_loss,
    joint_contrastive_loss,
    joint_distill_loss,
    scale_contrastive_loss,
    scale_distill_loss,
)


class LearningShapeletsCL:
    """
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        The keys are the length of the shapelets and the values the number of shapelets of
        a given length, e.g. {40: 4, 80: 4} learns 4 shapelets of length 40 and 4 shapelets of
        length 80.
    loss_func : torch.nn
        the loss function
    in_channels : int
        the number of input channels of the dataset
    num_classes : int
        the number of output classes.
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len,
                 loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True, l3=0.0, l4=0.0, T=0.1, alpha=0.0, is_ddp=False, checkpoint=False, seed=None,
                 shapelet_weight=None,configDir=None,config=None,beta=0.4, device=None):
        self.is_ddp = is_ddp
        self.checkpoint = checkpoint
        self.seed = seed
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if to_cuda and torch.cuda.is_available() else "cpu"
        )
        self.to_cuda = self.device.type == "cuda"
        if dist_measure == 'mix':
            self.model = LearningShapeletsModelMixDistances(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=self.to_cuda, checkpoint=checkpoint, device=self.device)
        else:
            self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=self.to_cuda, checkpoint=checkpoint,shapelet_weight=shapelet_weight,
                                            device=self.device)

            # self.model = Attention(shapelets_size_and_len=shapelets_size_and_len,
            #                                 in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
            #                                 to_cuda=to_cuda, checkpoint=checkpoint)

            # self.model = MMoE(shapelets_size_and_len=shapelets_size_and_len,expert_dim=32,n_expert=8,n_task=8,
            #                   in_channels = in_channels, num_classes=num_classes)

        self.model.to(self.device)

        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
        self.scheduler = None



        self.l3 = l3
        self.l4 = l4
        self.alpha = alpha
        self.beta = beta
        self.use_regularizer = False

        self.log_vars = nn.Parameter(
            torch.zeros(sum(shapelets_size_and_len.values()), device=self.device),
            requires_grad=True,
        ) # 可学习的权重列表 len=8

        #self.mask = MaskBlock(p=0.5)

        #self.bn = nn.BatchNorm1d(num_features=self.model.num_shapelets)
        #self.relu = nn.ReLU()

        #if self.to_cuda:
        #    self.mask.cuda()
        #    self.bn.cuda()
        #    self.relu.cuda()

        self.T = T
        self.configDir = configDir
        #self.r = 64

        #self.num_clusters = [0.01, 0.02, 0.04]

        # Matrix 传递
        self.C_accu_trans = None
        self.C_accu_Server = None
        self.Q = None
        self.Global_Model = None
        self.Selected_Scales = None
        self.Cached_Scale_Scores = None

        self.shapelet_weight = shapelet_weight
        self.config = config

    def set_device(self, device):
        self.device = torch.device(device)
        self.to_cuda = self.device.type == "cuda"
        self.model.to(self.device)
        self.log_vars.data = self.log_vars.data.to(self.device)
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if hasattr(value, "to"):
                        state[key] = value.to(self.device)
        if self.Global_Model is not None:
            self.Global_Model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _algo_name(self):
        return self.config.get('algo', 'fedcsl')

    def _uses_teacher_scale_set_algo(self):
        return self._algo_name() in ('fedcsl-onehot-fullteacher', 'fedcsl-onehot-splitteacher')

    @staticmethod
    def _normalize_scale_indices(scale_indices, num_shapelet_lengths):
        if scale_indices is None:
            return []
        if isinstance(scale_indices, (int, np.integer)):
            scale_indices = [int(scale_indices)]

        normalized = []
        for scale_idx in scale_indices:
            try:
                scale_idx = int(scale_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= scale_idx < num_shapelet_lengths and scale_idx not in normalized:
                normalized.append(scale_idx)
        return normalized

    def _resolve_scale_scores(self, x, pscore, num_shapelet_lengths, num_shapelet_per_length):
        if pscore is None or (hasattr(pscore, "__len__") and len(pscore) == 0):
            with torch.no_grad():
                _q_tmp = self.model(x, optimize=None, masking=False)
                _q_tmp = nn.functional.normalize(_q_tmp, dim=1)
                pscore = np.zeros(num_shapelet_lengths, dtype=np.float32)
                for _li in range(num_shapelet_lengths):
                    _qi = _q_tmp[:, _li * num_shapelet_per_length: (_li + 1) * num_shapelet_per_length]
                    pscore[_li] = float(_qi.pow(2).sum().item())
        pscore = np.asarray(pscore, dtype=np.float32).ravel()
        if pscore.size < num_shapelet_lengths:
            padded = np.zeros(num_shapelet_lengths, dtype=np.float32)
            padded[:pscore.size] = pscore
            pscore = padded
        elif pscore.size > num_shapelet_lengths:
            pscore = pscore[:num_shapelet_lengths]
        pscore = np.nan_to_num(pscore, nan=0.0, posinf=0.0, neginf=0.0)
        return pscore

    def _pick_onehot_scale(self, x, pscore, num_shapelet_lengths, num_shapelet_per_length):
        pscore = self._resolve_scale_scores(x, pscore, num_shapelet_lengths, num_shapelet_per_length)
        onehot_scale_idx = int(np.argmax(pscore))
        onehot_vec = np.zeros_like(pscore)
        if 0 <= onehot_scale_idx < len(onehot_vec):
            onehot_vec[onehot_scale_idx] = 1.0
        return onehot_scale_idx, onehot_vec

    def _pick_grouped_scales(self, x, pscore, num_shapelet_lengths, num_shapelet_per_length):
        pscore = self._resolve_scale_scores(x, pscore, num_shapelet_lengths, num_shapelet_per_length)
        if num_shapelet_lengths <= 1:
            return [0], pscore

        split = max(1, num_shapelet_lengths // 2)
        groups = [list(range(0, split))]
        if split < num_shapelet_lengths:
            groups.append(list(range(split, num_shapelet_lengths)))

        selected = []
        for group in groups:
            if not group:
                continue
            group_scores = pscore[group]
            best_idx = group[int(np.argmax(group_scores))]
            if best_idx not in selected:
                selected.append(int(best_idx))

        if not selected:
            selected.append(int(np.argmax(pscore)))
        return selected, pscore

    def _resolve_teacher_scale_indices(self, x, pscore, num_shapelet_lengths, num_shapelet_per_length):
        selected = self._normalize_scale_indices(self.Selected_Scales, num_shapelet_lengths)
        if selected:
            return selected
        selected, _ = self._pick_grouped_scales(
            x, pscore, num_shapelet_lengths, num_shapelet_per_length
        )
        return selected

    def _get_cached_scale_scores(self, x, num_shapelet_lengths, num_shapelet_per_length):
        if self.Cached_Scale_Scores is not None:
            cached = np.asarray(self.Cached_Scale_Scores, dtype=np.float32).ravel()
            if cached.size:
                return self._resolve_scale_scores(
                    x, cached, num_shapelet_lengths, num_shapelet_per_length
                )
        return None

    def _compute_selected_scale_losses(self, x_q, x_k, scale_indices, gamma, zeta, algo_name):
        device = x_q.device
        selected_scales = self._normalize_scale_indices(
            scale_indices, len(self.shapelets_size_and_len)
        )
        if not selected_scales:
            selected_scales = [0]

        loss = torch.tensor(0.0, device=device)
        loss_local_jointCLKD = torch.tensor(0.0, device=device)
        loss_global_CLKD_mutiscale = torch.tensor(0.0, device=device)
        loss_local_CLKD_mutiscale = torch.tensor(0.0, device=device)

        q_g_full = None
        k_g_full = None
        if self.Global_Model is not None and algo_name == 'fedcsl-onehot-fullteacher':
            q_g_full = self.Global_Model(x_q, optimize=None, masking=False)
            k_g_full = self.Global_Model(x_k, optimize=None, masking=False)

        for scale_idx in selected_scales:
            q = self.model.encode_scale(x_q, scale_idx, masking=False)
            k = self.model.encode_scale(x_k, scale_idx, masking=False)
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)

            loss += local_infonce_loss(q, k, self.loss_func, self.T) * gamma
            loss_local_CLKD_mutiscale += local_infonce_loss(q, k, self.loss_func, self.T) * 5.0

            if self.Global_Model is not None:
                if algo_name == 'fedcsl-onehot-fullteacher':
                    q_g = self.Global_Model.slice_scale_features(q_g_full, scale_idx)
                    k_g = self.Global_Model.slice_scale_features(k_g_full, scale_idx)
                    q_g = nn.functional.layer_norm(q_g, (q_g.shape[1],))
                    k_g = nn.functional.layer_norm(k_g, (k_g.shape[1],))
                else:
                    q_g = self.Global_Model.encode_scale(x_q, scale_idx, masking=False)
                    k_g = self.Global_Model.encode_scale(x_k, scale_idx, masking=False)

                q_g = nn.functional.normalize(q_g, dim=1)
                k_g = nn.functional.normalize(k_g, dim=1)

                if self.config["ablation"]["UseJointCL"]:
                    loss_local_jointCLKD += joint_contrastive_loss(q_g, k, self.loss_func, self.T)
                if self.config["ablation"]["UseJointKD"]:
                    loss_local_jointCLKD += joint_distill_loss(q, q_g, k, k_g, zeta)
                if self.config["ablation"]["UseScaleCL"]:
                    loss_global_CLKD_mutiscale += scale_contrastive_loss(
                        q_g, k, self.loss_func, self.T, weight=5.0
                    )
                if self.config["ablation"]["UseScaleKD"]:
                    loss_global_CLKD_mutiscale += scale_distill_loss(
                        q_g, q, k_g, k, weight=5.0, zeta=zeta
                    )

        num_selected = float(len(selected_scales))
        loss = loss / num_selected
        loss_local_jointCLKD = loss_local_jointCLKD / num_selected
        loss_global_CLKD_mutiscale = loss_global_CLKD_mutiscale / num_selected
        loss_local_CLKD_mutiscale = loss_local_CLKD_mutiscale / num_selected

        loss += loss_global_CLKD_mutiscale * zeta * 0.1
        loss += loss_local_jointCLKD * gamma * 0.5
        loss += loss_local_CLKD_mutiscale * gamma * 0.1

        zero = torch.tensor(0.0, device=device)
        return loss, zero, zero



    def update(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # C_accu_q 是长度为L=8的list，每个元素是一个tensor[40,40]
    # x (Bx())
    def update_CL(self, x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k):

        config = self.config

        augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1,dtype = np.int64))',
                             'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1,dtype = np.int64))',
                             'Pool(seed=np.random.randint(2 ** 32 - 1,dtype = np.int64))',
                             'Quantize(seed=np.random.randint(2 ** 32 - 1,dtype = np.int64))',
                             'TimeWarp(seed=np.random.randint(2 ** 32 - 1,dtype = np.int64))'
                             ]
        #augmentation_list = ['AddNoise()', 'Pool()', 'Quantize()', 'TimeWarp()']

        ts_l = x.size(2)

        aug1 = np.random.choice(augmentation_list, 1, replace=False)

        x_q = x.transpose(1,2).cpu().numpy()
        for aug in aug1:
            x_q = eval('tsaug.' + aug + '.augment(x_q)')
        x_q = torch.from_numpy(x_q).float()
        x_q = x_q.transpose(1,2)

        if self.to_cuda:
            x_q = x_q.to(self.device)


        aug2 = np.random.choice(augmentation_list, 1, replace=False)
        while (aug2 == aug1).all():
            aug2 = np.random.choice(augmentation_list, 1, replace=False)

        x_k = x.transpose(1,2).cpu().numpy()
        for aug in aug2:
            x_k = eval('tsaug.' + aug + '.augment(x_k)')
        x_k = torch.from_numpy(x_k).float()
        x_k = x_k.transpose(1,2)

        if self.to_cuda:
            x_k = x_k.to(self.device)



        #print(x_q, x_k)

        num_shapelet_lengths = len(self.shapelets_size_and_len)
        num_shapelet_per_length = self.num_shapelets // num_shapelet_lengths
        #-----------------checked
        pscore = None
        if config['ablation']['UseACF']:
            pscore = self._get_cached_scale_scores(
                x, num_shapelet_lengths, num_shapelet_per_length
            )
            if pscore is None:
                pscore = period_score(x, alpha=self.beta)

        # --- fedcsl-onehot: 每个 client 在本 batch 只激活一个尺度 ------------
        # 做法：把 pscore 退化为 one-hot（保留 argmax），让其它尺度的权重为 0；
        # 由于后续 loss_local/global_CLKD_mutiscale 以 precisions[length_i] 为权重，
        # 权重为 0 即等价于该尺度不参与对比/蒸馏。所有数据流保持一致，梯度路径不变。
        onehot_scale_idx = None
        selected_scale_indices = None
        algo_name = self._algo_name()
        if algo_name == 'fedcsl-onehot':
            onehot_scale_idx, pscore = self._pick_onehot_scale(
                x, pscore, num_shapelet_lengths, num_shapelet_per_length
            )
        elif self._uses_teacher_scale_set_algo():
            selected_scale_indices = self._resolve_teacher_scale_indices(
                x, pscore, num_shapelet_lengths, num_shapelet_per_length
            )

        with torch.autograd.set_detect_anomaly(False):
            gamma = config['model']['params'].get('gamma', 0.5)  # local, 默认值0.5
            zeta = 1 - gamma  # global

            if self._uses_teacher_scale_set_algo():
                loss, loss_cca, loss_sdl = self._compute_selected_scale_losses(
                    x_q, x_k, selected_scale_indices, gamma, zeta, algo_name
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                primary_scale = int(selected_scale_indices[0]) if selected_scale_indices else -1
                return [loss.item(), 0, loss_cca.item(), loss_sdl.item(), primary_scale], C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k

            # q,k  =(8x320)
            q = self.model(x_q, optimize=None, masking=False)
            k = self.model(x_k, optimize=None, masking=False)

            # 归一化 (供后续 SDL / CCA 使用)
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)

            # [基础项] 局部 InfoNCE (algo/contrastive/local_infonce.py)
            loss = local_infonce_loss(q, k, self.loss_func, self.T) * gamma
            loss_local_jointCLKD = torch.tensor(0.0, device=self.device)

            # 全局模型与本地模型的 Joint CL/KD
            _cur_algo = algo_name
            _is_fedcsl_like = _cur_algo in (
                'fedcsl',
                'fedcsl-onehot',
                'fedcsl-onehot-fullteacher',
                'fedcsl-onehot-splitteacher',
            )
            if self.Global_Model is not None and _is_fedcsl_like:
                k_g = self.Global_Model(x_k, optimize=None, masking=False)
                k_g = nn.functional.normalize(k_g, dim=1)
                q_g = self.Global_Model(x_q, optimize=None, masking=False)
                q_g = nn.functional.normalize(q_g, dim=1)

                # [UseJointCL] 全局↔本地 InfoNCE (algo/contrastive/joint_cl.py)
                if config["ablation"]["UseJointCL"]:
                    loss_local_jointCLKD += joint_contrastive_loss(q_g, k, self.loss_func, self.T)

                # [UseJointKD] 全局↔本地 KL 蒸馏 (algo/contrastive/joint_kd.py)
                if config["ablation"]["UseJointKD"]:
                    loss_local_jointCLKD += joint_distill_loss(q, q_g, k, k_g, zeta)


            q_sum = None
            q_square_sum = None
            k_sum = None
            k_square_sum = None

            loss_sdl = 0

            c_normalising_factor_q = self.alpha * c_normalising_factor_q + 1  # Ct
            c_normalising_factor_k = self.alpha * c_normalising_factor_k + 1
            #print(q.shape)
            #print(num_shapelet_lengths)
            precisions = []
            for i in range(num_shapelet_lengths):
                precision = torch.exp(-self.log_vars[i]) #方差倒数
                precisions.append(precision)

            # precision_tensor = torch.tensor(precisions)
            # precisions = F.softmax(precision_tensor, dim=0)


            scale_index = 0
            loss_global_CLKD_mutiscale = torch.tensor(0.0,device=self.device)
            loss_local_CLKD_mutiscale = torch.tensor(0.0,device=self.device)

            # 多尺度本地与全局对齐（见 algo/contrastive/*）。
            for length_i in range(num_shapelet_lengths):
                # 提取对应 scale 的表征切片
                qi = q[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                ki = k[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]

                # 尺度不确定性权重：UseACF=True 时用 pscore，否则常数 1。
                # - fedcsl-onehot 下 pscore 是 one-hot（仅 argmax=1，其余=0），
                #   使得其它尺度的 contrastive / KD 权重为 0（不参与）。
                if config['ablation']['UseACF'] and pscore is not None:
                    self.shapelet_weight = pscore
                    w = float(pscore[length_i]) if length_i < len(pscore) else 1.0
                    if not np.isfinite(w):
                        w = 1.0
                    if _cur_algo == 'fedcsl-onehot':
                        # one-hot: w ∈ {0, 1}，保持原样（不回退到 1）
                        precisions[length_i] = w * 5
                    else:
                        if w <= 0:
                            w = 1.0
                        precisions[length_i] = w * 5
                else:
                    precisions[length_i] = 1.0

                # [基础项] 本地 scale 对比 (algo/contrastive/local_infonce.py)
                loss_local_CLKD_mutiscale += precisions[length_i] * local_infonce_loss(
                    qi, ki, self.loss_func, self.T
                )

                if self.Global_Model is not None and _is_fedcsl_like:
                    qi_g = q_g[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                    ki_g = k_g[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]

                    # [UseScaleCL] 多尺度全局↔本地 InfoNCE (algo/contrastive/scale_cl.py)
                    if config["ablation"]["UseScaleCL"]:
                        loss_global_CLKD_mutiscale += scale_contrastive_loss(
                            qi_g, ki, self.loss_func, self.T, weight=precisions[length_i]
                        )

                    # [UseScaleKD] 多尺度全局↔本地 KL 蒸馏 (algo/contrastive/scale_kd.py)
                    if config["ablation"]["UseScaleKD"]:
                        loss_global_CLKD_mutiscale += scale_distill_loss(
                            qi_g, qi, ki_g, ki, weight=precisions[length_i], zeta=zeta
                        )


                # if q_sum!=None:
                #     print(len(x[0]))
                #     print(q.shape)
                #     print(q_sum.shape)
                #     print(qi.shape)
                if q_sum == None:
                    q_sum = qi
                    q_square_sum = qi * qi
                else:
                    q_sum = q_sum + qi
                    q_square_sum = q_square_sum + qi * qi

                C_mini_q = torch.matmul(qi.t(), qi) / (qi.shape[0] - 1)
                C_accu_t_q = self.alpha * C_accu_q[length_i] + C_mini_q
                C_appx_q = C_accu_t_q / c_normalising_factor_q
                loss_sdl += torch.norm(C_appx_q.flatten()[:-1].view(C_appx_q.shape[0] - 1, C_appx_q.shape[0] + 1)[:, 1:], 1).sum()
                #print(length_i)
                C_accu_q[length_i] = C_accu_t_q.detach()
                if k_sum == None:
                    k_sum = ki
                    k_square_sum = ki * ki
                else:
                    k_sum = k_sum + ki
                    k_square_sum = k_square_sum + ki * ki

                C_mini_k = torch.matmul(ki.t(), ki) / (ki.shape[0] - 1)
                C_accu_t_k = self.alpha * C_accu_k[length_i] + C_mini_k
                C_appx_k = C_accu_t_k / c_normalising_factor_k
                loss_sdl += torch.norm(C_appx_k.flatten()[:-1].view(C_appx_k.shape[0] - 1, C_appx_k.shape[0] + 1)[:, 1:], 1).sum()
                #print(length_i)
                C_accu_k[length_i] = C_accu_t_k.detach()

                # 计算对应scale的方差不确定性权重



                scale_index += 1



            loss_cca = 0.5 * torch.sum(q_square_sum - q_sum * q_sum / num_shapelet_lengths) + 0.5 * torch.sum(k_square_sum - k_sum * k_sum / num_shapelet_lengths)
            # 原文
            loss_csl = self.l3 * (loss_cca + self.l4 * loss_sdl)
            loss += loss_csl

            # print("loss_cca: ",loss_cca.item())
            #print("loss_sdl: ",loss_sdl.item())
            #print("loss: ",loss.item())

            if _is_fedcsl_like:
                loss_global_CLKD_mutiscale = loss_global_CLKD_mutiscale * zeta *0.1
                loss_local_jointCLKD =  loss_local_jointCLKD * gamma * 0.5
                loss_local_CLKD_mutiscale   =loss_local_CLKD_mutiscale *gamma * 0.1
                loss += loss_global_CLKD_mutiscale
                #print("loss_global_CLKD_mutiscale",loss_global_CLKD_mutiscale.item())
                loss += loss_local_jointCLKD
                #print("loss_local_jointCLKD:",loss_local_jointCLKD.item())
                loss += loss_local_CLKD_mutiscale
                #print("loss_local_CLKD_mutiscale",loss_local_CLKD_mutiscale.item())
                #print("loss_cca",loss_cca.item())
                #print("loss_sdl",loss_sdl.item())
                #print("loss",loss)
                #print("")

            if config.get('algo', 'fedcsl') == 'fedprox' and self.Global_Model != None:

                proximal_term = 0.0
                mu = 1e-5
                for w, w_global in zip(self.model.parameters(), self.Global_Model.parameters()):
                    proximal_term += torch.norm(w - w_global, p=2) ** 2  # L2 范数平方
                loss += (mu / 2) * proximal_term
                # print("loss_csl: ",loss_csl.item())
                # print("proximal_term: ",(mu / 2) * proximal_term.item())






            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return [loss.item(), 0, loss_cca.item(), loss_sdl.item(), 0], C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k



    def fine_tune(self, X, Y, epochs=1, batch_size=256, epoch_idx=-1):
        print("use finetune\n")
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.long).contiguous()

        # 检查数据是否为空
        if X.shape[0] == 0:
            print(f"警告：fine_tune数据集为空，返回空损失列表")
            return []

        train_ds = torch.utils.data.TensorDataset(X, Y)
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None
        
        # 如果数据量小于batch_size，使用drop_last=False，避免所有数据被丢弃
        drop_last = X.shape[0] >= batch_size
        if not drop_last:
            print(f"警告：fine_tune数据量({X.shape[0]})小于batch_size({batch_size})，使用drop_last=False")

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=drop_last)

        # set model in train mode
        self.model.train()

        losses_ce = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)

        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            for (x, y) in train_dl:

                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    #print("Training data", idx, " on cuda ", torch.cuda.current_device())
                loss_ce = self.update(x, y)
                losses_ce.append(loss_ce)
        return losses_ce


    def train(self, X, epochs=1, batch_size=256, epoch_idx=-1,lr=0.05):

        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            #X = torch.tensor(X, dtype=torch.float).contiguous() #origin
            X = np.array(X)                    # 合并成一个 numpy 数组
            X = torch.from_numpy(X).float()    # 转换为 tensor
            X = X.contiguous()                 # 保持内存连续

        # 检查数据是否为空
        if X.shape[0] == 0:
            print(f"警告：数据集为空，返回空损失列表")
            return []

        train_ds = torch.utils.data.TensorDataset(X, torch.arange(X.shape[0]))
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None

        # 如果数据量小于batch_size，使用drop_last=False，避免所有数据被丢弃
        drop_last = X.shape[0] >= batch_size
        if not drop_last:
            print(f"警告：数据量({X.shape[0]})小于batch_size({batch_size})，使用drop_last=False")

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=drop_last)

        # set model in train mode
        self.model.train()

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0



        self.model.train()

        # C_accu的清空需要放在此处，不然1个epoch就清零了。指望本地训练多个epoch
        c_normalising_factor_q = torch.tensor([0], dtype=torch.float, device=self.device)
        C_accu_q = [torch.tensor([0], dtype=torch.float, device=self.device) for _ in range(len(self.shapelets_size_and_len))]
        c_normalising_factor_k = torch.tensor([0], dtype=torch.float, device=self.device)
        C_accu_k = [torch.tensor([0], dtype=torch.float, device=self.device) for _ in range(len(self.shapelets_size_and_len))]


        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            # if self.C_accu_Server==None: #不采用全局相似度矩阵
            #     #print("不采用全局相似度矩阵！")
            c_normalising_factor_q = torch.tensor([0], dtype=torch.float, device=self.device)
            C_accu_q = [torch.tensor([0], dtype=torch.float, device=self.device) for _ in range(len(self.shapelets_size_and_len))]
            c_normalising_factor_k = torch.tensor([0], dtype=torch.float, device=self.device)
            C_accu_k = [torch.tensor([0], dtype=torch.float, device=self.device) for _ in range(len(self.shapelets_size_and_len))]

            for (x, idx) in train_dl:

                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.to(self.device)
                    #print("Training data", idx, " on cuda ", torch.cuda.current_device())




                if not self.use_regularizer:
                    # 一个batch 训练，更新相关矩阵,loss update
                    current_loss_ce, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k = self.update_CL(x, C_accu_q, c_normalising_factor_q, C_accu_k,
                                                                                                                         c_normalising_factor_k)

                    # C_accu_q 是一个list，存储每个尺度r下的C_accu tensor


                    losses_ce.append(current_loss_ce)
                else:
                    pass


            if not self.use_regularizer:
                progress_bar.set_description(f"Loss: {current_loss_ce}")
            else:
                if self.l1 > 0.0 and self.l2 > 0.0:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                 f"Loss sim: {current_loss_sim}")
                else:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
            if self.scheduler != None:
                self.scheduler.step()

        # 传给Server
        self.C_accu_trans = [(x + y )/2 for x,y in zip(C_accu_k,C_accu_q)]


        return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
        losses_ce, losses_dist)

    def transform(self, X, *, batch_size=512, result_type='tensor', normalize=False):
        # 先检查输入数据是否为空（在转换为tensor之前）
        if isinstance(X, (list, tuple)):
            if len(X) == 0:
                return np.array([]) if result_type == 'numpy' else torch.tensor([])
        elif isinstance(X, np.ndarray):
            if X.size == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
                return np.array([]) if result_type == 'numpy' else torch.tensor([])
        elif isinstance(X, torch.Tensor):
            if X.numel() == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
                return np.array([]) if result_type == 'numpy' else torch.tensor([])

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        # 再次检查转换后的tensor是否为空
        if X.numel() == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
            return np.array([]) if result_type == 'numpy' else torch.tensor([])

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shapelet_transform = []
        for (x, ) in dataloader:
            if self.to_cuda:
                x = x.to(self.device)
            with torch.no_grad():
            #shapelet_transform = self.model.transform(X)
                shapelet_transform.append(self.model(x, optimize=None).cpu())
        
        # 检查shapelet_transform是否为空
        if len(shapelet_transform) == 0:
            return np.array([]) if result_type == 'numpy' else torch.tensor([])
        
        shapelet_transform = torch.cat(shapelet_transform, 0)
        if normalize:
            shapelet_transform = nn.functional.normalize(shapelet_transform, dim=1)
        if result_type == 'tensor':
            return shapelet_transform
        return shapelet_transform.detach().numpy()

    def predict(self, X, *, batch_size=512):

        # 先检查输入数据是否为空（在转换为tensor之前）
        if isinstance(X, (list, tuple)):
            if len(X) == 0:
                return np.array([])
        elif isinstance(X, np.ndarray):
            if X.size == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
                return np.array([])
        elif isinstance(X, torch.Tensor):
            if X.numel() == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
                return np.array([])

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        # 再次检查转换后的tensor是否为空
        if X.numel() == 0 or (len(X.shape) > 0 and X.shape[0] == 0):
            return np.array([])

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        for (x, ) in dataloader:
            if self.to_cuda:
                x = x.to(self.device)
            with torch.no_grad():
            #shapelet_transform = self.model.transform(X)
                preds.append(self.model(x).cpu())
        
        # 检查preds是否为空
        if len(preds) == 0:
            return np.array([])
        
        preds = torch.cat(preds, 0)

        return preds.detach().numpy()


