import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

class MultiScaleConcatAttention(nn.Module):
    def __init__(self, in_channels, num_scales):
        super(MultiScaleConcatAttention, self).__init__()
        # 定义用于计算注意力分数的MLP层
        self.fc = nn.Sequential(
            nn.Linear(in_channels * num_scales, 512),  # 中间层可以调整大小
            nn.ReLU(),
            nn.Linear(512, num_scales)  # 输出num_scales个注意力分数
        )
        # 初始化一个softmax函数用于归一化注意力分数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, multi_scale_features):
        """
        输入:
            multi_scale_features: 包含多个尺度特征的列表，每个元素形状为 (batch_size, channels, features_dim)
        输出:
            fused_feature: 融合后的特征张量
            attention_weights: 各个尺度的注意力权重
        """
        batch_size = multi_scale_features[0].size(0)
        features_dim = multi_scale_features[0].size(2)

        # 将所有尺度的特征连接成一个张量
        concatenated_features = torch.cat(multi_scale_features, dim=1)  # shape: (batch_size, num_scales * channels, features_dim)

        # 计算全局平均池化后每个位置的特征表示
        pooled_features = F.adaptive_avg_pool1d(concatenated_features, 1).squeeze(-1)  # shape: (batch_size, num_scales * channels)

        # 计算每个尺度的注意力分数
        attention_scores = self.fc(pooled_features)  # shape: (batch_size, num_scales)

        # 归一化注意力分数以获得权重
        attention_weights = self.softmax(attention_scores)  # shape: (batch_size, num_scales)

        # 为了应用注意力权重，我们需要再次分割连接的特征张量
        split_sizes = [in_channels] * len(multi_scale_features)
        split_features = torch.split(concatenated_features, split_sizes, dim=1)

        # 应用注意力权重到原始特征上
        weighted_features = sum([split_features[i] * attention_weights[:, i].unsqueeze(-1).unsqueeze(-1)
                                 for i in range(len(split_features))])

        return weighted_features, attention_weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()
        # 定义用于计算注意力分数的MLP层
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),  # 中间层可以调整大小
            nn.ReLU(),
            nn.Linear(512, 1)  # 输出单个注意力分数
        )
        # 初始化一个softmax函数用于归一化注意力分数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """
        输入:
            features: 形状为 (batch_size, feature_dim) 的张量
        输出:
            attended_feature: 应用注意力后的特征张量
            attention_weights: 注意力权重
        """
        batch_size = features.size(0)
        
        # 计算每个样本的注意力分数
        attention_scores = self.fc(features).squeeze(-1)  # shape: (batch_size,)
        
        # 归一化注意力分数以获得权重
        attention_weights = self.softmax(attention_scores.unsqueeze(1))  # shape: (batch_size, 1)

        # 应用注意力权重到原始特征上
        attended_feature = features * attention_weights
        
        return attended_feature, attention_weights

class MultiScaleWeightedConcatAttention(nn.Module):
    def __init__(self, feature_dim, num_scales):
        super(MultiScaleWeightedConcatAttention, self).__init__()
        # 定义用于计算注意力分数的MLP层
        self.fc = nn.Sequential(
            # nn.Linear(feature_dim, 1)  # 中间层可以调整大小
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # 初始化一个softmax函数用于归一化注意力分数
        self.softmax = nn.Softmax(dim=0)
        self.num_scales = num_scales
        self.sigmoid = nn.Sigmoid()
    def forward(self, multi_scale_features):
        """
        输入:
            multi_scale_features: 包含多个尺度特征的列表，每个元素形状为 (batch_size, feature_dim)
        输出:
            fused_feature: 加权拼接后的特征张量
            attention_weights: 各个尺度的注意力权重
        """
        batch_size = multi_scale_features[0].size(0)

        # 初始化一个空列表存储各个尺度的注意力权重和加权特征
        weighted_features_list = []
        attention_weights_list = []
        attention_scores_list =[]
        # 对每个尺度的特征计算注意力权重，并应用该权重
        for scale_idx in range(self.num_scales):
            features = multi_scale_features[scale_idx]

            # 计算每个样本的注意力分数
            attention_scores = self.fc(features).squeeze(-1)  # shape: (batch_size)

            attention_scores_list.append(attention_scores)
        
        stacked_attention_scores = torch.stack(attention_scores_list, dim=1)  # shape: (batch_size, num_scales)
        # 归一化注意力分数以获得权重
        attention_weights = self.softmax(stacked_attention_scores)  # shape: (batch_size, num_scales)
        # print(attention_weights.shape)
        #exit(0)
        
        for scale_idx in range(self.num_scales):
            features = multi_scale_features[scale_idx]
            weight = attention_weights[:, scale_idx].unsqueeze(-1)
            # 应用注意力权重到原始特征上，避免原地操作
            weighted_features = features * weight * self.num_scales # shape: (batch_size, feature_dim)
            # print(weighted_features.shape)
            weighted_features_list.append(weighted_features)

        # 将所有尺度的加权特征拼接在一起
        fused_feature = torch.cat(weighted_features_list, dim=1)  # shape: (batch_size, num_scales * feature_dim)


        return fused_feature, attention_weights

class ShapeletsDistBlocksNoCat(nn.Module):
   
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='cosin', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocksNoCat, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        self.blocks = nn.ModuleList(
            [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                        in_channels=in_channels, to_cuda=self.to_cuda)
                for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        #self.msaBlock = MultiScaleAttention(in_channels=1, num_scales=8)
        self.msaBlock = MultiScaleWeightedConcatAttention(feature_dim=40, num_scales=8)
        
    def forward(self, x, masking=False):
       

        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        
        # 选择是否要 连接
        # for block in self.blocks:
        #     if self.checkpoint and self.dist_measure != 'cross-correlation':
        #         out = torch.cat((out, checkpoint(block, x, masking)), dim=2)
            
        #     else:
        #         out = torch.cat((out, block(x, masking)), dim=2)
            
        # 不连接
        outlist = []
        for block in self.blocks:
            out = block(x, masking)
            out = out.squeeze(dim=1)
            outlist.append(out)
            # print(out.shape)
            # exit(0)
        out,weights = self.msaBlock(outlist)
        print(weights)
        # exit(0)
        return out #[8,1,320]




class Attention(nn.Module):
   
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(Attention, self).__init__()

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocksNoCat(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        
        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                            #   nn.Linear(self.model.num_shapelets, 256),
                                            #   nn.ReLU(),
                                            #   nn.Linear(self.num_shapelets, 128)
                                            # nn.Linear(self.num_shapelets, 256),
                                            # nn.ReLU(),
                                            # nn.Linear(256, 128)
                                            # 这里有buggggggggg！！！！！！！
                                        )
        
        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))
                                              
        self.prodictor = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                        nn.Linear(self.num_shapelets, self.num_shapelets),
                                        nn.ReLU(),
                                        nn.Linear(self.num_shapelets, self.num_shapelets)) 

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False,isProdictor=False):
       
    
        # encoder
        x = self.shapelets_blocks(x, masking) # 
        #x = torch.squeeze(x, 1)  #[8,320]
        
        # test torch.cat
        #x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)

        if isProdictor:
            x = self.projection(x)
            x = self.prodictor(x)
            return x
        
        # projector
        x = self.projection(x)
        
        if optimize == 'acc':
            x = self.linear(x)
        
        
        return x # [batchsize, 320]

   
  