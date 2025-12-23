import torch
from torch import nn
from blocks import *

class Gate(nn.Module):
    def __init__(self,input_dim,output_dim,shapelets_size,in_channels = 1):
        super(Gate, self).__init__()
        
        self.encoder = Expert(input_dim,output_dim,shapelets_size,in_channels)
    def forward(self,x):
        x = self.encoder(x)

        x = x.squeeze(dim=1)

        return x
        


# 修改成csl中的encoder 模型，待完成
class Expert(nn.Module):
    def __init__(self,input_dim,output_dim,shapelets_size,in_channels = 1): #input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()
        
        p=0
        expert_hidden_layers = [64,32]
        self.expert_layer = MaxCosineSimilarityBlock(shapelets_size = shapelets_size, num_shapelets = 40, in_channels=1, to_cuda=True)

    def forward(self, x):
        out = self.expert_layer(x) #[bs,1,40]
        return out



class Expert_Gate(nn.Module):
    def __init__(self,feature_dim,expert_dim,shapelets_size_and_len,n_expert,n_task,in_channels=1,use_gate=True): #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        '''专家网络'''
        # for i in range(n_expert):
        #     setattr(self, "expert_layer"+str(i+1), Expert(feature_dim,expert_dim,in_channels=in_channels,
        #                                                   shapelets_size=shapelets_size)) #为每个expert创建一个DNN
        # self.expert_layers = [getattr(self,"expert_layer"+str(i+1)) for i in range(n_expert)]#为每个expert创建一个DNN
        
        i=0
        for shapelets_size, _ in shapelets_size_and_len.items():
            setattr(self, "expert_layer"+str(i+1), Expert(feature_dim,expert_dim,in_channels=in_channels,
                                                          shapelets_size=shapelets_size)) #为每个expert创建一个DNN
            i += 1
        self.expert_layers = [getattr(self,"expert_layer"+str(i+1)) for i in range(n_expert)]#为每个expert创建一个DNN
        
        '''门控网络'''
        for i in range(n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(Gate(feature_dim,expert_dim,in_channels=in_channels,
                                                                shapelets_size=shapelets_size),
                                                                nn.Linear(40, n_expert),
                                        					    nn.Softmax(dim=1))) 
        self.gate_layers = [getattr(self,"gate_layer"+str(i+1)) for i in range(n_task)]#为每个gate创建一个lr+softmax
        
    def forward(self, x):
        
        # input x(bs,)  -->output [bs,n_expert,expert_dim] / [bs,n_expert,multidim,expert_dim]
        

        if self.use_gate:
            # 构建多个专家网络
            E_net = [expert(x) for expert in self.expert_layers]

            E_net = torch.cat(([e[:,np.newaxis,:] for e in E_net]),dim = 1) # 维度 (bs,n_expert,expert_dim) [bs,n_expert,40]

            E_net = E_net.squeeze(dim=2) #[bs,n_expert,40]

            # 构建多个门网络
            gate_net = [gate(x) for gate in self.gate_layers]     # 维度 n_task个(bs,n_expert)
            # print(E_net.shape)
            #print(gate_net[0].shape)
            # exit(0)
            #towers计算：对应的门网络乘上所有的专家网络
            towers = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # 维度(bs,n_expert,1)
                print(g)
                tower = torch.matmul(E_net.transpose(1,2),g)# 维度 (bs,expert_dim,1)
                towers.append(tower.transpose(1,2).squeeze(1))           # 维度(bs,expert_dim) [bs,40]
                
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers = sum(E_net)/len(E_net)
            
            
       
        # exit(0)
        return towers #checked ok
    


# 对应learnshaplet
class MMoE(nn.Module):
	#feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    def __init__(self,num_classes,expert_dim,n_expert,n_task,shapelets_size_and_len,in_channels=1,use_gate=True): 
        super(MMoE, self).__init__()
        
        self.usetower = True
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.use_gate = True
        self.Expert_Gate = Expert_Gate(feature_dim=self.num_shapelets,expert_dim=expert_dim,
                                       n_expert=n_expert,n_task=n_task,in_channels=in_channels,
                                       use_gate=use_gate,shapelets_size_and_len=shapelets_size_and_len)
        
        # '''Tower1'''
        # p1 = 0 
        # hidden_layer1 = [64,32] #[64,32] 
        # self.tower1 = nn.Sequential(
        #     nn.Linear(expert_dim, hidden_layer1[0]),
        #     nn.ReLU(),
        #     nn.Dropout(p1),
        #     nn.Linear(hidden_layer1[0], hidden_layer1[1]),
        #     nn.ReLU(),
        #     nn.Dropout(p1),
        #     nn.Linear(hidden_layer1[1], 1))
        # '''Tower2'''
        # p2 = 0
        # hidden_layer2 = [64,32]
        # self.tower2 = nn.Sequential(
        #     nn.Linear(expert_dim, hidden_layer2[0]),
        #     nn.ReLU(),
        #     nn.Dropout(p2),
        #     nn.Linear(hidden_layer2[0], hidden_layer2[1]),
        #     nn.ReLU(),
        #     nn.Dropout(p2),
        #     nn.Linear(hidden_layer2[1], 1))
        
        hidden_layer2 = [64,32]
        # 特定 Task 的结构
        self.predictors = nn.ModuleList([
            nn.Sequential(
                #nn.BatchNorm1d(num_features=40),
                nn.Linear(40, 16),
                nn.ReLU(),
                nn.Dropout(0),
                nn.Linear(16,8),
            )
            for _ in range(len(shapelets_size_and_len))
        ])
        self.bn = nn.BatchNorm1d(num_features=320)
        
    def forward(self,x, optimize="acc",masking=False):
        
        towers = self.Expert_Gate(x) # task[i] 是 第i个任务的输出
        outlist = []
        
        self.usetower = False
        if self.usetower == False:
            for i in range(len(self.predictors)):
                outlist.append(towers[i])
        
        if self.use_gate and self.usetower:            
            # out1 = self.tower1(towers[0])
            # out2 = self.tower2(towers[1]) 
            for i in range(len(self.predictors)):
                outlist.append(self.predictors[i](towers[i]))
        # else:
        #     out1 = self.tower1(towers)
        #     out2 = self.tower2(towers)
            
            
        out_concat = torch.cat(outlist, dim=-1)
        print(out_concat.shape)
        out_concat = self.bn(out_concat)
        if optimize == "acc":
            self.linear(out_concat)
        
        return out_concat



# Model = MMoE(feature_dim=112,expert_dim=32,n_expert=4,n_task=2,use_gate=True,shapelets_size_and_len=shapelets_size_and_len)

# nParams = sum([p.nelement() for p in Model.parameters()])
# print('* number of parameters: %d' % nParams)


# class ShapeletsDistBlocksNoCat(nn.Module):
   
#     def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='cosin', to_cuda=True, checkpoint=False):
#         super(ShapeletsDistBlocksNoCat, self).__init__()
#         self.checkpoint = checkpoint
#         self.to_cuda = to_cuda
#         self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
#         self.in_channels = in_channels
#         self.dist_measure = dist_measure
#         self.blocks = nn.ModuleList(
#             [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
#                                         in_channels=in_channels, to_cuda=self.to_cuda)
#                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
#         #self.msaBlock = MultiScaleAttention(in_channels=1, num_scales=8)
#         #self.msaBlock = MultiScaleWeightedConcatAttention(feature_dim=40, num_scales=8)
        
#     def forward(self, x, masking=False):
       

#         out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        
#         # 选择是否要 连接
#         # for block in self.blocks:
#         #     if self.checkpoint and self.dist_measure != 'cross-correlation':
#         #         out = torch.cat((out, checkpoint(block, x, masking)), dim=2)
            
#         #     else:
#         #         out = torch.cat((out, block(x, masking)), dim=2)
            
#         # 不连接
#         outlist = []
#         for block in self.blocks:
#             out = block(x, masking)
#             out = out.squeeze(dim=1)
#             outlist.append(out)
#             # print(out.shape)
#             # exit(0)
#         out,weights = self.msaBlock(outlist)
#         print(weights)
#         # exit(0)
#         return out #[8,1,320]


