import torch

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# input: ema实例, ma_model, current_model
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# use 
# ema = EMA(beta=0.99)
# update_moving_average(ema, ma_model, current_model)


# 待修改
# def _retain_weight_scaler(self):
#     self.client_id_to_index = {c.cid: i for i, c in enumerate(self._clients)}

#     client_index = self.client_id_to_index[self.grouped_clients[0].cid]
#     weight_scaler = self.grouped_clients[0].weight_scaler if self.grouped_clients[0].weight_scaler else 0
#     scaler = torch.tensor((client_index, weight_scaler)).to(self.conf.device)
#     scalers = [torch.zeros_like(scaler) for _ in self.selected_clients]
#     dist.barrier()
#     dist.all_gather(scalers, scaler)
#     for i, client in enumerate(self._clients):
#         for scaler in scalers:
#             scaler = scaler.cpu().numpy()
#             if self.client_id_to_index[client.cid] == int(scaler[0]) and not client.weight_scaler:
#                 self._clients[i].weight_scaler = scaler[1]
    
# 计算模型l2距离            
def calculate_divergence(old_model, new_model, typ="L2"):
        size = 0
        total_distance = 0
        old_dict = old_model.state_dict()
        new_dict = new_model.state_dict()
        for name, param in old_model.named_parameters():
            #if 'conv' in name and 'weight' in name: 先计算所有层的距离
            total_distance += calculate_distance(old_dict[name].detach().clone().view(1, -1),
                                                        new_dict[name].detach().clone().view(1, -1),
                                                        typ)
            size += 1
        distance = total_distance / size
        #logger.info(f"Model distance: {distance} = {total_distance}/{size}")
        return distance
    
def calculate_distance(m1, m2, typ="L2"):
    if typ == "L2":
        return torch.dist(m1, m2, 2)