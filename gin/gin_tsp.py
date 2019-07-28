from config import device, use_dense
import torch
from gin.mlp import MLP
import numpy as np

# avoid nan on torch.sqrt(0) in torch.std
def my_std(a, mean):
    return torch.sqrt((a - mean).pow(2).mean() + 1e-10)

# policy and value network for MCTS
class GIN3(torch.nn.Module):
    def __init__(self, layer_num=2, feature=8, M=1, dropout=0.5):
        super(GIN3, self).__init__()
        self.layers = torch.nn.ModuleList()
        for l in range(layer_num):
            self.layers.append(MLP(
                2 if l == 0 else feature,
                feature
            ))
        self.value_output_layer = MLP(feature, M)
        self.policy_output_layer = MLP(feature, M)
        self.dropout = dropout

    # return (policy, value)
    def forward(self, adj,  dist_from_prev, dist_to_start, force_dense=False):
        # degree information
        # cur = torch.tensor(adj.sum(axis=0).reshape(adj.shape[0], 1))
        cur = torch.Tensor([dist_from_prev, dist_to_start]).t()
        # 注意: なんかadj[i][i] = 1みたいなのを消した
        adj = torch.from_numpy(adj).to(device)
        cur = cur.to(device)
        for i, layer in enumerate(self.layers):
            cur = layer(cur, adj)
            cur = torch.nn.functional.relu(cur)
            cur = torch.nn.functional.dropout(cur, self.dropout, training=self.training)
        
        policy = torch.nn.functional.softmax(self.policy_output_layer(cur, adj), dim=0)[:, 0]
        value = self.value_output_layer(cur, adj)[:, 0]
        # normalize value (mean 0, std 1)
        value_mean = value.mean()
        normalized_value = (value - value_mean) / my_std(value, value_mean)
        return policy.cpu(), normalized_value.cpu()
