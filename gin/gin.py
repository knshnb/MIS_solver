from config import device, use_dense
import torch
from gin.mlp import MLP
import numpy as np

# GIN is not supported on GPU
class GIN(torch.nn.Module):
    def __init__(self, layer_num=2, feature=8, M=1, dropout=0.5):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for l in range(layer_num):
            self.layers.append(MLP(
                1 if l == 0 else feature,
                M if l == layer_num - 1 else feature
            ))
        self.dropout = dropout

    def forward(self, adj):
        for i in range(adj.shape[0]):
            adj[i][i] = 1
        adj = torch.from_numpy(adj)

        x = torch.ones((adj.shape[0], 1), dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
                # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        return torch.nn.functional.softmax(x, dim=0)

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
                1 if l == 0 else feature,
                feature
            ))
        self.value_output_layer = MLP(feature, M)
        self.policy_output_layer = MLP(feature, M)
        self.dropout = dropout

    # return (policy, value)
    def forward(self, adj, force_dense=False):
        if use_dense or force_dense:
            if not use_dense:
                adj = adj.todense().A
            else:
                adj = adj.copy()
            for i in range(adj.shape[0]):
                adj[i][i] = 1
            adj = torch.from_numpy(adj).to(device)
        else:
            n, _ = adj.shape
            x = adj.row.tolist()
            y = adj.col.tolist()
            for i in range(n):
                x.append(i)
                y.append(i)
            m = len(x)
            adj = torch.sparse.FloatTensor(torch.LongTensor([x, y]), torch.Tensor(np.ones(m)), torch.Size(list(adj.shape))).to(device)

        x = torch.ones((adj.shape[0], 1), dtype=torch.float32)
        x = x.to(device)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        
        policy = torch.nn.functional.softmax(self.policy_output_layer(x, adj), dim=0)[:, 0]
        value = self.value_output_layer(x, adj)[:, 0]
        # normalize value (mean 0, std 1)
        value_mean = value.mean()
        normalized_value = (value - value_mean) / my_std(value, value_mean)
        return policy.cpu(), normalized_value.cpu()
