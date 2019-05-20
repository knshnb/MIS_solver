from config import device
import torch
from gin.mlp import MLP

# GPU対応はGIN3だけ
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
    def forward(self, adj):
        adj = adj.copy()
        for i in range(adj.shape[0]):
            adj[i][i] = 1
        adj = torch.from_numpy(adj).to(device)

        x = torch.ones((adj.shape[0], 1), dtype=torch.float32)
        x = x.to(device)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        
        policy = torch.nn.functional.softmax(self.policy_output_layer(x, adj), dim=0)[:, 0]
        value = self.value_output_layer(x, adj)[:, 0]
        # TODO: valueを正規化して出力したい！
        return policy.cpu(), value.cpu()
