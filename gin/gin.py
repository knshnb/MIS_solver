import torch
from gin.mlp import MLP

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

class GIN2(torch.nn.Module):
    def __init__(self, layer_num=2, feature=8, M=1, dropout=0.5):
        super(GIN2, self).__init__()
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
        return x[:,0]
