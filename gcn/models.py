import numpy as np
import torch
from gcn.layers import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self, layer_num=2, feature=8, M=1, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.layers.append(GraphConvolution(feature, feature))
        self.layers.append(GraphConvolution(feature, M))
        self.feature = feature
        self.dropout = dropout

    def forward(self, adj):
        adj = torch.from_numpy(adj)

        x = torch.ones((adj.shape[0], self.feature), dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
        return torch.nn.functional.softmax(x, dim=0)
