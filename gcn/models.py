import numpy as np
import torch
from gcn.layers import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self, layer_num=2, feature=8, M=1, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for l in range(layer_num - 1):
            self.layers.append(GraphConvolution(feature if l != 0 else 1, feature))
        self.layers.append(GraphConvolution(feature, M))
        self.feature = feature
        self.dropout = dropout

    def forward(self, adj):
        adj = torch.from_numpy(adj)

        x = torch.ones((adj.shape[0], 1), dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
                # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        return torch.nn.functional.softmax(x, dim=0)
