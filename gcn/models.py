import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, feature=8, M=8, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(feature, feature)
        self.gc2 = GraphConvolution(feature, M)
        self.feature = feature
        self.dropout = dropout

    def forward(self, adj):
        adj = torch.from_numpy(adj)

        x = torch.ones((adj.shape[0], self.feature), dtype=torch.float32)
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.softmax(x, dim=0)
