from environ.mis_env import MISEnv
from config import *
import numpy as np
# xp = chainer.cuda.cupy if use_gpu else np
xp = np

"""
- act: graph -> (vertex, probability)
    return vertex and probability based on the policy (mix of best_act and predict_act)
- best_act: graph -> vertex
    return the best vertex if exist else -1
- predict_act: graph -> (vertex, probability)
    return vertex and probability based on the policy
"""

def calc_B(adj):
    n, _ = adj.shape
    adj += xp.eye(n)
    D = xp.diag(1 / xp.sqrt(xp.sum(adj, axis=0)))
    return D.dot(adj.dot(D)).astype(xp.float32)

class GCNPolicy:
    def __init__(self, gcn):
        self.model = gcn
        if use_gpu:
            get_device = 0
            chainer.cuda.get_device(get_device).use()
            self.model.to_gpu()

    def best_act(self, adj):
        if use_dense:
            ds = xp.sum(adj, axis=0)
            return xp.argmin(ds) if xp.min(ds) <= 1 else -1
        else:
            n, _ = adj.shape
            ds = xp.zeros(n)
            for i in range(adj.row.size):
                ds[int(adj.row[i])] += 1
                ds[int(adj.col[i])] += 1
            ds /= 2
            return xp.argmin(ds) if xp.min(ds) <= 1 else -1

    def predict_act(self, adj):
        n, _ = adj.shape
        if normalize_adj : adj = calc_B(adj)
        prob = self.model(adj)
        if use_gpu:
            v = np.random.choice(n, p=prob.data[:, 0].get())
        else:
            v = np.random.choice(n, p=prob.data[:, 0].detach().numpy())
        return v, prob[v, 0]

    def act(self, adj):
        # ba = self.best_act(adj)
        # return (ba, xp.array(1, dtype=xp.float32)) if ba != -1 else self.predict_act(adj)
        return self.predict_act(adj)
    