import copy
import numpy as np
import torch
from environ.mis_env import MISEnv
from gin.gin import GIN3

# p(s) = gnn(s).policy
# v(s) = gnn(s).value
INF = 100000  # Q(s,a)の初期値をINF + v(s)[a]で初期化
ALPHA = 0.5  # ucb(s,a) = Q(s,a) + ALPHA * |V| * P(s,a) / (1 + N(s,a))
TAU = 0.5  # N(s,a)からpiを求めるときの温度
# TODO: TAUを学習が進むに連れて小さくしていく

class MCTSNode:
    @staticmethod
    def set_gnn(gnn):
        MCTSNode.gnn = gnn

    def __init__(self, graph, idx=-1, parent=None):
        n, _ = graph.shape
        self.graph = graph
        self.parent = parent
        self.children = [None for _ in range(n)]
        self.idx = idx

        self.visit_cnt = np.zeros(n, dtype=np.float32)
        self.max_return = -1
        if not self.is_end():
            with torch.no_grad():
                self.P, self.Q = MCTSNode.gnn(self.graph)
            self.P = self.P.detach().numpy()
            self.Q = self.Q.detach().numpy()
            self.Q += INF

    def is_end(self):
        return self.graph.shape[0] == 0

    def state_value(self):
        if self.is_end():
            return 0.
        ret = self.Q.max()
        assert ret >= INF
        return ret - INF
        # return 0. if self.is_end() else self.Q.max()

    # selfがNoneでないときのみ呼ばれる
    def best_child(self):
        n, _ = self.graph.shape
        ucb = self.Q + ALPHA * n * self.P / (1 + self.visit_cnt)
        return np.argmax(ucb)

    def pi(self):
        pow_tau = np.power(self.visit_cnt, 1 / TAU)
        assert (self.visit_cnt >= 0).all()
        return pow_tau / pow_tau.sum()
