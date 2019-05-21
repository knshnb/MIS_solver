import copy
import numpy as np
import torch
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter
from utils.randomplay import randomplay, make_adj_set
from utils.nodehash import NodeHash
from utils.gnnhash import GNNHash

# p(s) = gnn(s).policy
# v(s) = gnn(s).value
ALPHA = 2  # ucb(s,a) = Q(s,a) + ALPHA * |V| * P(s,a) / (1 + N(s,a))
# TAU = 0.5  # N(s,a)からpiを求めるときの温度
EPS = 1e-10
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
        if not self.is_end():
            hash = NodeHash.hash(self.graph)
            if GNNHash.has(hash):
                self.P, self.Q = GNNHash.get(hash)
            else:                
                Timer.start('gnn')
                with torch.no_grad():
                    self.P, self.Q = MCTSNode.gnn(self.graph)
                Timer.end('gnn')
                self.P = self.P.detach().numpy()
                self.Q = self.Q.detach().numpy()
                GNNHash.save(hash, self.P, self.Q.copy())

            if NodeHash.has(hash):
                self.reward_mean, self.reward_std = NodeHash.get(hash)
            else:
                # ランダム試行でrewardの平均、分散を求める
                NUM = max(100, 2 * n)
                rewards = np.empty(NUM)
                ss = make_adj_set(graph)
                # TODO?: 並列化
                Timer.start('sample')
                Counter.count('sample')
                for i in range(NUM):
                    rewards[i] = randomplay(ss)
                Timer.end('sample')
                self.reward_mean = rewards.mean()
                # stdを0にしないようにEPSを足す
                self.reward_std = rewards.std(ddof=1) + EPS
                assert not np.isnan(self.reward_std)
                NodeHash.save(hash, self.reward_mean, self.reward_std)

    def is_end(self):
        return self.graph.shape[0] == 0

    def state_value(self):
        if self.is_end():
            return 0.
        else:
            return self.Q.max()

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / self.reward_std

    # selfがNoneでないときのみ呼ばれる
    def best_child(self):
        n, _ = self.graph.shape
        ucb = self.Q + ALPHA * np.sqrt(self.visit_cnt.sum()) * self.P / (1 + self.visit_cnt)
        return np.argmax(ucb)

    def pi(self, TAU):
        pow_tau = np.power(self.visit_cnt, 1 / TAU)
        assert (self.visit_cnt >= 0).all()
        return pow_tau / pow_tau.sum()
