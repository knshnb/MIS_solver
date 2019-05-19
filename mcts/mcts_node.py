import copy
import numpy as np
import torch
from environ.mis_env import MISEnv
from gin.gin import GIN3

# p(s) = gnn(s).policy
# v(s) = gnn(s).value
ALPHA = 2  # ucb(s,a) = Q(s,a) + ALPHA * |V| * P(s,a) / (1 + N(s,a))
TAU = 0.5  # N(s,a)からpiを求めるときの温度
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
        self.max_return = -1
        if not self.is_end():
            with torch.no_grad():
                self.P, self.Q = MCTSNode.gnn(self.graph)
            self.P = self.P.detach().numpy()
            self.Q = self.Q.detach().numpy()

            # ランダム試行でrewardの平均、分散を求める
            env = MISEnv()
            env.set_graph(self.graph)
            NUM = max(100, 2 * n)
            rewards = np.empty(NUM)
            for i in range(NUM):
                g = env.reset()
                done = False
                while not done:
                    action = np.random.randint(g.shape[0])
                    g, reward, done, info = env.step(action)
                rewards[i] = reward
            self.reward_mean = rewards.mean()
            # stdを0にしないようにEPSを足す
            self.reward_std = rewards.std(ddof=1) + EPS
            assert not np.isnan(self.reward_std)

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

    def pi(self):
        pow_tau = np.power(self.visit_cnt, 1 / TAU)
        assert (self.visit_cnt >= 0).all()
        return pow_tau / pow_tau.sum()
