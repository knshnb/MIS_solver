import copy
import numpy as np
import torch
from gin.gin_tsp import GIN3
from utils.timer import Timer
from utils.counter import Counter
from utils.randomplay import randomplay_tsp
from utils.gnnhash import GNNHash

# p(s) = gnn(s).policy
# v(s) = gnn(s).value
ALPHA = 2  # ucb(s,a) = Q(s,a) + ALPHA * sqrt(N(s,a)) * P(s,a) / (1 + N(s,a))
EPS = 1e-10

class TSPMCTSNode:
    def __init__(self, graph, dist_from_prev, dist_to_start, mcts, idx=-1, parent=None):
        n, _ = graph.shape
        self.graph = graph
        self.dist_from_prev = dist_from_prev
        self.dist_to_start = dist_to_start
        self.parent = parent
        self.children = [None for _ in range(n)]
        self.mcts = mcts
        self.idx = idx

        self.visit_cnt = np.zeros(n, dtype=np.float32)
        if not self.is_end():
            hash = self.mcts.nodehash.hash(self.graph, self.dist_from_prev, self.dist_to_start)
            if self.mcts.gnnhash.has(hash):
                self.P, self.Q = self.mcts.gnnhash.get(hash)
            else:                
                Timer.start('gnn')
                with torch.no_grad():
                    self.P, self.Q = self.mcts.gnn(self.graph, self.dist_from_prev, self.dist_to_start)
                Timer.end('gnn')
                self.P = self.P.detach().numpy()
                self.Q = self.Q.detach().numpy()
                self.mcts.gnnhash.save(hash, self.P, self.Q.copy())

            if self.mcts.nodehash.has(hash):
                self.reward_mean, self.reward_std = self.mcts.nodehash.get(hash)
            else:
                # calculate reward mean and std by random sampling
                NUM = min(max(10, 2 * n), 100)
                if self.mcts.performance: NUM = 10
                rewards = np.empty(NUM)
                Timer.start('sample')
                Counter.count('sample')
                for i in range(NUM):
                    rewards[i] = randomplay_tsp(graph, self.dist_from_prev, self.dist_to_start)
                Timer.end('sample')
                self.reward_mean = rewards.mean()
                # std shoud not be 0!
                self.reward_std = rewards.std(ddof=1) + EPS
                assert not np.isnan(self.reward_std)
                self.mcts.nodehash.save(hash, self.reward_mean, self.reward_std)

    def is_end(self):
        return self.graph.shape[0] == 0

    def state_value(self):
        if self.is_end():
            return 0.
        else:
            return self.reward_mean + self.Q.max() * self.reward_std

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / self.reward_std

    def best_child(self):
        n, _ = self.graph.shape
        ucb = self.Q + ALPHA * np.sqrt(self.visit_cnt.sum()) * self.P / (1 + self.visit_cnt)
        return np.argmax(ucb)

    def pi(self, TAU):
        pow_tau = np.power(self.visit_cnt, 1 / TAU)
        assert (self.visit_cnt >= 0).all()
        return pow_tau / pow_tau.sum()
