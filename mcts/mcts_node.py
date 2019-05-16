import copy
import numpy as np
from environ.mis_env import MISEnv
from gin.gin import GIN3

ALPHA = 50
TAU = 2
class MCTSNode:
    gnn = GIN3()
    def __init__(self, graph, parent=None):
        n, _ = graph.shape
        self.graph = graph
        self.parent = parent
        self.children = [None for _ in range(n)]

        self.visit_cnt = np.zeros(n)
        self.rewards = []
        self.policy = None
        self.value = None

    def is_leaf(self):
        return self.graph.shape[0] == 0

    def best_child(self):
        if self.policy is None:
            assert self.value is None
            self.policy, self.value = MCTSNode.gnn(self.graph)
            self.policy = self.policy.detach().numpy()
            self.value = self.value.detach().numpy()

        for i in range(len(self.children)):
            if self.children[i] is not None:
                self.value[i] = np.mean(self.children[i].rewards)

        ucb = self.value + ALPHA * self.policy / (1 + self.visit_cnt)
        return np.argmax(ucb)

    def explore(self, v):
        self.visit_cnt[v] += 1
        if self.children[v] is None:
            env = MISEnv()
            env.set_graph(self.graph)
            next_graph, r, done, info = env.step(v)
            self.children[v] = MCTSNode(next_graph, parent=self)
        return self.children[v]

    def pi(self):
        pow_tau = np.power(self.visit_cnt, 1 / TAU)
        assert (self.visit_cnt >= 0).all()
        return pow_tau / pow_tau.sum()

def rollout(root_node):
    node = root_node
    ans = 0
    while not node.is_leaf():
        v = node.best_child()
        node = node.explore(v)
        ans += 1
    node.rewards.append(0)
    for r in range(ans):
        node = node.parent
        node.rewards.append(r + 1)
    assert node is root_node
    return ans

def search(root_node, iter_num=100):
    ans = []
    for i in range(iter_num):
        ans.append(rollout(root_node))
    return ans
