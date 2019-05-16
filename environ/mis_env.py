import numpy as np
from config import *
from utils.timer import Timer
# xp = chainer.cuda.cupy if use_gpu else np
xp = np

"""
graph is represented by adjacency matrix
- original_graph: original graph
- A: current subgraph
- to_vertex: index of A -> index of graph
- ans: current answer for MIS
- reward: current number of vertices in answer
"""
class MISEnv:
    def __init__(self):
        pass

    def set_graph(self, graph):
        self.original_graph = graph
        self.reset()

    def reset(self):
        self.A = self.original_graph
        self.to_vertex = xp.arange(self.A.shape[0], dtype=xp.int)
        self.ans = []
        self.reward = 0  # number of vertices already counted in the solution
        return self.A

    def step(self, action):  # action: index of a vertex
        Timer.start('env')
        self.ans.append(self.to_vertex[action])
        # delete neighbors
        mask = self.A[action] == 0
        # delete itself
        mask[action] = False
        self.to_vertex = self.to_vertex[mask]
        self.A = self.A[mask][:, mask]
        self.reward += 1
        assert self.A.shape[0] == 0 or self.A.shape[0] == self.A.shape[1]
        Timer.end('env')
        return self.A, self.reward, self.A.shape[0] == 0, {'ans': self.ans}
