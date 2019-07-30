import numpy as np
from utils.timer import Timer
xp = np

class TSPEnv:
    def __init__(self):
        pass

    def initialize_with_start_node(self, A, start_node):
        n = A.shape[0]
        mask = np.arange(n) != start_node
        dist_from_prev = A[start_node][mask]
        dist_to_start = A[start_node][mask]
        A = A[mask][:,mask]
        return A, dist_from_prev, dist_to_start

    def set_graph(self, graph, start_node, dist_from_prev, dist_to_start):
        self.start_node = start_node
        self.original_graph = graph
        self.original_dist_from_prev = dist_from_prev
        self.original_dist_to_start = dist_to_start
        self.reset()

    def reset(self):
        self.A = self.original_graph
        self.dist_from_prev = self.original_dist_from_prev
        self.dist_to_start = self.original_dist_to_start
        n = self.A.shape[0]
        vs = np.arange(n + 1)
        mask = vs != self.start_node
        vs = vs[mask]
        self.to_vertex = vs
        self.ans = []
        return self.A, self.dist_from_prev, self.dist_to_start

    def step(self, action):  # action: index of a vertex
        Timer.start('env')
        n = self.A.shape[0]
        self.ans.append(self.to_vertex[action])
        dif = 0
        dif -= self.dist_from_prev[action]

        if n == 1:
            dif -= self.dist_to_start[action]
        
        mask = np.arange(n) != action
        self.to_vertex = self.to_vertex[mask]
        self.dist_from_prev = self.A[action][mask]
        self.dist_to_start = self.dist_to_start[mask]
        self.A = self.A[mask][:, mask]

        assert self.A.shape[0] == 0 or self.A.shape[0] == self.A.shape[1]
        Timer.end('env')
        return self.A, self.dist_from_prev, self.dist_to_start, dif, self.A.shape[0] == 0, {'ans': self.ans}
