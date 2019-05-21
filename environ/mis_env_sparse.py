import numpy as np
from scipy.sparse import coo_matrix
from config import *
from timer import Timer

"""
graph is represented by adjacency matrix
- original_graph: original graph
- A: current subgraph
- to_vertex: index of A -> index of graph
- ans: current answer for MIS
- reward: current number of vertices in answer
"""
class MISEnv_Sparse:
    def __init__(self):
        pass

    def set_graph(self, graph):
        self.original_graph = graph
        self.reset()
    
    def reset(self):
        self.A = self.original_graph
        self.to_vertex = np.arange(self.A.shape[0], dtype=np.int)
        self.ans = []
        self.reward = 0
        return self.A
    
    def step(self, action):
        Timer.start('env')
        self.ans.append(self.to_vertex[action])
        self.reward += 1
        n, _ = self.A.shape
        row = self.A.row
        col = self.A.col
        m = row.size
        ss = [set() for _ in range(n)]
        for i in range(m):
            a = int(row[i])
            b = int(col[i])
            ss[a].add(b)
            ss[b].add(a)
        mask = np.full(n, True)
        mask[action] = False
        for i in range(n):
            if i in ss[action]:
                mask[i] = False
        self.to_vertex = self.to_vertex[mask]
        x = []
        y = []
        N = 0
        M = 0
        mp = {}
        for i in range(n):
            if mask[i]:
                mp[i] = N
                N += 1
        for i in range(m):
            a = int(row[i])
            b = int(col[i])
            if mask[a] and mask[b]:
                x.append(mp[a])
                y.append(mp[b])
                M += 1
        self.A = coo_matrix((np.ones(M, dtype=np.float32), (np.array(x), np.array(y))), shape=(N, N))
        Timer.end('env')
        return self.A, self.reward, N == 0, {'ans': self.ans}
