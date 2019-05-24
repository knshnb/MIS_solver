import numpy as np
from scipy.sparse import coo_matrix
from config import *
# xp = chainer.cuda.cupy if use_gpu else np
xp = np

class Graph:
    def __init__(self, n, is_dense):
        self.n = n
        self.m = 0
        self.is_dense = is_dense
        self.tmp = [set() for _ in range(n)]
    
    def add_edge(self, a, b):
        assert 0 <= a < self.n and 0 <= b < self.n
        self.tmp[a].add(b)
        self.tmp[b].add(a)

    def build(self):
        if self.is_dense:
            self.adj = xp.zeros((self.n, self.n), dtype=xp.float32)
            for a in range(self.n):
                for b in self.tmp[a]:
                    if a < b:
                        self.adj[a, b] = 1
                        self.adj[b, a] = 1
                        self.m += 1
        else:
            x = []
            y = []
            for a in range(self.n):
                for b in self.tmp[a]:
                    if a < b:
                        x.append(a)
                        y.append(b)
                        x.append(b)
                        y.append(a)
                        self.m += 1
            self.adj = coo_matrix((np.ones(2 * self.m, dtype=np.float32), (np.array(x), np.array(y))), shape=(self.n, self.n))

def generate_random_graph(n, m):
    g = Graph(n, use_dense)
    acc = 0
    while acc < m:
        # don't use xp!
        a = np.random.randint(n)
        b = np.random.randint(n)
        if a != b and a not in g.tmp[b]:
            g.add_edge(a, b)
            acc += 1
    g.build()
    assert g.m == m
    return g

def read_graph(filename):
    f = open(filename)
    text = f.readlines()
    n, m = map(int, text[0].split())
    g = Graph(n, use_dense)
    print("Start reading file {}".format(filename))
    for i in range(m):
        a, b = map(int, text[1 + i].split())
        g.add_edge(a, b)
    print("Finish reading file {}".format(filename))
    g.build()
    f.close()
    return g

def write_graph(graph, filename):
    n, _ = graph.adj.shape
    if graph.is_dense:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if graph.adj[i, j]:
                    edges.append((i, j))
    else:
        edges = []
        for i in range(graph.adj.row.size):
            a = graph.adj.row[i]
            b = graph.adj.col[i]
            if a < b:
                edges.append((a, b))
    f = open(filename, 'w')
    m = len(edges)
    f.write("{} {}\n".format(n, m))
    for i in range(m):
        f.write("{} {}\n".format(edges[i][0], edges[i][1]))
    f.close()

def read_test_graphs(size):
    return [read_graph("data/random/{}_{}_{}".format(size, int(size * 2.5), idx)).adj for idx in range(10)]

if __name__ == "__main__":
    # (n, m)
    GRAPHS = [
        (10000, 25000),
        (1000, 2500),
        (100, 250),
        (10, 25),
    ]
    for graph in GRAPHS:
        for idx in range(10):
            g = generate_random_graph(graph[0], graph[1])
            write_graph(g, "data/random/{}_{}_{}".format(graph[0], graph[1], idx))
