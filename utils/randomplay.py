import numpy as np
from config import use_dense

def make_adj_set_sparse(adj):
    n, _ = adj.shape
    row = adj.row
    col = adj.col
    m = row.size
    ss = [set() for _ in range(n)]
    for i in range(m):
        a = row[i]
        b = col[i]
        ss[a].add(b)
        ss[b].add(a)
    return ss

def make_adj_set(adj):
    if not use_dense:
        return make_adj_set_sparse(adj)
    n, _ = adj.shape
    ss = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                ss[i].add(j)
                ss[j].add(i)
    return ss

def randomplay(ss):
    n = len(ss)
    vs = [i for i in range(n)]
    np.random.shuffle(vs)
    ng = set()
    ret = 0
    for v in vs:
        if not v in ng:
            ng.add(v)
            ng |= ss[v]
            ret += 1
    return ret
