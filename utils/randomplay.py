import numpy as np

def randomplay(adj):
    n, _ = adj.shape
    ss = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                ss[i].add(j)
                ss[j].add(i)
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
