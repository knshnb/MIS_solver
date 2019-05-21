from utils.nodehash import NodeHash
from utils.graph import Graph
import numpy as np

# SET use_dense == True when running this code!!!!!!!
NodeHash.init(50)

for _ in range(100):
    n = np.random.randint(1, 51)
    sparse = Graph(n, False)
    dense = Graph(n, True)

    for i in range(2 * n):
        adj = [set() for _ in range(n)]
        a = np.random.randint(n)
        b = np.random.randint(n)
        if a != b and b not in adj[a]:
            sparse.add_edge(a, b)
            dense.add_edge(a, b)
            adj[a].add(b)
            adj[b].add(a)
    
    sparse.build()
    dense.build()

    hash1 = NodeHash.hash_sparse(sparse.adj)
    hash2 = NodeHash.hash(dense.adj)

    assert hash1 == hash2
    # print(hash1, hash2)
