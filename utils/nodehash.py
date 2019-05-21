from utils.timer import Timer
from config import use_dense
import numpy as np

class NodeHash:
    def __init__(self, MAX_N):
        Timer.start('init hash')
        self.items = {}
        self.memo = {}
        acc = 0
        mod = 998244353
        save = {}
        b = 1
        for i in range(MAX_N + 1):
            save[i * (i - 1) // 2] = i
        for i in range(MAX_N * (MAX_N - 1) // 2 + 1):
            if i in save:
                self.memo[save[i]] = acc
            acc += b
            if acc >= mod:
                acc -= mod
            b <<= 1
            if b >= mod:
                b -= mod
        Timer.end('init hash')

    def hash_sparse(self, adj):
        Timer.start('hash')
        n, _ = adj.shape
        row = adj.row
        col = adj.col
        val = adj.data.astype(int)
        m = row.size
        mod = 998244353
        ret = self.memo[n]
        for i in range(m):
            r = row[i]
            c = col[i]
            if r > c: continue
            k = (2 * n - 1 - r) * r // 2 + c - 1 - r
            k = int(k)
            ret -= pow(2, k, mod)
            ret += (1 + val[i]) * pow(2, k, mod)
            ret %= mod
        Timer.end('hash')
        return ret, n

    def hash(self, adj):
        if not use_dense:
            return self.hash_sparse(adj)
        Timer.start('hash')
        n, _ = adj.shape
        ret = 0
        mod = 998244353
        A = adj.astype(int)
        b = 1
        for i in range(n):
            for j in range(i + 1, n):
                ret += (1 + A[i][j]) * b
                # if you use weighted matrix, % here
                while ret >= mod:
                    ret -= mod
                b <<= 1
                if b >= mod:
                    b -= mod
        Timer.end('hash')
        return ret, n

    def has(self, hash):
        return hash in self.items
    
    def save(self, hash, reward_mean, reward_std):
        self.items[hash] = [reward_mean, reward_std]
    
    def get(self, hash):
        return self.items[hash]
