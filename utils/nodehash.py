from timer import Timer
from config import use_dense
import numpy as np

class NodeHash:
    items = {}
    memo = {}

    @staticmethod
    def init(MAX_N):
        Timer.start('init hash')
        acc = 0
        mod = 998244353
        save = {}
        b = 1
        for i in range(MAX_N + 1):
            save[i * (i - 1) // 2] = i
        for i in range(MAX_N * (MAX_N - 1) // 2 + 1):
            if i in save:
                NodeHash.memo[save[i]] = acc
            acc += b
            if acc >= mod:
                acc -= mod
            b <<= 1
            if b >= mod:
                b -= mod
        Timer.end('init hash')

    @staticmethod
    def hash_sparse(adj):
        n, _ = adj.shape
        row = adj.row
        col = adj.col
        val = adj.data.astype(int)
        m = row.size
        mod = 998244353
        ret = NodeHash.memo[n]
        for i in range(m):
            r = row[i]
            c = col[i]
            if r > c: continue
            k = (2 * n - 1 - r) * r // 2 + c - 1 - r
            k = int(k)
            ret -= pow(2, k, mod)
            ret += (1 + val[i]) * pow(2, k, mod)
            ret %= mod
        return ret, n

    @staticmethod
    def hash(adj):
        if not use_dense:
            return NodeHash.hash_sparse(adj)
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

    @staticmethod
    def has(hash):
        return hash in NodeHash.items
    
    @staticmethod
    def save(hash, reward_mean, reward_std):
        NodeHash.items[hash] = [reward_mean, reward_std]
    
    @staticmethod
    def get(hash):
        return NodeHash.items[hash]
