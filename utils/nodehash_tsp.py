from utils.timer import Timer
from config import use_dense
import numpy as np

class NodeHashTSP:
    def __init__(self):
        self.items = {}

    def hash(self, adj, dist_from_prev, dist_to_start):
        Timer.start('hash')
        n, _ = adj.shape
        ret = 0
        mod = 998244353
        b = 1
        # これ片方だけでよくないか
        for i in range(n):
            ret += adj[0][i] * b
            ret %= mod
            b <<= 1
            if b >= mod:
                b -= mod
        for i in range(n):
            ret += dist_from_prev[i] * b
            ret %= mod
            b <<= 1
            if b >= mod:
                b -= mod
        for i in range(n):
            ret += dist_to_start[i] * b
            ret %= mod
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
