from timer import Timer

class NodeHash:
    items = {}

    @staticmethod
    def hash(adj):
        Timer.start('hash')
        n, _ = adj.shape
        ret = 0
        mod = 998244353
        b = 1
        for i in range(n):
            for j in range(i + 1, n):
                ret += (1 + adj[i][j]) * b
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