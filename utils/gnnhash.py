class GNNHash:
    def __init__(self):
        self.items = {}

    def has(self, hash):
        return hash in self.items
    
    def save(self, hash, p, q):
        self.items[hash] = [p, q]
    
    def get(self, hash):
        return self.items[hash].copy()
    
    def clear(self):
        self.items = {}
