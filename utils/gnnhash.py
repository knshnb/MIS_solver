class GNNHash:
    items = {}

    @staticmethod
    def has(hash):
        return hash in GNNHash.items
    
    @staticmethod
    def save(hash, p, q):
        GNNHash.items[hash] = [p, q]
    
    @staticmethod
    def get(hash):
        return GNNHash.items[hash]
    
    @staticmethod
    def clear():
        GNNHash.items = {}
