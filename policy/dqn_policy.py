import numpy as np
import copy

"""
- act: graph -> vertex
"""

class DQNPolicy:
    def __init__(self, dqn):
        self.model = dqn
        self.update()
    
    def act(self, adj, epsilon):
        n, _ = adj.shape
        if np.random.random() < epsilon:
            return np.random.randint(n)
        val = self.model(adj)
        # print("val", val)
        return np.argmax(val.detach().numpy())

    def update(self):
        self.oldmodel = copy.deepcopy(self.model)
