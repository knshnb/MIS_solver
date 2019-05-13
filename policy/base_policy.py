import numpy as np

"""
- act: graph -> (vertex, probability)
    return vertex and probability based on the policy (mix of best_act and predict_act)
- best_act: graph -> vertex
    return the best vertex if exist else -1
- predict_act: graph -> (vertex, probability)
    return vertex and probability based on the policy
"""
class BasePolicy:
    def best_act(self, graph):
        degrees = graph.sum(axis=0)
        return degrees.argmin() if degrees.min() <= 1 else -1

    def predict_act(self, graph):
        return 0, 1.0

    def act(self, graph):
        ba = self.best_act(graph)
        return (ba, 1.0) if ba != -1 else self.predict_act(graph)
