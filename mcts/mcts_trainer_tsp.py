import os
import pickle
import numpy as np
import torch
from environ.tsp_env import TSPEnv
from mcts.mcts_tsp import MCTSTSP
from utils.graph import read_euc_2d_graph
from utils.timer import Timer
from utils.gnnhash import GNNHash
from utils.nodehash_tsp import NodeHashTSP

class MCTSTSPTrainer:
    def __init__(self, gnn, test_graphs, filename):
        self.mcts = MCTSTSP(gnn)
        self.test_graphs = test_graphs
        self.test_result = []
        self.filename = filename

    # rollout until the end
    def train1(self, graph, TAU, batch_size=10, iter_p=2):
        self.mcts.train(graph, TAU, batch_size=batch_size, iter_p=iter_p)

    # rollout only until leaf
    def train2(self, graph, TAU, batch_size=10, iter_p=2):
        self.mcts.train(graph, TAU, batch_size=batch_size, stop_at_leaf=True, iter_p=2)

    def test(self):
        result = [self.mcts.search(graph) for graph in self.test_graphs]
        print(result)
        self.test_result.append(result)

    def save_test_result(self):
        os.makedirs("log", exist_ok=True)
        with open("log/{}.pickle".format(self.filename), mode="wb") as f:
            pickle.dump(self.test_result, f)

    def save_model(self, suffix='final'):
        os.makedirs("model", exist_ok=True)
        torch.save(self.mcts.gnn.state_dict(), "model/{}_{}.pth".format(self.filename, suffix))
