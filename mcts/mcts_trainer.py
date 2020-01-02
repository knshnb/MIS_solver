import os
import pickle
import numpy as np
from config import use_dense
import torch
from environ.mis_env import MISEnv
from environ.mis_env_sparse import MISEnv_Sparse
from mcts.mcts import MCTS
from utils.graph import read_graph
from utils.timer import Timer
from utils.gnnhash import GNNHash
from utils.nodehash import NodeHash


class MCTSTrainer:
    def __init__(self, gnn, test_graphs, filename):
        self.mcts = MCTS(gnn)
        self.test_graphs = test_graphs
        self.test_result = []
        self.filename = filename

    # rollout until the end
    def train1(self, graph, TAU, batch_size=10, iter_p=2):
        self.mcts.train(graph, TAU, batch_size=batch_size, iter_p=iter_p)

    # rollout only until leaf
    def train2(self, graph, TAU, batch_size=10, iter_p=2):
        self.mcts.train(graph, TAU, batch_size=batch_size,
                        stop_at_leaf=True, iter_p=2)

    def test(self):
        result = [self.mcts.search(graph) for graph in self.test_graphs]
        print(result)
        self.test_result.append(result)

    def save_test_result(self):
        os.makedirs("log", exist_ok=True)
        with open("log/{}.pickle".format(self.filename), mode="wb") as f:
            pickle.dump(self.test_result, f)

    def save_model(self):
        os.makedirs("model", exist_ok=True)
        torch.save(self.mcts.gnn.state_dict(),
                   "model/{}.pth".format(self.filename))
