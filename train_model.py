import os
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def train_and_save(train_graph, test_graph, filename, iter=100):
    gnn = GIN3(layer_num=2)
    gnn.to(device)
    mcts = MCTS(gnn)

    Timer.start('all')
    for i in range(iter):
        print("epoch: ", i)
        train_ans = mcts.search(train_graph)
        print(train_ans)
        print("train ans mean", np.mean(train_ans))

        test_ans = mcts.search(test_graph)
        print(test_ans)
        print("test ans mean", np.mean(test_ans))

        print(mcts.gnn(train_graph))
        mcts.train(train_graph, 10 * 0.95 ** i)

    Timer.end('all')
    Timer.print()
    torch.save(gnn.state_dict(), filename)

if __name__ == "__main__":
    graph0 = read_graph("data/random/100_250_0").adj
    graph1 = read_graph("data/random/100_250_1").adj

    filename = "hoge"
    os.makedirs("model", exist_ok=True)
    train_and_save(graph0, graph1, "model/" + filename, iter=100)
