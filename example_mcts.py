from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from timer import Timer

if __name__ == "__main__":
    graph0 = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=np.float32)
    graph = read_graph("data/random/100_250_0").adj
    # graph = read_graph("data/random/10_25_0").adj

    gnn = GIN3(layer_num=2)
    gnn.to(device)
    mcts = MCTS(gnn)

    Timer.start('all')

    for i in range(10):
        print("epoch: ", i)
        ans = mcts.search(graph)
        print(ans)
        print("ans mean", np.mean(ans))
        print(mcts.gnn(graph))

        mcts.train(graph, 5 * 0.99 ** i)

    Timer.end('all')
    Timer.print()
