from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def use_model(graph, filename, iter=100):
    gnn = GIN3(layer_num=2)
    gnn.load_state_dict(torch.load(filename))
    gnn.to(device)
    gnn.eval()
    mcts = MCTS(gnn)

    Timer.start('all')

    result = mcts.search(graph, iter_num=iter)
    print(result)

    Timer.end('all')
    Timer.print()

if __name__ == "__main__":
    graph0 = read_graph("data/random/1000_2500_0").adj
    graph1 = read_graph("data/random/100_250_1").adj

    use_model(graph0, "model/hoge", iter=100)
