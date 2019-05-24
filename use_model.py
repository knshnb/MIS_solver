import multiprocessing
from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def best_gin(idx):
    gnn = GIN3(layer_num=6, feature=8)
    gnn.load_state_dict(torch.load("model/modified_p5_{}th.pth".format(idx)))
    gnn.to(device)
    gnn.eval()
    return gnn

def best_gins():
    return [best_gin(idx) for idx in range(10)]

def use_model(t):
    gnn, name, graph = t
    np.random.seed()

    mcts = MCTS(gnn, performance=True)

    Timer.start('all')

    result = mcts.search_for_exp(graph, time_limit=10 * 60, min_iter_num=100)
    print("graph: {}, result: {}".format(name, result))
    print("max: ", max(result))

    Timer.end('all')
    Timer.print()

    return max(result)

if __name__ == "__main__":
    gnns = best_gins()

    filename = "random/10_25_0"
    graph = read_graph("data/" + filename).adj
    print(filename)

    results = {}
    pool = Pool()
    results = pool.map(use_model, [(gnn, filename, graph) for gnn in gnns])
    pool.close()
    pool.join()
    print(results)

    print("file name: {} final max: {}".format(filename, max(results)))

    # for gnn in gnns:
    #     use_model(gnn, filename, graph)
