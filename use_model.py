import multiprocessing
from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def best_gin1():
    gnn = GIN3(layer_num=2, feature=8)
    gnn.load_state_dict(torch.load("model/train1/save_100_100_2.pth"))
    gnn.to(device)
    gnn.eval()
    return gnn

def best_gin2():
    gnn = GIN3(layer_num=6, feature=8)
    gnn.load_state_dict(torch.load("model/decay_train2_p5_0th.pth"))
    gnn.to(device)
    gnn.eval()
    return gnn

def use_model(gnn, name, graph, iter_num=1000):
    # seedを初期化しないと全部同じになってしまう！
    np.random.seed()

    gnn = best_gin2()
    mcts = MCTS(gnn, performance=True)

    Timer.start('all')

    result = mcts.search(graph, iter_num=iter_num)
    # result = mcts.best_search2(graph, TAU=TAU, iter_p=iter_p)
    # result = mcts.best_search2(graph)
    # result = mcts.greedy_v_search(graph)
    print("graph: {}, result: {}".format(name, result))

    Timer.end('all')
    Timer.print()

if __name__ == "__main__":
    graphs = {
        "100_250_0": read_graph("data/random/100_250_0").adj,
        "cora": read_graph("data/cora").adj,
        "siteseer": read_graph("data/citeseer").adj,
    }
    gnns = [
        best_gin1(),
        best_gin2(),
    ]

    print(multiprocessing.cpu_count())
    pool = Pool()
    for name, graph in graphs.items():
        for gnn in gnns:
            pool.apply_async(use_model, args=(gnn, name, graph))
    pool.close()
    pool.join()

    # use_model("cora", 0.3)
