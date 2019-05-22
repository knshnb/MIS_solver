import multiprocessing
from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def best_gin():
    gnn = GIN3(layer_num=5, feature=9)
    gnn.load_state_dict(torch.load("model/train2/new_train2_optuna_tmp_100_5_9_0.987863061152576.pth"))
    gnn.to(device)
    gnn.eval()
    return gnn

def use_model(graph, TAU, iter_p=1):
    # seedを初期化しないと全部同じになってしまう！
    np.random.seed()

    gnn = best_gin()
    mcts = MCTS(gnn)

    Timer.start('all')

    # result = mcts.search(graph, iter_num=10)
    # result = mcts.best_search2(graph, TAU=TAU, iter_p=iter_p)
    result = mcts.policy_search(graph)
    print("TAU: {}, result: {}".format(TAU, result))

    Timer.end('all')
    Timer.print()

if __name__ == "__main__":
    graph0 = read_graph("data/random/1000_2500_0").adj
    graph1 = read_graph("data/random/100_250_0").adj
    cora = read_graph("data/cora").adj

    print(multiprocessing.cpu_count())
    pool = Pool()
    for i in range(32):
        TAU = 0.1 + 0.5 * 0.8 ** i
        pool.apply_async(use_model, args=(graph1, TAU))
    pool.close()
    pool.join()

    # use_model(graph1, 0.3)
