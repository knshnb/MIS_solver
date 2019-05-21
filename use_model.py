import multiprocessing
from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

def use_model(graph, filename, TAU, iter_p=1):
    # seedを初期化しないと全部おなじになる！
    np.random.seed()

    gnn = GIN3(layer_num=2)
    gnn.load_state_dict(torch.load(filename))
    gnn.to(device)
    gnn.eval()
    mcts = MCTS(gnn)

    Timer.start('all')

    result = mcts.best_search(graph, TAU=TAU, iter_p=iter_p)
    print("TAU: {}, result: {}".format(TAU, result))

    Timer.end('all')
    Timer.print()

if __name__ == "__main__":
    graph0 = read_graph("data/random/1000_2500_0").adj
    graph1 = read_graph("data/random/100_250_1").adj

    print(multiprocessing.cpu_count())
    pool = Pool()
    for i in range(8):
        TAU = 0.1 + 0.5 * 0.8 ** i
        pool.apply_async(use_model, args=(graph1, "model/hoge", TAU))
    pool.close()
    pool.join()
