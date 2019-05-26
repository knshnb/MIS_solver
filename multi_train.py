from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph, generate_random_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer import MCTSTrainer
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter

def train(idx):
    np.random.seed()
    torch.manual_seed(idx)
    test_graphs = [read_graph("data/random/100_250_{}".format(i)).adj for i in range(5)]

    gnn = GIN3(layer_num=6)
    gnn.to(device)
    trainer = MCTSTrainer(gnn, test_graphs, "joe_train1_p5_0.98_100_n300m750_{}th".format(idx))

    Timer.start('all')

    for i in range(100):
        print("epoch: ", i)
        graph = generate_random_graph(300, 750).adj
        Timer.start('test')
        trainer.test()
        Timer.end('test')

        Timer.start('train')
        trainer.train1(graph, 10 * 0.98 ** i, iter_p=5)
        Timer.end('train')

    Timer.start('test')
    trainer.test()
    Timer.end('test')

    Timer.end('all')
    Timer.print()
    Counter.print()

    trainer.save_model()
    trainer.save_test_result()

if __name__ == "__main__":
    pool = Pool()
    pool.map(train, list(range(10)))
    pool.close()
    pool.join()
