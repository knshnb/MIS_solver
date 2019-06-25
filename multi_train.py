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

file_identifier = "experiment1"
layer_num = 6
feature = 8
test_graph = "100_250"
node = 100
edge = 250
iter_p = 5
epoch = 50
train_method = "train2"

file_prefix = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(file_identifier, layer_num, feature, test_graph, node, edge, iter_p, epoch, train_method)

def train(idx):
    np.random.seed()
    torch.manual_seed(idx)
    test_graphs = [read_graph("data/random/{}_{}".format(test_graph, i)).adj for i in range(5)]

    gnn = GIN3(layer_num=layer_num, feature=feature)
    gnn.to(device)
    trainer = MCTSTrainer(gnn, test_graphs, "{}_{}th".format(file_prefix, idx))

    Timer.start('all')

    for i in range(epoch):
        print("epoch: ", i)
        graph = generate_random_graph(node, edge).adj
        Timer.start('test')
        trainer.test()
        Timer.end('test')

        Timer.start('train')
        tmp = 0.01 ** (1 / epoch)
        # 10 * tmp^epoch ~= 0.1
        if train_method == "train1":
            trainer.train1(graph, 10 * tmp ** i, iter_p=iter_p)
        elif train_method == "train2":
            trainer.train2(graph, 10 * tmp ** i, iter_p=iter_p)
        else:
            print("no such method")
            assert False
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
    print("train start {}".format(file_prefix))
    pool = Pool()
    pool.map(train, list(range(10)))
    pool.close()
    pool.join()
