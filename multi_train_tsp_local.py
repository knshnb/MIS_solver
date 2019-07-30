from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_euc_2d_graph, generate_random_graph_tsp, convert_coordinates_to_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer_tsp import MCTSTSPTrainer
from gin.gin_tsp import GIN3
from utils.timer import Timer
from utils.counter import Counter
import datetime

file_identifier = "tsp"
layer_num = 10
feature = 8
node = 40
iter_p = 5
epoch = 100
train_method = "train1"

file_prefix = "{}_{}_{}_{}_{}_{}_{}_{}".format(file_identifier, layer_num, feature, node, iter_p, epoch, train_method, datetime.datetime.today())

# test_files = ['data/tsp/tsp_001_a280', 'data/tsp/tsp_002_berlin52', 'data/tsp/tsp_003_bier127']

def train(idx):
    np.random.seed()
    torch.manual_seed(idx)
    # test_graphs = [read_euc_2d_graph(f) for f in test_files]
    # test_graphs = [generate_random_graph_tsp(node) for _ in range(1)]
    N = 50
    coords = []
    for i in range(N):
        coords.append([i, 0])
    test_graphs = [convert_coordinates_to_graph(N, coords)]

    gnn = GIN3(layer_num=layer_num, feature=feature)
    gnn.to(device)
    trainer = MCTSTSPTrainer(gnn, test_graphs, "{}_{}th".format(file_prefix, idx))

    Timer.start('all')

    orig_graph, scale = test_graphs[0]
    for i in range(epoch):
        print("epoch: ", i)
        # graph, scale = generate_random_graph_tsp(node)
        graph = orig_graph.adj
        if True:
            Timer.start('test')
            trainer.test()
            Timer.end('test')
            trainer.save_model(i)
            trainer.save_test_result(i)

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
    # print("train start {}".format(file_prefix))
    # pool = Pool()
    # pool.map(train, list(range(8)))
    # pool.close()
    # pool.join()
    train(0)
