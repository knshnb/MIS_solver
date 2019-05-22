from config import device
import numpy as np
import torch
from utils.graph import read_graph, generate_random_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer import MCTSTrainer
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter

if __name__ == "__main__":
    test_graphs = [read_graph("data/random/1000_2500_{}".format(i)).adj for i in range(5)]

    gnn = GIN3(layer_num=2)
    gnn.to(device)
    trainer = MCTSTrainer(gnn, test_graphs, "test")

    Timer.start('all')

    for i in range(2):
        print("epoch: ", i)
        graph = generate_random_graph(100, 250).adj
        trainer.train(graph, 10 * 0.97 ** i)
        trainer.test()

    Timer.end('all')
    Timer.print()
    Counter.print()

    trainer.save_model()
    trainer.save_test_result()
