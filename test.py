from gin.gin import GIN3
from mcts2.mcts_trainer import MCTS_Trainer
from utils.graph import *

if __name__ == '__main__':
    graph0 = Graph(4, is_dense=True)
    graph0.add_edge(0, 1)
    graph0.add_edge(0, 2)
    graph0.add_edge(0, 3)
    graph0.add_edge(2, 1)
    graph0.add_edge(2, 3)
    graph0.build()

    graph1 = read_graph("data/random/100_250_0").adj
    # graph1 = read_graph("data/random/100_250_1").adj
    # graph1 = graph0.adj

    gnn = GIN3(layer_num=2)
    trainer = MCTS_Trainer(gnn)
    
    # print("no train:", trainer.test(graph1))
    for i in range(1000):
        print("epoch:", i + 1)
        trainer.train(graph1)
        print("ans:", trainer.test(graph1))
    