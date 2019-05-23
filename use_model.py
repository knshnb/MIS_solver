import multiprocessing
from multiprocessing import Pool
from config import device
import numpy as np
import torch
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from utils.timer import Timer

# layer_num = 6, feature = 8
best_model_names = [
    "model/decay_train2_p5_0th.pth",
    "model/final_train2_p5_9th.pth",
    "model/train1_100_p5_0th.pth",
    "model/train1_100_p2_1th.pth",
    "model/train1_100_p2_4th.pth",
    "model/train1_100_p2_3th.pth",
    "model/final_train1_p5_8th.pth",
    "model/nemui_train1_p5_16th.pth",
    "model/final_train1_p5_1th.pth",
    "model/final_train1_p5_3th.pth",
]

def best_gin1():
    gnn = GIN3(layer_num=2, feature=8)
    gnn.load_state_dict(torch.load("model/train1/save_100_100_2.pth"))
    gnn.to(device)
    gnn.eval()
    return gnn

def best_gin2(filename):
    gnn = GIN3(layer_num=6, feature=8)
    gnn.load_state_dict(torch.load(filename))
    gnn.to(device)
    gnn.eval()
    return gnn

def use_model(gnn, name, graph, iter_num=100):
    # seedを初期化しないと全部同じになってしまう！
    np.random.seed()

    mcts = MCTS(gnn, performance=True)

    Timer.start('all')

    result = mcts.search(graph, iter_num=iter_num)
    print("graph: {}, result: {}".format(name, result))

    Timer.end('all')
    Timer.print()

if __name__ == "__main__":
    graphs = {
        # "10_25_0": read_graph("data/random/10_25_0").adj,
        # "100_250_0": read_graph("data/random/100_250_0").adj,
        # "cora": read_graph("data/cora").adj,
        # "siteseer": read_graph("data/citeseer").adj,

        # "pubmed": read_graph("data/pubmed").adj,

        # "1000_2500_0": read_graph("data/random/1000_2500_0").adj,
        # "1000_2500_1": read_graph("data/random/1000_2500_1").adj,
        # "1000_2500_2": read_graph("data/random/1000_2500_2").adj,
        # "1000_2500_3": read_graph("data/random/1000_2500_3").adj,
        # "1000_2500_4": read_graph("data/random/1000_2500_4").adj,
        # "1000_2500_5": read_graph("data/random/1000_2500_5").adj,
        # "1000_2500_6": read_graph("data/random/1000_2500_6").adj,
        # "1000_2500_7": read_graph("data/random/1000_2500_7").adj,
        # "1000_2500_8": read_graph("data/random/1000_2500_8").adj,
        # "1000_2500_9": read_graph("data/random/1000_2500_9").adj,

        # "gen400-p0.9-55": read_graph("data/gen400-p0.9-55").adj,

        # "ego-facebook": read_graph("data/facebook2").adj,
        # "ego-gplus": read_graph("data/gplus2").adj,
        # "ego-twitter": read_graph("data/twitter2").adj,

        "gen200_p0.9_44.clq-compliment": read_graph("data/gen200_p0.9_44.clq-compliment").adj,
        "gen200_p0.9_55.clq-compliment": read_graph("data/gen200_p0.9_55.clq-compliment").adj,
        "gen400_p0.9_55.clq-compliment": read_graph("data/gen400_p0.9_55.clq-compliment").adj,
        "gen400_p0.9_65.clq-compliment": read_graph("data/gen400_p0.9_65.clq-compliment").adj,
        "gen400_p0.9_75.clq-compliment": read_graph("data/gen400_p0.9_75.clq-compliment").adj,

        "btcalpha": read_graph("data/btcalpha").adj,
        "btcotc": read_graph("data/btcotc").adj,
    }
    gnns = [best_gin2(name) for name in best_model_names]
    gnns.append(best_gin1())

    print(multiprocessing.cpu_count())
    pool = Pool()
    for name, graph in graphs.items():
        for gnn in gnns:
            pool.apply_async(use_model, args=(gnn, name, graph))
    pool.close()
    pool.join()

    # for name, graph in graphs.items():
    #     for gnn in gnns:
    #         use_model(gnn, name, graph)
