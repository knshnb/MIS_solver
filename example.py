from gcn.models import GCN
from gin.gin import GIN
from policy.gnn_policy import GNNPolicy
from trainer.trainer import Trainer
from utils.graph import read_graph, generate_random_graph

if __name__ == "__main__":
    # gnn = GCN(layer_num=3)
    gnn = GIN(layer_num=5)
    policy = GNNPolicy(gnn)
    trainer = Trainer(policy)
    g = read_graph("data/random/100_250_0").adj
    trainer.train(g, iter=1000, batch=10, print_log=True)
