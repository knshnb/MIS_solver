from gcn.models import GCN
from policy.gcn_policy import GCNPolicy
from trainer.trainer import Trainer
from utils.graph import read_graph, generate_random_graph

if __name__ == "__main__":
    gcn = GCN(layer_num=10)
    policy = GCNPolicy(gcn)
    trainer = Trainer(policy)
    g = read_graph("data/random/100_250_0").adj
    trainer.train(g, iter=1000, batch=10, print_log=True)
