from gcn.models import GCN
from policy.gcn_policy import GCNPolicy
from trainer.trainer import Trainer
from utils.graph import read_graph, generate_random_graph

if __name__ == "__main__":
    gcn = GCN()
    policy = GCNPolicy(gcn)
    trainer = Trainer(policy)
    g = generate_random_graph(100, 250).adj
    trainer.train(g, iter=10, batch=10, print_log=True)