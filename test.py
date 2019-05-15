from gcn.models import GCN2
from policy.dqn_policy import DQNPolicy
from trainer.dqn_trainer import DQNTrainer
from utils.graph import read_graph, generate_random_graph

if __name__ == "__main__":
    gcn = GCN2()
    policy = DQNPolicy(gcn)
    trainer = DQNTrainer(policy)
    g = read_graph("data/random/100_250_0").adj
    trainer.run(g)
