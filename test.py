from gcn.models import GCN2
from gin.gin import GIN2
from policy.dqn_policy import DQNPolicy
from trainer.dqn_trainer import DQNTrainer
from utils.graph import *

if __name__ == "__main__":
    gcn = GIN2()
    policy = DQNPolicy(gcn)
    trainer = DQNTrainer(policy)
    # g = read_graph("data/random/100_250_0")
    g = Graph(4, True)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.build()
    # trainer.oracle(g.adj)
    # trainer.policy.update()
    trainer.run(g.adj)
