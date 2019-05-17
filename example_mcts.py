import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts.mcts_node import MCTSNode, rollout, search
from utils.graph import read_graph

def mcts(root_node, iter_p=2):
    n, _ = root_node.graph.shape
    for i in range(n * iter_p):
        rollout(root_node)
    action = np.random.choice(n, p=root_node.pi())
    return action

def train(graph):
    optimizer = torch.optim.Adam(MCTSNode.gnn.parameters(), lr=0.01)
    env = MISEnv()
    env.set_graph(graph)

    nodes = []
    done = False
    while not done:
        node = MCTSNode(graph)
        action = mcts(node)
        nodes.append(node)
        graph, reward, done, info = env.step(action)
    for i in range(len(nodes)):
        optimizer.zero_grad()
        for batch in range(10):
            idx = np.random.randint(len(nodes))
            node = nodes[idx]
            p, v = MCTSNode.gnn(node.graph)

            mse = torch.nn.MSELoss()
            pi = torch.tensor(node.pi(), dtype=torch.float32)
            a = mse(v, torch.tensor(node.value))
            # TODO: use cross entropy loss for b
            b = mse(pi, p)
            loss = a + b
            loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # graph = np.array([
    #     [0, 1, 0, 1],
    #     [1, 0, 1, 1],
    #     [0, 1, 0, 1],
    #     [1, 1, 1, 0],
    # ], dtype=np.float32)
    graph = read_graph("data/random/100_250_0").adj
    for i in range(1000):
        train(graph)
        print(search(MCTSNode(graph)))
        print(MCTSNode.gnn(graph))