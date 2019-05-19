import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts2.mcts_node import MCTSNode

def normalize(arr):
    if arr == []: return []
    a = np.array(arr, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-5)
    return a.tolist()

class MCTS_Trainer:
    def __init__(self, gnn):
        MCTSNode.set_gnn(gnn)
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
        self.alpha = 4
        self.tau = 0.5

    # n_rewards: nodeのQを計算するときに正規化する指標に使う過去のreward. 正規化されたもの
    # idx: 今のrolloutが何番目の試行か。0-indexed
    def rollout(self, node, n_rewards, idx):
        reward = 0
        while 1:
            if node.is_end(): break
            v = node.best_ucb(self.alpha, n_rewards)
            if node.children[v] is None:
                env = MISEnv()
                env.set_graph(node.graph)
                graph, *_ = env.step(v)
                node.children[v] = MCTSNode(graph, idx=v, parent=node)
            node = node.children[v]
            reward += 1
        # normalize reward
        while node.parent is not None:
            v = node.idx
            assert v >= 0
            node = node.parent
            node.cnt[v] += 1
            node.Qidx[v].append(idx)
        return reward
    
    def get_improved_pi(self, graph, k=5):
        root = MCTSNode(graph)
        rewards = []
        assert not root.is_end()
        for i in range(graph.shape[0] * k):
            n_rewards = normalize(rewards)
            rewards.append(self.rollout(root, n_rewards, i))
        return root.pi(self.tau)
    
    def train(self, graph, batch_size=10):
        mse = torch.nn.MSELoss()
        env = MISEnv()
        env.set_graph(graph)
        graphs = []
        actions = []
        pis = []
        done = False
        while not done:
            n, _ = graph.shape
            pi = self.get_improved_pi(graph)
            action = np.random.choice(n, p=pi)
            graphs.append(graph)
            actions.append(action)
            pis.append(pi)
            graph, reward, done, info = env.step(action)
        
        T = len(graphs)
        z = torch.tensor(1)
        for i in range(T):
            loss = torch.zeros(1)
            for _ in range(batch_size):
                idx = np.random.randint(T)
                n, _ = graphs[idx].shape
                p, v = MCTSNode.gnn(graphs[idx])
                if n == 1:
                    # make v zero ()
                    v = (v - v.mean()) / 1e-5
                else:
                    v = (v - v.mean()) / (v.std() + 1e-5)
                loss += mse(z, v[actions[idx]])
                loss -= (torch.Tensor(pis[idx]) * torch.log(p + 1e-5)).sum()
            loss /= batch_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self, graph):
        env = MISEnv()
        env.set_graph(graph)
        while 1:
            n, _ = graph.shape
            pi = self.get_improved_pi(graph)
            action = np.random.choice(n, p=pi)
            graph, reward, done, info = env.step(action)
            if done:
                return reward
