import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts2.mcts_node import MCTSNode
from torchviz import make_dot, make_dot_from_trace

def inspect_params():
    params = list(MCTSNode.gnn.parameters())
    for param in params:
        assert not torch.isnan(param.max()).item()

def visualize(loss):
    make_dot(loss, params=dict(MCTSNode.gnn.named_parameters())).view()
    exit()

def normalize(arr):
    if arr == []: return []
    if len(arr) == 1: return [1.0]
    a = np.array(arr, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-5)
    return a.tolist()

class MCTS_Trainer:
    def __init__(self, gnn):
        MCTSNode.set_gnn(gnn)
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
        self.alpha = 4
        self.tau = 0.5
        self.res = []

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
    
    def get_improved_pi(self, graph, k=2):
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
        self.res.append(reward)
        z = torch.tensor(normalize(self.res)[-1])
        print(z)
        print(MCTSNode.newvisit / MCTSNode.allvisit, MCTSNode.newvisit, MCTSNode.allvisit)
        for i in range(T):
            loss = torch.tensor(0.0)
            for _ in range(batch_size):
                idx = np.random.randint(T)
                n, _ = graphs[idx].shape
                p, v = MCTSNode.gnn(graphs[idx])
                # torchのstdを使うと逆伝播でnanが出ることがある
                # assert not torch.isnan(v.std(unbiased=False)).item()
                # v = (v - v.mean()) / (v.std(unbiased=False) + 1e-5)
                std = v.detach().numpy().std() + 1e-5
                v = (v - v.mean()) / std
                loss += mse(z, v[actions[idx]])
                loss -= (torch.Tensor(pis[idx]) * torch.log(p + 1e-5)).sum()
            loss /= batch_size
            self.optimizer.zero_grad()
            loss.backward()
            if np.random.random() < 0.01:
                inspect_params()
            self.optimizer.step()
            if np.random.random() < 0.01:
                inspect_params()

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
