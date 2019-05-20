import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts.mcts_node import MCTSNode
from utils.graph import read_graph
from timer import Timer
from utils.gnnhash import GNNHash

EPS = 1e-30  # cross entropy lossをpi * log(EPS + p)で計算 (log(0)回避)

class MCTS:
    def __init__(self, gnn):
        MCTSNode.set_gnn(gnn)
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
        self.gnn = gnn

    # parantのQ(s,a), N(s,a)を更新
    def update_parant(self, node, V):
        par = node.parent
        normalized_V = par.normalize_reward(V)
        if par.visit_cnt[node.idx] == 0:
            # Qが初期値から一度も更新されていない
            par.Q[node.idx] = normalized_V
        else:
            self.update_Q(par, normalized_V, node.idx, method="mean")
        par.visit_cnt[node.idx] += 1

    def update_Q(self, node, V, idx, method="mean"):
        if method == "mean":
            node.Q[idx] = (node.Q[idx] * node.visit_cnt[idx] + V) / (node.visit_cnt[idx] + 1)
        elif method == "max":
            node.Q[idx] = max(node.Q[idx], V)
        elif method == "min":
            node.Q[idx] = min(node.Q[idx], V)
        else:
            assert False

    def rollout(self, root_node, stop_at_leaf=False):
        node = root_node
        v = -1
        finish = False
        while not finish:
            if node.is_end(): break
            v = node.best_child()
            if node.children[v] is None:
                env = MISEnv()
                env.set_graph(node.graph)
                next_graph, r, done, info = env.step(v)
                node.children[v] = MCTSNode(next_graph, idx=v, parent=node)
                if stop_at_leaf:
                    finish = True
            node = node.children[v]

        # Vをrootに向かって伝播させていく
        V = node.state_value()
        while node is not root_node:
            V += 1
            self.update_parant(node, V)
            node = node.parent
        return V

    # MCTSによって改善されたpiを返す
    def get_improved_pi(self, root_node, TAU, iter_p=2):
        assert not root_node.is_end()
        n, _ = root_node.graph.shape
        for i in range(max(100, n * iter_p)):
            self.rollout(root_node)
        return root_node.pi(TAU)

    def train(self, graph, TAU, batch_size=10):
        GNNHash.clear()
        mse = torch.nn.MSELoss()
        env = MISEnv()
        env.set_graph(graph)

        graphs = []
        actions = []
        pis = []
        means = []
        stds = []
        done = False
        while not done:
            n, _ = graph.shape
            node = MCTSNode(graph)
            means.append(node.reward_mean)
            stds.append(node.reward_std)
            pi = self.get_improved_pi(node, TAU)
            action = np.random.choice(n, p=pi)
            graphs.append(graph)
            actions.append(action)
            pis.append(pi)
            graph, reward, done, info = env.step(action)

        T = len(graphs)
        idxs = [i for i in range(T)]
        np.random.shuffle(idxs)
        i = 0
        while i < T:
            size = min(batch_size, T - i)
            self.optimizer.zero_grad()
            loss = torch.Tensor([0])
            for j in range(i, i + size):
                idx = idxs[j]
                Timer.start('gnn')
                p, v = MCTSNode.gnn(graphs[idx])
                Timer.end('gnn')
                n, _ = graphs[idx].shape
                # mean, stdを用いて正規化
                z = torch.tensor(((T - idx) - means[idx]) / stds[idx])
                loss += mse(z, v[actions[idx]]) - (torch.tensor(pis[idx]) * torch.log(p + EPS)).sum()
            loss /= size
            loss.backward()
            self.optimizer.step()
            i += size

    def search(self, graph, iter_num=100):
        root_node = MCTSNode(graph)
        ans = []
        for i in range(iter_num):
            ans.append(self.rollout(root_node))
        return ans
