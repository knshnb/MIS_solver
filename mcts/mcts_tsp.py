import time
import numpy as np
import torch
from environ.tsp_env import TSPEnv
from mcts.tsp_mcts_node import TSPMCTSNode
from utils.timer import Timer
from utils.gnnhash import GNNHash
from utils.nodehash_tsp import NodeHashTSP

EPS = 1e-30  # cross entropy loss: pi * log(EPS + p) (in order to avoid log(0))

class MCTSTSP:
    def __init__(self, gnn, performance=False):
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.003, weight_decay=1e-6)
        self.gnn = gnn
        self.nodehash = NodeHashTSP()
        self.gnnhash = GNNHash()
        # max reward of root in rollout
        self.root_max = -1e100
        self.performance = performance

    # update Q(s,a), N(s,a) of parent
    def update_parent(self, node, V):
        par = node.parent
        normalized_V = par.normalize_reward(V)
        if par.visit_cnt[node.idx] == 0:
            # Q is an initial value
            par.Q[node.idx] = normalized_V
        else:
            self.update_Q(par, normalized_V, node.idx, method="mean")
        par.visit_cnt[node.idx] += 1

    def update_Q(self, node, V, idx, method):
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
        dists = []
        while not finish:
            if node.is_end(): break
            v = node.best_child()
            if node.graph.shape[0] == 1:
                dists.append(node.dist_from_prev[v] + node.dist_to_start[v])
            else:
                dists.append(node.dist_from_prev[v])
            if node.children[v] is None:
                env = TSPEnv()
                # start_nodeどっかで保持しておきたいね。とりあえずここは無理やり0にしておく
                env.set_graph(node.graph, 0, node.dist_from_prev, node.dist_to_start)
                next_graph, next_dist_from_prev, next_dist_to_start, r, done, info = env.step(v)
                node.children[v] = TSPMCTSNode(next_graph, next_dist_from_prev, next_dist_to_start, self, idx=v, parent=node)
                if stop_at_leaf:
                    finish = True
            node = node.children[v]

        # backpropagate V
        V = node.state_value()
        idx = len(dists) - 1
        while node is not root_node:
            V -= dists[idx]
            idx -= 1
            self.update_parent(node, V)
            node = node.parent
        self.root_max = max(self.root_max, V)
        return V

    # return improved pi by MCTS
    def get_improved_pi(self, root_node, TAU, iter_p=2, stop_at_leaf=False):
        assert not root_node.is_end()
        self.root_max = -1e100
        n, _ = root_node.graph.shape
        for i in range(min(500, max(50, n * iter_p))):
            self.rollout(root_node, stop_at_leaf=stop_at_leaf)
        return root_node.pi(TAU)

    def train(self, graph, TAU, batch_size=10, iter_p=2, stop_at_leaf=False, start_node=0):
        self.gnnhash.clear()
        mse = torch.nn.MSELoss()
        env = TSPEnv()
        graph, dist_from_prev, dist_to_start = env.initialize_with_start_node(graph, start_node)
        env.set_graph(graph, start_node, dist_from_prev, dist_to_start)

        states = []
        actions = []
        pis = []
        means = []
        stds = []
        done = False
        while not done:
            n, _ = graph.shape
            node = TSPMCTSNode(graph, dist_from_prev, dist_to_start, self)
            means.append(node.reward_mean)
            stds.append(node.reward_std)
            pi = self.get_improved_pi(node, TAU, iter_p=iter_p, stop_at_leaf=stop_at_leaf)
            action = np.random.choice(n, p=pi)
            states.append([graph, dist_from_prev, dist_to_start])
            actions.append(action)
            pis.append(pi)
            graph, dist_from_prev, dist_to_start, reward, done, info = env.step(action)

        T = len(states)
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
                graph, dist_from_prev, dist_to_start = states[idx]
                p, v = self.gnn(graph, dist_from_prev, dist_to_start, True)
                Timer.end('gnn')
                # normalize z with mean, std
                z = torch.tensor(((T - idx) - means[idx]) / stds[idx])
                loss += mse(z, v[actions[idx]]) - (torch.tensor(pis[idx]) * torch.log(p + EPS)).sum()
            loss /= size
            loss.backward()
            self.optimizer.step()
            i += size

    # rollout iter_num times
    def search(self, graph, iter_num=10, start_node=0):
        env = TSPEnv()
        graph, dist_from_prev, dist_to_start = env.initialize_with_start_node(graph, start_node)
        root_node = TSPMCTSNode(graph, dist_from_prev, dist_to_start, self)
        ans = []
        for i in range(iter_num):
            r = self.rollout(root_node)
            if self.performance: print(r)
            ans.append(r)
        return ans
