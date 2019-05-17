import numpy as np
import torch
from environ.mis_env import MISEnv
from mcts.mcts_node import MCTSNode, INF
from utils.graph import read_graph

class MCTS:
    def __init__(self, gnn):
        MCTSNode.set_gnn(gnn)
        self.optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
        self.gnn = gnn

    # parantのQ(s,a), N(s,a)を更新
    def update_parant(self, node, V):
        par = node.parent
        par.Q[node.idx] = V if par.Q[node.idx] >= INF else max(par.Q[node.idx], V)
        par.visit_cnt[node.idx] += 1

    def rollout(self, root_node, stop_at_leaf=True):
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
    def get_improved_pi(self, graph, iter_p=5):
        root_node = MCTSNode(graph)
        assert not root_node.is_end()
        for i in range(iter_p * 2):
            self.rollout(root_node)
        return root_node.pi()

    def train(self, graph):
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
        for i in range(T):
            self.optimizer.zero_grad()
            for batch in range(10):
                idx = np.random.randint(T)
                p, v = MCTSNode.gnn(graphs[idx])

                n, _ = graphs[idx].shape
                z = torch.tensor(T - idx)
                # 解のサイズの方は適当に(n / 3)で割ってスケールを合わせておく
                loss = mse(z, v[actions[idx]]) / (max(1., n / 3)) + mse(torch.tensor(pis[idx]), p)
                # TODO: use cross entropy loss
                loss.backward()
            self.optimizer.step()

    def search(self, graph, iter_num=100):
        root_node = MCTSNode(graph)
        ans = []
        for i in range(iter_num):
            ans.append(self.rollout(root_node, stop_at_leaf=False))
        return ans
