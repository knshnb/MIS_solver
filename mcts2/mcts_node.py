import numpy as np

class MCTSNode:
    @staticmethod
    def set_gnn(gnn):
        MCTSNode.gnn = gnn

    @staticmethod
    def get_gnn():
        return MCTSNode.gnn
    
    # idx: 親の何番目の子か。親がない場合は-1
    def __init__(self, graph, idx=-1, parent=None, beta=0.3):
        n, _ = graph.shape
        self.n = n
        self.graph = graph
        self.parent = parent
        self.children = [None] * n
        self.idx = idx

        # 各子を何度訪れたか
        self.cnt = np.zeros(n, dtype=np.int)
        # 将来Qを計算するとき用
        # これまでのrolloutで何番目のものに寄与したかを保持しておく
        self.Qidx = [[] for _ in range(n)]

        if not self.is_end():
            # Pは各子を選ぶ確率。Vは各子の評価値
            # PとVはいずれも長さnの配列
            # Qの初期値をVを正規化して初期化
            self.P, V = MCTSNode.gnn(self.graph)
            self.P = self.P.detach().numpy()
            V = V.detach().numpy()
            self.cnt += 1
            # ここはnumpyなのでstd()がバグらない
            self.Qini = beta * (V - V.mean()) / (V.std() + 1e-5)

    def update_Q(self, n_rewards):
        self.Qtmp = [[self.Qini[i]] for i in range(self.n)]
        for i in range(self.n):
            for j in self.Qidx[i]:
                self.Qtmp[i].append(n_rewards[j])
        for i in range(self.n):
            assert self.cnt[i] == len(self.Qtmp[i])
        self.Q = np.empty(self.n, dtype=np.float32)
        for i in range(self.n):
            self.Q[i] = sum(self.Qtmp[i])

    def is_end(self):
        return self.graph.shape[0] == 0

    def best_ucb(self, alpha, n_rewards):
        self.update_Q(n_rewards)
        k = alpha * np.sqrt(self.cnt.sum()) / (1 + self.cnt)
        ucb = self.Q + k * self.P
        # print("UCB_DEBUG", k, self.Q, self.P, ucb)
        # print(ucb)
        return np.argmax(ucb)
    
    def pi(self, tau):
        assert self.cnt.sum()
        pow_tau = np.power(self.cnt, 1 / tau)
        return pow_tau / pow_tau.sum()
