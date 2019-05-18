import numpy as np

class MCTSNode:
    @staticmethod
    def set_gnn(gnn):
        MCTSNode.gnn = gnn

    @staticmethod
    def get_gnn():
        return MCTSNode.gnn
    
    # idx: 親の何番目の子か。親がない場合は-1
    def __init__(self, graph, idx=-1, parent=None):
        n, _ = graph.shape
        self.n = n
        self.graph = graph
        self.parent = parent
        self.children = [None] * n
        self.idx = idx

        # 各子を何度訪れたか
        self.cnt = np.zeros(n, dtype=np.int)
        # 各子のscoreの合計
        self.sum = np.zeros(n, dtype=np.float32)

        if not self.is_end():
            # Pは各子を選ぶ確率。Vは各子の評価値
            # PとVはいずれも長さnの配列
            # Qの初期値をVを正規化して初期化
            self.P, V = MCTSNode.gnn(self.graph)
            self.P = self.P.detach().numpy()
            V = V.detach().numpy()
            self.cnt += 1
            self.sum += (V - V.mean()) / (V.std() + 1e-5)

    def update_Q(self): 
        self.Q = self.sum / self.cnt

    def is_end(self):
        return self.graph.shape[0] == 0

    def best_ucb(self, alpha):
        self.update_Q()
        ucb = self.Q + alpha * self.P / self.cnt
        # print(ucb)
        return np.argmax(ucb)
    
    def pi(self, tau):
        assert self.cnt.sum()
        pow_tau = np.power(self.cnt, 1 / tau)
        return pow_tau / pow_tau.sum()
