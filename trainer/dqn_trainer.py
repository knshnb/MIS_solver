import torch
from environ.mis_env import MISEnv
from utils.graph import *
import numpy as np
from torchviz import make_dot
from collections import deque

class DQNTrainer:
    def __init__(self, policy):
        self.env = MISEnv()
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=1e-3)
        # print(dict(self.policy.model.named_parameters()))
        # exit()
        self.cnt = 0
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.memory = []
        self.eps = 1

    def play(self, graph, eps):
        self.memory = []
        self.env.set_graph(graph)
        prev = graph
        done = False
        while done == False:
            v = self.policy.act(graph, eps)
            graph, reward, done, info = self.env.step(v)
            self.memory.append((prev, graph, v))
            prev = graph

    def learn(self, batch=10):
        np.random.shuffle(self.memory)
        size = len(self.memory)
        i = 0
        while i < size:
            N = min(size - i, batch)
            ys = torch.empty(N)
            xs = torch.empty(N)
            for j in range(N):
                pre, cur, v = self.memory[i + j]
                ys[j] = 1 if cur.shape[0] == 0 else 1 + np.max(self.policy.oldmodel(cur).detach().numpy())
                xs[j] = self.policy.model(pre)[v]
            loss = self.loss_fn(xs, ys)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += N
        self.policy.update()

    def solution(self, graph):
        self.env.set_graph(graph)
        while 1:
            v = self.policy.act(graph, 0)
            graph, reward, done, info = self.env.step(v)
            if done: return reward

    def run(self, graph, epoch=10000):
        for ep in range(epoch):
            self.play(graph, self.eps)
            self.learn()
            self.eps *= 0.999
            if (ep % 10 == 0):
                print("Epoch: {}, Score: {}, Eps: {}".format(ep + 1, self.solution(graph), self.eps))
                if (ep % 100 == 0):
                    print(self.policy.model(graph))

    def oracle(self, graph):
        four = graph
        one = np.zeros((1, 1), dtype=np.float32)
        ans_four = np.array([1,2,1,2], dtype=np.float32)
        ans_one = np.array([1], dtype=np.float32)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for i in range(1000):
            k = self.policy.model(four if i & 1 else one)
            ans = ans_four if i & 1 else ans_one
            loss = loss_fn(k, torch.Tensor(ans))
            self.optimizer.zero_grad()
            loss.backward()
            print(i, loss, k)
            if i % 100 == 0:
                print(self.policy.model(four))
                print(self.policy.model(one))
            self.optimizer.step()
