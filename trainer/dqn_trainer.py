import torch
from environ.mis_env import MISEnv
from utils.graph import *
import numpy as np

class DQNTrainer:
    def __init__(self, policy, test_graphs=[]):
        self.env = MISEnv()
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=0.01)
        self.test_graphs = test_graphs
        self.rewards = []
        self.cnt = 0

    def train(self, graph, batch=10, C=10, print_log=True):
        self.env.set_graph(graph)
        graphs = []
        vertices = []
        graphs.append(graph.copy())
        done = False
        while done == False:
            v = self.policy.act(graph)
            vertices.append(v)
            graph, reward, done, info = self.env.step(v)
            if not done: graphs.append(graph.copy())
            
            # sample
            j = np.random.randint(len(vertices))
            yj = 1 if done and j == len(vertices) - 1 else 1 + np.max(self.policy.oldmodel(graphs[j + 1]).detach().numpy())
            # yjをtorchのvariableにするとまずそう
            loss = (yj - self.policy.model(graphs[j])[vertices[j]]) ** 2
            loss.backward()
            self.optimizer.step()
            self.cnt += 1
            if self.cnt == C:
                self.policy.update()
                self.cnt = 0

    def solution(self, graph):
        self.env.set_graph(graph)
        while 1:
            v = self.policy.act(graph, epsilon=0)
            graph, reward, done, info = self.env.step(v)
            if done: return reward

    def run(self, graph, epoch=10000):
        for ep in range(epoch):
            self.train(graph)
            print("Epoch: {}, Score: {}".format(ep + 1, self.solution(graph)))
