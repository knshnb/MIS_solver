import time
import torch
from environ.mis_env import MISEnv
from utils.graph import *
from config import *
import numpy as np
# xp = chainer.cuda.cupy if use_gpu else np
xp = np

class Trainer:
    def __init__(self, policy, test_graphs=[]):
        self.env = MISEnv() if use_dense else MISEnv_Sparse()
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=0.01)
        self.test_graphs = test_graphs
        self.rewards = []

    def train(self, adj, iter=10, batch=10, print_log=True):
        self.env.set_graph(adj)
        reward_sum = 0
        for epoch in range(iter):
            rewards = torch.empty(batch)
            log_probs = torch.zeros(batch)
            for n in range(batch):
                graph = self.env.reset()
                done = False
                while done == False:
                    action, prob = self.policy.act(graph)
                    log_probs[n] += torch.log(prob)
                    graph, reward, done, info = self.env.step(action)
                
                rewards[n] = reward
            if print_log: print(rewards)
            reward_sum += rewards.detach().numpy().sum()
            reward_mean = reward_sum / ((epoch + 1) * batch)

            loss = - ((rewards - reward_mean) * log_probs).mean()
            loss.backward()
            self.optimizer.step()

    def solution(self, adj):
        g = adj
        self.env.set_graph(g)
        while 1:
            v, prob = self.policy.act(g)
            g, r, finish, info = self.env.step(v)
            if finish:
                break
        return r
