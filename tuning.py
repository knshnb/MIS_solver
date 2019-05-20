from config import device
import numpy as np
import torch
import os
from utils.graph import read_graph
from mcts.mcts import MCTS
from gin.gin import GIN3
from timer import Timer
import optuna

# これを最小化するようなパラメータを見つけてくれる
def objective(trial):
    layer_num = trial.suggest_int('layer_num', 2, 20)
    feature = trial.suggest_int('feature', 5, 10)

    gnn = GIN3(layer_num=layer_num, feature=feature)
    gnn.to(device)
    mcts = MCTS(gnn)

    graph = read_graph("data/learn/50_125_0").adj

    Timer.start('all')
    ans = []
    for i in range(100):
        print("epoch: ", i)
        train_ans = mcts.search(graph)
        print(train_ans)
        print("train ans mean", np.mean(train_ans))
        ans.append(train_ans)
        print(mcts.gnn(graph))
        mcts.train(graph, 10 * 0.95 ** i)
    Timer.end('all')
    score = 0
    coef = 1
    for s in reversed(ans):
        score += s * coef
        coef *= 0.9

    Timer.print()
    torch.save(gnn.state_dict(), "model/optuna_tmp_{}_{}".format(layer_num, feature))
    
    return -score


if __name__ == '__main__':
    os.makedirs("model", exist_ok=True)
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)

    print("params_{}".format(study.best_params))
    print("value_{}".format(study.best_value))
