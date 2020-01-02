from config import device
import numpy as np
import torch
import os
from utils.graph import read_graph, generate_random_graph
from mcts.mcts import MCTS
from mcts.mcts_trainer import MCTSTrainer
from gin.gin import GIN3
from utils.timer import Timer
from utils.counter import Counter
import optuna

# minimize the return value


def objective(trial):
    test_graphs = [read_graph(
        "data/random/1000_2500_{}".format(i)).adj for i in range(5)]

    layer_num = trial.suggest_int('layer_num', 2, 20)
    feature = trial.suggest_int('feature', 5, 10)
    beta = trial.suggest_uniform('beta', 0.99, 0.997)

    gnn = GIN3(layer_num=layer_num, feature=feature)
    gnn.to(device)
    trainer = MCTSTrainer(gnn, test_graphs, "optuna_tmp_100_tran2_{}_{}_{}".format(
        layer_num, feature, beta))

    Timer.start('all')
    ans = []
    for i in range(500):
        print("epoch: ", i)
        graph = generate_random_graph(100, 250).adj
        trainer.train2(graph, 10 * beta ** i)
        trainer.test()
    Timer.end('all')
    score = 0
    coef = 1
    for all_rewards in reversed(trainer.test_result):
        coef *= 0.9
        for rewards in all_rewards:
            score += 10 * coef * np.max(rewards)
            score += coef * np.mean(rewards)

    Timer.print()
    trainer.save_model()
    trainer.save_test_result()

    return -score


if __name__ == '__main__':
    Timer.disable()
    Counter.disable()
    os.makedirs("model", exist_ok=True)
    study = optuna.create_study()
    study.optimize(objective, timeout=3600 * 8, n_jobs=-1)

    print("params_{}".format(study.best_params))
    print("value_{}".format(study.best_value))
