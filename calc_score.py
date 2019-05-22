import pickle
import numpy as np

def score(filename):
    with open(filename, mode="rb") as f:
        data = pickle.load(f)
    acc = 0
    for rewards in data[-1]:
        acc += np.max(rewards)
    return acc

if __name__ == "__main__":
    files = [
        "log/multi_train2/train2_100_p2_0th.pickle",
        "log/multi_train2/train2_100_p2_1th.pickle",
        "log/multi_train2/train2_100_p2_2th.pickle",
        "log/multi_train2/train2_100_p2_3th.pickle",
        "log/multi_train2/train2_100_p2_4th.pickle",
    ]
    for file in files:
        print(score(file))
