import numpy as np
import matplotlib.pyplot as plt

def multi1000():
    for idx in range(4):
        data = np.loadtxt("logs/GCN3_{}.txt".format(idx))
        mean = data.mean(axis=1)
        plt.plot(np.arange(mean.shape[0]), mean)
    plt.show()

def single1000():
    for idx in range(4):
        data = np.loadtxt("logs/one_GCN3_{}.txt".format(idx))
        mean = data.mean(axis=1)
        plt.plot(np.arange(mean[:500].shape[0]), mean[:500])
    plt.show()

if __name__ == "__main__":
    single1000()