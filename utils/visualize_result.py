import pickle

file = input()
f = open(file, 'rb')
data = pickle.load(f)

res = [[], [], []]
T = len(data)
for i in range(T):
    for j in range(1):
        res[j].append(min(data[i][j]))

import matplotlib.pyplot as plt

for i in range(1):
    plt.plot(res[i])

plt.show()