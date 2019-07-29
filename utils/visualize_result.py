import pickle

file = input()
f = open(file, 'rb')
data = pickle.load(f)

res = [[], [], []]
T = len(data)
for i in range(T):
    for j in range(3):
        res[j].append(-max(data[i][j]))

import matplotlib.pyplot as plt

for i in range(3):
    plt.plot(res[i])

plt.show()