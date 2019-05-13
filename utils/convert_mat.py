import glob
from graph import *
from config import *
from scipy.sparse import csc_matrix
import numpy as np
import scipy as sp

# for https://github.com/knshnb/NPHard
def save_mat(inputname, outputname):
    g = read_graph(filename)
    n = g.n
    row = []
    col = []
    if use_dense:
        for i in range(n):
            for j in range(i + 1, n):
                if g.adj[i, j]:
                    row.append(i)
                    col.append(j)
                    row.append(j)
                    col.append(i)
        mat = csc_matrix((np.ones(len(row), dtype=np.float64), (np.array(row), np.array(col))), shape=(n, n))
    else:
        mat = csc_matrix((np.ones(len(g.adj.row), dtype=np.float64), (g.adj.row, g.adj.col)), shape=(n, n))

    data = {}
    data['adj'] = mat
    sp.io.savemat(outputname, data)

if __name__ == "__main__":
    filenames = glob.glob("data/random/*")
    for filename in filenames:
        save_mat(filename, filename + ".mat")
