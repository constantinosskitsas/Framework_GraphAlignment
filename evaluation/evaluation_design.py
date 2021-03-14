from random import random
import numpy as np

#regal need +1
def remove_edges_directed(adj):
    count = 0
    print(np.shape(adj))
    for i in range(np.shape(adj)[0]):
        k = np.nonzero(adj[i])
        o = 1
        megethos = np.shape(k)[1]  #
        testxx = adj[i]
        al = (testxx.nonzero())
        # print(al)
        t = al[0]
        for j in t:
            test = adj[j]
            test1 = test.nonzero()
            test12 = test1[0]
            test2 = len(test12)
            tixeros = random()
            if tixeros <= 0.25 and megethos - o > 0 and test2 > 1:
                adj[i][j] = 0
                adj[j][i] = 0
                o = o + 1
                count = count + 1
    print(count)
    return adj
