from math import log2, floor
import numpy as np


def order_match(A, B):
    a = [np.sum(row) for row in A]
    b = [np.sum(row) for row in B]

    n = len(a)
    m = len(b)

    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(log2(m))
    for ap in a_p:
        while(e < m and
              abs(b_p[e][1] - ap[1]) < abs(b_p[s][1] - ap[1])
              ):
            e += 1
            s += 1
        ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

    # L = np.zeros((n, m), int)

    li = []
    lj = []
    w = []
    for i, bj in enumerate(ab_m):
        for j in bj:
            d = 1 - abs(a[i]-b[j]) / a[i]
            if (d >= 0):
                li.append(i)
                lj.append(j)
                w.append(d)

    return li, lj, w


if __name__ == "__main__":

    n = 10
    m = 11

    A = np.random.randint(2, size=(n, n))
    B = np.random.randint(2, size=(m, m))

    li, lj, w = order_match(A, B)

    print(A, B, li, lj, w, sep="\n\n")
