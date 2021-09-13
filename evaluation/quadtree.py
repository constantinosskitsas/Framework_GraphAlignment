import numpy as np
import torch


def printquadtree(quadtree):
    with np.printoptions(precision=2, linewidth=1000, suppress=True):
        for node in quadtree:
            print(node)


def quad_propagate(target, source, i, dim=0):
    _i = i << 1
    if dim == 0:
        xx = torch.amax(source[_i: _i + 2], dim=dim)
    else:
        xx = torch.amax(source[:, _i: _i + 2], dim=dim)
    if dim == 0:
        target[i] = torch.amax(xx.view(-1, 2), dim=1)
    else:
        target[:, i] = torch.amax(xx.view(-1, 2), dim=1)


def dequad(array):
    M = torch.empty((array.shape[0]//2, array.shape[1]//2))

    for i in range(M.shape[1]):

        quad_propagate(M, array, i)
        # _i = i << 1
        # xx = torch.amax(array[_i: _i + 2], dim=0)
        # M[i] = torch.amax(xx.view(-1, 2), dim=1)

    return M


def build_quadtree(matrix):
    quadtree = [matrix]
    while quadtree[-1].shape != (1, 1):
        node = dequad(quadtree[-1])
        quadtree.append(node)

    return quadtree


def lookup(quadtree):
    x = y = 0
    for node in quadtree[::-1]:
        x *= 2
        y *= 2
        _x, _y = np.unravel_index(
            torch.argmax(node[x:x+2, y:y+2]),
            shape=(2, 2)
        )
        x += _x
        y += _y
        # assert quadtree[-1][0, 0] == node[x, y]
    return x, y


def reconstruct_pytorch(quadtree, arg):
    x, y = arg
    quadtree[0][x] = float('-inf')
    quadtree[0][:, y] = float('-inf')
    for i in range(1, len(quadtree)):
        current_node = quadtree[i]
        prev_node = quadtree[i-1]

        x >>= 1

        quad_propagate(current_node, prev_node, x)

        # _x = x << 1  # _x is divisible by 2

        # xx = torch.amax(prev_node[_x: _x + 2], dim=0)
        # current_node[x] = torch.amax(xx.view(-1, 2), dim=1)

        # xx = prev_node[_x: _x + 2].T.reshape(-1, 4)
        # current_node[x] = torch.amax(xx, dim=1)

        y >>= 1

        quad_propagate(current_node, prev_node, y, dim=1)

        # _y = y << 1  # _y is divisible by 2

        # yy = torch.amax(prev_node[:, _y:_y + 2], dim=1)
        # current_node[:, y] = torch.amax(yy.view(-1, 2), dim=1)

        # yy = prev_node[:, _y:_y + 2].reshape(-1, 4)
        # current_node[:, y] = torch.amax(yy, dim=1)


def superfast_binbin_torch(M):
    M = torch.from_numpy(M)
    # torch.set_num_threads(35)
    n = M.shape[0]

    quadtree = build_quadtree(M)

    ma = np.zeros(n, int)
    mb = np.zeros(n, int)

    for _ in range(n):

        argmax = lookup(quadtree)

        reconstruct_pytorch(quadtree, argmax)

        x, y = argmax
        ma[x] = x
        mb[x] = y

    return ma, mb


if __name__ == "__main__":

    import sys
    import time
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    n = 2 ** k

    M = torch.rand(n, n, dtype=torch.float32)

    algs = [
        superfast_binbin_torch,
        # superfast,
    ]

    Ms = [
        M.numpy(),
        # M.clone().numpy(),
        # -M.clone().numpy(),
    ]
    args = []
    for alg, Mc in zip(algs, Ms):
        timings = []

        timings.append(time.time())
        args.append(alg(Mc))
        timings.append(time.time())

        print([timings[i] - timings[i-1] for i in range(1, len(timings))])

    if len(args) > 1:
        assert np.all(args[0][1] == args[1][1])

    print("n:", n)

# python - m memory_profiler test.py
