import numpy as np
import torch


def maxsquare(arr, x, y):
    # return np.max(arr[x: x+1, y: y+1])
    x1 = x+1
    y1 = y+1
    return max(
        arr[x, y],
        arr[x1, y],
        arr[x, y1],
        arr[x1, y1],
    )

    # val1 = arr[x, y]
    # val2 = arr[x+1, y]
    # val3 = arr[x, y+1]
    # val4 = arr[x+1, y+1]
    # max1 = val1 if val1 > val2 else val2
    # max2 = val3 if val3 > val4 else val4
    # return max1 if max1 > max2 else max2


# def fun(i, a_name, a_shape, M_name, M_shape):
#     a_m = shared_memory.SharedMemory(name=a_name)
#     a = np.ndarray(a_shape, buffer=a_m.buf)
#     M_m = shared_memory.SharedMemory(name=M_name)
#     M = np.ndarray(M_shape, buffer=M_m.buf)
#     _i = i << 1
#     xx = np.amax(a[_i: _i + 2], axis=0)
#     M[i] = np.amax(xx.reshape(-1, 2), axis=1)


def quad(array):
    M = np.empty((array.shape[0]//2, array.shape[1]//2))

    # sh_a = shared_memory.SharedMemory(create=True, size=array.nbytes)
    # np.ndarray(array.shape, dtype=array.dtype, buffer=sh_a.buf)[:] = array

    # sh_M = shared_memory.SharedMemory(create=True, size=M.nbytes)
    # M = np.ndarray(M.shape, dtype=array.dtype, buffer=sh_M.buf)
    # with Pool(5) as p:

    #     args = [
    #         (i, sh_a.name, array.shape, sh_M.name, M.shape) for i in range(M.shape[1])
    #     ]
    #     p.map(fun, args)
    # print(M)
    for i in range(M.shape[1]):
        # for j in range(M.shape[0]):
        #     M[j, i] = maxsquare(array, 2*j, 2*i)
        _i = i << 1
        xx = np.amax(array[_i: _i + 2], axis=0)
        M[i] = np.amax(xx.reshape(-1, 2), axis=1)

    return M  # , maxarg


def quad_torch(array):
    M = torch.empty((array.shape[0]//2, array.shape[1]//2))

    for i in range(M.shape[1]):

        _i = i << 1
        xx = torch.amax(array[_i: _i + 2], dim=0)
        M[i] = torch.amax(xx.reshape(-1, 2), dim=1)

    return M


def build_quadtree(matrix, _torch=False):
    quadtree = [matrix]
    # indexes = []
    while quadtree[-1].shape != (1, 1):
        # node, _max = quad(quadtree[-1])
        node = quad_torch(quadtree[-1]) if _torch else quad(quadtree[-1])
        quadtree.append(node)
        # indexes.append(_max)
    # timings.append(time())
    return quadtree  # , indexes


def printquadtree(quadtree):
    with np.printoptions(precision=2, linewidth=1000, suppress=True):
        for node in quadtree:
            print(node)


def reconstruct(quadtree, arg):
    x, y = arg
    quadtree[0][x] = -np.inf
    quadtree[0][:, y] = -np.inf
    # quadtree[0][x] = -1.0
    # quadtree[0][:, y] = -1.0
    for i in range(1, len(quadtree)):
        # x //= 2
        # y //= 2
        current_node = quadtree[i]
        prev_node = quadtree[i-1]
        x >>= 1
        y >>= 1
        for j in range(current_node.shape[0]):
            # quadtree[i][x, j] = maxsquare(quadtree[i-1], x*2, j*2)
            # quadtree[i][j, y] = maxsquare(quadtree[i-1], j*2, y*2)
            j2 = j << 1
            current_node[x, j] = maxsquare(prev_node, x << 1, j2)
            current_node[j, y] = maxsquare(prev_node, j2, y << 1)


def reconstruct_h(quadtree, arg):
    x, y = arg
    quadtree[0][x] = -np.inf
    quadtree[0][:, y] = -np.inf
    x_prop = range(quadtree[1].shape[0])
    y_prop = range(quadtree[1].shape[0])

    for i in range(1, len(quadtree)):
        prev_node = quadtree[i-1]
        current_node = quadtree[i]
        # next_node = quadtree[i+1]
        x >>= 1
        y >>= 1
        # print(x_prop)
        # print(y_prop)
        x_prop_new = set()
        for x_p in x_prop:
            val = current_node[x, x_p]
            newval = maxsquare(prev_node, x << 1, x_p << 1)
            if newval < val:
                current_node[x, x_p] = newval
                x_prop_new.add(x_p >> 1)
        x_prop = x_prop_new

        y_prop_new = set()
        for y_p in y_prop:
            val = current_node[y_p, y]
            newval = maxsquare(prev_node, y_p << 1, y << 1)
            if newval < val:
                current_node[y_p, y] = newval
                y_prop_new.add(y_p >> 1)
        y_prop = y_prop_new


def reconstruct_numpy(quadtree, arg):
    x, y = arg

    quadtree[0][x] = -np.inf
    quadtree[0][:, y] = -np.inf
    for i in range(1, len(quadtree)):
        current_node = quadtree[i]
        prev_node = quadtree[i-1]

        x >>= 1
        _x = x << 1  # _x is divisible by 2
        xx = np.amax(prev_node[_x: _x + 2], axis=0)
        current_node[x] = np.amax(xx.reshape(-1, 2), axis=1)

        y >>= 1
        _y = y << 1  # _y is divisible by 2
        yy = np.amax(prev_node[:, _y:_y + 2], axis=1)
        current_node[:, y] = np.amax(yy.reshape(-1, 2), axis=1)


def reconstruct_pytorch(quadtree, arg):
    x, y = arg
    # quadtree = [torch.from_numpy(arr) for arr in quadtree]
    quadtree[0][x] = float('-inf')
    quadtree[0][:, y] = float('-inf')
    for i in range(1, len(quadtree)):
        current_node = quadtree[i]
        prev_node = quadtree[i-1]

        x >>= 1
        _x = x << 1  # _x is divisible by 2
        xx = torch.amax(prev_node[_x: _x + 2], dim=0)
        current_node[x] = torch.amax(xx.reshape(-1, 2), dim=1)

        y >>= 1
        _y = y << 1  # _y is divisible by 2
        yy = torch.amax(prev_node[:, _y:_y + 2], dim=1)
        current_node[:, y] = torch.amax(yy.reshape(-1, 2), dim=1)


# reconstruct_numpy = reconstruct_pytorch


def lookup(quadtree, _torch=False):
    x = y = 0
    for node in quadtree[::-1]:
        x *= 2
        y *= 2
        _x, _y = np.unravel_index(
            torch.argmax(node[x:x+2, y:y+2]
                         ) if _torch else np.argmax(node[x:x+2, y:y+2]),
            shape=(2, 2)
        )
        # print(_x, _y)
        x += _x
        y += _y
        # assert quadtree[-1][0, 0] == node[x, y]
    return x, y


# def quadtreepop(quadtree):

#     val = quadtree[-1][0, 0]
#     argmax = lookup(quadtree)
#     # timings.append(time())
#     reconstruct(quadtree, argmax)
#     # timings.append(time())
#     return argmax, val


def superfast_binbin(M):

    n = M.shape[0]

    quadtree = build_quadtree(M)

    # timings.append(time())

    ma = np.zeros(n, int)
    mb = np.zeros(n, int)
    # ma = torch.zeros(n, int)
    # mb = torch.zeros(n, int)

    for _ in range(n):
        # for _ in range(100):
        # argmax, maxval = quadtreepop(quadtree)

        # val = quadtree[-1][0, 0]
        argmax = lookup(quadtree)
        # timings.append(time())
        reconstruct_numpy(quadtree, argmax)
        # reconstruct_pytorch(quadtree, argmax)

        # timings.append(time())
        x, y = argmax
        ma[x] = x
        mb[x] = y
    # printquadtree(quadtree)
    return ma, mb


def superfast_binbin_torch(M):

    n = M.shape[0]

    quadtree = build_quadtree(M, _torch=True)

    ma = np.zeros(n, int)
    mb = np.zeros(n, int)

    for _ in range(n):

        argmax = lookup(quadtree, _torch=True)

        reconstruct_pytorch(quadtree, argmax)

        x, y = argmax
        ma[x] = x
        mb[x] = y

    return ma, mb


# def writer_proc(queue, _M):
#     _n = _M.shape[0]
#     quadtree = build_quadtree(_M)

#     for _ in range(_n):
#         argmax = lookup(quadtree)
#         maxval = quadtree[-1][0, 0]
#         reconstruct_numpy(quadtree, argmax)
#         queue.put((argmax, maxval))
#     queue.put((None, -np.inf))


# def ultrafast(M, num_proc=4):
#     from multiprocessing import Queue, Process
#     import time

#     n = M.shape[0]

#     pqueues = [Queue(500) for _ in range(num_proc)]
#     writer_ps = []
#     for i, pqueue in enumerate(pqueues):
#         if i == 0:
#             sl = M[:n//2, :n//2]
#         if i == 1:
#             sl = M[:n//2, n//2:]
#         if i == 2:
#             sl = M[n//2:, :n//2]
#         if i == 3:
#             sl = M[n//2:, n//2:]

#         writer_ps.append(Process(target=writer_proc, args=(pqueue, sl)))
#         writer_ps[-1].daemon = True
#         # Launch reader_proc() as a separate python process
#         writer_ps[-1].start()

#     time.sleep(1)

#     ma = np.zeros(n, int)
#     mb = np.zeros(n, int)

#     # print("reading..")
#     msgs = [queue.get() for queue in pqueues]
#     imax = max(enumerate(msgs), key=lambda x: x[1][1])[0]
#     # print(imax)
#     for _ in range(n):

#         maxval = msgs[imax]

#         argmax, _ = maxval

#         if imax == 0:
#             argmax = argmax
#         if imax == 1:
#             argmax = (argmax[0], argmax[1]+n//2)
#         if imax == 2:
#             argmax = (argmax[0]+n//2, argmax[1])
#         if imax == 3:
#             argmax = (argmax[0]+n//2, argmax[1]+n//2)

#         print(f"reader", argmax)
#         print(f"reader", msgs)
#         # if (maxval == -1):
#         #     break
#         msgs[imax] = pqueues[imax].get()
#         imax = max(enumerate(msgs), key=lambda x: x[1][1])[0]

#         x, y = argmax
#         ma[x] = x
#         mb[x] = y

#     [writer_p.join() for writer_p in writer_ps]

#     return ma, mb


if __name__ == "__main__":

    import sys
    import time
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    test = len(sys.argv) > 2

    n = 2 ** k
    # print(n)
    # np.random.seed(1)
    # print(np.random.random(5))
    # exit()
    # M = np.random.random(n*n).reshape(n, n).astype(dtype=np.float32)
    M = torch.rand(n, n, dtype=torch.float32)
    # M = np.arange(n*n).reshape(n, n).astype(dtype=dtype)
    # print(M)

    algs = [
        superfast_binbin_torch,
        superfast_binbin,
        # matching.superfast,
        # ultrafast,
    ]

    Ms = [
        M.clone(),
        M.clone().numpy(),
        # M.clone().numpy(),
    ]
    args = []
    for alg, Mc in zip(algs, Ms):
        timings = []

        timings.append(time.time())
        args.append(alg(Mc))
        timings.append(time.time())

        print([timings[i] - timings[i-1] for i in range(1, len(timings))])

    if not test:
        assert np.all(args[0][1] == args[1][1])
    # print(args1)
    # print(args2)

    print("n:", n)

# python - m memory_profiler test.py
