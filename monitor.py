import psutil
import numpy as np
import sys
import matplotlib.pyplot as plt


def save(path, vals, dims, ylabel):
    vals = np.array(vals)
    np.savetxt(f"{path}.txt", vals, fmt="%.2f")

    vals = vals.reshape(vals.shape[0], -1)

    plt.figure()
    for val, dim in zip(vals.T, dims):
        plt.plot(val, label=dim)

    plt.ylabel(ylabel)
    plt.xlabel("Time[s]")

    plt.legend()
    plt.savefig(f"{path}.png")


def monitor(path, interval=1):

    cpu = []
    load = []
    mem = []

    try:
        while True:
            # print(1)
            cpu.append(psutil.cpu_percent(interval=interval))
            load.append(psutil.getloadavg())
            mem.append(psutil.virtual_memory().used / (1024 * 1024)) # MiB
    # except KeyboardInterrupt:
    except:
        save(f"{path}/cpu", cpu, ["cpu"], "usage[%]")
        save(f"{path}/load", load, ["avg1", "avg5", "avg15"], "load[cores]")
        save(f"{path}/mem", mem, ["used"], "memory[MiB]")

        # plt.plot(cpu)
        # plt.show()
        # # print(3)
        # np.save(f"{path}/load", np.array(load))
        # plt.plot(load)
        # plt.show()
        # # print(4)
        # np.save(f"{path}/mem", np.array(mem))

    # print(cpu)
    # print(load)
    # print(mem)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    monitor(path)
