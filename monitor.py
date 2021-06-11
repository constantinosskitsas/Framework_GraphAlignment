import psutil
import numpy as np
import sys
import matplotlib.pyplot as plt


def save(path, vals):
    np.save(path, np.array(vals))
    plt.plot(vals)

    # plt.xlabel(xlabel)
    # plt.xticks(dim3)
    # if plot_type == 1:
    #     plt.ylabel("Accuracy")
    #     plt.ylim([-0.1, 1.1])
    # else:
    #     plt.ylabel("Time[s]")
    #     # plt.yscale('log')

    # plt.legend()
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
            mem.append(psutil.virtual_memory().used)
    # except KeyboardInterrupt:
    except:
        save(f"{path}/cpu", np.array(cpu))
        save(f"{path}/load", np.array(load))
        save(f"{path}/mem", np.array(mem))

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
