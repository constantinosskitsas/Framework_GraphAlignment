import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ''' sar -g 1 -1 > load.txt '''

    vals = np.loadtxt("D:/Thesis/load.txt", skiprows=3,
                      usecols=(2, 4, 5, 6))[:-1].T
    labels = ('runq-sz', 'avg-1', 'avg-5', 'avg-15')
    ylabel = "Load"

    ''' sar -r 1 -1 > memory.txt '''

    # vals = np.loadtxt("D:/Thesis/memory.txt",
    #                   skiprows=3, usecols=(3, 4))[:-1].T
    # labels = ('available', 'used')
    # ylabel = "Memory[kb]"

    plt.figure()
    for i, val in enumerate(vals):
        print(val)
        plt.plot(np.arange(val.size), val, label=labels[i])
        # plt.plot(val, np.arange(val.size))
    plt.xlabel('Time[s]')
    plt.ylabel(ylabel)

    # plt.xticks(dim3)
    # if plot_type == 1:
    #     plt.ylabel("Accuracy")
    #     plt.ylim([-0.1, 1.1])
    # else:
    #     plt.ylabel("Time[s]")
    # plt.yscale('log')

    plt.legend()
    plt.show()

    # plt.savefig(f"{filename}_{dim1[i1]}.png")
