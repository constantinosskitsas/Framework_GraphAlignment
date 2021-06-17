import psutil
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save(path, vals, dims, ylabel):
    vals = np.array(vals)
    # np.savetxt(f"{path}.txt", vals, fmt="%.2f")

    vals = vals.reshape(vals.shape[0], -1)

    with PdfPages('foo.pdf') as pdf:
        plt.figure()
        for val, dim in zip(vals.T, dims):
            plt.plot(val, label=dim)

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        plt.ylabel(ylabel)
        plt.xlabel("Time[s]")

        plt.legend()

        plt.title("Cone-align")
        # plt.title("Grasp")
        plt.margins(x=0)
        pdf.savefig()


def monitor(path, interval=1):

    cpu = []
    load = []
    mem = []

    try:
        while True:
            # print(1)
            cpu.append(psutil.cpu_percent(interval=interval))
            load.append(psutil.getloadavg())
            mem.append(psutil.virtual_memory().used / (1024 * 1024))  # MiB
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


def ff1():
    xx2 = [1024, 2048, 4096, 8192, 16384,
           32768, 65536, 131072]

    xx = [1, 2, 3, 4, 5, 6, 7, 8]

    data = [
        (
            "GWL", [
                12.63850846,
                65.85141096,
                484.8533538,
                2514.297266], [
                    551.66,
                2149.98,
                2174.2,
                8479.86
            ]
        ),
        (
            'Cone', [
                3.323810196,
                13.05297151,
                63.92402992,
                278.6144972,
                1588.104114], [
                    184.66,
                356.01,
                1231.88,
                4933.95,
                18773.69
            ]
        ),
        (
            'Grasp', [
                4.781353331,
                10.72093945,
                26.54429522,
                105.5368177,
                500.3921397,
                2696.998351], [
                    116.21,
                90.44,
                464.8,
                1742.59,
                6874.17,
                27057.85
            ]
        ),
        (
            'Regal', [
                1.304475355,
                3.005040646,
                7.306168747,
                17.10120873,
                40.31162243,
                103.954876,
                305.9771749], [
                    106.26,
                107.58,
                302.1,
                1119.93,
                4271.82,
                18238.02,
                72044.47
            ]
        ),
        (
            'Lrea', [
                0.06236333847,
                0.2968031883,
                0.9521621227,
                3.479673719,
                12.87085423,
                50.21864805,
                196.7129255,
                774.7873389], [
                    82.65,
                75.26,
                151.2,
                240.62,
                2616.67,
                10326.22,
                41056.74,
                164161.58
            ]
        ),
        (
            'NSD', [
                0.03149690628,
                0.1982555389,
                1.019071817,
                3.855656719,
                14.75504041,
                58.17519765,
                251.3450502,
                928.309432], [
                    92.2,
                61.37,
                254.31,
                968.91,
                3900.4,
                15475.84,
                61581.95,
                244210.38
            ]
        ),
        (
            'Isorank', [
                2.0518507,
                10.69510326,
                61.63217468,
                393.503893,
                3157.031604], [
                    140.75,
                293.78,
                1081.94,
                4297.19,
                17192.26
            ]
        )
    ]
    lines = [
        ("x-c", "GWL"),
        ("o-g", "Cone"),
        ("v-r", "Grasp"),
        ("s-k", "Regal"),
        ("D-m", "Lrea"),
        ("p-y", "Nsd"),
        ("*-b", "Isorank")
    ]
    with PdfPages('foo.pdf') as pdf:
        # plt.figure()
        fig, ax = plt.subplots()
        # ax.set_xscale('log', base=2)
        ax2 = ax.twinx()

        ax.set_yscale('log')
        ax.set_ylabel("memory[MiB]")
        ax.set_xlabel("number of nodes")
        ax2.set_yscale('log')
        ax2.set_ylabel("time[s]")

        offset = -3
        width = 0.12

        for i, (_, time, memory) in enumerate(data):
            style, name = lines[i]
            ax2.plot(xx[:len(time)], time, style, label=name)

            val = [x + (offset + i) * width for x in xx[:len(memory)]]
            ax.bar(val, memory, label=name, width=0.09, color=style[-1])

            # ax.bar(memory, label=name)

        # data = [23, 45, 56, 78, 213]
        # plt.bar([1, 2, 3, 4, 5], data)
        # plt.show()
        # plt.xscale('symlog')

        plt.xticks(xx, labels=xx2)

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        # plt.ylabel(ylabel)
        # plt.xlabel("Time[s]")

        # plt.legend()

        plt.title("Size experiment")
        # plt.title("Grasp")
        plt.margins(x=0)

        # plt.show()
        pdf.savefig()


def ff2():
    xx2 = [10, 100, 1000, 10000]

    xx = [1, 2, 3, 4]

    data = [
        # (
        #     "GWL", [
        #         12.63850846,
        #         65.85141096,
        #         484.8533538,
        #         2514.297266], [
        #             551.66,
        #         2149.98,
        #         2174.2,
        #         8479.86
        #     ]
        # ),
        (
            'Cone', [
                1593.634008,
                2914.040459], [
                18603.07,
                20456.98
            ]
        ),
        (
            'Grasp', [
                512.459086,
                464.4194816,
                459.7869436,
                455.8949021], [
                6888.82,
                6715.59,
                6741.79,
                6825.8
            ]
        ),
        (
            'Regal', [
                74.20772839,
                248.9018573,
                3531.261194], [
                3824.33,
                26223.23,
                68678.62
            ]
        ),
        (
            'Lrea', [
                12.85661378,
                12.83879142,
                12.83131166,
                12.84513092], [
                2778.09,
                2540.51,
                2541.14,
                2714.07
            ]
        ),
        (
            'NSD', [
                16.8975801,
                18.23648343,
                19.9915678,
                32.28288283], [
                4041.86,
                3985.04,
                4533.38,
                5977.29
            ]
        ),
        (
            'Isorank', [
                3422.350885,
                3424.97045,
                3397.386357,
                3409.972073], [
                17124.91,
                17005.97,
                16923.82,
                16418.27
            ]
        )
    ]
    lines = [
        # ("x-c", "GWL"),
        ("o-g", "Cone"),
        ("v-r", "Grasp"),
        ("s-k", "Regal"),
        ("D-m", "Lrea"),
        ("p-y", "Nsd"),
        ("*-b", "Isorank")
    ]
    with PdfPages('foo.pdf') as pdf:
        # plt.figure()
        fig, ax = plt.subplots()
        # ax.set_xscale('log', base=2)
        ax2 = ax.twinx()

        ax.set_yscale('log')
        ax.set_ylabel("memory[MiB]")
        ax.set_xlabel("average degree")
        ax2.set_yscale('log')
        ax2.set_ylabel("time[s]")

        offset = -3
        width = 0.10

        for i, (_, time, memory) in enumerate(data):
            style, name = lines[i]
            ax2.plot(xx[:len(time)], time, style, label=name)

            val = [x + (offset + i) * width for x in xx[:len(memory)]]
            ax.bar(val, memory, label=name, width=0.09, color=style[-1])

            # ax.bar(memory, label=name)

        # data = [23, 45, 56, 78, 213]
        # plt.bar([1, 2, 3, 4, 5], data)
        # plt.show()
        # plt.xscale('symlog')

        plt.xticks(xx, labels=xx2)

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        # plt.ylabel(ylabel)
        # plt.xlabel("Time[s]")

        # plt.legend()

        plt.title("Degree experiment")
        # plt.title("Grasp")
        plt.margins(x=0)

        # plt.show()
        pdf.savefig()


def ff3():
    xx2 = [128,	256, 384, 512, 640, 768]

    xx = [1, 2, 3, 4, 5, 6]

    data = [
        (
            "running time",
            [137.546263,  144.7983146, 152.6686841,
                162.4423513, 174.9348392, 189.9256386]
        ),
        (
            "noise level 0.01",
            [0.6091359247, 0.6704629859, 0.6963604853,
                0.7082941322, 0.7129239911, 0.7203763308]
        ),
        (
            "noise level 0.03",
            [0.543723694, 0.6120326814, 0.6329537014,
                0.6514978955, 0.6562020302, 0.6677147809]
        ),
        (
            "noise level 0.05",
            [0.5360485269, 0.6038375836, 0.6312205992,
                0.6441445902, 0.6516959643, 0.662837336]
        ),
    ]
    # lines = [
    #     # ("x-c", "GWL"),
    #     ("o-g", "Cone"),
    #     ("v-r", "Grasp"),
    #     ("s-k", "Regal"),
    #     ("D-m", "Lrea"),
    #     ("p-y", "Nsd"),
    #     ("*-b", "Isorank")
    # ]
    with PdfPages('foo.pdf') as pdf:
        # plt.figure()
        fig, ax = plt.subplots()
        # ax.set_xscale('log', base=2)
        ax2 = ax.twinx()

        # ax.set_yscale('log')
        ax.set_ylabel("accuracy[%]")
        # ax2.set_yscale('log')
        ax2.set_ylabel("time[s]")

        ax2.set_ylim([0, 200])

        ax.set_ylim([0, 1])

        offset = -1
        width = 0.10

        (name, time) = data[0]
        ax2.plot(xx[:len(time)], time, "k", label=name)

        for i, (name, time) in enumerate(data[1:]):

            val = [x + (offset + i) * width for x in xx[:len(time)]]
            ax.bar(val, time, label=name, width=0.09)

            # ax.bar(memory, label=name)

        # data = [23, 45, 56, 78, 213]
        # plt.bar([1, 2, 3, 4, 5], data)
        # plt.show()
        # plt.xscale('symlog')

        plt.xticks(xx, labels=xx2)

        ax.set_xlabel("dimensionality")

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        # plt.ylabel(ylabel)
        # plt.xlabel("Time[s]")

        # fig.legend(loc='lower left')
        # plt.legend(loc='lower left')
        # plt.legend()
        # ax.legend()
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower left")
        # ax2.legend()

        plt.title("Facebook Cone-Align")
        # plt.title("Grasp")
        plt.margins(x=0)

        # plt.show()
        pdf.savefig()


def ff4():

    xx2 = [8, 15, 20, 25, 30, 35, 40, 45]

    xx = [1, 2, 3, 4, 5, 6, 7, 8]

    data = [
        (
            "running time",
            [1.57, 2.85, 3.80, 4.71, 5.63, 6.56, 7.49, 8.42]
        ),
        (
            "noise level 0.01",
            [0.3542956177, 0.4907650409, 0.519485021, 0.5510522407,
                0.5670462986, 0.5699430552, 0.5716761575, 0.5823471156]
        ),
        (
            "noise level 0.03",
            [0.1532062392, 0.2018816539, 0.2247833622, 0.2445407279,
                0.2538499629, 0.2627383016, 0.2696707106, 0.2758108443]
        ),
        (
            "noise level 0.05",
            [0.1041099282, 0.1356771478, 0.1505570686, 0.1594949245,
                0.1677890567, 0.1688784353, 0.1720722951, 0.175513741]
        ),
    ]
    # lines = [
    #     # ("x-c", "GWL"),
    #     ("o-g", "Cone"),
    #     ("v-r", "Grasp"),
    #     ("s-k", "Regal"),
    #     ("D-m", "Lrea"),
    #     ("p-y", "Nsd"),
    #     ("*-b", "Isorank")
    # ]
    with PdfPages('foo.pdf') as pdf:
        # plt.figure()
        fig, ax = plt.subplots()
        # ax.set_xscale('log', base=2)
        ax2 = ax.twinx()

        # ax.set_yscale('log')
        ax.set_ylabel("accuracy[%]")
        # ax2.set_yscale('log')
        ax2.set_ylabel("time[s]")

        ax.set_ylim([0, 1])

        offset = -1
        width = 0.10

        (name, time) = data[0]
        ax2.plot(xx[:len(time)], time, "k", label=name)

        for i, (name, time) in enumerate(data[1:]):

            val = [x + (offset + i) * width for x in xx[:len(time)]]
            ax.bar(val, time, label=name, width=0.09)

            # ax.bar(memory, label=name)

        # data = [23, 45, 56, 78, 213]
        # plt.bar([1, 2, 3, 4, 5], data)
        # plt.show()
        # plt.xscale('symlog')

        plt.xticks(xx, labels=xx2)

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        # plt.ylabel(ylabel)
        ax.set_xlabel("rank")

        # fig.legend(loc='lower left')
        # plt.legend(loc='lower left')
        # plt.legend()
        # ax.legend()
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)
        # ax2.legend(lines + lines2, labels + labels2, loc="lower left")
        # ax2.legend()

        plt.title("Facebook LREA")
        # plt.title("Grasp")
        plt.margins(x=0)

        # plt.show()
        pdf.savefig()


def ff5():

    xx = [1, 2, 3, 4, 5, 6]

    # xx2 = [1,
    #        2,
    #        4,
    #        6,
    #        8,
    #        10]

    # data = [
    #     (
    #         "L+gt", [
    #             0.7296,
    #             0.5172,
    #             0.2624,
    #             0.1172,
    #             0.0524,
    #             0.0136
    #         ]
    #     ),
    #     (
    #         "L", [
    #             0.012,
    #             0.012,
    #             0.0148,
    #             0.0144,
    #             0.0136,
    #             0.0092
    #         ]
    #     ),
    #     (
    #         "gt-coverage", [
    #             0.1776,
    #             0.3348,
    #             0.588,
    #             0.778,
    #             0.9168,
    #             0.9856
    #         ]
    #     )
    # ]

    xx2 = [0.25,
           0.5,
           0.75,
           1,
           1.25,
           1.5]

    data = [
        (
            "L+gt", [
                0.7076,
                0.4748,
                0.296,
                0.1876,
                0.102,
                0.0756
            ]
        ),
        (
            "L", [
                0.0168,
                0.0348,
                0.058,
                0.0768,
                0.0792,
                0.0672
            ]
        ),
        (
            "gt-coverage", [
                0.192,
                0.384,
                0.576,
                0.754,
                0.944,
                0.982
            ]
        )
    ]
    with PdfPages('foo.pdf') as pdf:
        # plt.figure()
        fig, ax = plt.subplots()
        # ax.set_xscale('log', base=2)
        ax2 = ax.twinx()

        # ax.set_yscale('log')
        ax.set_ylabel("accuracy[%]")
        ax.set_ylim([0, 1])
        # ax2.set_yscale('log')
        ax2.set_ylabel("coverage[%]")
        ax2.set_ylim([0, 1])

        # offset = -3
        # width = 0.12
        name, time = data[-1]
        ax2.plot(xx, time, "--y", label=name)

        for i, (name, time) in enumerate(data[:-1]):
            # style, name = lines[i]

            ax.plot(xx[:len(time)], time, label=name)

            # ax2.plot(xx[:len(time)], time, label=name)

            # val = [x + (offset + i) * width for x in xx[:len(memory)]]
            # ax.bar(val, memory, label=name, width=0.09, color=style[-1])

            # ax.bar(memory, label=name)

        # data = [23, 45, 56, 78, 213]
        # plt.bar([1, 2, 3, 4, 5], data)
        # plt.show()
        # plt.xscale('symlog')

        plt.xticks(xx, labels=xx2)

        ax.set_xlabel("k")

        # plt.plot(val, label=dim)
        # plt.axhline(y=40, color='r', linestyle='--')

        # plt.ylabel(ylabel)
        # plt.xlabel("Time[s]")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        plt.title("Net-Align convergance")
        # plt.title("Grasp")
        plt.margins(x=0)

        # plt.show()
        pdf.savefig()


if __name__ == "__main__":
    # path = sys.argv[1] if len(sys.argv) > 1 else "."
    # monitor(path)

    # path = r"D:\Thesis\_results\scalingskadi\10_10k_mon\170\mon\CONE_2\cpu.txt"
    # path = r"D:\Thesis\_results\scalingskadi\14_mon\140\mon\cpu.txt"
    # vals = np.loadtxt(path)
    # print(vals[:-70].tolist())
    # # print(vals[:-170])
    # # print(vals[:-180])

    # save("", vals[:-70], ["cpu"], "usage[%]")

    # path = r"D:\Thesis\_results\scalingskadi\15_mon\146\mon\load.txt"
    # vals = np.loadtxt(path)

    # save("", vals[:2200, 0], ["avg1", "avg5", "avg15"], "load[cores]")

    ff2()
