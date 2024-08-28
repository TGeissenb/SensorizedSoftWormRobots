import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

import EPFLcolors


def bounding_hull(points, n=100):
    x = points[0, :]
    y = points[1, :]
    min_x = np.min(x)
    max_x = np.max(x)

    val = []
    low_bound = []
    upp_bound = []

    x_samp = np.linspace(min_x, max_x, n)

    for i in range(1, n):
        val.append(1/2 * (x_samp[i - 1] + x_samp[i]))
        low_bound.append(np.min(y[(x > x_samp[i - 1]) & (x < x_samp[i])]))
        upp_bound.append(np.max(y[(x > x_samp[i - 1]) & (x < x_samp[i])]))

    return val, low_bound, upp_bound


if __name__ == "__main__":

    path = r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\SlowCycleResponse"

    # Read files
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json") and not "v2" in file:
                filenames.append(os.path.join(root, file))

    # Read data from files and store in common directory
    data = {"Bending Angle": [],
            "Curvature": [],
            "Resistance": []}
    for file in filenames:
        with open(file, 'r') as f:
            data_set = json.load(f)
            for key in data.keys():
                data[key].append(data_set[key])

    # Synchronize frames and resistance measurements
    for i in np.arange(len(data["Resistance"])):
        f = interp1d(np.linspace(0, len(data["Bending Angle"][i]), num=len(data["Resistance"][i])), data["Resistance"][i])
        res = f(np.arange(len(data["Bending Angle"][i])))
        data["Resistance"][i] = res

    for key in data.keys():
        data[key] = np.array(list(itertools.chain.from_iterable(data[key])))

    # Make plots with sorted data
    k = 25
    fig, ax = plt.subplots(1, 1)

    data_array = np.array([data["Curvature"], data["Bending Angle"], data["Resistance"]])
    data_sort = data_array[:, data_array[0, :].argsort()]

    poly = np.poly1d(np.polyfit(data_sort[0, :], data_sort[2, :], 3))
    ax.plot(data_sort[0, ::k], poly(data_sort[0, ::k]),
            color=EPFLcolors.colors[2])

    val, low_bound, upp_bound = bounding_hull(np.array([data_sort[0, :], data_sort[2, :]]))
    ax.fill_between(val, low_bound, upp_bound, color=EPFLcolors.colors[0], alpha=0.3)
    ax.plot(val, low_bound, color=EPFLcolors.colors[1])
    ax.plot(val, upp_bound, color=EPFLcolors.colors[1])

    ax.set_xlabel("Curvature [-]")
    ax.set_ylabel("Resistance [$M\Omega$]")
    ax.text(0.95, 0.1, "Sample size: n = {0:d}".format(len(data["Resistance"])), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)
    plt.legend(["Cubic fit", "Raw data"])

    import tikzplotlib
    tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\CarbonBlackSensor\CurvatureResistance.tex")

    plt.show()

    fig, ax = plt.subplots(1, 1)

    data_sort = data_array[:, data_array[1, :].argsort()]

    poly = np.poly1d(np.polyfit(data_sort[1, :], data_sort[2, :], 5))
    ax.plot(data_sort[1, ::k], poly(data_sort[1, ::k]),
            color=EPFLcolors.colors[2])

    val, low_bound, upp_bound = bounding_hull(np.array([data_sort[1, :], data_sort[2, :]]))
    ax.fill_between(val, low_bound, upp_bound, color=EPFLcolors.colors[0], alpha=0.3)
    ax.plot(val, low_bound, color=EPFLcolors.colors[1])
    ax.plot(val, upp_bound, color=EPFLcolors.colors[1])

    ax.set_xlabel("Bending Angle [°]")
    ax.set_ylabel("Resistance [$M\Omega$]")
    ax.text(0.95, 0.1, "Sample size: n = {0:d}".format(len(data["Resistance"])), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)
    plt.legend(["Cubic fit", "Raw data"])

    import tikzplotlib
    tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\CarbonBlackSensor\AngleResistance.tex")

    plt.show()
