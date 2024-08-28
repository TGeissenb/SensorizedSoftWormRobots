import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import itertools

import EPFLcolors
from CalibrationResult import bounding_hull


dir = r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\Blobing\Test"

step = 1
step2 = 100
tot_corr = [0.0, 0.0, 0.0]
n_samples = 0
data = [[], [], [], []]

# Read files
paths = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".json"):
            paths.append(os.path.join(root, file))

fig, ax = plt.subplots(nrows=1, ncols=1)

for path in paths:

    with open(path, 'r') as f:
        obstruction = json.load(f)

    obstruction["val"] = np.array(obstruction["val"])
    obstruction["time"] = np.array(obstruction["time"])

    df = pd.read_csv(path[:-5] + ".csv", sep=",")

    # Interpolations
    corrs = []
    for time_offset in range(0, 2, 2):
        corr = 0.0

        t = (df["Time"] - df["Time"][0]) / 1000
        t = np.array(t.tolist())
        t2 = obstruction["time"] - obstruction["time"][0]

        t = t[t < t2[len(t2) - 1]]
        t = t[time_offset:]

        obs_int = []
        for i in range(5):
            obs = (obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0]
            fobs = interp1d(t2, obs)

            obs_int.append(fobs(t))

        sens1 = ((df["Sensor1"] - df["Sensor1"][0]) * 5/1024).tolist()
        sens1 = sens1[time_offset:len(t)+time_offset]
        sens2 = ((df["Sensor2"] - df["Sensor2"][0]) * 5/1024).tolist()
        sens2 = sens2[time_offset:len(t)+time_offset]
        sens3 = ((df["Sensor3"] - df["Sensor3"][0]) * 5/1024).tolist()
        sens3 = sens3[time_offset:len(t)+time_offset]
        sens4 = ((df["Sensor4"] - df["Sensor4"][0]) * 5/1024).tolist()
        sens4 = sens4[time_offset:len(t)+time_offset]
        sens5 = ((df["Sensor5"] - df["Sensor5"][0]) * 5/1024).tolist()
        sens5 = sens5[time_offset:len(t)+time_offset]
        sens6 = ((df["Sensor6"] - df["Sensor6"][0]) * 5/1024).tolist()
        sens6 = sens6[time_offset:len(t)+time_offset]

        corr += np.corrcoef(obs_int[4], sens1)[0, 1]
        corr += np.corrcoef(obs_int[3], sens4)[0, 1]
        corr += np.corrcoef(obs_int[1], sens2)[0, 1]

        corrs.append(corr)

    # Show best shift
    idx = max(enumerate(corrs), key=lambda x: x[1])[0]
    print("Start Offset: {0:.2f}s".format(t[idx]))

    t = (df["Time"] - df["Time"][0])/1000
    t2 = obstruction["time"] - obstruction["time"][0]

    t = t[t < t2[len(t2) - 1]]
    t = t[idx:]

    obs_int = []
    for i in range(5):
        obs = (obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0]
        fobs = interp1d(t2, obs)

        obs_int.append(fobs(t))

    sens1 = ((df["Sensor1"] - df["Sensor1"][0]) * 5/1024)[idx:len(t)+idx]
    sens2 = ((df["Sensor2"] - df["Sensor2"][0]) * 5/1024)[idx:len(t)+idx]
    sens3 = ((df["Sensor3"] - df["Sensor3"][0]) * 5/1024)[idx:len(t)+idx]
    sens4 = ((df["Sensor4"] - df["Sensor4"][0]) * 5/1024)[idx:len(t)+idx]
    sens5 = ((df["Sensor5"] - df["Sensor5"][0]) * 5/1024)[idx:len(t)+idx]
    sens6 = ((df["Sensor6"] - df["Sensor6"][0]) * 5/1024)[idx:len(t)+idx]

    total_obs = []
    for i in range(5):
        total_obs.append((obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0])

    total_obs = np.array(total_obs)

    tot_corr[0] += np.corrcoef(obs_int[4], sens1)[0, 1] * len(sens1)
    tot_corr[1] += np.corrcoef(obs_int[3], sens4)[0, 1] * len(sens4)
    tot_corr[2] += np.corrcoef(obs_int[1], sens2)[0, 1] * len(sens2)
    n_samples += len(sens1)

    data[0].append(obs_int[4])
    data[0].append(obs_int[3])
    data[0].append(obs_int[1])

    data[1].append(sens1.tolist())
    data[1].append(sens4.tolist())
    data[1].append(sens2.tolist())

    data[2].append(obs_int[4][1:] - obs_int[4][:-1])
    data[2].append(obs_int[3][1:] - obs_int[3][:-1])
    data[2].append(obs_int[1][1:] - obs_int[1][:-1])

    data[3].append(np.array(sens1.tolist())[1:] - np.array(sens1.tolist())[:-1])
    data[3].append(np.array(sens4.tolist())[1:] - np.array(sens4.tolist())[:-1])
    data[3].append(np.array(sens2.tolist())[1:] - np.array(sens2.tolist())[:-1])

    #ax.scatter((obs_int[4])[::step], sens1[::step], color=EPFLcolors.colors[0])
    #ax.scatter((obs_int[3])[::step], sens4[::step], color=EPFLcolors.colors[0])
    #ax.scatter((obs_int[1])[::step], sens2[::step], color=EPFLcolors.colors[0])

data[0] = list(itertools.chain.from_iterable(data[0]))
data[1] = list(itertools.chain.from_iterable(data[1]))
data[2] = list(itertools.chain.from_iterable(data[2]))
data[3] = list(itertools.chain.from_iterable(data[3]))

data_array = np.array(data[0:2])
data_sort = data_array[:, data_array[0, :].argsort()]

val, low_bound, upp_bound = bounding_hull(np.array([data_sort[0, :], data_sort[1, :]]), 50)
ax.fill_between(val, low_bound, upp_bound, color=EPFLcolors.colors[0], alpha=0.3)
ax.plot(val, low_bound, color=EPFLcolors.colors[1])
ax.plot(val, upp_bound, color=EPFLcolors.colors[1])

poly = np.poly1d(np.polyfit(data_sort[0, :], data_sort[1, :], 1))
ax.plot(data_sort[0, ::step2], poly(data_sort[0, ::step2]),
        color=EPFLcolors.colors[2])

for k in tot_corr:
    print("Correlation: {0:.2f}".format(k/n_samples))

ax.text(0.3, -7, r"$\rho = {0:.2f}$".format(sum(tot_corr) / (3*n_samples)), horizontalalignment='center')
ax.text(0.3, -9, "(n = {0:d})".format(3*n_samples), horizontalalignment='center')
ax.text(0.3, -1, "y = {0:.2f}x - {1:.2f}".format(poly[1], -poly[0]), horizontalalignment='center', color=EPFLcolors.colors[2])

ax.set_xlabel("Visibility [-]")
ax.set_ylabel("Sensor Signal [V]")

plt.savefig(dir + r"\BlobingProximity.png")
import tikzplotlib
tikzplotlib.save(dir + r"\BlobingProximity.tex")

plt.show()

