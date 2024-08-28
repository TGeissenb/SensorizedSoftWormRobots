import numpy as np
import os
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import json

import EPFLcolors


path1 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs_Unidirectional"
path2 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs_Bidirectional"

params = ["spacing_x", "spacing_y", "ground_friction", "base_length", "actuator_length", "base", "mass",
          "center_of_mass", "stiffness"]

fig, axs = plt.subplots(figsize=(16, 8), nrows=len(params), ncols=len(params))

logs_uni = []
logs_bi = []

# Vectorize data as [params; distance]
for file in os.listdir(path1):
    with open(os.path.join(path1, file), 'r') as json_file:
        parameters = json.load(json_file)

    trajectories = np.array(parameters["Blob Trajectory"])
    orientations = np.array(parameters["Orientation"])

    data = []
    for i, key1 in enumerate(params):
        data.append(parameters["Parameters"][key1])

    if np.max(trajectories[:, 2]) < 2:
        data.append(-np.min(trajectories[:, 1]))
    else:
        data.append(-1)

    logs_uni.append(data)

# Vectorize data as [params; distance]
for file in os.listdir(path2):
    with open(os.path.join(path2, file), 'r') as json_file:
        parameters = json.load(json_file)

    trajectories = np.array(parameters["Blob Trajectory"])
    orientations = np.array(parameters["Orientation"])

    data = []
    for i, key1 in enumerate(params):
        data.append(parameters["Parameters"][key1])

    if np.max(trajectories[:, 2]) < 2:
        data.append(-np.min(trajectories[:, 1]))
    else:
        data.append(-1)

    logs_bi.append(data)

logs_uni = np.array(logs_uni)
logs_bi = np.array(logs_bi)

# Plot solution spaces
for i, key1 in enumerate(params):
    for j, key2 in enumerate(params):
        sc = axs[i, j].scatter(logs_bi[:, i], logs_bi[:, j], c=logs_bi[:, -1], edgecolor='none', s=1)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
fig.colorbar(sc, cax=cbar_ax)
plt.show()

sorted_idcs_uni = np.argpartition(logs_uni[:, -1], 4)
sorted_idcs_uni = sorted_idcs_uni[np.argsort(logs_uni[sorted_idcs_uni, -1])]
sorted_idcs_uni = sorted_idcs_uni[::-1]

sorted_idcs_bi = np.argpartition(logs_bi[:, -1], 4)
sorted_idcs_bi = sorted_idcs_bi[np.argsort(logs_bi[sorted_idcs_bi, -1])]
sorted_idcs_bi = sorted_idcs_bi[::-1]

fig2, ax1 = plt.subplots(1, 1)
bar_labels = [str(x) for x in sorted_idcs_uni]
ax1.scatter(np.arange(len(sorted_idcs_uni)), logs_uni[sorted_idcs_uni, -1], color=EPFLcolors.colors[0])
bar_labels = [str(x) for x in sorted_idcs_bi]
ax1.scatter(np.arange(len(sorted_idcs_bi)), logs_bi[sorted_idcs_bi, -1], color=EPFLcolors.colors[2])
#ax1.set_xticklabels(bar_labels, rotation=90)
ax1.set_xlabel("Configuration number")
ax1.set_ylabel("Travelled distance in target direction")
ax1.legend(["Unidirectional bending", "Bidirectional bending"])
#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\SoMo\Unidirectional_Bidirectional.tex")

fig3, axs2 = plt.subplots(1, len(params), sharey="all")
for i, key1 in enumerate(params):
    axs2[i].scatter(logs_uni[:, i], logs_uni[:, -1], c=[EPFLcolors.colors[0]] * len(logs_uni[:, -1]), edgecolor='none', s=1)
    axs2[i].scatter(logs_bi[:, i], logs_bi[:, -1], c=[EPFLcolors.colors[2]] * len(logs_bi[:, -1]), edgecolor='none', s=1)
    axs2[i].set_xlabel(key1)

axs2[0].set_ylabel("Travelled distance in target direction")
#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\SoMo\Unidirectional_Bidirectional_SolSpace.tex")
