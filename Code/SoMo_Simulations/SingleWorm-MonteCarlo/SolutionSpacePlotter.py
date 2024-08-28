import numpy as np
import os
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import json

import EPFLcolors


path1 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs_Comparison_unidirectional"
path2 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs_Comparison_bidirectional"
path3 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs_Bidirectional"

params = ["spacing_x", "spacing_y", "ground_friction", "base_length", "actuator_length", "base", "mass",
          "center_of_mass", "stiffness"]

fig, axs = plt.subplots(figsize=(16, 8), nrows=len(params), ncols=len(params))

logs_uni = []
logs_bi = []
logs_bi_nocomp = []

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

# Vectorize data as [params; distance]
for file in os.listdir(path3):
    with open(os.path.join(path3, file), 'r') as json_file:
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

    logs_bi_nocomp.append(data)

logs_uni = np.array(logs_uni)
logs_bi = np.array(logs_bi)
logs_bi_nocomp = np.array(logs_bi_nocomp)

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
bar_labels = [str(x) for x in sorted_idcs_bi]
ax1.scatter(np.arange(len(sorted_idcs_bi)), logs_uni[sorted_idcs_bi, -1], color=EPFLcolors.colors[0])
bar_labels = [str(x) for x in sorted_idcs_bi]
ax1.scatter(np.arange(len(sorted_idcs_bi)), logs_bi[sorted_idcs_bi, -1], color=EPFLcolors.colors[2])
#ax1.set_xticklabels(bar_labels, rotation=90)
ax1.set_xlabel("Configuration number")
ax1.set_ylabel("Travelled distance in target direction")
ax1.legend(["Unidirectional bending", "Bidirectional bending"])
plt.show()

bid = 0
dist_bid = 0.0
dist_avg = 0.0
for k in sorted_idcs_bi:
    dist_bid += logs_bi[k, -1] - logs_uni[k, -1]
    dist_avg += logs_bi[k, -1]
    if logs_bi[k, -1] >= logs_uni[k, -1]:
        bid += 1
print("Bidirectional bending superior in {0:.2f}% of cases".format(bid/len(sorted_idcs_bi) * 100))
print("Bidirectional bending superior with {0:.2f} more distance travelled on average".format(dist_bid/len(sorted_idcs_bi)))
print("Bidirectional bending achieves average travelled distance of {0:.2f}".format(dist_avg/len(sorted_idcs_bi)))

#import tikzplotlib
#tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\SoMo\Unidirectional_Bidirectional.tex")

logs_bi = logs_bi[np.where(logs_bi[:, -1] >= dist_avg/len(sorted_idcs_bi))]
logs_bi_nocomp = logs_bi_nocomp[np.where(logs_bi_nocomp[:, -1] >= dist_avg/len(sorted_idcs_bi))]
fig3, axs2 = plt.subplots(1, len(params), sharey="all")
for i, key1 in enumerate(params):
    if key1 in ["base_length", "actuator_length"]:
        factor = 50
    else:
        factor = 1

    axs2[i].scatter(factor*logs_bi[:, i], logs_bi[:, -1], edgecolor='none', s=1)
    axs2[i].plot([np.min(factor * logs_bi[:, i]), np.max(factor * logs_bi[:, i])], [0, 0])
    axs2[i].scatter(factor*logs_bi_nocomp[:, i], logs_bi_nocomp[:, -1], edgecolor='none', s=1)
    axs2[i].plot([np.min(factor * logs_bi_nocomp[:, i]), np.max(factor * logs_bi_nocomp[:, i])], [0, 0])
    axs2[i].set_xlabel(key1)

axs2[0].set_ylabel("Travelled distance in target direction")
#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\SoMo\Unidirectional_Bidirectional_SolSpace.tex")
