import numpy as np
import os
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import json

import EPFLcolors


path1 = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\BlobSimulationOrientationControl\Logs_PhaseShift"

params = ["obs_height", "spacing_x", "ground_friction", "base_length", "actuator_length", "mass",
          "center_of_mass", "stiffness", "torque"]

fig, axs = plt.subplots(figsize=(16, 8), nrows=len(params), ncols=len(params))

logs = []

# Vectorize data as [params; distance]
for file in os.listdir(path1):
    with open(os.path.join(path1, file), 'r') as json_file:
        parameters = json.load(json_file)

    trajectories = np.array(parameters["Blob Trajectory"])

    data = []
    for i, key1 in enumerate(params):
        data.append(parameters["Parameters"][key1])

    if np.max(trajectories[:, :, 2]) < 10:
        goals = -np.min(trajectories[:, :, 1], axis=0)
        data.append(np.mean(goals))
        data.append(np.std(goals))
    else:
        data.append(-1)
        data.append(-1)

    logs.append(data)

logs = np.array(logs)

# Plot solution spaces
for i, key1 in enumerate(params):
    for j, key2 in enumerate(params):
        sc = axs[i, j].scatter(logs[:, i], logs[:, j], c=logs[:, -2], edgecolor='none', s=1)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.7])
fig.colorbar(sc, cax=cbar_ax)
plt.show()

fig3, axs2 = plt.subplots(1, len(params), sharey="all")
for i, key1 in enumerate(params):
    axs2[i].scatter(logs[:, i], logs[:, -2], c=[EPFLcolors.colors[0]] * len(logs[:, -2]), edgecolor='none', s=1)
    axs2[i].set_xlabel(key1)

axs2[0].set_ylabel("Travelled distance in target direction")
plt.show()

fig4, axs3 = plt.subplots(1, 1)
axs3.scatter(logs[:, -1], logs[:, -2], c=[EPFLcolors.colors[0]] * len(logs[:, -2]), edgecolor='none', s=5)
axs3.set_xlabel("Standard deviation of worm end locations")
axs3.set_ylabel("Travelled distance in target direction")
plt.show()

#import tikzplotlib
#tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\SoMo\Blob_ObjectiveFunction.tex")

k = (np.where(logs[:, -2] == np.max(logs[:, -2])))[0][0]
print("Best parameters: " + (os.listdir(path1)[k]))
for i, key in enumerate(params):
    val = logs[k, i].astype(np.float32)
    print(key + ": " + str(val))
