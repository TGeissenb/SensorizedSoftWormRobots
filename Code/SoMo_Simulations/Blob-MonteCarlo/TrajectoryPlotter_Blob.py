import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json

jsons = [r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\BlobSimulationOrientationControl\Logs_PhaseShift\20240717_104355_Blob_Log.json"]


fig, ax = plt.subplots(nrows=1, ncols=1)
#ax = ax.reshape(-1)
ax = [ax]

for j, jsonf in enumerate(jsons):
    with open(jsonf, 'r') as json_file:
        parameters = json.load(json_file)

    trajectories = np.array(parameters["Blob Trajectory"])

    # Static Plotting
    if len(trajectories.shape) == 3:
        for i in range(trajectories.shape[1]):
            ax[j].plot(-trajectories[:, i, 0], -trajectories[:, i, 1])
    elif len(trajectories.shape) == 2:
        ax[0].plot(-trajectories[:, 0], -trajectories[:, 1])

    #ax[j].set_xlim([-10, 60])
    #ax[j].set_ylim([-30, 10])

    #ax[j].set_title("Obstacle spacing: {0:d}".format(j + 2))

plt.axis("equal")
plt.gca().invert_yaxis()
plt.show()

