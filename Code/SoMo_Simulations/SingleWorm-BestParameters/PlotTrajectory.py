import numpy as np
import os
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import json

path = r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\LocomotingThreeChannelWorm\Logs\20240616_113701_Blob_Log.json"

fig, axs = plt.subplots(figsize=(16, 8), nrows=1, ncols=1)

with open(path, 'r') as json_file:
    parameters = json.load(json_file)

trajectories = np.array(parameters["Blob Trajectory"])
orientations = np.array(parameters["Orientation"])

axs.plot(-trajectories[:, 0], -trajectories[:, 1])
axs.axis("equal")

plt.show()
