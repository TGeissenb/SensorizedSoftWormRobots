import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import EPFLcolors

path = r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\InstronBlobing\SensorData"

filenames = []
for root, dirs, files in os.walk(path):
    for file in files:
        filenames.append(os.path.join(root, file))


forces = {"2": [],
          "4": [],
          "6": [],
          "8": []}


for file in filenames:

    df_sensor = pd.read_csv(file, sep=",")
    df_instron = pd.read_csv(file.replace("SensorData", "Instron_EntanglementStrength"), sep=",")

    n_worms = len(df_sensor.keys()) - 1

    forces["{0:d}".format(n_worms)].append(np.max(df_instron["Force"]))

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.violinplot([forces["2"], forces["4"], forces["6"], forces["8"]],
              showmeans=False, showmedians=True, showextrema=True)

ax.set_xticks(np.arange(1, 5), labels=forces.keys())
ax.set_xlabel("Number of entangled worms")
ax.set_ylabel("Maximum force until disentanglement [N]")

#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Phototransistor Spined Worm\DisentanglementForce.tex")
