import numpy as np
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import EPFLcolors
from CurvatureComputation_v2 import bbox

key_frames = 4
live_monitor = False
video_path = r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\Blobing\Test\20240724_Blobing_v6.mp4"

cap = cv2.VideoCapture(video_path)

lower_color_bounds = np.array((0, 10, 105), dtype=np.uint8, ndmin=1)
#lower_color_bounds = np.array((150, 150, 15), dtype=np.uint8, ndmin=1)
upper_color_bounds = np.array((70, 90, 255), dtype=np.uint8, ndmin=1)

obstruction = {"time": [],
               "val": [[], [], [], [], []]}

outvid = cv2.VideoWriter(video_path[:-4] + "_Mask.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, (int(cap.get(3)), int(cap.get(4))))

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(3))
height = int(cap.get(4))
i = 0

while(cap.isOpened()):
    ret, orig_frame = cap.read()

    if i % key_frames != 0:
        i += 1
        continue

    if ret:
        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(orig_frame, lower_color_bounds, upper_color_bounds)

        # Filter out noise from mask
        mask_filtered = cv2.GaussianBlur(mask, (3, 3), 0)
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, np.ones((30, 30), dtype=np.uint8))
        _, mask_filtered = cv2.threshold(mask_filtered, 245, 255, cv2.THRESH_BINARY)

        mask_rgb = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
        frame = orig_frame & mask_rgb

        # Find bounding box of mask
        rmin, rmax, cmin, cmax = bbox(mask)
        dr = rmax - rmin
        rows = [[rmin, int(rmin + 0.2*dr)],
                [int(rmin + 0.2*dr), int(rmin + 0.4*dr)],
                [int(rmin + 0.4*dr), int(rmin + 0.6*dr)],
                [int(rmin + 0.6*dr), int(rmin + 0.8*dr)],
                [int(rmin + 0.8*dr), rmax]]

        for row in rows:
            frame = cv2.rectangle(frame, (cmin, row[0]), (cmax, row[1]), (0, 0, 255), 3)

        obstruction["time"].append(i/60.0)
        obstruction["val"][0].append(mask[rows[0][0]:rows[0][1], :].sum().item())
        obstruction["val"][1].append(mask[rows[1][0]:rows[1][1], :].sum().item())
        obstruction["val"][2].append(mask[rows[2][0]:rows[2][1], :].sum().item())
        obstruction["val"][3].append(mask[rows[3][0]:rows[3][1], :].sum().item())
        obstruction["val"][4].append(mask[rows[4][0]:rows[4][1], :].sum().item())

        if live_monitor:
            cv2.imshow('newFrame', frame)
        outvid.write(frame)

    else:
        break

    i += 1
    print("Frame {0:d} of {1:d} analyzed. \t {2:.2f}% completed".format(i, frames, i/float(frames) * 100.0))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
outvid.release()
cv2.destroyAllWindows()

with open(video_path[:-4] + ".json", 'w') as f:
    json.dump(obstruction, f)

#with open(video_path[:-4] + ".json", 'r') as f:
#    obstruction = json.load(f)

obstruction["val"] = np.array(obstruction["val"])
obstruction["time"] = np.array(obstruction["time"])

df = pd.read_csv(video_path[:-4] + ".csv", sep=",")

fig, axs = plt.subplots(nrows=7, ncols=1, sharex='all', constrained_layout=True)

# Interpolations
corrs = []
for time_offset in range(0, 100, 2):
    corr = 0.0

    t = (df["Time"] - df["Time"][0]) / 1000
    t2 = obstruction["time"] - obstruction["time"][0]

    t = t[t < t2[len(t2) - 1]]
    t = t[time_offset:]

    obs_int = []
    for i in range(5):
        obs = (obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0]
        fobs = interp1d(t2, obs)

        obs_int.append(fobs(t))

    sens1 = ((df["Sensor1"] - df["Sensor1"][0]) * 5/1024)[0:len(t)]
    sens2 = ((df["Sensor2"] - df["Sensor2"][0]) * 5/1024)[0:len(t)]
    sens3 = ((df["Sensor3"] - df["Sensor3"][0]) * 5/1024)[0:len(t)]
    sens4 = ((df["Sensor4"] - df["Sensor4"][0]) * 5/1024)[0:len(t)]
    sens5 = ((df["Sensor5"] - df["Sensor5"][0]) * 5/1024)[0:len(t)]
    sens6 = ((df["Sensor6"] - df["Sensor6"][0]) * 5/1024)[0:len(t)]

    corr += np.corrcoef(obs_int[4], sens1)[0, 1]
    corr += np.corrcoef(obs_int[3], sens4)[0, 1]
    corr += np.corrcoef(obs_int[1], sens2)[0, 1]

    corrs.append(corr)

# Show best shift
idx = max(enumerate(corrs), key=lambda x: x[1])[0]
print("Start Offset: {0:d}".format(idx))

t = (df["Time"] - df["Time"][0])/1000
t2 = obstruction["time"] - obstruction["time"][0]

t = t[t < t2[len(t2) - 1]]
t = t[idx:]

obs_int = []
for i in range(5):
    obs = (obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0]
    fobs = interp1d(t2, obs)

    obs_int.append(fobs(t))

sens1 = ((df["Sensor1"] - df["Sensor1"][0]) * 5/1024)[0:len(t)]
sens2 = ((df["Sensor2"] - df["Sensor2"][0]) * 5/1024)[0:len(t)]
sens3 = ((df["Sensor3"] - df["Sensor3"][0]) * 5/1024)[0:len(t)]
sens4 = ((df["Sensor4"] - df["Sensor4"][0]) * 5/1024)[0:len(t)]
sens5 = ((df["Sensor5"] - df["Sensor5"][0]) * 5/1024)[0:len(t)]
sens6 = ((df["Sensor6"] - df["Sensor6"][0]) * 5/1024)[0:len(t)]

total_obs = []
for i in range(5):
    total_obs.append((obstruction["val"][i, :] - obstruction["val"][i, 0])/obstruction["val"][i, 0])

total_obs = np.array(total_obs)

step = 20

axs[0].plot(obstruction["time"][::step],
            ((np.sum(obstruction["val"], axis=0) - np.sum(obstruction["val"][:, 0])) / np.sum(obstruction["val"][:, 0]))[::step],
            color=EPFLcolors.colors[0])
axs[0].set_ylabel("Visibility of sensorized worm")

ax1 = axs[1].twinx()
ax1.plot(t[::step], (obs_int[4])[::step], color=EPFLcolors.colors[0])
axs[1].plot(t[::step], sens1[::step], color=EPFLcolors.colors[2], label="Tip Front")
axs[1].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[4], sens1)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[1].transAxes)
axs[1].set_ylabel("Tip Front [V]")
ax1.set_ylabel("Visibility")

ax2 = axs[2].twinx()
ax2.plot(t[::step], (obs_int[3])[::step], color=EPFLcolors.colors[0])
axs[2].plot(t[::step], sens4[::step], color=EPFLcolors.colors[2], label="Mid Front")
axs[2].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[3], sens4)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[2].transAxes)
axs[2].set_ylabel("Mid Front [V]")
ax2.set_ylabel("Visibility")

ax3 = axs[3].twinx()
ax3.plot(t[::step], (obs_int[1])[::step], color=EPFLcolors.colors[0])
axs[3].plot(t[::step], sens2[::step], color=EPFLcolors.colors[2], label="Base Front")
axs[3].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[1], sens2)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[3].transAxes)
axs[3].set_ylabel("Base Front [V]")
ax3.set_ylabel("Visibility")

axs[4].plot(t[::step], sens5[::step], label="Tip Rear")
axs[4].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[4], sens5)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[4].transAxes)
axs[4].set_ylabel("Tip Rear [V]")

axs[5].plot(t[::step], sens3[::step], label="Mid Rear")
axs[5].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[3], sens3)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[5].transAxes)
axs[5].set_ylabel("Mid Rear [V]")

axs[6].plot(t[::step], sens6[::step], label="Base Rear")
axs[6].text(0.97, 0.15, r'$\rho = ${0:.2f}'.format(np.corrcoef(obs_int[1], sens6)[0, 1]),
            horizontalalignment='right', verticalalignment='center', transform=axs[6].transAxes)
axs[6].set_ylabel("Base Rear [V]")

axs[6].set_xlabel("Time [s]")

plt.savefig(video_path[:-4] + ".png")
import tikzplotlib
tikzplotlib.save(video_path[:-4] + ".tex")

plt.show()
