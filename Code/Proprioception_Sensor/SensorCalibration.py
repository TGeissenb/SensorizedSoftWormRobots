import numpy as np
import pandas as pd
import cv2
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import LinearRegression
import json

import EPFLcolors
from Utils.CurvatureComputation_v2 import closest_point, angle_between


key_frames = 15
live_monitor = False
video_path = r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\SlowCycleResponse\20240728_CycleResponse_v10.mp4"

df = pd.read_csv(video_path[:-4] + ".csv", sep=",")

cap = cv2.VideoCapture(video_path)

lower_color_bounds = np.array((15, 35, 165), dtype=np.uint8, ndmin=1)
#lower_color_bounds = np.array((150, 150, 150), dtype=np.uint8, ndmin=1)
upper_color_bounds = np.array((255, 255, 255), dtype=np.uint8, ndmin=1)

cg = []
avg_curv = []
bend_angle = []

outvid = cv2.VideoWriter(video_path[:-4] + "_Mask.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         120/key_frames, (int(cap.get(3)), int(cap.get(4))))

outvid2 = cv2.VideoWriter(video_path[:-4] + "_Skeleton.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         120/key_frames, (int(cap.get(3)), int(cap.get(4))))

outvid3 = cv2.VideoWriter(video_path[:-4] + "_Curve.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         120/key_frames, (640, 480))

outvid4 = cv2.VideoWriter(video_path[:-4] + "_DynamicPlot.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         120/key_frames, (640, 480))

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(3))
height = int(cap.get(4))
i = 0
fig, ax = plt.subplots(nrows=1, ncols=1)
fig2 = plt.figure(constrained_layout=True)
gs = fig2.add_gridspec(2, 2)
axs0 = fig2.add_subplot(gs[0, 0])
axs1 = fig2.add_subplot(gs[0, 1])
axs2 = fig2.add_subplot(gs[1, :])
fig3, axs = plt.subplots(nrows=2, ncols=1)
axs[0].invert_yaxis()
ax2 = axs[1].twinx()

while(cap.isOpened()):
    ret, orig_frame = cap.read()

    if i % key_frames != 0:
        i += 1
        continue

    if ret:
        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(orig_frame, lower_color_bounds, upper_color_bounds)

        # Filter out noise from mask
        mask_filtered = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, np.ones((21, 21), dtype=np.uint8))
        _, mask_filtered = cv2.threshold(mask_filtered, 200, 255, cv2.THRESH_BINARY)

        mask_rgb = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
        frame = orig_frame & mask_rgb

        # Compute center of gravity
        cg.append(scipy.ndimage.center_of_mass(mask_rgb))

        # Find skeleton of mask
        skeleton = skeletonize(mask_filtered, method="lee").astype(dtype=np.uint8)
        skeleton_rgb = cv2.cvtColor(255 * skeleton, cv2.COLOR_GRAY2BGR)
        skeleton_mask = orig_frame & skeleton_rgb

        # Get positions of skeleton points and order them according to distance
        x, y = np.where(skeleton == 1)

        x_ord = [x[0]]
        y_ord = [y[0]]
        x = np.delete(x, 0, axis=0)
        y = np.delete(y, 0, axis=0)
        idx = closest_point(np.array([x, y]).T, (x_ord[-1], y_ord[-1]), threshold=10)
        while idx is not None and x.shape[0] > 1:
            x_ord.append(x[idx])
            y_ord.append(y[idx])
            x = np.delete(x, idx, axis=0)
            y = np.delete(y, idx, axis=0)
            idx = closest_point(np.array([x, y]).T, (x_ord[-1], y_ord[-1]), threshold=10)

        x_ord = np.array(x_ord)
        y_ord = np.array(y_ord)

        # Fitting a curve
        t = np.linspace(0, 1, len(x_ord))
        fx = UnivariateSpline(t, x_ord, k=3, s=None)
        fy = UnivariateSpline(t, y_ord, k=3, s=None)
        coeff_x = fx.get_coeffs()
        coeff_y = fy.get_coeffs()

        # Differentiate and compute curvature
        fx_ = fx.derivative()
        fx__ = fx_.derivative()
        fy_ = fy.derivative()
        fy__ = fy_.derivative()

        k = abs(fx_(t) * fy__(t) - fy_(t) * fx__(t)) / (fx_(t) ** 2 + fy_(t) ** 2) ** 1.5

        # Compute average curvature within percentiles
        #lower_thresh = np.percentile(k, 10, method="hazen")
        #upper_thresh = np.percentile(k, 90, method="hazen")
        #k_filtered = k[np.where(np.logical_and(k >= lower_thresh, k <= upper_thresh))]
        avg_curv.append(np.mean(k))

        # Compute bending angle
        vec1 = np.array([- (fx(0.05) - fx(0)), (fy(0.05) - fy(0))])
        vec2 = np.array([- (fx(1) - fx(0.95)), (fy(1) - fy(0.95))])
        alpha = angle_between(vec1, vec2)
        bend_angle.append(alpha * 180/np.pi)

        # Plot mathematical curve
        axs0.cla()
        axs1.cla()
        axs2.cla()
        axs0.plot(t, fx(t), label="x(t)", color=EPFLcolors.colors[2])
        axs0.plot(t, fy(t), label="y(t)", color=EPFLcolors.colors[3])
        axs0.set_xlabel("t")
        axs0.set_ylabel("x/y")
        axs0.set_xlim([0, 1])
        axs0.set_ylim([0, max([int(cap.get(3)), int(cap.get(4))])])
        axs1.plot(t, k, label="$\kappa(t)$", color=EPFLcolors.colors[0])
        axs1.plot(t, np.mean(k) * np.ones(len(t)), label="$\kappa_{avg}$", color=EPFLcolors.colors[1])
        axs1.set_xlabel("t")
        axs1.set_ylabel("$\kappa$")
        axs1.set_xlim([0, 1])
        axs1.set_ylim([0, 0.04])
        axs2.scatter(fy(t), fx(t), label="(y/x)", color=EPFLcolors.colors[0])
        stepsize = 0.3
        axs2.plot([fy(0), fy(0) - stepsize * (fx(0.05) - fx(0))], [fx(0), fx(0) + stepsize * (fy(0.05) - fy(0))], color="k")
        axs2.plot([fy(1), fy(1) - stepsize * (fx(1) - fx(0.95))], [fx(1), fx(1) + stepsize * (fy(1) - fy(0.95))], color="k")
        axs2.text(0.85 * fy(0.5), fx(0.5), "{0:.2f}°".format(alpha * 180/np.pi))
        axs2.set_xlabel("y")
        axs2.set_ylabel("x")
        axs2.axis("equal")
        axs2.set_xlim([0, width])
        axs2.set_ylim([height, -height])

        lines_labels = [ax.get_legend_handles_labels() for ax in fig2.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig2.legend(lines, labels)

        canvas = FigureCanvas(fig2)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

        # Dynamic plot of curvature
        axs[0].cla()
        axs[1].cla()
        cg_mat = np.array(cg)
        #axs[0].plot(cg_mat[:, 1], cg_mat[:, 0])
        #axs[0].set_xlabel("X [Pixel]")
        #axs[0].set_ylabel("Y [Pixel]")
        #axs[0].axis("equal")
        axs[0].axis("off")
        axs[0].set_xlim([0, width])
        axs[0].set_ylim([height, -height])

        fcap1 = UnivariateSpline(np.linspace(0, frames/key_frames, len(df["Resistance"])), df["Resistance"],
                                 k=3, s=0)

        axs[1].plot(range(len(avg_curv)), avg_curv, color=EPFLcolors.colors[0])
        ax2.plot(np.linspace(0, len(avg_curv), len(avg_curv)), fcap1(np.linspace(0, len(avg_curv), len(avg_curv))),
                 color=EPFLcolors.colors[2])
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Average Curvature [-]")
        ax2.set_ylabel("Measured Resistance [$\Omega$]")
        axs[1].set_xlim([0, frames/key_frames])
        axs[1].set_ylim([0, 0.04])
        ax2.set_ylim([500000, 3000000])
        custom_lines = [Line2D([0], [0], color=EPFLcolors.colors[0], lw=4),
                        Line2D([0], [0], color=EPFLcolors.colors[2], lw=4)]
        axs[1].legend(custom_lines, ["Curvature", "Resistance 1", "Resistance 2"], bbox_to_anchor=(0.05, 0.95), loc='upper left')

        canvas = FigureCanvas(fig3)
        canvas.draw()
        mat2 = np.array(canvas.renderer._renderer)
        mat2 = cv2.cvtColor(mat2, cv2.COLOR_RGB2BGR)

        if live_monitor:
            cv2.imshow('newFrame', skeleton_mask)
        outvid.write(frame)
        outvid2.write(skeleton_mask)
        outvid3.write(mat)
        outvid4.write(mat2)

    else:
        break

    i += 1
    print("Frame {0:d} of {1:d} analyzed. \t {2:.2f}% completed".format(i, frames, i/float(frames) * 100.0))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
outvid.release()
outvid2.release()
cv2.destroyAllWindows()

cg = np.array(cg)
bend_angle_norm = bend_angle - bend_angle[0]

plt.close(fig)
plt.close(fig2)
plt.close(fig3)
"""
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].invert_yaxis()

axs[0].plot(cg[:, 1], cg[:, 0])
axs[0].set_xlabel("X [Pixel]")
axs[0].set_ylabel("Y [Pixel]")

ax2 = axs[1].twinx()
# axs[1].plot(range(len(bend_angle)), bend_angle_norm, color=EPFLcolors.colors[0])
axs[1].plot(range(len(avg_curv)), avg_curv, color=EPFLcolors.colors[0])
ax2.plot(np.linspace(0, len(bend_angle), len(df["Capacitance"])), df["Capacitance"]*10**12, color=EPFLcolors.colors[2])
axs[1].set_xlabel("Frame")
#axs[1].set_ylabel("Bending Angle [°]")
axs[1].set_ylabel("Average Curvature [-]")
ax2.set_ylabel("Measured Capacitance [pF]")
"""
fig, ax = plt.subplots(nrows=1, ncols=1)
ax2 = ax.twinx()
ax.plot(range(len(avg_curv)), avg_curv, color=EPFLcolors.colors[0])
ax2.plot(np.linspace(0, len(avg_curv), len(df["Resistance"])), df["Resistance"], color=EPFLcolors.colors[2])
ax.set_xlabel("Frame")
ax.set_ylabel("Curvature [-]")
ax2.set_ylabel("Measured Resistance [$\Omega$]")

fig.savefig(video_path[:-4] + "_Summary.png")
plt.show()

#import tikzplotlib
#tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Bending_Sensor\CBBendingSensor.tex")

with open(video_path[:-4] + ".json", 'w') as f:
    json.dump({"Curvature": avg_curv,
               "Bending Angle": bend_angle,
              "Resistance": df["Resistance"].tolist()}, f)
