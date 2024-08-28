# Import
import EPFLcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter

l = 150

value_dict = {"Resistance": [],
              "Resistance_Orig": [],
              "Resistance_Cycle": [],
              "Max_Resistance": [],
              "Min_Resistance": [],
              "Max_Resistance_Orig": [],
              "Min_Resistance_Orig": [],
              "Strain": [],
              "Strain_Cycle": []}

plot_val = {"Block": [],
            "Cycles": [],
            "NumCycles": []}

#for i in range(200):

cycle = 1
block = 99
last_resist = 0
with open(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\StrainSensor\CycleTest\20240710_CycleTest_Sensor.csv") as fp:
    Lines = fp.readlines()
    for count, line in enumerate(Lines):

        value = line.split(",")[0]
        try:
            resist = float(value)
            if (len(value_dict["Resistance"]) < 250 or resist - max(value_dict["Resistance"][-250:-10]) < 2500000) and resist < 75000000:
                value_dict["Resistance"].append(resist)
            else:
                print("Line {0:d}: {1}".format(count, value))
                value_dict["Resistance"].append(50000000)

            value_dict["Resistance_Orig"].append(resist)

            if resist > last_resist and block < 0:
                cycle += 1
                block = 90
            else:
                block = block - 1

            last_resist = resist
            value_dict["Resistance_Cycle"].append(cycle)
        except:
            pass

"""
    plot_val["Block"].append(i)
    plot_val["Cycles"].append(value_dict["Resistance_Cycle"][-1])

for key in ["Block", "Cycles"]:
    plot_val[key] = np.array(plot_val[key])

for i in plot_val["Cycles"]:
    plot_val["NumCycles"].append(np.count_nonzero(plot_val["Cycles"] == i))

print(plot_val["Cycles"][82])
print(plot_val["Cycles"][175])

plt.plot(plot_val["Block"], plot_val["NumCycles"])
plt.show()
"""

res_cyc_array = np.array(value_dict["Resistance_Cycle"])
res_array = np.array(value_dict["Resistance"])

for cyc in range(1, cycle+1):
    value_dict["Max_Resistance_Orig"].append(
        np.max(res_array[np.where(res_cyc_array == cyc)]))
    value_dict["Min_Resistance_Orig"].append(
        np.min(res_array[np.where(res_cyc_array == cyc)]))

value_dict["Min_Resistance_Orig"][-1] = value_dict["Min_Resistance_Orig"][-2]

value_dict["Max_Resistance"] = savgol_filter(value_dict["Max_Resistance_Orig"], 100, 3)
value_dict["Min_Resistance"] = savgol_filter(value_dict["Min_Resistance_Orig"], 100, 3)

# reading csv file
df_instron = pd.read_csv(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\05 Sensors\Spined Phototransistor Worm\StrainSensor\CycleTest\20240710_CycleTest_Instron.csv",
                        sep=",")

cycle = 1
block = 4293
last_strain = 0
for disp in df_instron["Displacement"]:
    value_dict["Strain"].append(disp/l)
    if disp/l > last_strain and block < 0:
        cycle += 1
        block = 3000
    else:
        block = block - 1

    last_strain = disp / l
    value_dict["Strain_Cycle"].append(cycle)

for key in value_dict.keys():
    value_dict[key] = np.array(value_dict[key])

num_cycles = max(value_dict["Resistance_Cycle"])
print("Number of Cycles: {0:d}".format(num_cycles))

strain = value_dict["Strain"][np.where(value_dict["Strain_Cycle"] < num_cycles)]
res = value_dict["Resistance"][np.where(value_dict["Resistance_Cycle"] < num_cycles)]

strain_one = value_dict["Strain"][np.where(value_dict["Strain_Cycle"] == 1)]
res_one = value_dict["Resistance"][np.where(value_dict["Resistance_Cycle"] == 1)]


t1 = np.linspace(0, 1, len(strain))
t2 = np.linspace(0, 1, len(res))
t = np.linspace(0, 0.99999, len(res))

f1 = interpolate.interp1d(t1, strain)
f2 = interpolate.interp1d(t2, res)

t1_one = np.linspace(0, 1, len(strain_one))
t2_one = np.linspace(0, 1, len(res_one))

f1_one = interpolate.interp1d(t1_one, strain_one)
f2_one = interpolate.interp1d(t2_one, res_one)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(value_dict["Strain_Cycle"], value_dict["Strain"], color=EPFLcolors.colors[0])

ax1.set_xlabel("Strain Cycle [-]")
ax1.set_ylabel("Strain [-]")
plt.show()

plot_x = []
plot_y = []
for i in range(max(value_dict["Resistance_Cycle"])):
    plot_x.append(i + 1)
    plot_x.append(i + 1)
    plot_y.append(value_dict["Min_Resistance_Orig"][i])
    plot_y.append(value_dict["Max_Resistance_Orig"][i])

fig, ax1 = plt.subplots(nrows=1, ncols=1)
#ax1.plot(value_dict["Resistance_Cycle"], value_dict["Resistance_Orig"], color=EPFLcolors.colors[2])
#ax1.plot(value_dict["Resistance_Cycle"], value_dict["Resistance"][::10], color=EPFLcolors.colors[0], alpha=0.3,
#         label="Raw Data")
ax1.plot(plot_x, plot_y, color=EPFLcolors.colors[0], alpha=0.3, label="Raw Data")
ax1.plot(np.arange(len(value_dict["Max_Resistance"])), value_dict["Max_Resistance"], color=EPFLcolors.colors[1],
         label="Resistance at 100% Strain")
ax1.plot(np.arange(len(value_dict["Min_Resistance"])), value_dict["Min_Resistance"], color=EPFLcolors.colors[1],
         label="Resistance at 0% Strain")
ax1.set_xlabel("Resistance Cycle [-]")
ax1.set_ylabel("Resistance [Ohm]")
plt.legend()
plt.show()

#import tikzplotlib
#tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\CarbonBlackSensor\CycleTest.tex")

fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(f1_one(t2_one), res_one, color=EPFLcolors.colors[0], alpha=0.3, label="Raw Data")
ax1.plot(f1_one(t2_one), savgol_filter(res_one, 30, 3), color=EPFLcolors.colors[1],
         label="Smoothed Resistance")
ax1.annotate("", xy=(0, 25000000), xytext=(0, 35000000), arrowprops=dict(arrowstyle="<|-|>", linewidth=1, color='k'))
ax1.text(0.1, 30000000, "Change in zero strain resistance", horizontalalignment='left')
ax1.set_xlabel("Strain [-]")
ax1.set_ylabel("Resistance [Ohm]")
plt.legend()
#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\CarbonBlackSensor\FirstHysteresis.tex")

x = f1(t2) * (f1(t+0.000001) - f1(t)) / abs(f1(t+0.000001) - f1(t))
z_plus = np.polyfit(x[np.where(x >= 0)], res[np.where(x >= 0)], 5)
f_plus = np.poly1d(z_plus)
z_minus = np.polyfit(x[np.where(x <= 0)], res[np.where(x <= 0)], 5)
f_minus = np.poly1d(z_minus)

strain_hyst = [[], []]
res_hyst = [[], []]
for i in t:
    strain_hyst[0].append(i)
    strain_hyst[1].insert(0, i)
    res_hyst[0].append(f_plus(i))
    res_hyst[1].insert(0, f_minus(-i))

strain_hyst = np.array(strain_hyst).flatten()
res_hyst = np.array(res_hyst).flatten()

fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(f1(t2)[::10], res[::10], color=EPFLcolors.colors[0], alpha=0.3, label="Raw Data")
#ax1.plot(f1(t2), f_plus(f1(t2)), color=EPFLcolors.colors[1], label="Median Resistance")
#ax1.plot(f1(t2), f_minus(-f1(t2)), color=EPFLcolors.colors[1], label="Median Resistance")
ax1.plot(strain_hyst[::100], savgol_filter(res_hyst[::100], 50, 3), color=EPFLcolors.colors[1],
         label="Median Resistance")
for i in range(1, 8):
    ax1.annotate("", xy=(strain_hyst[i*5000], res_hyst[i*5000]), xytext=(strain_hyst[i*5000+10], res_hyst[i*5000+10]),
                 arrowprops=dict(arrowstyle="<|-",
                                 color=EPFLcolors.colors[1]))
ax1.plot(f1(t2)[::100], np.median(value_dict["Min_Resistance"]) * (1+f1(t2)[::100])/(1-0.45*f1(t2)[::100]),
         color=EPFLcolors.colors[2], label="Gauge Factor")
ax1.set_xlabel("Strain [-]")
ax1.set_ylabel("Resistance [Ohm]")
plt.legend()
#plt.show()

import tikzplotlib
tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\CarbonBlackSensor\MedianResistance.tex")

print((value_dict["Min_Resistance"][0] - value_dict["Min_Resistance"][-1]) / value_dict["Min_Resistance"][0])
print((value_dict["Max_Resistance"][0] - value_dict["Max_Resistance"][-1]) / value_dict["Max_Resistance"][0])

plot_x = []
plot_y = []
for i in range(max(value_dict["Resistance_Cycle"])):
    plot_x.append(i + 1)
    plot_x.append(i + 1)
    plot_y.append(value_dict["Min_Resistance_Orig"][i]/value_dict["Min_Resistance_Orig"][i] - 1)
    plot_y.append(value_dict["Max_Resistance_Orig"][i]/value_dict["Min_Resistance_Orig"][i] - 1)

fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.plot(plot_x, plot_y, color=EPFLcolors.colors[0], alpha=0.3, label="Raw Data")
ax1.plot(np.arange(len(value_dict["Max_Resistance"])), value_dict["Max_Resistance"]/value_dict["Min_Resistance"] - 1, color=EPFLcolors.colors[1],
         label="Resistance at 100% Strain")
ax1.plot(np.arange(len(value_dict["Min_Resistance"])), value_dict["Min_Resistance"]/value_dict["Min_Resistance"] - 1, color=EPFLcolors.colors[1],
         label="Resistance at 0% Strain")
ax1.set_xlabel("Resistance Cycle [-]")
ax1.set_ylabel(r"$\frac{\Delta R}{R_0}$")
plt.legend()
plt.show()

dat = value_dict["Max_Resistance"]/value_dict["Min_Resistance"] - 1
print((dat[0] - dat[-1]) / dat[0])

print((res_one[0] - res_one[-1]) / res_one[0])
