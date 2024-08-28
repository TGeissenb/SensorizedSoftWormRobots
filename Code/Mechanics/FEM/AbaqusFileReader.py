import numpy as np
import matplotlib.pyplot as plt
import EPFLcolors


file1 = open(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\06 FEM\Abaqus Whole Worm\Worm_Linear_2.rpt", 'r')
file2 = open(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\06 FEM\Abaqus Whole Worm\Worm_Nonlinear_1_85.rpt", 'r')
Lines1 = file1.readlines()
Lines2 = file2.readlines()
file1.close()
file2.close()

scaling = 1000

# Store node positions
xyz1 = []
uvw1 = []

# Read file
for line in Lines1:
    id = line[20:31].strip()
    x = line[32:48].strip()
    y = line[48:64].strip()
    z = line[64:80].strip()
    u = line[80:96].strip()
    v = line[96:112].strip()
    w = line[112:128].strip()

    print("Id: {} \t x: {} \t y: {} \t z:{} \t u: {} \t v: {} \t w: {}".format(id, x, y, z, u, v, w))

    try:
        xyz1.append([-float(x) *scaling, float(z) *scaling, -float(y) *scaling])
        uvw1.append([-float(u) *scaling, float(w) *scaling, -float(v) *scaling])
    except:
        pass

# Convert to numpy array
xyz1 = np.array(xyz1)
uvw1 = np.array(uvw1)

# Get bounding box of initial worm
print("X: \t {} \t {}".format(xyz1.min(0)[0], xyz1.max(0)[0]))
print("Y: \t {} \t {}".format(xyz1.min(0)[1], xyz1.max(0)[1]))
print("Z: \t {} \t {}".format(xyz1.min(0)[2], xyz1.max(0)[2]))

# Store node positions
xyz2 = []
uvw2 = []

# Read file
for line in Lines2:
    id = line[20:31].strip()
    x = line[32:48].strip()
    y = line[48:64].strip()
    z = line[64:80].strip()
    u = line[80:96].strip()
    v = line[96:112].strip()
    w = line[112:128].strip()

    print("Id: {} \t x: {} \t y: {} \t z:{} \t u: {} \t v: {} \t w: {}".format(id, x, y, z, u, v, w))

    try:
        xyz2.append([-float(x) *scaling, float(z) *scaling, -float(y) *scaling])
        uvw2.append([-float(u) *scaling, float(w) *scaling, -float(v) *scaling])
    except:
        pass

# Convert to numpy array
xyz2 = np.array(xyz2)
uvw2 = np.array(uvw2)

# Plot nodal positions in 3D
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(projection="3d")

ax1.scatter(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], s=0.01, marker='.', edgecolors=None, color=EPFLcolors.colors[5])
ax1.scatter(uvw1[:, 0], uvw1[:, 1], uvw1[:, 2], s=0.01, marker='.', edgecolors=None, color=EPFLcolors.colors[2])
ax1.scatter(uvw2[:, 0], uvw2[:, 1], uvw2[:, 2], s=0.01, marker='.', edgecolors=None, color=EPFLcolors.colors[3])

theta = np.linspace(0, 6.27 * np.pi/180, 100)
ax1.plot(2010.37 - 2010.37 * np.cos(theta), 2010.37 * np.sin(theta), linewidth=8, color=EPFLcolors.colors[0])

ax1.legend(["Undeformed nodes", "Deformed nodes linear", "Deformed nodes nonlinear", "Analytical result"])
ax1.set_xlabel("X [mm]")
ax1.set_ylabel("Y [mm]")
ax1.set_zlabel("Z [mm]")

ax1.axis("equal")

plt.show()

###################################################################################################

file1 = open(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\07 Simulation\Abaqus\Worm_Linear.rpt", 'r')
Lines = file1.readlines()
file1.close()

scaling = 1000

# Store node positions
xyz = []
uvw = []

# Read file
for line in Lines:
    id = line[15:25].strip()
    x = line[25:41].strip()
    y = line[41:58].strip()
    z = line[58:73].strip()
    u = line[73:89].strip()
    v = line[89:105].strip()
    w = line[105:121].strip()

    print("Id: {} \t x: {} \t y: {} \t z:{} \t u: {} \t v: {} \t w: {}".format(id, x, y, z, u, v, w))

    try:
        xyz.append([-float(x) *scaling, -float(y) *scaling, -float(z) *scaling])
        uvw.append([-float(u) *scaling, -float(v) *scaling, -float(w) *scaling])
    except:
        pass

# Convert to numpy array
xyz = np.array(xyz)
uvw = np.array(uvw)

# Get bounding box of initial worm
print("X: \t {} \t {}".format(xyz.min(0)[0], xyz.max(0)[0]))
print("Y: \t {} \t {}".format(xyz.min(0)[1], xyz.max(0)[1]))
print("Z: \t {} \t {}".format(xyz.min(0)[2], xyz.max(0)[2]))

# Plot nodal positions in 3D
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(projection="3d")

ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.01, marker='.', edgecolors=None, color=EPFLcolors.colors[5])
ax1.scatter(uvw[:, 0], uvw[:, 1], uvw[:, 2], s=0.01, marker='.', edgecolors=None, color=EPFLcolors.colors[2])

theta = np.linspace(0, 32.86 * np.pi/180, 100)
ax1.plot(395.21 - 395.21 * np.cos(theta), 395.21 * np.sin(theta), linewidth=7, color=EPFLcolors.colors[0])

ax1.legend(["Undeformed nodes", "Deformed nodes linear", "Analytical result"])
ax1.set_xlabel("X [mm]")
ax1.set_ylabel("Y [mm]")
ax1.set_zlabel("Z [mm]")

ax1.axis("equal")

plt.show()
