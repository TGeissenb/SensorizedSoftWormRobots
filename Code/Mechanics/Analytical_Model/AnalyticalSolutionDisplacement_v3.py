import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.optimize import newton, fsolve
import json
import os
import random

import EPFLcolors
from utils import matrix_padding, mean
import tikzplotlib

from HyperparameterSearch import goldensection

# Colors
top = mpl.colormaps['Blues'].resampled(128)
bottom = mpl.colormaps['Reds'].resampled(128)

newcolors = np.vstack((top(np.linspace(1, 0, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='RedBlue')


def analytical_model(R, d, r, l, p_i, p_o, disc_t, disc_r, E, nu, c1, c2, c3, c4, c5, c6, plots=False, save_plots=False, nlgeom=True):

    theta = np.arange(disc_t, 2 * np.pi, disc_t)

    # Analytical solution for wall thickness
    t = ((R * np.sin(- 2 * np.arctan2((np.sqrt(R ** 2 / np.tan(theta) ** 2 - d ** 2 + R ** 2) + R / np.tan(theta) * (
                theta - np.pi) / abs(theta - np.pi)), (d + R)))
          * 1 / np.sin(theta) * (theta - np.pi) / abs(theta - np.pi)) - r)

    if plots:
        plt.plot(theta, t)
        plt.xlabel('Theta [rad]')
        plt.ylabel('t [mm]')
        if save_plots:
            tikzplotlib.save(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Wall_Thickness.tex")
        else:
            plt.show()

    # Compute stresses using Lamé's formula for linear elasticity dependent of thickness t(theta) and radial position k
    sigma_r = []
    sigma_t = []
    sigma_a = []
    for ti in t:
        radial_coordinate = np.arange(r, r + ti, disc_r)
        stress_r_dist = []
        stress_t_dist = []
        stress_a_dist = []
        for k in radial_coordinate:
            # Lamé's formula
            sig_r = (r ** 2 * p_i - (r + ti) ** 2 * p_o) / ((r + ti) ** 2 - r ** 2) - (
                        (p_i - p_o) * r ** 2 * (r + ti) ** 2) / (((r + ti) ** 2 - r ** 2) * k ** 2)
            sig_t = (r ** 2 * p_i - (r + ti) ** 2 * p_o) / ((r + ti) ** 2 - r ** 2) + (
                        (p_i - p_o) * r ** 2 * (r + ti) ** 2) / (((r + ti) ** 2 - r ** 2) * k ** 2)
            sig_a = (r ** 2 * (p_i - p_o)) / ((r + ti) ** 2 - r ** 2)
            if nlgeom:
                stress_r_dist.append(sig_r)
                stress_t_dist.append(sig_t)
                stress_a_dist.append(c1 * np.power(sig_a, 5) +
                                     c2 * np.power(sig_a, 3) +
                                     c3 * np.power(sig_a, 1))
            else:
                stress_r_dist.append(sig_r)
                stress_t_dist.append(sig_t)
                stress_a_dist.append(sig_a)
        sigma_r.append(stress_r_dist)
        sigma_t.append(stress_t_dist)
        sigma_a.append(stress_a_dist)

    # Pad stress matrices
    sigma_r = np.array(matrix_padding(sigma_r))
    sigma_t = np.array(matrix_padding(sigma_t))
    sigma_a = np.array(matrix_padding(sigma_a))

    if plots:
        min_stress = min(np.nanmin(sigma_r), np.nanmin(sigma_t), np.nanmin(sigma_a))
        max_stress = max(np.nanmax(sigma_r), np.nanmax(sigma_t), np.nanmax(sigma_a))
        lim_stress = max(abs(min_stress), abs(max_stress))

        fig = plt.figure(figsize=(8, 4.5))
        plt.rcParams.update({'font.size': 16})
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(sigma_a).T, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_stress, vmax=lim_stress)
        # plt.title("Axial stress")
        bar = fig.colorbar(pc)
        bar.set_label(r'Axial stress [$\frac{N}{mm^2}$]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Axial_Stress.png")
        else:
            plt.show()

        fig = plt.figure(figsize=(8, 4.5))
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(sigma_r).T * 1e6, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_stress, vmax=lim_stress)
        # plt.title("Radial stress")
        bar = fig.colorbar(pc)
        bar.set_label(r'Radial stress [$\frac{N}{mm^2}$]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Radial_Stress.png")
        else:
            plt.show()

        fig = plt.figure(figsize=(8, 4.5))
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(sigma_t).T * 1e6, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_stress, vmax=lim_stress)
        # plt.title("Circumferential stress")
        bar = fig.colorbar(pc)
        bar.set_label(r'Circumferential stress [$\frac{N}{mm^2}$]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Circumferential_Stress.png")
        else:
            plt.show()

    # Linear elastic model
    strain_r = 1 / E * sigma_r - nu / E * sigma_t - nu / E * sigma_a
    strain_t = 1 / E * sigma_t - nu / E * sigma_r - nu / E * sigma_a
    strain_a = 1 / E * sigma_a - nu / E * sigma_r - nu / E * sigma_t

    if plots:
        min_strain = min(np.nanmin(strain_r), np.nanmin(strain_t), np.nanmin(strain_a))
        max_strain = max(np.nanmax(strain_r), np.nanmax(strain_t), np.nanmax(strain_a))
        lim_strain = max(abs(min_strain), abs(max_strain))

        fig = plt.figure(figsize=(8, 4.5))
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(strain_a).T, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_strain, vmax=lim_strain)
        # plt.title("Axial strain")
        bar = fig.colorbar(pc)
        bar.set_label('Axial strain [-]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Axial_Strain.png")
        else:
            plt.show()

        fig = plt.figure(figsize=(8, 4.5))
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(strain_r).T, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_strain, vmax=lim_strain)
        # plt.title("Radial strain")
        bar = fig.colorbar(pc)
        bar.set_label('Radial strain [-]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Radial_Strain.png")
        else:
            plt.show()

        fig = plt.figure(figsize=(8, 4.5))
        ax1 = plt.subplot(projection="polar")
        circ = mpatches.Circle((0, 0), np.ceil(R + d), facecolor=EPFLcolors.colors[5], transform=ax1.transData._b)
        ax1.add_patch(circ)
        pc = ax1.pcolormesh(theta, np.arange(r, r + max(t), disc_r), np.array(strain_t).T, shading='nearest',
                            cmap=newcmp,
                            vmin=-lim_strain, vmax=lim_strain)
        # plt.title("Circumferential strain")
        bar = fig.colorbar(pc)
        bar.set_label('Circumferential strain [-]')
        if save_plots:
            plt.savefig(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Circumferential_Strain.png")
        else:
            plt.show()

        plt.rcParams.update({'font.size': 10})

    # Fit plane to strain field to enforce planar cross-section boundary condition
    A = np.zeros((strain_a.size, 4))
    for row in range(strain_a.shape[0]):
        if len(strain_a.shape) > 1:
            for col in range(strain_a.shape[1]):
                A[row * len(strain_a[row]) + col, :] = \
                    [(r + disc_r * col) * np.cos(theta[row]), (r + disc_r * col) * np.sin(theta[row]), 1,
                     strain_a[row][col]]
        else:
            A[row, :] = \
                [(r + t[row] / 2) * np.cos(theta[row]), (r + t[row] / 2) * np.sin(theta[row]), 1,
                 strain_a[row]]

    A = A[~np.isnan(A).any(axis=1)]

    b = A[:, 3]
    A = A[:, 0:3]

    fit = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
    if plots:
        ax = plt.figure().add_subplot(projection='3d')
        if len(strain_a.shape) > 1:
            for i, tau in enumerate(theta):
                ax.plot(np.linspace(r, r + t[i], len(strain_a[i])) * np.cos(tau),
                        np.linspace(r, r + t[i], len(strain_a[i])) * np.sin(tau), strain_a[i])
        else:
            ax.plot((r + t / 2) * np.cos(theta), (r + t / 2) * np.sin(theta), strain_a)
            ax.scatter(A[:, 0], A[:, 1])

    X, Y = np.meshgrid(np.arange(-R, R), np.arange(-R, R))
    Z = np.zeros(X.shape)
    for k in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[k, c] = fit[0] * X[k, c] + fit[1] * Y[k, c] + fit[2]

    if plots:
        ax.plot_wireframe(X, Y, Z, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Axial Strain')
        if save_plots:
            tikzplotlib.save(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Fitted_Plane.tex")
        else:
            plt.show()

    # Use fitted plane
    strain_thin = fit[2] + fit[0] * (r + t[0] / 2)
    strain_thick = fit[2] + fit[0] * (- r - t[int(len(strain_a) / 2)] / 2)

    center = (((r + t[0] / 2) * (1 + strain_thick) + (r + t[int(len(strain_a) / 2)] / 2) * (1 + strain_thin)) /
              (strain_thin - strain_thick))
    # print("Curling center {0:.2f}mm from air channel of the worm".format(center))

    curl = (1 + strain_thin) * l / (center + r + t[0] / 2)

    if plots:
        phi = np.arange(0, curl, 0.001)
        plt.plot(center * np.cos(phi), center * np.sin(phi), c="k")
        plt.plot((center + 2 * R) * np.cos(phi), (center + 2 * R) * np.sin(phi), c="k")
        plt.plot([(center) * np.cos(curl), (center + 2 * R) * np.cos(curl)],
                 [(center) * np.sin(curl), (center + 2 * R) * np.sin(curl)], c="k")

        plt.plot((center + R + d - r) * np.cos(phi), (center + R + d - r) * np.sin(phi), c="b")
        plt.plot((center + R + d + r) * np.cos(phi), (center + R + d + r) * np.sin(phi), c="b")
        plt.axis("equal")
        plt.axis("off")

        if save_plots:
            tikzplotlib.save(
                r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Curl.tex")
        else:
            plt.show()

    return (c4 * np.power(curl, 5) +
            c5 * np.power(curl, 3) +
            c6 * np.power(curl, 1)) * 180/np.pi


if __name__ == "__main__":

    plots = False
    save_plots = False
    epochs = 25
    nparticles = 1000
    start_limits = [[0, 3 / np.power(0.3, 5)],
                    [0, 3 / np.power(0.3, 3)],
                    [0, 3 / np.power(0.3, 1)],
                    [0, 3],
                    [0, 3],
                    [0, 3]]
    nlgeom = True

    # Discretization
    disc_t = 0.3
    disc_r = 0.1

    # Inputs
    E = 0.4485
    nu = 0.47

    # Particle Swarm Optimization Weights
    w1 = 0.5
    w2 = 0.2
    w3 = 0.3

    config = [{"R": 2, "d": 0.5, "r": 1, "l": 232, "curl": [0.00000000e+00,
                                                            -1.44875396e-01,
                                                            1.97971673e-01,
                                                            8.18724686e-01,
                                                            1.53241378e+00,
                                                            3.32232585e+00,
                                                            5.39582420e+00,
                                                            1.28936854e+01,
                                                            2.20657077e+01]},
              {"R": 3, "d": 1.5, "r": 1, "l": 220, "curl": [0.00000000e+00,
                                                            -2.27610266e-02,
                                                            2.88504193e-01,
                                                            6.87519943e-01,
                                                            1.32824511e+00,
                                                            2.51903789e+00,
                                                            4.17917946e+00,
                                                            7.15888112e+00,
                                                            1.25058338e+01,
                                                            2.08138212e+01,
                                                            3.53436402e+01,
                                                            6.37370151e+01]},
              {"R": 3.5, "d": 1.5, "r": 1, "l": 173, "curl": [0.00000000e+00,
                                                              -1.60986402e-01,
                                                              -8.84284856e-02,
                                                              2.88630207e-01,
                                                              6.63670214e-01,
                                                              1.02159714e+00,
                                                              1.70731583e+00,
                                                              2.66222806e+00,
                                                              4.12025021e+00,
                                                              5.97115605e+00,
                                                              8.51240039e+00,
                                                              1.21439367e+01,
                                                              1.65385633e+01,
                                                              2.33999040e+01,
                                                              3.01196154e+01,
                                                              3.92904575e+01,
                                                              4.84242637e+01,
                                                              6.23201605e+01,
                                                              8.57303324e+01,
                                                              1.06011763e+02,
                                                              1.23923298e+02,
                                                              1.46886176e+02]},
              {"R": 4, "d": 1.75, "r": 1, "l": 172, "curl": [0.00000000e+00,
                                                             -7.24754130e-03,
                                                             3.54135754e-01,
                                                             3.34523304e-01,
                                                             2.85535314e-01,
                                                             2.89698721e-01,
                                                             7.26587747e-01,
                                                             1.32625439e+00,
                                                             1.76672449e+00,
                                                             2.57655185e+00,
                                                             3.70335789e+00,
                                                             5.46243106e+00,
                                                             7.91279565e+00,
                                                             1.09433203e+01,
                                                             1.55270349e+01,
                                                             2.09382699e+01,
                                                             2.75299437e+01,
                                                             3.64463134e+01,
                                                             4.54664729e+01,
                                                             6.81300002e+01,
                                                             9.43350365e+01,
                                                             1.21038009e+02,
                                                             1.44071132e+02,
                                                             1.69743511e+02]},
              ]

    # Initialize or read particles
    if os.path.exists(r"C:\Users\tobia\PycharmProjects\MasterThesis\FEM_Analysis\Optimization_Log_v3.json"):
        with open(r"C:\Users\tobia\PycharmProjects\MasterThesis\FEM_Analysis\Optimization_Log_v3.json", 'r') as f:
            load_dict = json.load(f)
            particles_time = load_dict["particles"]
            errors_time = load_dict["errors"]
            errors_time_array = np.array(errors_time)
            particles_time_array = np.array(particles_time)
            particles = particles_time[-1]
            start_epochs = len(particles_time) - 1

    else:
        particles_time = []
        particles = []
        for i in range(nparticles):
            particles.append([random.uniform(start_limits[0][0], start_limits[0][1]),
                              random.uniform(start_limits[1][0], start_limits[1][1]),
                              random.uniform(start_limits[2][0], start_limits[2][1]),
                              random.uniform(start_limits[3][0], start_limits[3][1]),
                              random.uniform(start_limits[4][0], start_limits[4][1]),
                              random.uniform(start_limits[5][0], start_limits[5][1]),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0])
        particles_time.append(particles)
        errors_time = []
        start_epochs = 0

    # Iterate over number of optimizations
    print("Starting at epoch {0:d}".format(start_epochs + 1))
    for i in range(epochs - start_epochs):
        errors = []

        if i > 0:
            particles = new_particles

        # Compute errors for all particles
        for num_par, particle in enumerate(particles):

            cumulative_error = 0.0
            num_errors = 0

            for conf in config[1:]:
                for k in range(len(conf["curl"])):
                    model_curl = analytical_model(R=conf["R"],
                                                  d=conf["d"],
                                                  r=conf["r"],
                                                  l=conf["l"],
                                                  p_i=0.007 * (k + 1),
                                                  p_o=0.0,
                                                  disc_t=disc_t,
                                                  disc_r=disc_r,
                                                  E=E,
                                                  nu=nu,
                                                  c1=particle[0],
                                                  c2=particle[1],
                                                  c3=particle[2],
                                                  c4=particle[3],
                                                  c5=particle[4],
                                                  c6=particle[5],
                                                  plots=plots,
                                                  save_plots=save_plots,
                                                  nlgeom=nlgeom)

                    cumulative_error += np.power(model_curl - conf["curl"][k], 2) * (k+1)
                    num_errors += 1

            errors.append(np.power(cumulative_error / num_errors, 0.5))
            if (num_par+1) % 10 == 0:
                print("Particle {0:d} finished".format(num_par+1))

        errors_time.append(errors)

        if i == epochs - 1:
            break

        # Find best particles
        errors_time_array = np.array(errors_time)
        particles_time_array = np.array(particles_time)

        if errors_time_array.shape[0] > 1:
            best_indv_pos_idcs = np.argmin(errors_time_array, axis=0)
        else:
            best_indv_pos_idcs = np.zeros(errors_time_array.shape[1])
        best_indv_pos = []
        for j in range(best_indv_pos_idcs.shape[0]):
            best_indv_pos.append(particles_time_array[int(best_indv_pos_idcs[j]), int(j)])
        best_glob_pos_idx = np.where(errors_time_array == np.min(errors_time_array))
        best_glob_pos = particles_time_array[int(best_glob_pos_idx[0][0]), int(best_glob_pos_idx[1][0])]

        # Update particle motion
        new_particles = []
        for j in range(len(particles)):
            new_particles.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            new_particles[j][6] = w1 * particles[j][6] + w2 * (best_indv_pos[j][0] - particles[j][0]) + w3 * (best_glob_pos[0] - particles[j][0])
            new_particles[j][7] = w1 * particles[j][7] + w2 * (best_indv_pos[j][1] - particles[j][1]) + w3 * (best_glob_pos[1] - particles[j][1])
            new_particles[j][8] = w1 * particles[j][8] + w2 * (best_indv_pos[j][2] - particles[j][2]) + w3 * (best_glob_pos[2] - particles[j][2])
            new_particles[j][9] = w1 * particles[j][9] + w2 * (best_indv_pos[j][3] - particles[j][3]) + w3 * (best_glob_pos[3] - particles[j][3])
            new_particles[j][10] = w1 * particles[j][10] + w2 * (best_indv_pos[j][4] - particles[j][4]) + w3 * (best_glob_pos[4] - particles[j][4])
            new_particles[j][11] = w1 * particles[j][11] + w2 * (best_indv_pos[j][5] - particles[j][5]) + w3 * (best_glob_pos[5] - particles[j][5])
            new_particles[j][0] = particles[j][0] + new_particles[j][6]
            new_particles[j][1] = particles[j][1] + new_particles[j][7]
            new_particles[j][2] = particles[j][2] + new_particles[j][8]
            new_particles[j][3] = particles[j][3] + new_particles[j][9]
            new_particles[j][4] = particles[j][4] + new_particles[j][10]
            new_particles[j][5] = particles[j][5] + new_particles[j][11]

        particles_time.append(new_particles)

        print("Epoch {0:d}/{1:d}:\nBest Coefficients: C1 = {2:.5f}, C2 = {3:.5f}, C3 = {4:.5f}, C4 = {5:.5f}, C5 = {6:.5f}, C6 = {7:.5f}\nError = {8:.5f}".format(
            i+1+start_epochs, epochs, best_glob_pos[0], best_glob_pos[1], best_glob_pos[2], best_glob_pos[3], best_glob_pos[4], best_glob_pos[5], np.min(errors_time_array)
        ))

        with open(r"C:\Users\tobia\PycharmProjects\MasterThesis\FEM_Analysis\Optimization_Log_v3.json", 'w') as f:
            json.dump({"particles": particles_time,
                           "errors": errors_time}, f)

    print(particles_time_array[np.where(errors_time_array == np.min(errors_time_array))][0])

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    axs0 = fig.add_subplot(gs[0, :])
    axs1 = fig.add_subplot(gs[1, 0])
    axs2 = fig.add_subplot(gs[1, 1])
    axs3 = fig.add_subplot(gs[1, 2])
    axs4 = fig.add_subplot(gs[2, 0])
    axs5 = fig.add_subplot(gs[2, 1])
    axs6 = fig.add_subplot(gs[2, 2])

    # Get errors over time
    min_error = []
    max_error = []
    for i in range(errors_time_array.shape[0]):
        min_error.append(np.min(errors_time_array[i, :]))
        max_error.append(np.max(errors_time_array[i, :]))

    # Get particle values over time
    particles_min = np.zeros((errors_time_array.shape[0], 6))
    particles_max = np.zeros((errors_time_array.shape[0], 6))
    for i in range(errors_time_array.shape[0]):
        particles_min[i, :] = np.min(particles_time_array[i, :, 0:6], axis=0)
        particles_max[i, :] = np.max(particles_time_array[i, :, 0:6], axis=0)

    axs0.plot(np.arange(errors_time_array.shape[0]), min_error, color=EPFLcolors.colors[2])
    axs0.plot(np.arange(errors_time_array.shape[0]), max_error, color=EPFLcolors.colors[2])
    axs0.fill_between(np.arange(errors_time_array.shape[0]), min_error, max_error,
                      color=EPFLcolors.colors[2], alpha=0.3)
    axs0.set_xlabel("Epoch")
    axs0.set_ylabel("Error [°]")
    #axs0.set_yscale('log')

    for i, axs in enumerate([axs1, axs2, axs3, axs4, axs5, axs6]):
        axs.plot(np.arange(errors_time_array.shape[0]), particles_min[:, i], color=EPFLcolors.colors[0])
        axs.plot(np.arange(errors_time_array.shape[0]), particles_max[:, i], color=EPFLcolors.colors[0])
        axs.fill_between(np.arange(errors_time_array.shape[0]), particles_min[:, i], particles_max[:, i],
                         color=EPFLcolors.colors[0], alpha=0.3)
        axs.set_xlabel("Epoch")
        axs.set_ylabel("C{0:d}".format(i + 1))
        axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        #axs.set_yscale('log')

    plt.show()
    for i in range(len(min_error)):
        print("Iteration {0:d}: {1:.4f}%".format(i+1, (max_error[i] - min_error[i]) / min_error[i] * 100))

    #import tikzplotlib
    #tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\Convergence.tex")

