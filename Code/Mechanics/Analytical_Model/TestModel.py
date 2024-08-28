import numpy as np
import matplotlib.pyplot as plt
import json

from AnalyticalSolutionDisplacement_v3 import analytical_model
import EPFLcolors


if __name__ == "__main__":

    data = "Training"

    plots = False
    save_plots = False
    nlgeom = True

    # Discretization
    disc_t = 0.1
    disc_r = 0.1

    # Inputs
    E = 0.4485
    nu = 0.47

    if data == "Training":
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
                                                                 1.69743511e+02]}
                  ]

    else:
        config = []

    with open(r"C:\Users\tobia\PycharmProjects\MasterThesis\FEM_Analysis\Optimization_Log_v3.json", 'r') as f:
        load_dict = json.load(f)
        particles_time_array = np.array(load_dict["particles"])
        errors_time_array = np.array(load_dict["errors"])
        particle = particles_time_array[np.where(errors_time_array == np.min(errors_time_array))][0]

    curl_results = []

    for conf in config:
        curl_design = []
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
                                          plots=False,
                                          save_plots=False,
                                          nlgeom=True)

            curl_design.append(model_curl)
            print("Pressure {0:d}/{1:d} evaluated".format(k+1, len(conf["curl"])))

        curl_results.append(curl_design)

    fig, axs = plt.subplots(nrows=1, ncols=len(config), sharex='all', sharey='all')

    if len(config) == 1:
        axs = [axs]

    for i, conf in enumerate(config):

        axs[i].plot((np.arange(len(conf["curl"])) + 1) * 0.007, conf["curl"], label="Test data", color=EPFLcolors.colors[0])
        axs[i].plot((np.arange(len(curl_results[i])) + 1) * 0.007, curl_results[i], label="Model results", color=EPFLcolors.colors[2])
        axs[i].text(0.05, 0.95, 'R: {0:.2f}\nr: {1:.2f}\nd: {2:.2f}\nL: {3:.2f}'.format(conf["R"], conf["r"], conf["d"], conf["l"]),
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=axs[i].transAxes)

        axs[i].set_xlabel("Pressure [kPa]")
        axs[i].set_ylabel("Bending angle [°]")
        #axs[i].set_ylim([-5, 360])

    plt.show()

    #import tikzplotlib
    #tikzplotlib.save(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\02 Images\Self-Made\Analytical_Model\TrainingFit.tex")

    print(particles_time_array[np.where(errors_time_array == np.min(errors_time_array))][0])
