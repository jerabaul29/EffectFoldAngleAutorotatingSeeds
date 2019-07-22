import numpy as np
from figure_2D_maps import perform_2D_map_analysis
import pickle
import matplotlib
import matplotlib.pyplot as plt

min_KL = 0.1
max_KL = 3.2
# min_KL = 0.8
# max_KL = 3.6
nbr_KLs = 40
step_KLs = (max_KL - min_KL) / nbr_KLs
KL_range = np.arange(min_KL, max_KL + step_KLs, step_KLs)

added_AOA_camber_deg = 2.50
list_psi = [0.0, 15.0, 35.0]
show = False

# to generate the data
if False:

    for angle_base in list_psi:

        list_model_Re_psi = []
        list_model_St_psi = []
        KL_range_valid = []

        for crrt_KL in KL_range:
            try:
                crrt_Re, crrt_St = perform_2D_map_analysis(KL=crrt_KL, angle_base=angle_base, added_AOA_camber_deg=added_AOA_camber_deg, show=show)
                list_model_Re_psi.append(crrt_Re)
                list_model_St_psi.append(crrt_St)
                KL_range_valid.append(crrt_KL)
            except:
                pass

        dict_model_data_psi = {}
        dict_model_data_psi["KL"] = KL_range_valid
        dict_model_data_psi["Re"] = np.array(list_model_Re_psi)
        dict_model_data_psi["St"] = np.array(list_model_St_psi)


        with open('model_data_camber_psi_{}.pkl'.format(angle_base), 'wb') as handle:
            pickle.dump(dict_model_data_psi, handle, protocol=pickle.HIGHEST_PROTOCOL)
     

# to plot
colors = ["red", "black", "blue"]
linestyles = ['-', ':', '--']


if True:   
    plt.figure()
    matplotlib.rcParams.update({'font.size': 16})
    
    list_nbr_points = [40, 35, 30]

    for ind, angle_base in enumerate(list_psi):   

        with open('model_data_camber_psi_{}.pkl'.format(angle_base), 'rb') as handle:
            dict_model_data = pickle.load(handle)

        if ind == 0:
            plt.plot(dict_model_data["KL"][: list_nbr_points[ind]], dict_model_data["Re"][: list_nbr_points[ind]], linestyle=linestyles[ind], label=r"$\psi = {} ^\circ$".format(angle_base), color=colors[ind])
        else:
            plt.plot(dict_model_data["KL"][: list_nbr_points[ind]], dict_model_data["Re"][: list_nbr_points[ind]], linestyle=linestyles[ind], label=r"${} ^\circ$".format(angle_base), color=colors[ind])
    
    # plt.grid()
    plt.xlabel('KL')
    plt.ylabel("Re")
    plt.legend(ncol=3, borderpad=0.2, columnspacing=0.3)
    plt.tight_layout()
    #
    plt.gca().invert_yaxis()
    plt.ylim([18000, 3500])
    # plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
    plt.savefig('Figures/Re_effect_psi.pdf')
    
    plt.figure()
    matplotlib.rcParams.update({'font.size': 24})
    
    for ind, angle_base in enumerate(list_psi):   

        with open('model_data_camber_psi_{}.pkl'.format(angle_base), 'rb') as handle:
            dict_model_data = pickle.load(handle)

        plt.plot(dict_model_data["KL"][: list_nbr_points[ind]], dict_model_data["St"][: list_nbr_points[ind]], linestyle=linestyles[ind], label=r"$\psi = {} ^\circ$".format(angle_base), color=colors[ind])
        
    
    # plt.grid()
    # plt.ylim([14000, 0])
    plt.xlabel('KL')
    plt.ylabel("St")
    # plt.legend()
    plt.tight_layout()
    #
    # plt.gca().invert_yaxis()
    # plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
    plt.savefig('Figures/Re_effect_psi_St.pdf')

plt.show()


