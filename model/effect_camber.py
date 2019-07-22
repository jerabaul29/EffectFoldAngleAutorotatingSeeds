import numpy as np
from figure_2D_maps import perform_2D_map_analysis
import pickle
import matplotlib
import matplotlib.pyplot as plt

min_KL = 0.7
max_KL = 3.2
# min_KL = 0.8
# max_KL = 3.6
nbr_KLs = 40
step_KLs = (max_KL - min_KL) / nbr_KLs
KL_range = np.arange(min_KL, max_KL + step_KLs, step_KLs)

angle_base = 0
list_added_AOA_camber_deg = [0.0, 2.5, 4.0]
show = False

# to generate the data
if True:

    for added_AOA_camber_deg in list_added_AOA_camber_deg:

        list_model_Re_psi0 = []
        list_model_St_psi0 = []
        KL_range_valid = []

        for crrt_KL in KL_range:
            try:
                crrt_Re, crrt_St = perform_2D_map_analysis(KL=crrt_KL, angle_base=angle_base, added_AOA_camber_deg=added_AOA_camber_deg, show=show)
                list_model_Re_psi0.append(crrt_Re)
                list_model_St_psi0.append(crrt_St)
                KL_range_valid.append(crrt_KL)
            except:
                pass

        dict_model_data_psi_0 = {}
        dict_model_data_psi_0["KL"] = KL_range_valid
        dict_model_data_psi_0["Re"] = np.array(list_model_Re_psi0)
        dict_model_data_psi_0["St"] = np.array(list_model_St_psi0)


        with open('model_data_psi_camber_{}.pkl'.format(added_AOA_camber_deg), 'wb') as handle:
            pickle.dump(dict_model_data_psi_0, handle, protocol=pickle.HIGHEST_PROTOCOL)
     

# to plot
colors = ["red", "black", "blue"]
linestyles = ['-', ':', '--']


if True:   
    plt.figure()
    matplotlib.rcParams.update({'font.size': 16})

    for ind, added_AOA_camber_deg in enumerate(list_added_AOA_camber_deg):   

        with open('model_data_psi_camber_{}.pkl'.format(added_AOA_camber_deg), 'rb') as handle:
            dict_model_data = pickle.load(handle)

        if ind == 0:
            plt.plot(dict_model_data["KL"], dict_model_data["Re"], linestyle=linestyles[ind], label=r"$\Delta \alpha = {} ^\circ$".format(added_AOA_camber_deg), color=colors[ind])
        else:
            plt.plot(dict_model_data["KL"], dict_model_data["Re"], linestyle=linestyles[ind], label=r"${} ^\circ$".format(added_AOA_camber_deg), color=colors[ind])
    
    # plt.grid()
    # plt.ylim([14000, 0])
    plt.xlabel('KL')
    plt.ylabel("Re")
    plt.legend(ncol=3, fontsize=14)
    plt.tight_layout()
    #
    plt.gca().invert_yaxis()
    # plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
    plt.savefig('Figures/Re_effect_camber.pdf')
    
    plt.figure()
    matplotlib.rcParams.update({'font.size': 24})

    for ind, added_AOA_camber_deg in enumerate(list_added_AOA_camber_deg):   

        with open('model_data_psi_camber_{}.pkl'.format(added_AOA_camber_deg), 'rb') as handle:
            dict_model_data = pickle.load(handle)


        plt.plot(dict_model_data["KL"], dict_model_data["St"], linestyle=linestyles[ind], label=r"$\Delta \alpha = {} ^\circ$".format(added_AOA_camber_deg), color=colors[ind])
    
    # plt.grid()
    # plt.ylim([14000, 0])
    plt.xlabel('KL')
    plt.ylabel("St")
    # plt.legend()
    plt.tight_layout()
    #
    # plt.gca().invert_yaxis()
    # plt.savefig('Figures/illustration_model_psi_0.pdf', bbox_inches='tight')
    plt.savefig('Figures/Re_effect_camber_St.pdf')

plt.show()


