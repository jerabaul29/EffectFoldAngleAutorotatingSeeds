import numpy as np
from figure_2D_maps import perform_2D_map_analysis
import pickle

min_KL = 0.1
max_KL = 2.8
# min_KL = 0.8
# max_KL = 3.6
nbr_KLs = 40
step_KLs = (max_KL - min_KL) / nbr_KLs
KL_range = np.arange(min_KL, max_KL + step_KLs, step_KLs)

angle_base = 35
added_AOA_camber_deg = 2.5
show = False

list_model_Re_psi0 = []
list_model_St_psi0 = []

for crrt_KL in KL_range:
    crrt_Re, crrt_St = perform_2D_map_analysis(KL=crrt_KL, angle_base=angle_base, added_AOA_camber_deg=added_AOA_camber_deg, show=show)
    list_model_Re_psi0.append(crrt_Re)
    list_model_St_psi0.append(crrt_St)

dict_model_data_psi_0 = {}
dict_model_data_psi_0["KL"] = KL_range
dict_model_data_psi_0["Re"] = np.array(list_model_Re_psi0)
dict_model_data_psi_0["St"] = np.array(list_model_St_psi0)

with open('model_data_psi_35.pkl', 'wb') as handle:
    pickle.dump(dict_model_data_psi_0, handle, protocol=pickle.HIGHEST_PROTOCOL)
