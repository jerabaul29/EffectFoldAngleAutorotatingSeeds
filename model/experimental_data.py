"""
The data obtained from the image analysis of the falling seeds videos.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from parse import parse
from scipy import stats

if __name__ == "__main__":
    print_flag = True
else:
    print_flag = False

##########################################
# the renormalisation stuff
L_ref = 0.06
nu = 1.0e-6
Re_normalisation = -L_ref / nu / 1000.0

##########################################
# Data with 3 and 10 repetitions

if print_flag:
    print("\n\nLook at experiments with 3 or 10 repetitions\n")

path_to_load = ''
alldata_dict = pickle.load(open(path_to_load + 'all_results_experiments.pkl'))


def find_u_mean(alpha, curvature, weight, nr_of_experiments):
    u_mean = 0
    if nr_of_experiments == 3:
        for i in range(1, nr_of_experiments+1):
            u_mean += alldata_dict[alpha][curvature][weight][i]['u mean']
    else:
        for i in range(0, nr_of_experiments):
            u_mean += alldata_dict[alpha][curvature][weight][i]['u mean']
    u_mean /= float(nr_of_experiments)
    return u_mean

def find_f_mean(alpha, curvature, weight, nr_of_experiments):
    f_mean = 0
    if nr_of_experiments == 3:
        for i in range(1, nr_of_experiments+1):
            f_mean += alldata_dict[alpha][curvature][weight][i]['f mean']
    else:
        for i in range(0, nr_of_experiments):
            f_mean += alldata_dict[alpha][curvature][weight][i]['f mean']

    f_mean /= float(nr_of_experiments)
    return f_mean

KL_liste = []
f_list = []
u_list = []
St_list = []
if print_flag:
    print 'alpha', 'radius_of_curvature', 'mass_[g]', 'rotation_[Hz]', 'speed_[m/s]'
for key1 in alldata_dict.keys():
    for key2 in alldata_dict[key1].keys():
        key2mark = key2
        if key2 == 10000:
            key2mark = 1000
        if key1 == 35:
            f = find_f_mean(key1, key2, 2.16, 3)
            u = find_u_mean(key1, key2, 2.16, 3)
            St = f * 60e-3 / u
            KL = 60.0 / key2mark
            if print_flag:
                print 'psi =', key1, ', KL =', KL, ', f =', f, ' [Hz], U =', u, '[m/s]', ', St =', St
        if key1 == 0:
            f = find_f_mean(key1, key2, 2.16, 10)
            u = find_u_mean(key1, key2, 2.16, 10)
            St = f * 60e-3 / u
            KL = 60.0 / key2mark
            if print_flag:
                print 'psi =', key1, ', KL =', KL, ', f =', f, ' [Hz], U =', u, '[m/s]', ', St =', St
        KL_liste.append(KL)
        u_list.append(u)
        f_list.append(f)
        St_list.append(St)

KL_list = np.asarray(KL_liste)
u_list = np.asarray(u_list)
f_list = np.asarray(f_list)
St_list = np.asarray(St_list)
Re_list = u_list * Re_normalisation * 1000  # because this is in m instead of mm for the rest

##########################################
# Data from videos used in figures 10 and 12:

if print_flag:
    print("\n\nLook at experiments for fig 10\n")

file_pickle_dictionary = 'dict_all_results.pkl'

with open(file_pickle_dictionary) as current_file:
    dictionary_Richard = pickle.load(current_file)

list_radius_curvature = np.array([100, 65, 50, 35, 30, 26, 23, 20, 18, 16, 15])
KL_total = 60. / list_radius_curvature
list_vertical_velocities_Richard1 = []
list_vertical_velocities_Richard2 = []
list_vertical_velocities_Richard3 = []
list_frequency_Richard1 = []
list_frequency_Richard2 = []
list_frequency_Richard3 = []

for current_radius_curvature in list_radius_curvature:
    current_key = 'seed_' + str(current_radius_curvature) + 'mm_1mean_vertical_velocity'
    current_keyf = 'seed_' + str(current_radius_curvature) + 'mm_1averaged_seed_rotation'
    #print("seed radius of curvature: " + str(current_radius_curvature) + " | rotation: " + str(dictionary_Richard[current_key]))
    list_vertical_velocities_Richard1.append(dictionary_Richard[current_key])
    list_frequency_Richard1.append(dictionary_Richard[current_keyf])

for current_radius_curvature in list_radius_curvature:
    current_key = 'seed_' + str(current_radius_curvature) + 'mm_2mean_vertical_velocity'
    current_keyf = 'seed_' + str(current_radius_curvature) + 'mm_2averaged_seed_rotation'
    #print("seed radius of curvature: " + str(current_radius_curvature) + " | rotation: " + str(dictionary_Richard[current_key]))
    list_vertical_velocities_Richard2.append(dictionary_Richard[current_key])
    list_frequency_Richard2.append(dictionary_Richard[current_keyf])

for current_radius_curvature in list_radius_curvature:
    current_key = 'seed_' + str(current_radius_curvature) + 'mm_3mean_vertical_velocity'
    current_keyf = 'seed_' + str(current_radius_curvature) + 'mm_2averaged_seed_rotation'
    #print("seed radius of curvature: " + str(current_radius_curvature) + " | rotation: " + str(dictionary_Richard[current_key]))
    list_vertical_velocities_Richard3.append(dictionary_Richard[current_key])
    list_frequency_Richard3.append(dictionary_Richard[current_keyf])

# frequencies found manually from videos
list_frequency_Richard1[4] = 1.33
list_frequency_Richard2[4] = 1.339
list_frequency_Richard3[4] = 1.331
list_frequency_Richard1[3] = 1.375
list_frequency_Richard2[3] = 1.339
list_frequency_Richard3[3] = 1.327
list_frequency_Richard1[2] = 1.579
list_frequency_Richard2[2] = 1.552
list_frequency_Richard3[2] = 1.552
list_frequency_Richard1[1] = 1.818
list_frequency_Richard2[1] = 1.846
list_frequency_Richard3[1] = 1.818
list_frequency_Richard1[0] = 2.333
list_frequency_Richard2[0] = 2.264
list_frequency_Richard3[0] = 2.308

total_velocities = (np.asarray(list_vertical_velocities_Richard1) + np.asarray(list_vertical_velocities_Richard2) + np.asarray(list_vertical_velocities_Richard3)) / 3.0
total_frequencies = (np.asarray(list_frequency_Richard1) + np.asarray(list_frequency_Richard2) + np.asarray(list_frequency_Richard3)) / 3.0
total_strouhal = -60 * total_frequencies / total_velocities
total_Re = total_velocities * Re_normalisation

if print_flag:
    for i in range(len(KL_total)):
        print '1/R= %1.2f mm -> KL = %1.1f -> U = %1.2f mm/s -> Re = %6.0f -> f = %1.2f Hz -> St = %1.2f' %(list_radius_curvature[i], KL_total[i], total_velocities[i], total_Re[i], total_frequencies[i], total_strouhal[i])
