from scipy import misc
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from tqdm import tqdm
import pickle

#################################################################################
# all functions ----------------------------------------------------------------

class generateDataOnClick:
    def __init__(self, verbose=0):
        self.position_on_click_accumulator = []
        self.verbose = verbose

    def position_on_click(self, event):
        x, y = event.x, event.y
        if event.button == 1:
            if event.inaxes is not None:
                if self.verbose > 0:
                    print 'data coords:' + str(event.xdata) + " , " + str(event.ydata)
                self.position_on_click_accumulator.append((event.xdata, event.ydata))
                plt.axvline(event.xdata, color='r')
                plt.show()

    def return_positions(self):
        return self.position_on_click_accumulator


def plot_raw_results_folder_process(list_pos_seed, list_true_wing_tip, polynomial_calibration_vertical=None, sampling_frequency=30.0, valid_index=(-1, -1), debug=False, f_max=5):

    # generate data to use
    pos_seed = np.asarray(list_pos_seed, dtype=np.float32)
    pos_tips = np.asarray(list_true_wing_tip, dtype=np.float32)

    use_calibration = False

    if valid_index[0] > -1:
        ind_min = valid_index[0]
        ind_max = valid_index[1]
        pos_seed = pos_seed[ind_min:ind_max, :]
        pos_tips = pos_tips[ind_min:ind_max, :]

    number_of_indices = pos_seed.shape[0]
    time_vector = np.arange(0, number_of_indices, 1.0) / sampling_frequency

    if debug:
        print time_vector



    # Fourier analysis for determination of the angular frequency
    (f, Pxx_tips_1x) = signal.periodogram(pos_tips[:, 1] - pos_seed[:, 0], fs=sampling_frequency)
    (f_2, Pxx_tips_2x) = signal.periodogram(pos_tips[:, 3] - pos_seed[:, 0], fs=sampling_frequency)
    #fft_adjuster = np.ones(len(f))
    #fft_adjuster[0:int(len(f)/3.)] = np.linspace(0.1, 20, int(len(f)/3.) - 0)
    #Pxx_tips_2x *= fft_adjuster

    #plt.figure()
    #plt.plot(f, fft_adjuster)
    #plt.show()

    if debug:
        print "f shape:"
        print f.shape
        print "Pxx_tips_1x shape:"
        print Pxx_tips_1x.shape

    ind_max_spectrum_left = np.argmax(Pxx_tips_1x[1:-2])
    frequency_max_spectrum_left = f[ind_max_spectrum_left + 1]

    ind_max_spectrum_right = np.argmax(Pxx_tips_2x[1:-2])
    frequency_max_spectrum_right = f_2[ind_max_spectrum_right + 1]
    '''
    plt.figure()
    plt.plot(pos_seed[:, 0], pos_seed[:, 1], marker='o', color='k')
    plt.plot(pos_tips[:, 1], pos_tips[:, 0], marker='o', color='r')
    plt.plot(pos_tips[:, 3], pos_tips[:, 2], marker='o', color='b')
    plt.xlabel("x (pxls)")
    plt.ylabel("y (pxls)")
    #'''
    #'''
    plt.figure()
    # plt.plot(pos_seed[:,1])
    plt.plot(time_vector, pos_seed[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pxls, not calibrated)')


    plt.figure()
    plt.plot(f, Pxx_tips_1x)
    plt.xlabel("f (Hz)")
    plt.ylabel("PSD position left wing tip")
    plt.xlim([0, f_max])


    generate_data_on_click_object = generateDataOnClick()
    plt.connect('button_press_event', generate_data_on_click_object.position_on_click)

    plt.show()

    selected_positions_pixels = generate_data_on_click_object.return_positions()
    frequency_max_spectrum_left = (selected_positions_pixels[0][0])



    generate_data_on_click_object = generateDataOnClick()
    plt.figure()
    plt.plot(pos_seed[:, 0], pos_seed[:, 1], marker='o', color='k')
    plt.plot(pos_tips[:, 1], pos_tips[:, 0], marker='o', color='r')
    plt.plot(pos_tips[:, 3], pos_tips[:, 2], marker='o', color='b')
    plt.xlabel("x (pxls)")
    plt.ylabel("y (pxls)")

    plt.figure()
    # plt.plot(pos_seed[:,1])
    plt.plot(time_vector, pos_seed[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pxls, not calibrated)')

    plt.figure()
    plt.plot(f_2, Pxx_tips_2x)
    plt.xlabel("f (Hz)")
    plt.ylabel("PSD position right wing tip")
    plt.xlim([0, f_max])
    plt.connect('button_press_event', generate_data_on_click_object.position_on_click)
    plt.show()

    selected_positions_pixels = generate_data_on_click_object.return_positions()
    frequency_max_spectrum_right = (selected_positions_pixels[0][0])
    #'''
    '''
    plt.figure()
    plt.imshow(misc.imread('/home/richaraf/Desktop/flying_seeds/seed_26mm_3/00000128.png'))
    plt.plot(pos_seed[:, 0], pos_seed[:, 1], marker='o', color='g')
    plt.plot(pos_tips[:, 1], pos_tips[:, 0], marker='o', color='r')
    plt.plot(pos_tips[:, 3], pos_tips[:, 2], marker='o', color='b')
    plt.xlabel("x (pxls)")
    plt.ylabel("y (pxls)")
    #'''
    print " "
    print "Information seed rotation"
    print "Frequency max spectrum left wing tip oscillation: " + str(frequency_max_spectrum_left)
    print "Frequency rotation seed: " + str(frequency_max_spectrum_left / 2.0)
    print "Frequency max spectrum right wing tip oscillation: " + str(frequency_max_spectrum_right)
    print "Frequency rotation seed: " + str(frequency_max_spectrum_right / 2.0)
    averaged_seed_rotation = 0.5 * (frequency_max_spectrum_right / 2.0 + frequency_max_spectrum_left / 2.0)
    print "Mean frequency rotation seed: " + str(averaged_seed_rotation)
    print " "

    if polynomial_calibration_vertical is not None:
        print "Using vertical calibration"
        p = np.poly1d(polynomial_calibration_vertical)
        use_calibration = True

        if debug:
            print pos_seed

        pos_seed[:, 1] = p(pos_seed[:, 1])
        pos_tips[:, 0] = p(pos_tips[:, 0])
        pos_tips[:, 2] = p(pos_tips[:, 2])

        if debug:
            print pos_seed

        # figure identified points
        '''
        plt.figure()
        plt.plot(pos_seed[:, 0], pos_seed[:, 1], 'o', color='r')
        plt.plot(pos_tips[:, 1], pos_tips[:, 0], 'o', color='b')
        plt.plot(pos_tips[:, 3], pos_tips[:, 2], 'o', color='g')
        plt.xlabel("x (pxls)")
        plt.ylabel("y (mm)")
        plt.show()
        #'''
        # look at vertical velocity by time evolution of height
        '''
        import matplotlib
        matplotlib.rcParams.update({'font.size': 28})
        plt.figure()
        # plt.plot(pos_seed[:,1])
        plt.plot(time_vector, np.asarray(pos_seed[:, 1]) / 1000., linewidth=4)
        plt.xlabel('Time [s]')
        plt.ylabel('H [m]')
        plt.savefig('../Oppsummering/figures/ConstantU.png', bbox_inches='tight')
        plt.show()
        #'''

        # look at the vertical velocity by differenciation
        vertical_velocity = (pos_seed[1:, 1] - pos_seed[0:-1, 1]) * sampling_frequency

        '''
        plt.figure()
        # plt.plot(vertical_velocity)
        plt.plot(time_vector[1:], vertical_velocity)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (mm / s, calibrated)')
        plt.show()
        #'''
        mean_vertical_velocity = (pos_seed[-1, 1] - pos_seed[0, 1]) * sampling_frequency / pos_seed.shape[0]

        print " "
        print "Information seed vertical velocity"
        print "Seed mean vertical velocity (mm/s): " + str(mean_vertical_velocity)

    else:
        print "Using no calibration"
        # figure identified points
        plt.figure()
        plt.plot(pos_seed[:, 0], pos_seed[:, 1], marker='o', color='k')
        plt.plot(pos_tips[:, 1], pos_tips[:, 0], marker='o', color='r')
        plt.plot(pos_tips[:, 3], pos_tips[:, 2], marker='o', color='b')
        plt.xlabel("x (pxls)")
        plt.ylabel("y (pxls)")
        plt.show()

        # look at vertical velocity by time evolution of height
        plt.figure()
        # plt.plot(pos_seed[:,1])
        plt.plot(time_vector, pos_seed[:, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Position (pxls, not calibrated)')

        # look at the vertical velocity by differenciation
        vertical_velocity = (pos_seed[1:, 1] - pos_seed[0:-1, 1]) * sampling_frequency

        plt.figure()
        # plt.plot(vertical_velocity)
        plt.plot(time_vector[1:], vertical_velocity)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (pxl / s, not calibrated)')

    return (pos_seed, pos_tips, use_calibration, averaged_seed_rotation, mean_vertical_velocity)


def load_one_result(result_name):
    with open(path + list_cases[ind_case] + '/' + result_name + '.pkl', 'r') as crrt_file:
            result_data = pickle.load(crrt_file)
    return result_data

#################################################################################

# analysis of the data
path = '/home/richaraf/Desktop/stroboscopic_strouhal/'
sampling_frequency = 30

# loads the calibration --------------------------------------------------------
poly_fit_calibration = np.load(path + 'poly_fit_calibration.npy')

# load list of all cases -------------------------------------------------------
list_cases = []
for file_name in os.listdir(path):
    if fnmatch.fnmatch(file_name, 'seed_35mm_2*'):
        list_cases.append(file_name)
list_cases = sorted(list_cases)

print "Cases to process:"
for crrt_case in list_cases:
    print crrt_case

print " "
nbr_cases = len(list_cases)
print "Number of cases: " + str(nbr_cases)

dict_all_results = {}

# analyse all cases ------------------------------------------------------------
for ind_case in range(nbr_cases):
    # for ind_case in [5]:

    print ""
    print "------------------------------------------------------------"
    print "Analysing case: " + str(list_cases[ind_case])

    path_to_images = path + list_cases[ind_case] + '/'

    print "Load generated data"

    list_pos_seed = load_one_result('list_pos_seed')
    list_width_data_seed = load_one_result('list_width_data_seed')
    list_true_wing_tip = load_one_result('list_true_wing_tip')

    print "Load valid range data"
    valid_range = np.genfromtxt(path_to_images + "valid_range.csv", delimiter=",")

    min_valid_range = int(valid_range[0])
    max_valid_range = int(valid_range[1])

    (pos_seed, pos_tips, use_calibration, averaged_seed_rotation, mean_vertical_velocity) = plot_raw_results_folder_process(
        list_pos_seed, list_true_wing_tip, polynomial_calibration_vertical=poly_fit_calibration,
        sampling_frequency=sampling_frequency, valid_index=(min_valid_range, max_valid_range), debug=False, f_max=5)

    dict_all_results[list_cases[ind_case] + "pos_seed"] = pos_seed
    dict_all_results[list_cases[ind_case] + "pos_tips"] = pos_tips
    dict_all_results[list_cases[ind_case] + "use_calibration"] = use_calibration
    dict_all_results[list_cases[ind_case] + "averaged_seed_rotation"] = averaged_seed_rotation
    dict_all_results[list_cases[ind_case] + "mean_vertical_velocity"] = mean_vertical_velocity

# save the dictionary
with open(path + 'dict_all_results.pkl', 'w') as crrt_file:
        pickle.dump(dict_all_results, crrt_file, pickle.HIGHEST_PROTOCOL)

#print pos_seed
#print pos_tips
