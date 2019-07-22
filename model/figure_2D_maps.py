"""
For description of the script, see the README.md
"""

"""
Generate and plot the vertical force and forward moment maps for one seed, show them,
and find the equilibrium point.
"""

from compute_force_on_seed import ComputeForce
from generate_seed_profile import WingGeometry
from lift_drag_coefficients import interpolator_Cd, interpolator_Cl
import numpy as np
import matplotlib.pyplot as plt
from printind.printind_function import printi
import tqdm
from detect_level_line_intersections import detect_intersections


def perform_2D_map_analysis(KL=2.3, angle_base=0, added_AOA_camber_deg=2.5, show=True):
    # parameters for making things dimensional in the model computation
    L = 0.06
    nu = 1.0e-6
    g = 9.81
    seed_loading_kilograms = 0.0012  # standard mass loading for the experiments, taking into account lead minus buoyancy in water

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # the effective curvature

    # some renormalization stuff
    crrt_radius_curvature = L / KL

    # create wing profile
    wing_instance = WingGeometry(verbose=1, angle_base=angle_base, length_linear=0.015, total_length=0.075, radius_curvature=crrt_radius_curvature)
    wing_instance.compute_geometrical_properties()
    wing_instance.compute_segments_linear()
    wing_instance.compute_segments_curved()

    show_seed = False  # used during testing to check that the seed generated looks good

    if show_seed:
        wing_instance.plot_seed()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # create force analysis instance
    compute_force_instance = ComputeForce(verbose=0, added_AOA_camber_deg=added_AOA_camber_deg)  # standard camber effect
    compute_force_instance.set_wing_pitch(15)  # standard wing pitch
    compute_force_instance.set_wing_chord(0.015)
    compute_force_instance.set_seed_profile(wing_instance)
    compute_force_instance.set_interpolator_Cd(interpolator_Cd)
    compute_force_instance.set_interpolator_Cl(interpolator_Cl)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # compute 2D maps

    # seed loading
    vertical_force_loading = seed_loading_kilograms * g

    # range Re ie fall velocity
    Re_min = 3500
    Re_max = 17000
    nbr_points_Re = 25.0
    Re_reso = (Re_max - Re_min) / nbr_points_Re
    # range angular rate
    St_min = 0.10
    St_max = 0.9
    nbr_points_St = 25.0
    St_reso = (St_max - St_min) / nbr_points_St

    # arrays
    Re_array = np.arange(Re_min, Re_max, Re_reso)
    St_array = np.arange(St_min, St_max, St_reso)

    # meshgrids
    # x is fall_rate_array
    # y is angular_frequency_array
    X, Y = np.meshgrid(Re_array, St_array)

    # arrays of results
    forward_moment_array = np.zeros(X.shape)
    vertical_force_array = np.zeros(Y.shape)

    for index, x in tqdm.tqdm(np.ndenumerate(X)):
        # printi(str(index))

        current_fall_rate = X[index] * nu / L
        current_angular_frequency = Y[index] * current_fall_rate / L

        compute_force_instance.set_angular_frequency(current_angular_frequency)
        compute_force_instance.set_vertical_velocity(current_fall_rate)
        compute_force_instance.compute_force_all_elements()
        compute_force_instance.compute_resultant_force()

        crrt_forward_moment = compute_force_instance.return_forward_moment()
        crrt_vertical_force = compute_force_instance.return_vertical_force()

        # note: the factor 2 comes from the fact that the seed has 2 wings
        forward_moment_array[index] = 2.0 * crrt_forward_moment
        vertical_force_array[index] = 2.0 * crrt_vertical_force

    vertical_force_array = vertical_force_array - vertical_force_loading

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # title string
    title_string = 'loading ' + str(seed_loading_kilograms)

    amplification_factor = 50000.0

    level_curves = [-20.0, -15.0, -10.0, -5.0, 0.0, 1.5, 3.0]

    # dummy figure to generate the isolines at 0 level
    plt.figure()
    contour_1 = plt.contour(X, Y, amplification_factor * forward_moment_array, 6, colors='k', levels=[0.0])
    contour_2 = plt.contour(X, Y, 1000.0 * vertical_force_array, 6, colors='k', levels=[0.0])
    plt.close()

    intersection_points_2_x, intersection_points_2_y = detect_intersections(contour_1, contour_2)
    intersection_Re = intersection_points_2_x[0]
    intersection_St = intersection_points_2_y[0]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # plot forward moment
    level_curves = [-20.0, -15.0, -10.0, -5.0, 0.0, 3.0, 8.0, 20.0]
    plt.figure()
    plt.pcolor(X, Y, amplification_factor * forward_moment_array, cmap='coolwarm')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('scaled M / [N.m]')
    CS = plt.contour(X, Y, amplification_factor * forward_moment_array, 6, colors='k', levels=level_curves)
    plt.clabel(CS, fontsize=9, inline=1)
    plt.ylabel("St")
    plt.xlabel("Re")
    # plt.title("Forward moment map " + title_string)
    plt.savefig('Figures/forward_moment_colormap.pdf', bbox_inches='tight')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # plot vertical force
    plt.figure()
    plt.pcolor(X, Y, 1000.0 * vertical_force_array, cmap='coolwarm')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('scaled F / [N]')
    CS = plt.contour(X, Y, 1000.0 * vertical_force_array, 6, colors='k')
    plt.clabel(CS, fontsize=9, inline=1)
    plt.ylabel("St")
    plt.xlabel("Re")
    # plt.title("Total vertical force map " + title_string)
    plt.savefig('Figures/vertical_force_colormap.pdf', bbox_inches='tight')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # plot both moment and force and find point of equilibrium
    font_size_inside = 10
    font_size_outside = 14

    plt.figure()

    # moment
    levels = [-30.0, -15.0, -5.0, 0.0, 3.0, 8.0, 20.0]  # this is renormalized so that looks good
    CS = plt.contour(X, Y, 50000.0 * forward_moment_array, colors='r', label="moment", levels=levels)
    CS.levels
    fmt = {}
    for crrt_level in levels:
        fmt[crrt_level] = str(crrt_level)
    plt.clabel(CS, fmt=fmt, fontsize=font_size_inside, inline=1)
    # plt.clabel(CS)

    # force
    levels = [-10.0, -6.0, -3.0, 0.0, 6.0, 30.0, 60.0]
    CS = plt.contour(X, Y, 1000.0 * vertical_force_array, colors='b', label="force", levels=levels)
    CS.levels
    fmt = {}
    for crrt_level in levels:
        fmt[crrt_level] = str(crrt_level)
    plt.clabel(CS, fmt=fmt, fontsize=font_size_inside, inline=1)

    # intersection of 0 isolines
    # plt.plot([5630], [0.767], '.', markersize='20', color='k', label='equilibrium')
    # plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=20, color='k')
    plt.scatter(intersection_Re, intersection_St, s=30, color='k', label="Re: {:3.0f}\n St: {:0.2f}".format(intersection_Re, intersection_St))

    # add colors and label
    plt.plot([6000], [0.5], 'r', label='moment')
    plt.plot([6000], [0.5], 'b', label='force')

    # add labels and legend
    # plt.ylim([0.25, 0.65])
    plt.ylabel("St")
    plt.xlabel("Re")
    # plt.grid()
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fancybox=False, shadow=False)
    legend = plt.legend(loc='upper right', ncol=1, frameon=True, framealpha=0.9)
    # legend = plt.legend(ncol=1, frameon=True)
    # plt.title(title_string)
    # plt.xlim([2000, 8000])
    # plt.ylim([0.1, 0.6])

    # Define a class that forces representation of float to look a certain way
    # This remove trailing zero so '1.000' becomes '1.0'

    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')

    plt.savefig('Figures/level_curve_maps_loading_for_article.pdf')

    print("current KL: {}".format(KL))
    print("angle base: {}".format(angle_base))
    print("added_AOA_camber_deg {}".format(added_AOA_camber_deg))
    print("Re equilibrium: {}".format(intersection_Re))
    print("St equilibrium: {}".format(intersection_St))

    if show:
        plt.show()
    else:
        plt.close('all')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # save the data for later plots
    """
    np.save("Figures/X", X)
    np.save("Figures/Y", Y)
    np.save("Figures/vertical_force", vertical_force_array)
    np.save("Figures/forward_moment", forward_moment_array)
    """

    return(intersection_Re, intersection_St)


if __name__ == "__main__":
    perform_2D_map_analysis()
