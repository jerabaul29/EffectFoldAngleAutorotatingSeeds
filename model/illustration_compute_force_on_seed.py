"""
For description of the script, see the README.md
"""

"""
Illustrate the distribution of forces and moments on the wings for one seed geometry.
"""

from compute_force_on_seed import ComputeForce
from generate_seed_profile import WingGeometry
from lift_drag_coefficients import interpolator_Cd, interpolator_Cl

KL = 2.3
L = 0.06
crrt_radius_curvature = L / KL

nu = 1.0e-6

# those are obtained from the figure_2D_maps.py script, using the right seed parameters.
Re = 4777
St = 0.69

vertical_velocity = Re * nu / L
angular_frequency = St * vertical_velocity / L

gray_region = [0.015, 0.04]

wing_instance = WingGeometry(verbose=1, angle_base=0.0, length_linear=0.015, total_length=0.075, radius_curvature=crrt_radius_curvature, save_seed="./Figures/seed_profile.pdf")
wing_instance.compute_geometrical_properties()
wing_instance.compute_segments_linear()
wing_instance.compute_segments_curved()
wing_instance.plot_seed(gray_region=gray_region)

compute_force_instance = ComputeForce(verbose=1, added_AOA_camber_deg=2.5)
compute_force_instance.set_wing_pitch(15)
compute_force_instance.set_wing_chord(0.016)
compute_force_instance.set_seed_profile(wing_instance)
compute_force_instance.set_interpolator_Cd(interpolator_Cd)
compute_force_instance.set_interpolator_Cl(interpolator_Cl)

compute_force_instance.set_angular_frequency(angular_frequency)
compute_force_instance.set_vertical_velocity(vertical_velocity)
compute_force_instance.compute_force_all_elements()
# compute_force_instance.display_reduced_information(title_base="example_for_report")
compute_force_instance.display_reduced_information(title_base="Figures/example_for_report", gray_region=gray_region)
