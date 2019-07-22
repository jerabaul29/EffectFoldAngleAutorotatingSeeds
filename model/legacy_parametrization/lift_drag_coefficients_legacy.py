"""
For description of the script, see the README.md
"""

from interpolator_class import GeneratorCoefficient
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""A module to generate lift and drag coefficient values"""

"""
NOTE: THIS IS A LEGACY FILE; HERE FOR ILLUSTRATION REASONS, AS THIS WAS THE FIRST
PARAMETRIZATION USED; FOLLOWING REVIEW ADVICES, WE CHANGED THE PARAMETRIZATION (SEE
THE NON LEGACY FILE).

The challenge in getting lift / drag coefficient is that this is usually known at high
Re, while we are working at low Re.

However, it seems that differences in the curves between high and low Re are low
enough for our crude model (we do not expect to be better than, maybe, 15 percent
accuracy).
"""

matplotlib.rcParams.update({'font.size': 16})


matplotlib.rcParams.update({'font.size': 16})


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# tables for the interpolator --------------------------------------------------

# LIFT

# table for AOA, ie Angle Of Attack
# in degrees
array_AOA_1 = np.array([-140.0, -90.0, -50.0, -30.0, -20.0, -10.0, -06.00, 00.00, 06.00, 10.00, 20.00, 30.00, 50.00, 90.00, 140.0])

# table for CL, ie Lift coefficient
array_CLft = np.array([0.30000, 0.000, -0.300, -0.55, -0.60, -0.65, -0.65, 0.000, 0.650, 0.650, 0.600, 0.550, 0.300, 0.00, -0.300])

# DRAG

# table for AOA, ie Angle Of Attack
# in degrees
array_AOA_2 = np.array([-110.0, -90.00, -70.00, -30.00, -20.00, -10.00, -05.00, 00.00, 05.00, 10.00, 20.00, 30.00, 70.00, 90.00, 110.00])

# table for Cd
Cl_over_Cd = 7.0
drag_factor = 7.0 / Cl_over_Cd  # for experimenting on the effect of CL / CD
array_CDrag = drag_factor * np.array([1.52857143, 1.68142857, 1.52857143, 0.61142857, 0.38214286,
                                      0.15285714, 0.07642857, 0.03821429, 0.07642857, 0.15285714,
                                      0.38214286, 0.61142857, 1.52857143, 1.68142857, 1.52857143])

# build interpolator -----------------------------------------------------------

# scaling_factor = 2.0
scaling_factor = 1.0

interpolator_Cl = GeneratorCoefficient()
interpolator_Cl.setup_interpolator(array_AOA_1, array_CLft * scaling_factor)
interpolator_Cl.show_interpolation(show=False)
plt.xlabel("angle of attack [deg]")
plt.ylabel("$C_L$")
plt.tight_layout()
plt.savefig("Figures/Cl.pdf")
plt.show()

interpolator_Cd = GeneratorCoefficient()
interpolator_Cd.setup_interpolator(array_AOA_2, array_CDrag * scaling_factor)
interpolator_Cd.show_interpolation(show=False)
plt.xlabel("angle of attack [deg]")
plt.ylabel("$C_D$")
plt.tight_layout()
plt.savefig("Figures/Cd.pdf")
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# show the Cl / Cd characteristics ---------------------------------------------

array_alpha = np.arange(0, np.pi / 2.0, np.pi / 2.0 / 100)
cl_over_cd = interpolator_Cl.return_interpolated(array_alpha * 180.0 / np.pi) / interpolator_Cd.return_interpolated(array_alpha * 180.0 / np.pi)

plt.figure()
plt.plot(180.0 / np.pi * array_alpha, cl_over_cd)
plt.xlim([0, 30])
plt.xlabel("angle of attack [deg]")
plt.ylabel("C$_L$ / C$_D$")
plt.grid()
plt.tight_layout()
plt.savefig("Figures/Cl_over_Cd.pdf")
plt.show()

plt.figure()
plt.plot(180.0 / np.pi * array_alpha, interpolator_Cl.return_interpolated(array_alpha * 180.0 / np.pi))
plt.xlim([0, 90])
plt.xlabel("angle of attack [deg]")
plt.ylabel("C$_L$")
plt.grid()
#plt.savefig("Figures/Cl.pdf")
plt.tight_layout()
plt.show()
