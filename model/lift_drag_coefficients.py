"""
For description of the script, see the README.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 16})


class GeneratorCoefficient(object):

    def __init__(self, verbose=0):
        self.verbose = verbose

    def setup_interpolator(self, interpolator_function):

        self.interpolator = interpolator_function

    def show_interpolation(self, show=True, n_points=100):
        """display the interpolation."""

        x_min = -180
        x_max = 180
        step = float(x_max - x_min) / n_points

        if self.verbose > 0:
            printi("x_min: " + str(x_min))
            printi("x_max: " + str(x_max))
            printi("step : " + str(step))

        array_x_values = np.arange(x_min, x_max, step)
        array_interpolated_values = self.interpolator(array_x_values)

        plt.figure()
        plt.plot(array_x_values, array_interpolated_values, '--', label='coefficient')
        # plt.legend()
        # plt.grid()
        if show:
            plt.show()

    def return_interpolated(self, x_value):
        """return interpolated value at x_value."""

        return(self.interpolator(x_value))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# formula (12, 13) from 'Unsteady forces and flows in low Reynolds number hovering flight: two-dimensional computations vs robotic wing experiments'
# agrees well with the content of 'Rotational accelerations stabilize leading edge vortices on revolving fly wings'
# NOTE: it seems there is an error in the paper (?), should be a - in Eqn. 13. Otherwise does not agree with formula (14, 15) or with the plots

parametrization = 1


if parametrization == 1:
    coeff_regularization = 2.0
    
    def CL_function(alpha_degrees):
        return(0.225 + 1.58 * np.sin(2.13 * np.pi / 180 * alpha_degrees - 7.2 * np.pi / 180))


    def CD_function(alpha_degrees):
        return((1.92 - 1.55 * np.cos(2.04 * np.pi / 180 * alpha_degrees - 9.82 * np.pi / 180)) / coeff_regularization)


# formula (14, 15) from 'Unsteady forces and flows in low Reynolds number hovering flight: two-dimensional computations vs robotic wing experiments'
if parametrization == 2:
    coeff_regulatization = 3.0
    
    def CL_function(alpha_degrees):
        return(1.2 * np.sin(2.0 * np.pi / 180 * alpha_degrees))


    def CD_function(alpha_degrees):
        return((1.4 - 1.0 * np.cos(2.0 * np.pi / 180 * alpha_degrees)) / coeff_regulatization)


# to show or not the CL and CD parametrization
show_CL_CD = False

if __name__ == "__main__":
    show_CL_CD = True

interpolator_Cl = GeneratorCoefficient()
interpolator_Cl.setup_interpolator(CL_function)

if show_CL_CD:
    interpolator_Cl.show_interpolation(show=False)
    plt.xlabel("angle of attack [deg]")
    plt.ylabel("$C_L$")
    plt.tight_layout()
    plt.savefig("Figures/Cl.pdf")

interpolator_Cd = GeneratorCoefficient()
interpolator_Cd.setup_interpolator(CD_function)

if show_CL_CD:
    interpolator_Cd.show_interpolation(show=False)
    plt.xlabel("angle of attack [deg]")
    plt.ylabel("$C_D$")
    plt.tight_layout()
    plt.savefig("Figures/Cd.pdf")
    
if show_CL_CD:
    array_alpha = np.arange(0, 90, 90.0 / 100)
    array_Cd = interpolator_Cd.return_interpolated(array_alpha)
    array_Cl = interpolator_Cl.return_interpolated(array_alpha)
    
    plt.figure()
    matplotlib.rcParams.update({'font.size': 18})
    plt.plot(array_Cd, array_Cl)
    plt.xlabel("C_D")
    plt.ylabel("C_L")
    
    plt.figure()
    matplotlib.rcParams.update({'font.size': 18})
    plt.plot(array_alpha, array_Cl / array_Cd, linewidth=3.0)
    plt.xlabel("angle of attack [deg]")
    plt.ylabel("$C_L$ / $C_D$")
    plt.tight_layout()
    # plt.grid()
    plt.savefig("Figures/Cl_over_Cd.pdf")
    
    plt.figure()
    plt.plot(array_alpha, array_Cl, label="$C_L$", linewidth=3.0, linestyle="-")
    plt.plot(array_alpha, array_Cd, label="$C_D$", linewidth=3.0, linestyle="--")
    plt.xlabel("angle of attack [deg]")
    plt.ylabel("coefficient")
    plt.legend()
    plt.tight_layout()
    # plt.grid()
    plt.savefig("Figures/Cl_and_Cd.pdf")
    
    
    
    
    plt.show()

