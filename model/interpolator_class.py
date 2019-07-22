"""
For description of the script, see the README.md
"""

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from printind.printind_function import printi


class GeneratorCoefficient(object):

    def __init__(self, verbose=0):
        self.verbose = verbose

    def setup_interpolator(self, table_x_values, table_coefficient_values, kind='linear'):

        self.table_x_values = table_x_values
        self.table_coefficient_values = table_coefficient_values
        self.kind = kind

        self.interpolator = interp1d(self.table_x_values, self.table_coefficient_values, kind=self.kind)

    def show_interpolation(self, show=True, n_points=100):

        x_min = np.min(self.table_x_values)
        x_max = np.max(self.table_x_values)
        step = float(x_max - x_min) / n_points

        if self.verbose > 0:
            printi("x_min: " + str(x_min))
            printi("x_max: " + str(x_max))
            printi("step : " + str(step))

        array_x_values = np.arange(x_min, x_max, step)
        array_interpolated_values = self.interpolator(array_x_values)

        plt.figure()
        plt.plot(self.table_x_values, self.table_coefficient_values, 'o', label="data")
        plt.plot(array_x_values, array_interpolated_values, '--', label='interpolated')
        plt.legend()
        # plt.grid()
        if show:
            plt.show()

    def return_interpolated(self, x_value):

        return(self.interpolator(x_value))
