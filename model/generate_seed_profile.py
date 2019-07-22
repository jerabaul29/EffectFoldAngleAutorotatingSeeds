"""
For description of the script, see the README.md
"""

import numpy as np
import matplotlib.pyplot as plt
from printind.printind_function import printi
from matplotlib.patches import Rectangle


class WingGeometry(object):
    def __init__(self, angle_base=35.0, length_linear=0.015, total_length=0.0715, radius_curvature=0.03, verbose=0, save_seed=None):
        self.angle_base = angle_base  # angle between the linear part of the wing and the VERTICAL direction
        self.length_linear = length_linear
        self.total_length = total_length
        self.radius_curvature = radius_curvature
        self.verbose = verbose
        self.save_seed = save_seed

        self.list_s = []  # curvilinear length to start / end of wing segments
        self.list_s_middle = []  # curvilinear length to middle of segment
        self.list_R = []
        self.list_h = []
        self.list_phi = []  # angle between any wing segment and the HORIZONTAL direction
        self.list_size_element = []

    def compute_geometrical_properties(self):
        self.length_curved = self.total_length - self.length_linear

    def compute_segments_linear(self, n_segments=100):
        step = float(self.length_linear) / n_segments
        s_linear = np.arange(step, self.length_linear + step, step)

        self.list_s.append(0)
        radial_offset = 0.0025

        for current_s in s_linear:
            self.list_s.append(current_s)

            current_middle_s = current_s - step / 2.0
            self.list_s_middle.append(current_middle_s)

            self.list_R.append(current_middle_s * np.sin(self.angle_base * np.pi / 180) + radial_offset)
            self.list_h.append(current_middle_s * np.cos(self.angle_base * np.pi / 180))
            self.list_phi.append(90.0 - self.angle_base)

            self.list_size_element.append(step)

    def compute_segments_curved(self, n_segments=100):
        step = float(self.length_curved) / n_segments
        s_linear = np.arange(self.length_linear + step, self.total_length + step, step)

        theta_2 = -np.pi + self.angle_base * np.pi / 180.0

        R_end_linear = self.list_R[-1]
        h_end_linear = self.list_h[-1]

        for current_s in s_linear:
            self.list_s.append(s_linear)

            current_middle_s = current_s - step / 2.0
            self.list_s_middle.append(current_middle_s)

            current_angle_rad = float(current_middle_s - self.length_linear) / self.radius_curvature

            self.list_phi.append(90.0 - self.angle_base - current_angle_rad * 180.0 / np.pi)

            array_vector_start_to_current_middle = np.array([self.radius_curvature * (1 - np.cos(current_angle_rad)),
                                                             self.radius_curvature * np.sin(current_angle_rad)])

            rotation_matrix = np.array([[np.cos(theta_2), np.sin(theta_2)],
                                        [-np.sin(theta_2), np.cos(theta_2)]])

            additional_R_h = np.dot(rotation_matrix, array_vector_start_to_current_middle)

            self.list_R.append(R_end_linear - additional_R_h[0])
            self.list_h.append(h_end_linear - additional_R_h[1])

            self.list_size_element.append(step)

    def plot_seed(self, gray_region=None):
        linewidth = 3.0
        color = (0.5, 0.5, 0.5, 0.5)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.list_R, self.list_h, label='seed wing', linewidth=linewidth)
        plt.xlabel("R [m]")
        plt.ylabel("h [m]")
        plt.axis('equal')
        plt.legend()
        plt.grid()
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -100), gray_region[1] - gray_region[0], 200, color=color)
            ax.add_patch(rect)
            plt.ylim([-0.005, 0.05])
            plt.xlim([0.00, 0.05])
        plt.tight_layout()
        if self.save_seed is not None:
            plt.savefig(self.save_seed)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.list_s_middle, self.list_R, label='R')
        plt.plot(self.list_s_middle, self.list_h, label='h')
        plt.xlabel("s_middle [m]")
        plt.ylabel("[m]")
        plt.legend()
        plt.grid()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.list_s_middle, self.list_phi, label='phi')
        plt.xlabel("s_middle [m]")
        plt.ylabel("phi [deg]")
        plt.legend()
        plt.grid()
        plt.show()
