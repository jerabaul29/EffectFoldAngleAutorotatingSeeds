"""
For description of the script, see the README.md
"""

import numpy as np
from printind.printind_function import printi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ComputeForce(object):
    def __init__(self, verbose=0, rho=1000.0, added_AOA_camber_deg=0.0, radius_sphere=0.010):
        self.verbose = verbose
        self.rho = rho
        self.added_AOA_camber_deg = added_AOA_camber_deg
        self.radius_sphere = radius_sphere

    def set_vertical_velocity(self, vertical_velocity):
        self.vertical_velocity = vertical_velocity

    def set_angular_frequency(self, angular_frequency):
        self.angular_frequency = angular_frequency
        self.angular_rate = 2 * np.pi * self.angular_frequency

    def set_interpolator_Cl(self, interpolator_Cl):
        self.interpolator_Cl = interpolator_Cl

    def set_interpolator_Cd(self, interpolator_Cd):
        self.interpolator_Cd = interpolator_Cd

    def set_seed_profile(self, wing_instance):
        self.wing_instance = wing_instance

    def set_wing_pitch(self, wing_pitch):
        self.wing_pitch = wing_pitch

    def set_wing_chord(self, wing_chord):
        self.wing_chord = wing_chord

    def compute_force_all_elements(self):
        self.list_crrt_R = []
        self.list_projected_vertical_velocity = []
        self.list_AOA_deg = []
        self.list_total_velocity = []
        self.list_wind_angle_rad = []
        self.list_drag = []
        self.list_lift = []
        self.list_Cd = []
        self.list_Cl = []
        self.list_forward_force = []
        self.list_vertical_force = []
        self.list_forward_force_lift = []
        self.list_forward_force_drag = []
        self.list_vertical_force_lift = []
        self.list_vertical_force_drag = []
        self.list_base_coeff = []
        self.list_crrt_rotation_velocity = []
        self.list_forward_moments = []

        for (crrt_size, crrt_R, crrt_phi) in zip(self.wing_instance.list_size_element, self.wing_instance.list_R, self.wing_instance.list_phi):
            # horizontal velocity due to rotation of the seed
            crrt_rotation_velocity = self.angular_rate * crrt_R

            # angle of attack
            projected_vertical_velocity = self.vertical_velocity * np.cos(crrt_phi * np.pi / 180.0)
            wind_angle_rad = np.arctan2(projected_vertical_velocity, crrt_rotation_velocity)
            crrt_AOA_deg = wind_angle_rad * 180.0 / np.pi - self.wing_pitch

            # total velocity magnitude
            total_velocity = np.sqrt(crrt_rotation_velocity**2 + projected_vertical_velocity**2)

            # base coefficient for computation of lift and drag
            base_coeff = 0.5 * self.rho * (total_velocity**2) * self.wing_chord * crrt_size

            # compute lift and drag; careful about the orientation of crrt_AOA_deg! This is because of
            # direction of rotation vs. the sketches
            Cd = self.interpolator_Cd.return_interpolated(crrt_AOA_deg + self.added_AOA_camber_deg)
            Cl = self.interpolator_Cl.return_interpolated(crrt_AOA_deg + self.added_AOA_camber_deg)
            crrt_lift = base_coeff * Cl
            crrt_drag = base_coeff * Cd
            crrt_forward = np.sin(wind_angle_rad) * crrt_lift \
                - np.cos(wind_angle_rad) * crrt_drag
            crrt_vertical = np.cos(wind_angle_rad) * crrt_lift \
                + np.sin(wind_angle_rad) * crrt_drag

            crrt_forward_projected = crrt_forward
            crrt_vertical_projected = crrt_vertical * np.cos(crrt_phi * np.pi / 180.0)

            self.list_crrt_R.append(crrt_R)
            self.list_projected_vertical_velocity.append(projected_vertical_velocity)
            self.list_crrt_rotation_velocity.append(crrt_rotation_velocity)
            self.list_wind_angle_rad.append(wind_angle_rad)
            self.list_total_velocity.append(total_velocity)
            self.list_AOA_deg.append(crrt_AOA_deg)
            self.list_base_coeff.append(base_coeff)
            self.list_drag.append(crrt_drag)
            self.list_lift.append(crrt_lift)
            self.list_Cd.append(Cd)
            self.list_Cl.append(Cl)
            self.list_forward_force.append(crrt_forward_projected)
            self.list_vertical_force.append(crrt_vertical_projected)
            self.list_forward_moments.append(crrt_forward_projected * crrt_R)

    def compute_resultant_force(self):
        self.resultant_forward_moment = sum(self.list_forward_moments)
        self.resultant_vertical_force = sum(self.list_vertical_force)

    def display_reduced_information(self, title_base=None, gray_region=None):
        linewidth = 3.0
        color = (0.5, 0.5, 0.5, 0.5)
        
        
        
        # figure with profile and angle of attack ------------------------------
        fig, ax1 = plt.subplots()
        
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -0.002), gray_region[1] - gray_region[0], 0.06, color=color)
            ax1.add_patch(rect)
        
        ax1.plot(self.wing_instance.list_R, self.wing_instance.list_h, 'b', linewidth=linewidth * 2.0, label="seed wing", linestyle="-")
        ax1.set_xlabel('R [m]')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('h [m]', color='b')
        ax1.tick_params('y', colors='b')
        plt.ylim([-0.002, 0.052])
        plt.xlim([-0.002, 0.052])

        ax2 = ax1.twinx()
        ax2.plot(self.wing_instance.list_R, self.list_AOA_deg, 'r', linewidth=linewidth, linestyle="--", label="Angle of attack")
        ax2.set_ylabel('local angle of attack [deg]', color='r')
        ax2.tick_params('y', colors='r')
        
        ax2.plot([], [], 'b', linewidth=linewidth, linestyle="-", label="seed wing")
        
        plt.legend(loc="lower right")

        fig.tight_layout()
        
        plt.savefig(title_base + "combined_fig_1.pdf")
        # done -----------------------------------------------------------------
        
        
        
        # figure with vertical force and moment --------------------------------
        fig, ax1 = plt.subplots()
        
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -0.5), gray_region[1] - gray_region[0], 1.0, color=color)
            ax1.add_patch(rect)
        
        ax1.plot(self.wing_instance.list_R, np.array(self.list_vertical_force) / np.array(self.wing_instance.list_size_element), 'b', linewidth=linewidth, linestyle="-", label="vertical force")
        ax1.plot([0.0, 0.049], [0.0, 0.0], 'k', linewidth=linewidth * 0.75)
        ax1.set_xlabel('R [m]')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('Vertical force distribution [N/m]', color='b')
        ax1.tick_params('y', colors='b')
        plt.ylim([-0.3, 0.3])
        plt.xlim([0.00, 0.049])

        ax2 = ax1.twinx()
        ax2.plot(self.wing_instance.list_R, np.array(self.list_forward_moments) / np.array(self.wing_instance.list_size_element), 'r', linewidth=linewidth, linestyle="--", label="forward moment")
        ax2.set_ylabel('Forward moment distribution [N.m / m]', color='r')
        ax2.tick_params('y', colors='r')
        plt.ylim([-0.0044, 0.0045])
        
        ax2.plot([], [], 'b', linewidth=linewidth, linestyle="-", label="vertical force")
        
        plt.legend(loc="lower left")

        fig.tight_layout()
        
        plt.savefig(title_base + "combined_fig_2.pdf")
        # done -----------------------------------------------------------------
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -100), gray_region[1] - gray_region[0], 200, color=color)
            ax.add_patch(rect)
        plt.plot(self.wing_instance.list_R, np.array(self.list_vertical_force) / np.array(self.wing_instance.list_size_element), linewidth=linewidth)
        # plt.legend()
        plt.grid()
        plt.xlabel("R [m]")
        plt.ylabel("Vertical force distribution [N/m]")
        plt.ylim([-0.3, 0.4])
        plt.tight_layout()
        if title_base is not None:
            plt.savefig(title_base + "_forceDistribution.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -100), gray_region[1] - gray_region[0], 200, color=color)
            ax.add_patch(rect)
        plt.plot(self.wing_instance.list_R, np.array(self.list_forward_force) / np.array(self.wing_instance.list_size_element), linewidth=linewidth)
        # plt.legend()
        plt.grid()
        plt.ylim([-0.2, 0.1])
        plt.xlabel("R [m]")
        plt.ylabel("Forward force distribution [N/m]")
        plt.tight_layout()
        if title_base is not None:
            plt.savefig(title_base + "_forwardForceDistribution.pdf")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -100), gray_region[1] - gray_region[0], 200, color=color)
            ax.add_patch(rect)
        plt.plot(self.wing_instance.list_R, np.array(self.list_forward_moments) / np.array(self.wing_instance.list_size_element), linewidth=linewidth)
        # plt.legend()
        plt.grid()
        plt.xlabel("R [m]")
        plt.ylabel("Forward moment distribution [N.m / m]")
        plt.ylim([-0.008, 0.004])
        plt.tight_layout()
        if title_base is not None:
            plt.savefig(title_base + "_momentDistribution.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if gray_region is not None:
            rect = Rectangle((gray_region[0], -100), gray_region[1] - gray_region[0], 200, color=color)
            ax.add_patch(rect)
        plt.plot(self.wing_instance.list_R, self.list_AOA_deg, linewidth=linewidth)
        # plt.legend()
        plt.grid()
        plt.xlabel("R [m]")
        plt.ylabel("local angle of attack [deg]")
        plt.ylim([-20, 50])
        plt.tight_layout()
        if title_base is not None:
            plt.savefig(title_base + "_AOADistribution.pdf")
            
            
            
            
            
        plt.show()

    def return_forward_moment(self):
        return(self.resultant_forward_moment)

    def return_vertical_force(self):
        return(self.resultant_vertical_force)

    def display_all_results(self):
        plt.figure()
        plt.plot(self.wing_instance.list_R, np.array(self.list_AOA) * 180.0 / np.pi, label="AOA")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, np.array(self.list_wind_angle) * 180.0 / np.pi, label="wind angle")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_total_velocity, label="total_velocity")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_crrt_rotation_velocity, label="crrt_rotation_velocity")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_Cl, label="Cl")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_Cd, label="Cd")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_base_coeff, label="base_coeff")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_lift, label="lift")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_drag, label="drag")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, np.array(self.list_lift) / np.array(self.list_drag), label="lift/drag")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, self.list_forward_force_lift, label="forward lift")
        plt.plot(self.wing_instance.list_R, self.list_forward_force_drag, label="forward drag")
        plt.plot(self.wing_instance.list_R, self.list_vertical_force_lift, label="vertical lift")
        plt.plot(self.wing_instance.list_R, self.list_vertical_force_drag, label="vertical drag")
        plt.legend()
        plt.plot()

        plt.figure()
        plt.plot(self.wing_instance.list_R, np.array(self.list_vertical_force) / np.array(self.wing_instance.list_size_element), label="Vertical force")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.wing_instance.list_R, np.array(self.list_forward_force) / np.array(self.wing_instance.list_size_element), label="Forward force")
        plt.legend()
        plt.show()


"""
note that in compute_force_on_seed, both the rotation and the vertical velocity are positive for 'normal seed'. I.e.
positive vertical velocity means seed going down, positive frequency means seed rotating as in experiments.
"""
