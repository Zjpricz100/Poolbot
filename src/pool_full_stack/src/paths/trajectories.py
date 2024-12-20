#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:
    def __init__(self):
        """
        Parameters
        ----------
        total_time : float
        	desired duration of the trajectory in seconds 
        """
        pass

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        The function get_g_matrix from utils may be useful to perform some frame
        transformations.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_boundaries = [[-2, 2],
                           [-2, 2],
                           [-2, 2]]
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([max(pos_boundaries[0][0], min(target_positions[:, 0]) + pos_padding[0][0]), 
                        min(pos_boundaries[0][1], max(target_positions[:, 0]) + pos_padding[0][1])])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([max(pos_boundaries[1][0], min(target_positions[:, 1]) + pos_padding[1][0]), 
                        min(pos_boundaries[1][1], max(target_positions[:, 1]) + pos_padding[1][1])])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([max(pos_boundaries[2][0], min(target_positions[:, 2]) + pos_padding[2][0]), 
                        min(pos_boundaries[2][1], max(target_positions[:, 2]) + pos_padding[2][1])])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_boundaries = [[-2, 2],
                           [-2, 2],
                           [-2, 2]]
        vel_paddinTrajectory.g = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([max(vel_boundaries[0][0], min(target_velocities[:, 0]) + vel_padding[0][0]), 
                        min(vel_boundaries[0][1], max(target_velocities[:, 0]) + vel_padding[0][1])])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([max(vel_boundaries[1][0], min(target_velocities[:, 1]) + vel_padding[1][0]), 
                        min(vel_boundaries[1][1], max(target_velocities[:, 1]) + vel_padding[1][1])])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([max(vel_boundaries[2][0], min(target_velocities[:, 2]) + vel_padding[2][0]), 
                        min(vel_boundaries[2][1], max(target_velocities[:, 2]) + vel_padding[2][1])])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):
    def __init__(self, start_position, goal_position, target_velocity, desired_orientation = np.array([0, 1, 0, 0])):
        """
        start_position is 1x3 np array
        goal_position is 1x3 np array
        target_velocity is velocity we want to travel at from start position to goal_position
        """
        Trajectory.__init__(self)
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance = np.linalg.norm(self.goal_position - self.start_position)
        self.constant_velocity = target_velocity
        self.total_time = self.distance / self.constant_velocity
        print(self.total_time)
        self.direction_vector = (goal_position - start_position) / self.distance # normalized direction vector
        self.desired_orientation = desired_orientation
    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        if time >= self.total_time:
            time = self.total_time
        displacement = self.constant_velocity * time
        pos = self.start_position + displacement * self.direction_vector
        return np.hstack((pos, self.desired_orientation))
    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        if time >= self.total_time:
            return np.zeros(6)
        linear_vel = self.constant_velocity * self.direction_vector
        return np.hstack((linear_vel, np.zeros(3)))
    def target_acceleration(self, time):
        return np.zeros(6)

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """

    path = LinearTrajectory(np.array([0, 0, 0]), np.array([.1, .1, .1]), 1)
    # path = CircularTrajectory(np.array([0.2, 0.4, 0.6]), .3, 10)
    path.display_trajectory()