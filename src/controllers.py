#!/usr/bin/env python

"""
Starter script for lab1. 
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Lab imports
from utils import *

# ROS imports
try:
    import tf
    import tf2_ros
    import rospy
    import baxter_interface
    import intera_interface
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import RobotTrajectory
except:
    pass

NUM_JOINTS = 7

class Controller:

    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        limb : :obj:`sawyer_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin

        # Set this attribute to True if the present controller is a jointspace controller.
        self.is_joinstpace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly.  

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        """
        pass

    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)

    def plot_results(self, times, actual_positions, actual_velocities, target_positions, target_velocities): #FIX
        """
        Plots results.
        If the path is in joint space, it will plot both workspace and jointspace plots.
        Otherwise it'll plot only workspace

        Inputs:
        times : nx' :obj:`numpy.ndarray`
        actual_positions : nx7 or nx6 :obj:`numpy.ndarray`
            actual joint positions for each time in times
        actual_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            actual joint velocities for each time in times
        target_positions: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace positions for each time in times
        target_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace velocities for each time in times
        """

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)

        # Find the actual workspace positions and velocities
        actual_workspace_positions = np.zeros((len(times), 3))
        actual_workspace_velocities = np.zeros((len(times), 3))
        actual_workspace_quaternions = np.zeros((len(times), 4))

        for i in range(len(times)):
            positions_dict = joint_array_to_dict(actual_positions[i], self._limb)
            fk = self._kin.forward_position_kinematics(joint_values=positions_dict)
            
            actual_workspace_positions[i, :] = fk[:3]
            actual_workspace_velocities[i, :] = \
                self._kin.jacobian(joint_values=positions_dict)[:3].dot(actual_velocities[i])
            actual_workspace_quaternions[i, :] = fk[3:]
        # check if joint space
        if self.is_joinstpace_controller:
            # it's joint space

            target_workspace_positions = np.zeros((len(times), 3))
            target_workspace_velocities = np.zeros((len(times), 3))
            target_workspace_quaternions = np.zeros((len(times), 4))

            for i in range(len(times)):
                positions_dict = joint_array_to_dict(target_positions[i], self._limb)
                target_workspace_positions[i, :] = \
                    self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
                target_workspace_velocities[i, :] = \
                    self._kin.jacobian(joint_values=positions_dict)[:3].dot(target_velocities[i])
                target_workspace_quaternions[i, :] = np.array([0, 1, 0, 0])

            # Plot joint space
            plt.figure()
            # print len(times), actual_positions.shape()
            joint_num = len(self._limb.joint_names())
            for joint in range(joint_num):
                plt.subplot(joint_num,2,2*joint+1)
                plt.plot(times, actual_positions[:,joint], label='Actual')
                plt.plot(times, target_positions[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Position Error")
                plt.legend()

                plt.subplot(joint_num,2,2*joint+2)
                plt.plot(times, actual_velocities[:,joint], label='Actual')
                plt.plot(times, target_velocities[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Velocity Error")
                plt.legend()
            print("Close the plot window to continue")
            plt.show()

        else:
            # it's workspace
            target_workspace_positions = target_positions
            target_workspace_velocities = target_velocities
            target_workspace_quaternions = np.zeros((len(times), 4))
            target_workspace_quaternions[:, 1] = 1

        plt.figure()
        workspace_joints = ('X', 'Y', 'Z')
        joint_num = len(workspace_joints)
        for joint in range(joint_num):
            plt.subplot(joint_num,2,2*joint+1)
            plt.plot(times, actual_workspace_positions[:,joint], label='Actual')
            plt.plot(times, target_workspace_positions[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Position Error")
            plt.legend()

            plt.subplot(joint_num,2,2*joint+2)
            plt.plot(times, actual_velocities[:,joint], label='Actual')
            plt.plot(times, target_velocities[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Velocity Error")
            plt.legend()

        print("Close the plot window to continue")
        plt.show()

        # Plot orientation error. This is measured by considering the
        # axis angle representation of the rotation matrix mapping
        # the desired orientation to the actual orientation. We use
        # the corresponding angle as our metric. Note that perfect tracking
        # would mean that this "angle error" is always zero.
        angles = []
        for t in range(len(times)):
            quat1 = target_workspace_quaternions[t]
            quat2 = actual_workspace_quaternions[t]
            theta = axis_angle(quat1, quat2)
            angles.append(theta)

        plt.figure()
        plt.plot(times, angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Error Angle of End Effector (rad)")
        print("Close the plot window to continue")
        plt.show()

    def get_ik(self, pos):
        """
        takes in position (and orientation) and returns the pose IK for the robot
        
        Parameters
        ----------
        Inputs:
        pos : 6x': np.array of position

        Outputs:
        ik : 7x' : np.array of joint inputs
        """
        return self._kin.inverse_kinematics(pos[:3], pos[3:])
        

    def get_joint_velocity(self, vel):
        pi = self._kin.jacobian_pseudo_inverse()
        q_dot = (pi.dot(vel)).A1
        return q_dot

    def execute_path(self, traj, rate=200, timeout=None, log=False):
        """
        takes in a trajectory and moves the sawyer in order to follow the trajectory  

        Parameters
        ----------
        traj : :obj:`Trajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For plotting
        # print(path.joint_trajectory.joint_names)
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()

        # For interpolation
        total_time = traj.total_time
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            target_position = self.get_ik(traj.target_pose(t))
            target_velocity = self.get_joint_velocity(traj.target_velocity(t))
            target_acceleration = traj.target_acceleration(t) # this doesn't matter

            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position)
                actual_velocities.append(current_velocity)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if t > total_time: # make sure we don't go past total time
                self.stop_moving()
                break

        if log:
            self.plot_results(times, actual_positions, actual_velocities, target_positions, target_velocities)

        return True

class FeedforwardJointVelocityController(Controller):
    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        # TODO: Implement Feedforward control
        controller_velocity = target_velocity

        self._limb.set_joint_velocities(joint_array_to_dict(controller_velocity, self._limb))

class PIDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  This controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the sawyer's 
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the sawyer.
    """
    def __init__(self, limb, kin, Kp, Ki, Kd, Kw):
        """
        Parameters
        ----------
        limb : :obj:`sawyer_interface.Limb`
        kin : :obj:`sawyerKinematics`
        Kp : 7x' :obj:`numpy.ndarray` of proportional constants
        Ki: 7x' :obj:`numpy.ndarray` of integral constants
        Kd : 7x' :obj:`numpy.ndarray` of derivative constants
        Kw : 7x' :obj:`numpy.ndarray` of anti-windup constants
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Ki = np.diag(Ki)
        self.Kd = np.diag(Kd)
        self.Kw = 0
        
        self.integ_error = np.zeros(7)
        
        self.is_joinstpace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly. This method should call
        self._limb.joint_angle and self._limb.joint_velocity to get the current joint position and velocity
        and self._limb.set_joint_velocities() to set the joint velocity to something.  You may find
        joint_array_to_dict() in utils.py useful

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)
        
        # TODO: implement PID control to set the joint velocities. 
        #print(current_position)
        #print(current_velocity)
        pos_error = target_position - current_position
        vel_error = target_velocity - current_velocity
        self.integ_error = self.Kw * self.integ_error + pos_error
        p_out = self.Kp @ pos_error
        print(p_out)
        i_out = self.Ki @ self.integ_error
        print(i_out)
        d_out = self.Kd @ vel_error
        print(d_out)
        controller_velocity = target_velocity + p_out + i_out + d_out
        #print(controller_velocity)
        self._limb.set_joint_velocities(joint_array_to_dict(controller_velocity, self._limb))