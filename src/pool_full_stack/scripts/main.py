#!/usr/bin/env python
"""
Starter script for 106a lab7. 
Author: Chris Correa
"""
import sys
import argparse
import numpy as np
import rospkg
import roslaunch

from paths.trajectories import LinearTrajectory
from paths.paths import MotionPath
from paths.path_planner import PathPlanner
from controllers.controllers import ( 
    PIDJointVelocityController, 
    FeedforwardJointVelocityController
)
from utils.utils import *

from trac_ik_python.trac_ik import IK

import rospy
import tf2_ros
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics

def get_current_position(limb):
    """
    Get the current end-effector position of the robot.
    
    Parameters:
    -----------
    limb : intera_interface.Limb
        The limb interface object for the Sawyer robot.
    
    Returns:
    --------
    numpy.ndarray
        A 3-element array representing the [x, y, z] position of the end-effector.
    """
    # Create a tf2 buffer and listener
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    
    try:
        # Look up the transform from base to right_hand
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
        
        # Extract the position from the transform
        position = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ])
        
        return position
    
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get current position: {e}")
        return None



def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    if input('Would you like to tuck the arm? (y/n): ') == 'y':
        rospack = rospkg.RosPack()
        path = rospack.get_path('sawyer_full_stack')
        launch_path = path + '/launch/custom_sawyer_tuck.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        launch.start()
    else:
        print('Canceled. Not tucking the arm.')

def get_trajectory(limb, kin, direction, distance, ik_solver, args):
    """
    Returns an appropriate robot trajectory for the specified task.  You should 
    be implementing the path functions in paths.py and call them here
    
    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    tag_pos : 3x' :obj:`numpy.ndarray`

    direction: a (1x3) numpy array vector indicating what direction to go from robots current position
    distance: The amount of meters the robot will move in direction
        
    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """

    current_position = get_current_position(limb)
    target_position = current_position + direction * distance

    trajectory = LinearTrajectory(
        start_position = current_position,
        goal_position = target_position,
        target_velocity = 0.4
    )
    path = MotionPath(limb, kin, trajectory)
    return path.to_robot_trajectory(args.num_way, jointspace=True, extra_points=0)

def get_controller(controller_name, limb, kin):
    """
    Gets the correct controller from controllers.py

    Parameters
    ----------
    controller_name : string

    Returns
    -------
    :obj:`Controller`
    """
    if controller_name == 'open_loop':
        controller = FeedforwardJointVelocityController(limb, kin)
    elif controller_name == 'pid':
        Kp = 0.5 * np.array([0.4, 5, 1.7, 1.5, 2, 2, 3])
        Kd = 0.05 * np.array([2, 1.5, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.5, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help='Only "line" is supported')
    parser.add_argument('-controller_name', '-c', type=str, default='pid', help='Options: moveit, open_loop, pid. Default: pid')
    parser.add_argument('-rate', type=int, default=200, help='Control loop rate in Hz. Default: 200')
    parser.add_argument('-timeout', type=int, default=None, help='Timeout in seconds. Default: None')
    parser.add_argument('-num_way', type=int, default=50, help='Number of waypoints. Default: 50')
    parser.add_argument('--log', action='store_true', help='Log and plot controller performance')
    args = parser.parse_args()

    rospy.init_node('linear_motion_node')

    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")
    ik_solver = IK("base", "right_gripper_tip")

    # Get an appropriate RobotTrajectory for the linear task
    robot_trajectory = get_trajectory(
        limb, 
        kin, 
        np.array([-1, 0, 0]),
        0.1, # meters
        ik_solver, 
        args)

    # This is a wrapper around MoveIt! for you to use. We use MoveIt! to go to the start position
    planner = PathPlanner('right_arm')

    # By publishing the trajectory to the move_group/display_planned_path topic, you should
    # be able to view it in RViz. You will have to click the "loop animation" setting in
    # the planned path section of MoveIt! in the menu on the left side of the screen.
    pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
    if args.controller_name != "moveit":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

    if args.controller_name == "moveit":
        try:
            input('Press <Enter> to execute the trajectory using MOVEIT')
        except KeyboardInterrupt:
            sys.exit()
        # Uses MoveIt! to execute the trajectory.
        planner.execute_plan(robot_trajectory)
    else:
        controller = get_controller(args.controller_name, limb, kin)
        try:
            input('Press <Enter> to execute the trajectory using YOUR OWN controller')
        except KeyboardInterrupt:
            sys.exit()
        # execute the path using your own controller.
        done = controller.execute_path(
            robot_trajectory,
            rate=args.rate,
            timeout=args.timeout,
            log=args.log
        )
        if not done:
            print('Failed to move to position')
            sys.exit(0)

if __name__ == "__main__":
    main()
