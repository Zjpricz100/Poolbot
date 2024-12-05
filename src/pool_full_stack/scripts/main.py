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
    FeedforwardJointVelocityController,
)
from utils.utils import *

from trac_ik_python.trac_ik import IK

import rospy
import tf2_ros
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import MoveGroupCommander
from sawyer_pykdl import sawyer_kinematics


NUM_JOINTS = 7

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


def get_current_position_and_orientation(limb):
    """
    Get the current end-effector position and orientation of the robot, 
    returning them as a PoseStamped message.
    
    Parameters:
    -----------
    limb : intera_interface.Limb
        The limb interface object for the Sawyer robot.
    
    Returns:
    --------
    geometry_msgs/PoseStamped
        A PoseStamped message containing the position and orientation 
        of the end-effector with a timestamp and frame of reference.
    """
    # Create a tf2 buffer and listener
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    
    try:
        # Look up the transform from base to right_hand
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
        
        # Create PoseStamped message
        pose_stamped = PoseStamped()
        
        # Set the header with the frame of reference and timestamp
        pose_stamped.header.stamp = rospy.Time.now()  # Set the current timestamp
        pose_stamped.header.frame_id = 'base'  # Frame of reference
        
        # Set the position (Point)
        pose_stamped.pose.position.x = trans.transform.translation.x
        pose_stamped.pose.position.y = trans.transform.translation.y
        pose_stamped.pose.position.z = trans.transform.translation.z
        
        # Set the orientation (Quaternion)
        pose_stamped.pose.orientation.x = trans.transform.rotation.x
        pose_stamped.pose.orientation.y = trans.transform.rotation.y
        pose_stamped.pose.orientation.z = trans.transform.rotation.z
        pose_stamped.pose.orientation.w = trans.transform.rotation.w
        
        return pose_stamped
    
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get current position and orientation: {e}")
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

def get_trajectory(limb, kin, direction, distance, target_vel, ik_solver, args, ar_tag=False):
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
        target_velocity = target_vel,
        desired_orientation = [0, 1, 0, 0]
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
        Kp = 0.25 * np.array([0.4, 4, 1.7, 0.5, 2, 2, 3])
        Kd = 0.05 * np.array([2, 0.8, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.5, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    elif controller_name == 'pid_torque': # Gravity compensation
        Kp = 3 * np.array([0.4, 6, 1.7, 0.5, 2, 2, 3])
        Kd = 0.05 * np.array([2, 0.8, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.5, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointTorqueController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller

def exec_trajectory(robot_trajectory, pub, disp_traj, args, limb, kin, planner):
    disp_traj.trajectory.append(robot_trajectory)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
    if args.controller_name != "moveit":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

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

def lookup_tag(tag_number):
    """
    Given an AR tag number, this returns the position of the AR tag in the robot's base frame.

    Parameters
    ----------
    tag_number : int

    Returns
    -------
    3x' :obj:`numpy.ndarray`
        tag position
    """
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)

    try:
        # TODO: lookup the transform and save it in trans
        # The rospy.Time(0) is the latest available 
        # The rospy.Duration(10.0) is the amount of time to wait for the transform to be available before throwing an exception
        trans = tfBuffer.lookup_transform('base',f'ar_marker_{tag_number}', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)
        print("Retrying ...")

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    print(tag_pos)
    return np.array(tag_pos)

# Uses Move it and IK to move to the starting point of the table (designated by AR tag)
def setup_game(limb, kin, ik_solver, args):    
    # Wait for the IK service to become available
    rospy.wait_for_service('compute_ik')
    compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

    # Create a MoveGroupCommander for the right arm
    group = MoveGroupCommander("right_arm")

    # Grab ar_tag positions
    tag_pos = [lookup_tag(marker) for marker in args.ar_marker][0]
    print(tag_pos)
    while not rospy.is_shutdown():
        # Set up the first target pose
        request = GetPositionIKRequest()
        request.ik_request.group_name = "right_arm"
        link = "right_gripper_tip"
        request.ik_request.ik_link_name = link
        request.ik_request.pose_stamped.header.frame_id = "base"
        
        # Set request pose position to ar_tag pos
        request.ik_request.pose_stamped.pose.position.x = tag_pos[0]
        request.ik_request.pose_stamped.pose.position.y = tag_pos[1]
        request.ik_request.pose_stamped.pose.position.z = tag_pos[2] + 0.5
        request.ik_request.pose_stamped.pose.orientation.x = 0.0
        request.ik_request.pose_stamped.pose.orientation.y = 1.0
        request.ik_request.pose_stamped.pose.orientation.z = 0.0
        request.ik_request.pose_stamped.pose.orientation.w = 0.0

        try:
            # Compute IK for the first target
            response = compute_ik(request)
            print(response)
            if response.error_code.val == response.error_code.SUCCESS:
                waypoints = []
                
                # Append the first target pose
                waypoints.append(request.ik_request.pose_stamped.pose)

                # Plan a Cartesian path
                (plan, fraction) = group.compute_cartesian_path(
                    waypoints,   # waypoints to follow
                    0.01,        # eef_step
                    0.0          # jump_threshold
                )
                print(fraction)

                # User confirmation before executing
                user_input = input("Enter 'y' if the trajectory looks safe on RVIZ")
                if user_input == 'y':
                    group.execute(plan, wait=True)
                    rospy.sleep(1.0)

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


def moveToBall(ball_position, limb, kin, ik_solver, args):
    return None
    

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help='Only "line" is supported')
    parser.add_argument('-controller_name', '-c', type=str, default='pid', help='Options: moveit, open_loop, pid. Default: pid')
    parser.add_argument('-rate', type=int, default=200, help='Control loop rate in Hz. Default: 200')
    parser.add_argument('-timeout', type=int, default=None, help='Timeout in seconds. Default: None')
    parser.add_argument('-num_way', type=int, default=50, help='Number of waypoints. Default: 50')
    parser.add_argument('--log', action='store_true', help='Log and plot controller performance')
    parser.add_argument('-ar_marker', '-ar', nargs='+', help=
        'Which AR marker to use.  Default: 1'
    )
    args = parser.parse_args()

    rospy.init_node('linear_motion_node')

    substitute = "stp_022312TP99620_tip_1"

    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")
    ik_solver = IK("base", "right_gripper_tip") # for amir

    pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    disp_traj = DisplayTrajectory()

    # Get an appropriate RobotTrajectory for the linear task
    # This is a wrapper around MoveIt! for you to use. We use MoveIt! to go to the start position
    planner = PathPlanner('right_arm')
    '''
    
    curr_pos = get_current_position_and_orientation(limb)
    curr_pos.pose.position.z -= 0.4
    plan = planner.plan_to_pose(curr_pos)

    if args.controller_name != "moveit":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])
    '''
    setup_game(limb, kin, ik_solver, args)
    robot_trajectory = get_trajectory(
        limb, 
        kin, 
        np.array([-1, 0.5, 0]),
        0.1, # meters
        0.5,
        ik_solver, 
        args)
    
    exec_trajectory(robot_trajectory, pub, disp_traj, args, limb, kin, planner)

    robot_trajectory = get_trajectory(
        limb, 
        kin, 
        np.array([1, -0.5, 0]),
        0.1, # meters
        0.5,
        ik_solver, 
        args)

    exec_trajectory(robot_trajectory, pub, disp_traj, args, limb, kin, planner)

if __name__ == "__main__":
    main()
