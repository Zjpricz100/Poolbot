
import sys
import argparse
import numpy as np
import rospkg
import roslaunch

from geometry_msgs import PoseStamped

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

"""Class designed to handle the actuation components of Poolbot"""

class Commander:
    def __init__(self, limb, kin, tip_name):
        self.limb = limb
        self.kin = kin
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.ik_solver = IK("base", tip_name)
        
        # For visualizing trajectories
        self.pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        self.disp_traj = DisplayTrajectory()

        # For Moveit!
        self.planner = PathPlanner('righ_arm')

    def get_current_position_and_orientation(self):
        """
        Get the current end-effector position and orientation of the robot, 
        returning them as a PoseStamped message.
        
        Returns:
        --------
        geometry_msgs/PoseStamped
            A PoseStamped message containing the position and orientation 
            of the end-effector with a timestamp and frame of reference.
        """
        try:
            # Look up the transform from base to right_hand
            trans = self.tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
            
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
        
    def tuck(self):
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

    def get_trajectory(self, direction, distance, target_vel, args):
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

        current_position = self.get_current_position(self.limb)
        target_position = current_position + direction * distance

        trajectory = LinearTrajectory(
            start_position = current_position,
            goal_position = target_position,
            target_velocity = target_vel,
            desired_orientation = [0, 1, 0, 0]
        )
        path = MotionPath(self.limb, self.kin, trajectory)
        return path.to_robot_trajectory(args.num_way, jointspace=True, extra_points=0)
    
    def get_controller(self, controller_name):
        """
        Gets the correct controller from controllers.py

        Parameters
        ----------
        controller_name : string

        Returns
        -------
        :obj:`Controller`
        """
        limb = self.limb
        kin = self.kin
        if controller_name == 'open_loop':
            controller = FeedforwardJointVelocityController(limb, kin)
        elif controller_name == 'pid':
            Kp = 0.25 * np.array([0.4, 4, 1.7, 0.5, 2, 2, 3])
            Kd = 0.05 * np.array([2, 0.8, 2, 0.5, 0.8, 0.8, 0.8])
            Ki = 0.01 * np.array([1.4, 1.5, 1.4, 1, 0.6, 0.6, 0.6])
            Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
            controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
        else:
            raise ValueError('Controller {} not recognized'.format(controller_name))
        return controller

    def exec_trajectory(self, robot_trajectory, args):
        planner = self.planner
        
        # Move to the trajectory start position
        plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
        if args.controller_name != "moveit":
            plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        controller = self.get_controller(args.controller_name, self.limb, self.kin)
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

    def publish_trajectory(self, robot_trajectory):
        self.disp_traj.trajectory.append(robot_trajectory)
        self.disp_traj.trajectory_start = RobotState()
        self.pub.publish(self.disp_traj)

    

