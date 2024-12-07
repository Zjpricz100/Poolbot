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

import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
import tf.transformations as tft

"""Class designed to handle the actuation components of Poolbot"""

class Commander:
    def __init__(self, limb, kin, tip_name):
        self.limb = limb
        self.kin = kin
        self.tip_name = tip_name
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.ik_solver = IK("base", "right_hand")
        
        
        # For visualizing trajectories
        self.pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        self.disp_traj = DisplayTrajectory()

        # For Moveit!
        self.planner = PathPlanner('right_arm')

    def get_current_position_and_orientation(self, source = "base", target = "right_hand"):
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
            trans = self.tfBuffer.lookup_transform(source, target, rospy.Time(0), rospy.Duration(10.0))
            
            # Create PoseStamped message
            pose_stamped = PoseStamped()
            
            # Set the header with the frame of reference and timestamp
            pose_stamped.header.stamp = rospy.Time.now()  # Set the current timestamp
            pose_stamped.header.frame_id = source  # Frame of reference
            
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

    def convert_pose(self, source, target):
        print(type(source))
        try:
            # Check if the transform exists and is valid
            t = self.tfBuffer.transform(source, target, rospy.Duration(1.0))
        except tf2_ros.TransformException as e:
            rospy.logerr("Transform failed: %s", e)
            return None
        
        tn = PoseStamped()
        tn.header.stamp = rospy.get_rostime()
        tn.header.frame_id = "base"
        tn.pose.position.x = t.pose.position.x
        tn.pose.position.y = t.pose.position.y
        tn.pose.position.z = t.pose.position.z
        return tn

    def get_offset_point(self, ball_pos, curr_orientation):
        curr_orientation = [curr_orientation.orientation.x, curr_orientation.orientation.y, curr_orientation.orientation.z, curr_orientation.orientation.w]
        (roll, pitch, yaw) = tft.euler_from_quaternion(curr_orientation)
        angle_in_degrees = -10.0
        angle_in_radians = angle_in_degrees * (3.14159 / 180.0)  # Convert to radians
        new_pitch = pitch + angle_in_radians

        angle_in_degrees = -90
        angle_in_radians = angle_in_degrees * (3.14159 / 180.0)  # Convert to radians
        print(yaw)
        new_yaw = angle_in_radians

        rotation_quaternion = tft.quaternion_from_euler(roll, new_pitch, yaw + new_yaw)  # roll=0, pitch=angle, yaw=0
        print(new_yaw)
        r_z = np.array([[np.cos(new_yaw), -np.sin(new_yaw), 0],
                        [np.sin(new_yaw), np.cos(new_yaw), 0],
                        [0, 0, 1]])

        offset_vector = [-0.1, -0.01, 0.1]

        rotated_vector = r_z @ offset_vector
        print(rotated_vector)
        return rotation_quaternion, rotated_vector
        

    def move_to_ball(self, ball_color):
        ball_pose = rospy.wait_for_message(f"ball/{ball_color}", PoseStamped)
        print(ball_pose)

        curr_pos = self.get_current_position_and_orientation()
       
        rotation_quaternion, rotated_vector = self.get_offset_point(ball_pose, curr_pos.pose)

        rotated_orientation = Quaternion()
        rotated_orientation.x = rotation_quaternion[0]
        rotated_orientation.y = rotation_quaternion[1]
        rotated_orientation.z = rotation_quaternion[2]
        rotated_orientation.w = rotation_quaternion[3]

        #ball_pose.pose.orientation = rotated_orientation
        ball_pose.pose.orientation = rotated_orientation
        ball_pose.pose.position.x += rotated_vector[0]
        ball_pose.pose.position.y += rotated_vector[1]
        ball_pose.pose.position.z += rotated_vector[2]
        # ball_pose.pose.position.x += -0.1
        # ball_pose.pose.position.y += -0.01
        # ball_pose.pose.position.z += 0.1
        print(ball_pose)
        plan = self.planner.plan_to_pose(ball_pose)
        plan = self.planner.retime_trajectory(plan, 0.3)
        self.planner.execute_plan(plan[1])


if __name__ == "__main__":
    rospy.init_node('testing_node')
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")
    commander = Commander(limb, kin, "pole")
    commander.move_to_ball("purple")
