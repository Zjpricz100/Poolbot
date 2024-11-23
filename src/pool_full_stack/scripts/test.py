from sawyer_pykdl import sawyer_kinematics
import sys
import argparse
import numpy as np
import rospkg
import roslaunch
import time as time

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

def move_at_velocity(kin, limb, target_displacement, dir, step_size = 0.5, time_step = 0.1):
    dir = normalize(dir)
    v_linear = dir * step_size
    v_ang = np.array([0, 0, 0])
    v_des = np.concatenate((v_linear, v_ang))
    total_displacement = 0
    print("test")
    while total_displacement < target_displacement and not rospy.is_shutdown():
        pi = kin.jacobian_pseudo_inverse()
        q_dot = (pi.dot(v_des)).A1
        print(q_dot)
        command = joint_array_to_dict(q_dot, limb)
        print(command)
        limb.set_joint_velocities(command)

        total_displacement += np.linalg.norm(v_linear) * time_step
        print(total_displacement)
        print("moving")
        time.sleep(time_step)
    command = joint_array_to_dict(np.zeros(6), limb)
    limb.set_joint_velocities(command)



def main():
    rospy.init_node('sawyer_kinematics')
    kin = sawyer_kinematics('right')
    limb = intera_interface.Limb("right")
                                                #[x, y, z]
    move_at_velocity(kin, limb, 0.125, np.array([0, 1.0, 0.0])) # Facing sawyer, goes to the right
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    '''
    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)
    goal_position = current_position
    goal_position[0] += 0.5 
    
    trajectory = LinearTrajectory(current_position, goal_position, 0.5)
    path = MotionPath(limb, kin, ik_solver, trajectory)
    robot_trajectory =  path.to_robot_trajectory(num_way, True)
    plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
    controller = controllers.FeedforwardJointVelocityController(limb, kin)

    # controller = controllers.FeedforwardJointVelocityController(limb, kin)
    # controller.execute_path
    '''

if __name__ == "__main__":
    main()