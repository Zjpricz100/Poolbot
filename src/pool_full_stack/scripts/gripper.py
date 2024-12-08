import rospy

from intera_interface import gripper as robot_gripper

rospy.init_node('gripper_test')

right_gripper = robot_gripper.Gripper('right_gripper')

right_gripper.open()
rospy.sleep(1.0)

current_pos = right_gripper.get_position()
right_gripper.set_position(current_pos - 0.01)

rospy.sleep(1.0)