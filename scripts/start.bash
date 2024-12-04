xterm -hold -e 'rosrun intera_interface joint_trajectory_action_server.py' &
xterm -hold -e 'roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true' &
