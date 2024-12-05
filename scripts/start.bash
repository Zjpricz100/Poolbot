xterm -hold -e 'rosrun intera_interface joint_trajectory_action_server.py' &
xterm -hold -e 'roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true' &
xterm -hold -e 'cd ..; source devel/setup.bash; roslaunch lab4_cam ar_track.launch' &
xterm -hold -e 'cd ..; source devel/setup.bash; roslaunch lab4_cam run_cam.launch' &
