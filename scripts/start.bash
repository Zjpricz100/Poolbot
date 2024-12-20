xterm -hold -e 'source ~ee106a/sawyer_setup.bash; rosrun intera_interface joint_trajectory_action_server.py' &
xterm -hold -e 'source ~ee106a/sawyer_setup.bash; roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true' &
xterm -hold -e 'cd ..; source devel/setup.bash; roslaunch lab4_cam multi_ar_track.launch' &
xterm -hold -e 'source ~ee106a/sawyer_setup.bash; rosrun intera_examples camera_display.py -c head_camera' &
xterm -hold -e 'cd ..; source devel/setup.bash; roslaunch lab4_cam sawyer_camera_track.launch' &
xterm -hold -e 'cd ..; source devel/setup.bash; roslaunch lab4_cam run_cam.launch' &

