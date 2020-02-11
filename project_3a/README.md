# Project 3a

The `pursuit_evasion_ws` folder is a catikin workspace with one package called 'pursuit_evasion'. That package is located in `pursuit_evasion_ws/src`. This package contains `launch` and `world` folders which contain roslaunch and gazebo .world files.

# Task 1
We've created `smallmaze.world` with gazebo for our robots to play around in.

## How to roslaunch `apartment.launch`
`cd pursuit_evasion_ws`
`. devel/setup.bash`
`roslaunch pursuit_evasion_pkg apartment.launch`

This should be output in the terminal:
'[spawn_urdf-4] process has died [pid 9882, exit code 1, cmd /opt/ros/melodic/lib/gazebo_ros/spawn_model -urdf -model turtlebot3 -x 0.0 -y 3.5 -z 0.0 -param robot_description __name:=spawn_urdf __log:=/home/ch/.ros/log/4501b72e-4c2d-11ea-b41a-04d9f51dc43b/spawn_urdf-4.log].
log file: /home/ch/.ros/log/4501b72e-4c2d-11ea-b41a-04d9f51dc43b/spawn_urdf-4*.log'

but do not fear, it is a known error. There is actually nothing wrong and the error message can be ignored.

# Task 2
View a camera node
`roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch`

# Task 3
## How to control the bot with keyboard commands
`roslaunch turtlebot3_example turtlebot3_pointop_key.launch`


## How to test our traversal
`rosrun pursuit_evasion_pkg traverse_floor.py`
`rosservice call /gazebo/reset_world`
