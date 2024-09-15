# Base controller

Startup code for controling Gen3Lite in python. The default CMakeLists.txt file is configured to run python scripts using rospy lib - python integration for ROS.

## Notes on launch

* Use the following
```
rosrun base_controller example_moveit.py __ns:=gen3_lite
```
to launch the example python code, or other developments. The namespace convention is important to be set based on *gen3lite.launch*.

```
roslaunch base_controller example_moveit.launch
```
The above launch command is another way of launching python files while ensuring the correct name space in the launch file (modifiable) itself.

## Notes on simulation
Use the launch file *gen3lite.launch* in the *launch* folder to run a working simulation. Make sure to disable moveit there if you don't need it.

```
roslaunch base_controller gen3lite.launch
```

This launch file requires *moveit*, you can either disable it or look at [the main Readme file](../README.md) for installation instructions.