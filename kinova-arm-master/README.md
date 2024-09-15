# Kinova Gen3 Lite

## Instructions to install MoveIt

The simplest way to install MoveIt is from pre-built binaries (Debian):

```bash
sudo apt install ros-<distro>-moveit
```

E.g., distro=noetic

You can also add it to catkin workspace and build it yourself (the link might have changed).

```bash
sudo apt install python3-wstool python3-catkin-tools python3-rosdep


wstool init src
wstool merge -t src https://raw.githubusercontent.com/ros-planning/moveit/master/moveit.rosinstall
wstool update -t src
rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO}
catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release

```

## Instructions to install Conan

Just follow the instruction sequence below. Conan is necessary to run some ros_kortex examples.

```bash
    sudo python3 -m pip install conan==1.59
    conan config set general.revisions_enabled=1
    conan profile new default --detect > /dev/null
    conan profile update settings.compiler.libcxx=libstdc++11 default
```
