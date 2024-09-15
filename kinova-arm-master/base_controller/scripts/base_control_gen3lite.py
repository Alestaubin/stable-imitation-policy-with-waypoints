#!/usr/bin/env python

import rospy
import math
import rosparam
import argparse

from std_msgs.msg import Float64MultiArray

class BaseController:
    def __init__(self, activate: bool = False):

        # read config files
        self.is_activated = activate
        self.__config_file_address = f'../config/basic_config.yaml'
        self.__configs = rosparam.load_file(self.__config_file_address,
                                            default_namespace="base_controller")[0][0]

        # display the config with a demonstration for try/catch structures
        try:
            rospy.loginfo(f'Publishing to: {self.__configs["joint_position_controller_topic"]}')
        except KeyError:
            rospy.logwarn('Error reading configuration: joint_position_controller_topic')

        # create a publisher, this is where you should publish messages to joint controllers
        self.__velocity_publisher = rospy.Publisher(self.__configs["joint_position_controller_topic"],
                                                    Float64MultiArray, queue_size=10)

def main():
    rospy.init_node("base_controller", anonymous=True)
    bc_module = BaseController(True)

    rate = rospy.Rate(200)

    # main control loop
    while not rospy.is_shutdown():
        if not bc_module.is_activated:
            rospy.loginfo('Base controller is not activated.')
            rate.sleep()
            continue

        rate.sleep()
    rospy.signal_shutdown("Controller served its purpose.")


if __name__ == '__main__':
    try:
        main()
    except rospy.exceptions.ROSInterruptException:
        rospy.logerr(f'Shutting down peacefully due to a user interrupt.')
    else:
        rospy.logerr(f'Shutting down...')