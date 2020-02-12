#!/usr/bin/env python

"""
    Riley Tallman
    CSE 598 Perception in Robotics
    1/23/2020
    Description: This is a ROS node written in python that instructs
    the turtlesim_node from the turtlesim package to draw a capital 
    'T' to the screen. It initializes a node and then publishes Twist
    messages on the topic 'turtle1/cmd_vel' to draw the 'T'.
"""

from turtlebot3_controller import turtlebot3_controller


def moveAround(controller):

    controller.goToPoint(-5,0,0)



if __name__ == '__main__':
    controller = turtlebot3_controller()
    moveAround(controller)
    controller.shutdown()

