#!/usr/bin/env python

"""
    Riley Tallman
    CSE 598 Perception in Robotics
    1/23/2020
    Description: This is a ROS node written in python that instructs
    the turtlebot3 to draw a 'T' to the screen. After moving to a 
    separate room. It initializes a node and then publishes Twist
    messages on the topic 'turtle1/cmd_vel' to draw the 'T'.
"""

from turtlebot3_controller import turtlebot3_controller


def moveAround(controller):

    # 90 points towards +y 
    controller.goToPoint(0,-2,90)
    controller.goToPoint(0,0,-180)
    controller.goToPoint(-3,0,90)
    controller.goToPoint(-3,3,-180)

    # draw a 'T'
    controller.goToPoint(-6,3,90)
    controller.goToPoint(-6,6,-90)
    controller.goToPoint(-6,0,-90)



if __name__ == '__main__':
    controller = turtlebot3_controller()
    moveAround(controller)
    controller.shutdown()


