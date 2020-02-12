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

    # 90 points towards +y 
    controller.goToPoint(5,-3,-180)       # gets to the point
    controller.goToPoint(0,-2,-90)          # success!
    controller.goToPoint(1,0,0)
    controller.goToPoint(4.1,0,80)         # enter the room with a walking guy
    controller.goToPoint(5,4,-45)
    controller.goToPoint(6,3,-90)        # starts to move in a figure 8
    controller.goToPoint(6,0,180)
    controller.goToPoint(-4,0,0)
    controller.goToPoint(-1,0,90)
    controller.goToPoint(-1,5,0)
    controller.goToPoint(1,5,90)
    controller.goToPoint(2,5,-90)
    controller.goToPoint(0,5,-90)



if __name__ == '__main__':
    controller = turtlebot3_controller()
    moveAround(controller)
    controller.shutdown()


