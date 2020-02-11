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

import rospy
from geometry_msgs.msg import Twist, Vector3


def drawT():
    rospy.init_node('traverse_floor', anonymous=True)   # start a new node with name 'my_initials'
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=30)  # publisher object
    rate = rospy.Rate(100) # 1000hz
    # rospy.sleep(1)

    if rospy.is_shutdown():     # make sure ros is running
        import sys
        sys.exit()  # exit program.

    rate.sleep()
    startTime = rospy.get_time()
    log = "Commence drawing %s" % startTime
    rospy.loginfo(log)      # log the time

    while(rospy.get_time() < (startTime + 3.0)):      # for 3 seconds
        publisher.publish(Twist(Vector3(.5,0,0),Vector3(0,0,0)))        # move forward
        rate.sleep()

    # twist = Twist(Vector3(.3,0,0),Vector3(0,0,0))    # start with moving forward

    # while(not rospy.is_shutdown()):     # repeatedly publish whatever direction and angle we want, depending on the time


    #     if(rospy.get_time() > 5):
    #         twist = Twist(Vector3(.3,0,0),Vector3(0,0,.1))

    #     publisher.publish(twist)

    # publisher.publish(Twist(Vector3(.2,0,0),Vector3(0,0,0)))        # move forward
    # rospy.sleep(2)

    # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,0)))        # stop
    


if __name__ == '__main__':
    try:
        drawT()
    except rospy.ROSInterruptException:
        pass
