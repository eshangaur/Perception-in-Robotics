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
    rospy.init_node('my_initials', anonymous=True)   # start a new node with name 'my_initials'
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=30)  # publisher object
    rate = rospy.Rate(1) # 10hz
    # rospy.sleep(1)

    if not rospy.is_shutdown():     # make sure ros is running
        log = "Commence drawing %s" % rospy.get_time()
        rospy.loginfo(log)      # log what time we're drawing

        pi = 3.13       # the rotation is oddly close to radians?

        rate.sleep()        # the messages do not publish successfully without these for some reason
        publisher.publish(Twist(Vector3(2,0,0),Vector3(0,0,0)))        # move forward
        rate.sleep()
        

        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(1,0,0),Vector3(0,0,0)))        # move forward
        # rate.sleep()
        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(5,0,0),Vector3(0,0,0)))        # draw the top of the T
        # rate.sleep()
        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(1,0,0),Vector3(0,0,0)))        # move forward
        # rate.sleep()
        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(2,0,0),Vector3(0,0,0)))        # move forward
        # rate.sleep()

        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,-pi/2)))     # rotate 90 degrees to the right
        # rate.sleep()
        # publisher.publish(Twist(Vector3(5,0,0),Vector3(0,0,0)))        # draw the left side
        # rate.sleep()
        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(1,0,0),Vector3(0,0,0)))        # draw the bottom
        # rate.sleep()
        # publisher.publish(Twist(Vector3(0,0,0),Vector3(0,0,pi/2)))     # rotate 90 degrees to the left
        # rate.sleep()
        # publisher.publish(Twist(Vector3(5,0,0),Vector3(0,0,0)))        # draw the right side

        

if __name__ == '__main__':
    try:
        drawT()
    except rospy.ROSInterruptException:
        pass
