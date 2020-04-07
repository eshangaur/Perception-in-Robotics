#!/usr/bin/env python

'''
	This file contains a class for converting ROS images to
	OpenCV images. It was retrieved from
	http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
	on April 6th 2020.
'''


from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

	def __init__(self):
		print("ROS_to_OpenCV node initialized")
		self.image_pub = rospy.Publisher("CV_image",Image, queue_size=5)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw",Image,self.callback)

	def callback(self,ros_image):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			print(e)

		# print(type(cv_image))
		(rows,cols,channels) = cv_image.shape
		if cols > 60 and rows > 60 :
			cv2.circle(cv_image, (50,50), 10, 255)

		cv2.imshow("Image window", cv_image)
		cv2.waitKey(3)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

def main(args):
	rospy.init_node('ROS_img_to_CV', anonymous=False)  # we only need one of these nodes so make anonymous=False
	ic = image_converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	# cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
