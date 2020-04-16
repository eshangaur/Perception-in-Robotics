#!/usr/bin/env python

'''
	This file contains a class for converting ROS images to
	OpenCV images. It was retrieved from
	http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
	on April 6th 2020.


	How to run this file:
	1. cd ~/Documents/GitHub/CSE598-Perception-In-Robotics/project_3b/3b_ws/
	2. source devel/setup.bash
	3. roslaunch pursuit_evasion robot_amcl.launch map_file:=/home/ch/Documents/GitHub/CSE598-Perception-In-Robotics/project_3b/3b_ws/src/pursuit_evasion/maps/task1map.yaml
	4. rosrun pursuit_evasion person_follower.py
'''


from __future__ import print_function


import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError

MODELS_PATH = '/home/ch/Documents/GitHub/CSE598-Perception-In-Robotics/project_3b/3b_ws/src/pursuit_evasion/src/models'

import os
import sys
import numpy as np
import tensorflow as tf


sys.path.append(os.path.join(MODELS_PATH, 'research'))
sys.path.append(os.path.join(MODELS_PATH, 'research/slim'))
# print(sys.path)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class person_follower:

	def __init__(self):
		print('ROS_OpenCV_bridge initialized')
		self.image_pub = rospy.Publisher("/CV_image", Image, queue_size=5)
		self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=5)
		self.bridge = CvBridge()


		# >>> TF stuff >>>
		model = tf.saved_model.load(os.path.join(MODELS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29/saved_model/'))
		self.detection_model = model.signatures['serving_default']
		print('Vision Model Loaded')

		# >>> Patches >>>
		# patch tf1 into `utils.ops`
		utils_ops.tf = tf.compat.v1

		# Patch the location of gfile
		tf.gfile = tf.io.gfile

		# >>> Loading Label Map >>>
		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join(MODELS_PATH, 'research/object_detection/data/mscoco_label_map.pbtxt')
		self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


		image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)

		print("Subscribed to image_raw")


	'''
		This methdod performs these steps:
		1. Converts a ROS Image to a CV image
		2. runs an object detection inference on the image
		3. Draws the inference on the image
		4. publishes the drawn inference
		5. Follows the human
	'''
	def callback(self, ros_image):
		print('!callback initiated')
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			print(e)
		# print(type(cv_image))     # <type 'numpy.ndarray'>

		output_dict = self.run_inference_for_single_image(cv_image)
		self.draw_output(cv_image, output_dict)
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(3)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

		# >>> Follow the person >>>
		self.follow_person(output_dict)


	'''
		returns a dictionary containing the bounding boxes of detected objects in normalized coordinates.
		{
			u'detection_classes': array([1]), 
			u'detection_boxes': array([[3.0305982e-04, 3.1669766e-01, 6.3187075e-01, 6.9868499e-01]], dtype=float32), 
			u'detection_scores': array([0.96191406], dtype=float32), 
			'num_detections': 1
		}
	'''
	def run_inference_for_single_image(self, image):
		image = np.asarray(image)
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis, ...]

		# Run inference
		output_dict = self.detection_model(input_tensor)

		# All outputs are batches tensors.
		# Convert to numpy arrays, and take index [0] to remove the batch dimension.
		# We're only interested in the first num_detections.
		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key: value[0, :num_detections].numpy()
						for key, value in output_dict.items()}
		output_dict['num_detections'] = num_detections

		# detection_classes should be ints.
		output_dict['detection_classes'] = output_dict['detection_classes'].astype(
			np.int64)

		# Handle models with masks:
		if 'detection_masks' in output_dict:
			# Reframe the the bbox mask to the image size.
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					output_dict['detection_masks'], output_dict['detection_boxes'],
					image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
											tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

		return output_dict


	'''
		this method will directly draw the output of an inference on the image that is passed in
		nothing is returned.
	'''
	def draw_output(self, np_img, output_dict):
		vis_util.visualize_boxes_and_labels_on_image_array(
			np_img,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			self.category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)

	
	def follow_person(self, output_dict):
		'''
			only publish when we find a person
		'''
		move_cmd = Twist()		# all values are 0.0 by default

		index = -1
		for i in range(len(output_dict['detection_classes'])):
			if (output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.4):		# if we detected a person
				index = i	# keep the index
				break		# break the for loop

		if (index > -1):   # if we found a person
			box = output_dict['detection_boxes'][index]
			box_center = (box[1] + box[3])/2.0

			# 0.872 radians is 50 degrees
			move_cmd.angular.z = -1.01 * (box_center - 0.5)	# subtract 0.5 so that it is a value between -0.5 and 0.5
			move_cmd.linear.x = 0.21

			debug = False
			debug = True
			if(debug):
				print('box_center: ', box_center)
				print('angular.z: {:.2f}'.format(move_cmd.angular.z))
					
			# >>> publish the movement speeds >>>
			self.cmd_vel_pub.publish(move_cmd)

				

def main(args):
	rospy.init_node('person_follower', anonymous=False)  # we only need one of these nodes so make anonymous=False
	pf = person_follower()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
